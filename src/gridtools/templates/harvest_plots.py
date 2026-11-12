# ruff: noqa: RUF001
"""Contact sheets of harvested long chirps for review.

Needs the optional ``plot`` extra (matplotlib), imported lazily. Every
harvested template gets one cell: a spectrogram cutout of the combined
electrode signal with the extracted frequency overlaid (white: inside the
template support, grey: outside), and below it the final tapered template
(baseline-relative). The cell title carries the ``harvest_id`` used in
``review.csv``, so a bad template is vetoed by setting ``accept`` to False
in that row.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from gridtools.templates.harvest import (
    HARVEST_DELTA_FILE,
    HARVEST_METADATA_FILE,
    HARVEST_TRACES_FILE,
    REVIEW_FLAGS,
)

log = logging.getLogger(__name__)

Frame = Any


def load_harvest_traces(harvest_dir: Path) -> dict[str, Any]:
    """Read ``traces.npz`` written by :func:`~gridtools.templates.harvest.run_harvest`."""
    with np.load(harvest_dir / HARVEST_TRACES_FILE) as data:
        return {name: data[name] for name in data.files}


def _trace(traces: dict[str, Any], name: str, i: int) -> npt.NDArray[np.float64]:
    offsets = traces["offsets"]
    return np.asarray(traces[name][offsets[i] : offsets[i + 1]], dtype=np.float64)


def write_contact_sheets(
    out_dir: Path | str,
    *,
    per_page: int = 20,
    columns: int = 5,
    pad_s: float = 0.25,
    dpi: int = 80,
) -> list[Path]:
    """Draw every harvested template on numbered contact-sheet pages.

    Parameters
    ----------
    out_dir
        Harvest output directory (with ``harvest/``).
    per_page, columns
        Cells per page and per row.
    pad_s
        Context shown before the support start and after its end.
    dpi
        Resolution of the PNG pages.

    Returns
    -------
    list of pathlib.Path
        The written pages (``contact_sheets/sheet_NN.png``).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    out_dir = Path(out_dir)
    harvest_dir = out_dir / "harvest"
    meta: Frame = pd.read_parquet(harvest_dir / HARVEST_METADATA_FILE)
    delta = np.load(harvest_dir / HARVEST_DELTA_FILE, mmap_mode="r")
    traces = load_harvest_traces(harvest_dir)
    fs = float(traces["samplerate_hz"])
    sheet_dir = out_dir / "contact_sheets"
    sheet_dir.mkdir(exist_ok=True)
    for old in sheet_dir.glob("sheet_*.png"):
        old.unlink()
    n = len(meta)
    centre = delta.shape[1] // 2 if n else 0
    pages: list[Path] = []
    rows = int(np.ceil(per_page / columns))
    for page, first in enumerate(range(0, n, per_page)):
        fig = plt.figure(figsize=(4.2 * columns, 4.6 * rows))
        grid = fig.add_gridspec(2 * rows, columns, height_ratios=[1.6, 1.0] * rows)
        for cell, i in enumerate(range(first, min(first + per_page, n))):
            r, c = divmod(cell, columns)
            top = fig.add_subplot(grid[2 * r, c])
            bottom = fig.add_subplot(grid[2 * r + 1, c])
            row = meta.iloc[i]
            signal = _trace(traces, "signal", i)
            freq = _trace(traces, "smooth_freq", i)
            start = float(traces["window_start_s"][i])
            t = start + np.arange(signal.size) / fs
            align = float(row["align_abs_s"])
            lo = align + float(row["support_start_s"]) - pad_s
            hi = align + float(row["support_stop_s"]) + pad_s
            f, tt, power = spectrogram(signal, fs=fs, nperseg=512, noverlap=512 - 40)
            db = 10 * np.log10(power + 1e-30)
            baseline = float(row["baseline_hz"])
            fmax = baseline + float(row["excursion_hz"]) + 120
            band = (f >= baseline - 120) & (f <= fmax)
            shown = db[band]
            top.pcolormesh(
                tt + start,
                f[band],
                shown,
                shading="auto",
                cmap="magma",
                vmin=float(np.percentile(shown, 40)),
                vmax=float(shown.max()),
                rasterized=True,
            )
            inside = (t >= align + float(row["support_start_s"])) & (
                t <= align + float(row["support_stop_s"])
            )
            top.plot(t, np.where(inside, np.nan, freq), color="0.7", lw=0.5)
            top.plot(t, np.where(inside, freq, np.nan), color="w", lw=0.7)
            top.set_xlim(lo, hi)
            top.set_ylim(baseline - 120, fmax)
            flags = " ".join(f for f in REVIEW_FLAGS if f in row and bool(row[f]))
            top.set_title(
                f"#{int(row['harvest_id'])} {row['recording'][:10]} "
                f"t={float(row['onset_10_abs_s']):.2f} s "
                f"{row['individual'] or 'track ' + str(int(row['emitter_track_id']))}\n"
                f"{float(row['excursion_hz']):.0f} Hz, "
                f"{1e3 * float(row['duration_10_s']):.0f} ms"
                + (f"  [{flags}]" if flags else ""),
                fontsize=7,
                loc="left",
                color="C3" if flags else "k",
            )
            top.tick_params(labelsize=6)
            tau = (np.arange(delta.shape[1]) - centre) / fs
            bottom.plot(tau, delta[i], color="C0", lw=0.8)
            bottom.axhline(0, color="0.6", lw=0.4)
            bottom.set_xlim(
                float(row["support_start_s"]) - pad_s,
                float(row["support_stop_s"]) + pad_s,
            )
            bottom.tick_params(labelsize=6)
            bottom.set_ylabel("Δf (Hz)", fontsize=6)
        fig.suptitle(
            f"Harvested long chirps, page {page + 1} (cells {first}–"
            f"{min(first + per_page, n) - 1}); top: spectrogram, extracted EODf "
            "(white: template support); bottom: tapered template vs time from "
            "alignment point (s)",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        path = sheet_dir / f"sheet_{page + 1:02d}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        pages.append(path)
    log.info("%d contact sheet pages in %s", len(pages), sheet_dir)
    return pages


def write_rejection_examples(
    out_dir: Path | str,
    recordings_root: Path | str,
    *,
    per_reason: int = 6,
    seed: int = 0,
) -> Path | None:
    """Spectrograms of a few rejected candidates per reason (for the report).

    Parameters
    ----------
    out_dir
        Harvest output directory (with ``candidates.csv``).
    recordings_root
        Directory holding the recordings (read only).
    per_reason
        Examples per rejection reason.
    seed
        Seed for picking the examples.

    Returns
    -------
    pathlib.Path or None
        ``rejections.png``, or None without rejections.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    from gridtools.templates.harvest import open_recording_lenient

    out_dir = Path(out_dir)
    records: Frame = pd.read_csv(out_dir / "candidates.csv")
    rejected = records[records["status"] == "rejected"]
    if rejected.empty:
        return None
    rng = np.random.default_rng(seed)
    reasons = sorted(rejected["reason"].unique())
    fig, axes = plt.subplots(
        len(reasons),
        per_reason,
        figsize=(3.2 * per_reason, 2.6 * len(reasons)),
        squeeze=False,
    )
    opened: dict[str, Any] = {}
    for r, reason in enumerate(reasons):
        group = rejected[rejected["reason"] == reason]
        picks = group.iloc[
            np.sort(rng.choice(len(group), min(per_reason, len(group)), replace=False))
        ]
        for c in range(per_reason):
            ax = axes[r, c]
            if c >= len(picks):
                ax.axis("off")
                continue
            row = picks.iloc[c]
            name = str(row["recording"])
            if name not in opened:
                opened[name] = open_recording_lenient(Path(recordings_root) / name)[0]
            rec = opened[name]
            t0 = float(row["box_t_start_s"]) - 0.2
            t1 = float(row["box_t_end_s"]) + 0.2
            channel = (
                int(row["best_channel"])
                if "best_channel" in row and np.isfinite(row["best_channel"])
                else 0
            )
            x = rec.read_time(max(t0, 0.0), t1, [channel])[:, 0].astype(np.float64)
            f, tt, power = spectrogram(
                x, fs=rec.samplerate_hz, nperseg=1024, noverlap=984
            )
            band = (f > 500) & (f < 1700)
            ax.pcolormesh(
                tt + max(t0, 0.0),
                f[band],
                10 * np.log10(power[band] + 1e-30),
                shading="auto",
                cmap="magma",
                rasterized=True,
            )
            ax.set_title(f"{reason}\n{name[:10]} t={t0 + 0.2:.1f}", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.tight_layout()
    path = out_dir / "rejections.png"
    fig.savefig(path, dpi=70)
    plt.close(fig)
    return path
