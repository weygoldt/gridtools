"""Review plots of every background segment and the author's veto table.

:func:`write_review` draws one PNG per segment into ``<library>/review/``
(plus thumbnails and one HTML contact sheet per source) and merges
``review.csv``, in which the author can set ``accept`` to False (and add a
``note``) for segments that contain chirps or are unusable otherwise.
Existing ``accept``/``note`` values are never overwritten: rows are merged
by segment id. :class:`gridtools.background.BackgroundLibrary` drops
rejected segments by default, and a rebuild treats them like chirp
detections (their span is avoided and another window is chosen).

The spectrogram is the power summed over electrodes (Hann window of
``nfft`` = 2**round(log2(0.1024 fs)) points, i.e. 2048 at 20 kHz = 102 ms,
9.8 Hz bins, hop nfft / 8 = 12.8 ms). Chirps of *A. leptorhynchus* last
10-30 ms with 50-400 Hz excursions: with a 102-ms window they appear as a
short vertical streak above the fish's band plus a dip of the band, and a
12.8-ms hop puts at least one column on the shortest chirp; 9.8 Hz bins
still separate fish more than about 20 Hz apart (94 % of the time in the
2019/2020 competitions). Each 60-s segment is split into rows of 10 s so a
chirp spans a few pixels instead of a fraction of one. Colours span a fixed
65 dB above the median floor of the source, so levels compare across the
segments of a source.
"""

from __future__ import annotations

import html
import logging
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.fft
from numpy.lib.stride_tricks import sliding_window_view

from gridtools.background.library import read_manifest

__all__ = [
    "REVIEW_COLUMNS",
    "merge_review",
    "plot_segment",
    "read_review",
    "spectrogram_power",
    "write_review",
]

log = logging.getLogger(__name__)

REVIEW_COLUMNS = (
    "segment_id",
    "dataset",
    "recording",
    "start_s",
    "segment_class",
    "screened",
    "n_fish",
    "png",
    "accept",
    "note",
)

_ROW_S = 10.0
_F_MAX_HZ = 2000.0
_DB_SPAN = 65.0


def read_review(root: str | Path) -> pd.DataFrame | None:
    """The author's ``review.csv`` of a library, or None if there is none."""
    path = Path(root) / "review.csv"
    if not path.exists():
        return None
    df = pd.read_csv(
        path, dtype={"segment_id": str, "note": str}, keep_default_na=False
    )
    if "accept" not in df.columns:
        df["accept"] = True
    df["accept"] = df["accept"].map(_truthy)
    if "note" not in df.columns:
        df["note"] = ""
    return df


def _truthy(value: Any) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("", "1", "true", "t", "yes", "y", "1.0"):
        return True
    if text in ("0", "false", "f", "no", "n", "0.0"):
        return False
    msg = f"review.csv: cannot read accept value {value!r}"
    raise ValueError(msg)


def merge_review(root: str | Path, manifest: pd.DataFrame) -> pd.DataFrame:
    """Merge ``review.csv`` with a manifest and write it back (atomically).

    Rows of segments in the manifest are added with ``accept=True``; rows the
    author already has keep their ``accept`` and ``note``; rows of segments
    no longer in the manifest are kept (so a veto survives a rebuild that
    replaced the segment).

    Returns
    -------
    pandas.DataFrame
        The merged review table.
    """
    root = Path(root)
    old = read_review(root)
    new = pd.DataFrame(
        {
            "segment_id": manifest["segment_id"].astype(str),
            "dataset": manifest["dataset"],
            "recording": manifest["recording"],
            "start_s": manifest["start_s"],
            "segment_class": manifest["segment_class"],
            "screened": manifest["screened"],
            "n_fish": manifest["n_fish"],
            "png": [f"review/{s}.png" for s in manifest["segment_id"]],
            "accept": True,
            "note": "",
        }
    )
    if old is not None:
        ids = new["segment_id"].tolist()
        for column in ("accept", "note"):
            known = dict(zip(old["segment_id"], old[column], strict=True))
            new[column] = [
                known.get(i, v) for i, v in zip(ids, new[column], strict=True)
            ]
        gone = pd.DataFrame(old.loc[~old["segment_id"].isin(ids).to_numpy()])
        for column in new.columns:
            if column not in gone.columns:
                gone[column] = pd.NA
        new = pd.concat([new, gone[list(new.columns)]], ignore_index=True)
    new["accept"] = [_truthy(v) for v in new["accept"]]
    new["note"] = ["" if pd.isna(v) else str(v) for v in new["note"]]
    new = pd.DataFrame(new[list(REVIEW_COLUMNS)])
    fd, tmp = tempfile.mkstemp(prefix=".review.", suffix=".csv", dir=root)
    try:
        with os.fdopen(fd, "w", newline="") as f:
            new.to_csv(f, index=False)
        Path(tmp).replace(root / "review.csv")
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return new


def spectrogram_power(
    x: np.ndarray, samplerate_hz: float, f_max_hz: float = _F_MAX_HZ
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Electrode-summed power spectrogram (unit**2/Hz) up to ``f_max_hz``.

    Returns
    -------
    times_s : numpy.ndarray, shape (T,)
        Frame centres.
    freqs_hz : numpy.ndarray, shape (F,)
    power : numpy.ndarray, shape (F, T)
    """
    nfft = int(2 ** round(math.log2(0.1024 * samplerate_hz)))
    hop = nfft // 8
    window = np.hanning(nfft + 1)[:-1].astype(np.float32)
    scale = 2.0 / (samplerate_hz * float(np.sum(window.astype(np.float64) ** 2)))
    n_bins = int(f_max_hz * nfft / samplerate_hz) + 1
    n_frames = 1 + (x.shape[0] - nfft) // hop
    power = np.zeros((n_bins, n_frames))
    for e in range(x.shape[1]):
        col = np.asarray(x[:, e], dtype=np.float32)
        col = col - col.mean()
        frames = sliding_window_view(col, nfft)[::hop][:n_frames]
        spec = np.asarray(scipy.fft.rfft(frames * window, axis=1), dtype=np.complex64)
        spec = spec[:, :n_bins]
        power += (
            scale
            * (spec.real.astype(np.float64) ** 2 + spec.imag.astype(np.float64) ** 2).T
        )
    times = (hop * np.arange(n_frames) + nfft / 2) / samplerate_hz
    freqs = np.arange(n_bins) * samplerate_hz / nfft
    return times, freqs, power


def _clock(seconds: float) -> str:
    s = round(seconds)
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def plot_segment(
    root: Path, row: dict[str, Any], out_png: Path, vmin_db: float
) -> None:
    """Draw one segment's review figure (see module docstring)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gridtools.io import load_tracks, open_recording

    seg_dir = root / str(row["path"])
    rec = open_recording(seg_dir / "background.wav")
    x = rec.read(0, rec.n_samples)
    fs = rec.samplerate_hz
    times, freqs, power = spectrogram_power(x, fs)
    db = 10 * np.log10(np.maximum(power, 1e-300))
    tracks = load_tracks(seg_dir) if (seg_dir / "ident_v.npy").exists() else None
    duration = rec.duration_s
    n_rows = max(1, math.ceil(duration / _ROW_S - 1e-6))
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 1.9 * n_rows + 0.9), squeeze=False)
    colors = [matplotlib.colormaps["tab10"](k) for k in range(10)]
    image = None
    for r, ax in enumerate(axes[:, 0]):
        t0, t1 = r * _ROW_S, min(duration, (r + 1) * _ROW_S)
        sel = (times >= t0) & (times < t1)
        if sel.any():
            image = ax.imshow(
                db[:, sel],
                origin="lower",
                aspect="auto",
                extent=(times[sel][0], times[sel][-1], freqs[0], freqs[-1]),
                cmap="magma",
                vmin=vmin_db,
                vmax=vmin_db + _DB_SPAN,
                interpolation="nearest",
            )
        ax.set_xlim(t0, t0 + _ROW_S)
        ax.set_ylim(0, _F_MAX_HZ)
        ax.set_ylabel("Hz", fontsize=8)
        ax.tick_params(labelsize=7)
        if tracks is not None:
            for k, track in enumerate(tracks):
                f = track.frequency_at(np.array([t0, t1]), max_gap_s=2.0)
                inside = track.freqs_hz[(track.times_s >= t0) & (track.times_s < t1)]
                fallback = float(np.median(inside)) if inside.size else math.nan
                for xpos, value, marker in ((t0, f[0], ">"), (t0 + _ROW_S, f[1], "<")):
                    value = value if np.isfinite(value) else fallback
                    if np.isfinite(value):
                        ax.plot(
                            [xpos],
                            [value],
                            marker=marker,
                            color=colors[k % 10],
                            markersize=7,
                            clip_on=False,
                        )
    axes[-1, 0].set_xlabel("time in segment (s)", fontsize=8)
    fish = ""
    if row.get("n_fish"):
        eodf = ", ".join(
            f"{f:.0f}" for f in row.get("fish_eodf_hz", []) if f is not None
        )
        fish = f" | {int(row['n_fish'])} real fish ({eodf} Hz)"
    elif row.get("segment_class") == "fish_free":
        fish = " | fish-free"
    start = float(row["start_s"])
    fig.suptitle(
        f"{row['segment_id']}\n{row['dataset']} | {row['recording']} | start "
        f"{_clock(start)} ({start:.0f} s) | screened={bool(row['screened'])}{fish} | "
        f"floor {row['noise_psd_db']:.1f} dB, {int(row['n_electrodes'])} el., "
        f"{fs / 1000:g} kHz",
        fontsize=9,
    )
    if image is not None:
        cbar = fig.colorbar(image, ax=axes[:, 0].tolist(), pad=0.01, fraction=0.015)
        cbar.set_label("summed PSD (dB)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    from PIL import Image

    with Image.open(out_png) as im:
        thumb = im.convert("RGB")
        thumb.thumbnail((480, 480))
        thumb.save(out_png.with_name(out_png.stem + ".thumb.jpg"), quality=80)


def _plot_task(args: tuple[Path, dict[str, Any], Path, float]) -> tuple[str, float]:
    t = time.perf_counter()
    root, row, png, vmin = args
    plot_segment(root, row, png, vmin)
    return str(row["segment_id"]), time.perf_counter() - t


def _contact_sheets(root: Path, manifest: pd.DataFrame, review: pd.DataFrame) -> None:
    accept = dict(zip(review["segment_id"], review["accept"], strict=True))
    note = dict(zip(review["segment_id"], review["note"], strict=True))
    pages = []
    for dataset, rows in manifest.groupby("dataset", sort=True):
        cards = []
        for row in rows.to_dict("records"):
            sid = str(row["segment_id"])
            status = "" if accept.get(sid, True) else " <b>REJECTED</b>"
            text = html.escape(
                f"{row['recording']} @ {float(row['start_s']):.0f} s, {row['segment_class']}"
            )
            extra = html.escape(note.get(sid, "") or "")
            cards.append(
                f'<figure><a href="{sid}.png"><img src="{sid}.thumb.jpg" loading="lazy"></a>'
                f"<figcaption>{html.escape(sid)}<br>{text}{status}<br>{extra}</figcaption></figure>"
            )
        page = f"{dataset}.html"
        (root / "review" / page).write_text(
            "<!doctype html><meta charset=utf-8><title>"
            f"{html.escape(str(dataset))}</title><style>body{{font:12px sans-serif}}"
            "figure{display:inline-block;width:480px;margin:4px;vertical-align:top}"
            "img{width:480px}</style>"
            f"<h1>{html.escape(str(dataset))}: {len(rows)} segments</h1>"
            + "".join(cards)
        )
        pages.append((str(dataset), page, len(rows)))
    links = "".join(
        f'<li><a href="{p}">{html.escape(d)}</a> ({n} segments)</li>'
        for d, p, n in pages
    )
    (root / "review" / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Background review</title>"
        "<h1>Background segments</h1><p>Edit <code>review.csv</code> (column "
        "<code>accept</code>, <code>note</code>) to veto segments.</p>"
        f"<ul>{links}</ul>"
    )


def write_review(
    root: str | Path,
    *,
    workers: int = 6,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Plot every segment, write contact sheets and merge ``review.csv``.

    Parameters
    ----------
    root : path
        Library directory.
    workers : int
        Plotting processes (at most 6).
    overwrite : bool
        Redraw PNGs that already exist (default: only missing ones).

    Returns
    -------
    dict
        ``n_segments``, ``n_plotted``, ``wall_s`` and ``cpu_s`` (sum over
        plots).
    """
    root = Path(root)
    t0 = time.perf_counter()
    manifest = read_manifest(root / "manifest.parquet")
    review = merge_review(root, manifest)
    summed = 10 * np.log10(
        manifest["noise_psd"].to_numpy(dtype=float)
        * manifest["n_electrodes"].to_numpy(dtype=float)
    )
    floors = {
        str(d): float(np.median(summed[(manifest["dataset"] == d).to_numpy()]))
        for d in manifest["dataset"].unique()
    }
    tasks = []
    for row in manifest.to_dict("records"):
        png = root / "review" / f"{row['segment_id']}.png"
        if png.exists() and not overwrite:
            continue
        tasks.append((root, row, png, floors[str(row["dataset"])] - 5.0))
    cpu = 0.0
    if tasks:
        (root / "review").mkdir(exist_ok=True)
        with ProcessPoolExecutor(max_workers=max(1, min(workers, 6))) as pool:
            for _, seconds in pool.map(_plot_task, tasks, chunksize=4):
                cpu += seconds
    _contact_sheets(root, manifest, review)
    stats = {
        "n_segments": len(manifest),
        "n_plotted": len(tasks),
        "wall_s": round(time.perf_counter() - t0, 1),
        "cpu_s": round(cpu, 1),
    }
    log.info("review: %s", stats)
    return stats
