"""Report of a background library: inventory, levels, SNRs and demo figures.

:func:`write_report` writes ``report.md`` and ``figures/*.png`` into the
library directory (needs the ``plot`` extra). The real-fish SNRs come from
the manifest (same definitions as :func:`gridtools.background.scale_for_snr`),
the demo figures add one simulated chirping fish to real segments at three
SNRs.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gridtools.background.levels import (
    SnrDistribution,
    fish_signal_level,
    scale_for_snr,
)
from gridtools.background.library import BackgroundLibrary, read_manifest
from gridtools.background.review import spectrogram_power

__all__ = ["demo_mixture", "write_report"]

log = logging.getLogger(__name__)


def _q(values: np.ndarray, digits: int = 1) -> str:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "-"
    q05, q50, q95 = np.percentile(v, [5, 50, 95])
    return f"{q50:.{digits}f} ({q05:.{digits}f} to {q95:.{digits}f})"


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    """A column as a float array (NaN for missing values)."""
    return np.asarray(pd.to_numeric(df[name], errors="coerce"), dtype=float)


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def demo_mixture(
    segment_data: np.ndarray,
    samplerate_hz: float,
    positions_m: np.ndarray,
    level: float,
    snr_db: float,
    *,
    eodf_hz: float,
    duration_s: float = 10.0,
) -> np.ndarray:
    """A simulated fish with two chirps added to a background at ``snr_db``.

    The fish sits at the grid centre, heading along x, 10 cm above the
    electrode plane, EOD amplitude 1 (scene unit); chirps of +150 Hz / 20 ms
    at 3 s and +60 Hz / 12 ms at 7 s with a 30 % amplitude dip.

    Returns
    -------
    numpy.ndarray, shape (n, E)
    """
    from gridtools.sim.eod import APTERONOTUS_LEPTORHYNCHUS, synthesize_eod
    from gridtools.sim.field import dipole_gain

    n = round(duration_s * samplerate_hz)
    t = np.arange(n) / samplerate_hz
    excursion = np.zeros(n)
    dip = np.zeros(n)
    for centre, height, width in ((3.0, 150.0, 0.020), (7.0, 60.0, 0.012)):
        shape = np.exp(-0.5 * ((t - centre) / (width / 2.355)) ** 2)
        excursion += height * shape
        dip += 0.3 * shape
    eod = synthesize_eod(
        eodf_hz + excursion, 1.0 - dip, samplerate_hz, APTERONOTUS_LEPTORHYNCHUS
    )
    centre = positions_m.mean(axis=0)
    gains = dipole_gain(
        centre[None, :] + np.array([[0.05, 0.02]]),
        np.array([0.0]),
        positions_m,
        fish_depth_m=0.1,
        electrode_depth_m=0.0,
    )[0]
    fish = eod[:, None].astype(np.float64) * gains[None, :]
    fundamental = APTERONOTUS_LEPTORHYNCHUS.amplitudes[0] * gains
    signal = fish_signal_level([fundamental[None, :]])
    scale = scale_for_snr(level, signal, snr_db)
    return fish + scale * np.asarray(segment_data[:n], dtype=np.float64)


def _demo_figure(
    library: BackgroundLibrary, row: dict[str, Any], out: Path, snrs: tuple[float, ...]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    segment = library.segment(str(row["segment_id"]))
    fs = segment.samplerate_hz
    eodf = [f for f in row.get("fish_eodf_hz", []) if f is not None]
    target = 950.0 if not eodf else float(max(eodf)) + 80.0
    if target > 1300:
        target = float(min(eodf)) - 80.0
    raw = segment.read(0, round(10.0 * fs)).astype(np.float64)
    panels = [("real background (unscaled)", raw)]
    for snr in snrs:
        mix = demo_mixture(
            segment.data,
            fs,
            segment.geometry.positions_m,
            segment.level,
            snr,
            eodf_hz=target,
        )
        panels.append(
            (f"+ simulated fish at {snr:.0f} dB SNR (chirps at 3 s, 7 s)", mix)
        )
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(12, 2.3 * len(panels)), sharex=True
    )
    for ax, (title, x) in zip(axes, panels, strict=True):
        times, freqs, power = spectrogram_power(x, fs)
        db = 10 * np.log10(np.maximum(power, 1e-300))
        floor = float(np.median(db[(freqs > 500) & (freqs < 1500)]))
        ax.imshow(
            db,
            origin="lower",
            aspect="auto",
            extent=(times[0], times[-1], freqs[0], freqs[-1]),
            cmap="magma",
            vmin=floor - 5,
            vmax=floor + 60,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel("Hz")
    axes[-1].set_xlabel("s")
    fig.suptitle(
        f"{row['segment_id']} ({row['dataset']}; simulated fish at {target:.0f} Hz)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def _level_figure(manifest: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted(manifest["dataset"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for d in datasets:
        sub = pd.DataFrame(manifest.loc[(manifest["dataset"] == d).to_numpy()])
        axes[0].hist(_col(sub, "noise_psd_db"), bins=30, histtype="step", label=d)
        axes[1].hist(
            np.log10(np.maximum(_col(sub, "hum_amplitude"), 1e-12)),
            bins=30,
            histtype="step",
            label=d,
        )
        axes[2].hist(
            np.log10(1 + _col(sub, "click_rate_per_s") * 3600),
            bins=30,
            histtype="step",
            label=d,
        )
    axes[0].set_xlabel("noise floor N0 (dB re 1 unit^2/Hz; units differ by source)")
    axes[1].set_xlabel("log10 50-Hz hum amplitude (unit)")
    axes[2].set_xlabel("log10(1 + clicks per hour)")
    for ax in axes:
        ax.set_ylabel("segments")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def _snr_figure(
    real_snr: np.ndarray,
    level_diff: np.ndarray,
    implied: np.ndarray,
    dist: SnrDistribution,
    out: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    draws = np.asarray(dist.sample(rng, 100_000))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    bins = np.arange(-5, 95, 2.5)
    axes[0].hist(draws, bins=bins, density=True, alpha=0.5, label="default target SNR")
    axes[0].hist(
        real_snr, bins=bins, density=True, histtype="step", lw=2, label="real fish"
    )
    axes[0].set_xlabel(f"SNR (dB, strongest electrode, {7.32:.2f}-Hz bin)")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    bins = np.arange(-60, 95, 2.5)
    axes[1].hist(
        implied,
        bins=bins,
        density=True,
        alpha=0.5,
        label="loudest real fish - simulated fish\n(default SNRs on fish segments)",
    )
    axes[1].hist(
        level_diff,
        bins=bins,
        density=True,
        histtype="step",
        lw=2,
        label="louder - quieter real fish\n(same segment)",
    )
    axes[1].set_xlabel("level difference (dB)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def write_report(
    root: str | Path, *, seed: int = 0, n_demo_per_dataset: int = 2
) -> Path:
    """Write ``report.md`` and figures for a built library.

    Parameters
    ----------
    root : path
        Library directory.
    seed : int
        Seed of the demo choices and SNR draws.
    n_demo_per_dataset : int
        Demo segments per source dataset.

    Returns
    -------
    pathlib.Path
        The report.
    """
    root = Path(root)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    manifest = read_manifest(root / "manifest.parquet")
    if "accept" not in manifest.columns:
        manifest["accept"] = True
    library = BackgroundLibrary.load(root, include_rejected=True)
    recordings = (
        pd.read_csv(root / "recordings.csv")
        if (root / "recordings.csv").exists()
        else pd.DataFrame()
    )
    meta = json.loads((root / "library.json").read_text())
    dist = SnrDistribution()
    rng = np.random.default_rng(seed)

    lines: list[str] = []
    add = lines.append
    add("# Background library\n")
    add(
        f"`{root}`: {len(manifest)} segments, "
        f"{meta.get('size_bytes', _dir_size(root / 'segments')) / 1e9:.1f} GB. "
        "Built by `python -m gridtools.background build` (provenance per build in "
        "`library.json`). Definitions: `gridtools.background.levels`; selection: "
        "`gridtools.background.select`, `gridtools.background.field`; review "
        "PNGs and veto table: `review/index.html`, `review.csv`.\n"
    )

    # inventory
    add("## Inventory\n")
    add(
        "| source | recordings / deployments | used | segments (fish / fish-free) | "
        "screened | accepted | duration | size |"
    )
    add("|---|---|---|---|---|---|---|---|")
    for dataset, sub in manifest.groupby("dataset", sort=True):
        recs = (
            recordings[recordings["dataset"] == dataset]
            if len(recordings)
            else pd.DataFrame()
        )
        used = sub["recording"].nunique()
        classes = np.asarray(sub["segment_class"], dtype=str)
        n_fish = int(np.sum(classes == "fish"))
        n_free = int(np.sum(classes == "fish_free"))
        size = sum(_dir_size(root / str(p)) for p in sub["path"])
        add(
            f"| {dataset} | {len(recs)} | {used} | {n_fish} / {n_free} | "
            f"{int(np.sum(_col(sub, 'screened')))} | {int(np.sum(_col(sub, 'accept')))} | "
            f"{np.sum(_col(sub, 'duration_s')) / 60:.0f} min | {size / 1e9:.1f} GB |"
        )
    add("")
    if len(recordings) and "skipped" in recordings.columns:
        skipped = pd.DataFrame(recordings.loc[recordings["skipped"].notna().to_numpy()])
        if len(skipped):
            add(
                "Skipped: "
                + "; ".join(
                    f"{r['dataset']}/{r['recording']} ({r['skipped']})"
                    for r in skipped.to_dict("records")
                )
                + ".\n"
            )

    # field scan
    field = (
        pd.DataFrame(
            recordings.loc[
                recordings["dataset"].astype(str).str.startswith("field_").to_numpy()
            ]
        )
        if len(recordings)
        else pd.DataFrame()
    )
    if len(field):
        add("## Field deployments: fish-free scan\n")
        add(
            "Windows of segment length + 2 x margin, spread evenly over each "
            "deployment, tested for spectral lines (8 dB above the running median "
            "on any electrode, 300-2000 Hz). `scanned` is audio actually tested, "
            "`span` the deployments' total duration.\n"
        )
        add(
            "| source | site | deployments | span (h) | scanned (h) | windows | fish-free | kept |"
        )
        add("|---|---|---|---|---|---|---|---|")
        for key, group in field.groupby(["dataset", "site"], sort=True):
            dataset, site = tuple(key)  # type: ignore[misc]
            sub = pd.DataFrame(group)
            total = {
                c: float(np.nansum(_col(sub, c))) if c in sub.columns else 0.0
                for c in (
                    "span_h",
                    "scanned_h",
                    "n_probes",
                    "n_fish_free",
                    "n_segments",
                )
            }
            add(
                f"| {dataset} | {site} | {len(sub)} | {total['span_h']:.0f} | "
                f"{total['scanned_h']:.2f} | {total['n_probes']:.0f} | "
                f"{total['n_fish_free']:.0f} | {total['n_segments']:.0f} |"
            )
        add("")

    # levels
    _level_figure(manifest, figures / "levels.png")
    add("## Levels\n")
    add(
        "Background level = noise floor N0 (median over electrodes of the "
        "median non-tonal PSD in 500-1500 Hz) x 7.32 Hz (one bin of the v1 "
        "4096-point spectrogram). Units: competitions in ADC volts, field "
        "loggers in int16 full scale (not comparable across sources; the SNR "
        "scaling absorbs the unit).\n"
    )
    add(
        "| source | N0 (dB re unit^2/Hz) | electrode spread (dB) | 50-Hz hum (unit) | clicks/h |"
    )
    add("|---|---|---|---|---|")
    for dataset, sub in manifest.groupby("dataset", sort=True):
        add(
            f"| {dataset} | {_q(sub['noise_psd_db'].to_numpy())} | "
            f"{_q(sub['noise_psd_spread_db'].to_numpy())} | "
            f"{_q(sub['hum_amplitude'].to_numpy(), 7)} | "
            f"{_q(sub['click_rate_per_s'].to_numpy() * 3600, 0)} |"
        )
    add("\n![levels](figures/levels.png)\n")

    # real SNR and the default distribution
    fish = pd.DataFrame(manifest.loc[(manifest["segment_class"] == "fish").to_numpy()])
    real = np.array(
        [
            s
            for lst in fish["fish_snr_db"]
            for s in lst
            if s is not None and math.isfinite(s)
        ]
    )
    diffs = _col(fish, "real_level_diff_db")
    diffs = diffs[np.isfinite(diffs)]
    real_max = _col(fish, "real_snr_max_db")
    real_max = real_max[np.isfinite(real_max)]
    targets = np.asarray(dist.sample(rng, 50 * max(1, real_max.size)))
    implied = np.repeat(real_max, 50) - targets if real_max.size else np.empty(0)
    _snr_figure(real, diffs, implied, dist, figures / "snr.png")
    add("## SNR\n")
    add(
        "SNR = power of a fish's fundamental on its strongest electrode "
        "(median over time) / background level, in dB. Real fish: measured "
        "along their tracks in every fish segment (Parseval over the line's "
        "main lobe, floor subtracted).\n"
    )
    add(
        "| source | real fish (fish x segments) | SNR (dB) | louder - quieter fish (dB) |"
    )
    add("|---|---|---|---|")
    for dataset, sub in fish.groupby("dataset", sort=True):
        vals = np.array(
            [
                s
                for lst in sub["fish_snr_db"]
                for s in lst
                if s is not None and math.isfinite(s)
            ]
        )
        add(
            f"| {dataset} | {vals.size} | {_q(vals)} | "
            f"{_q(_col(pd.DataFrame(sub), 'real_level_diff_db'))} |"
        )
    add(f"| all | {real.size} | {_q(real)} | {_q(diffs)} |\n")
    lo, hi = dist.range_db
    p5 = float(np.percentile(real, 5)) if real.size else math.nan
    add(
        f"Default `SnrDistribution`: with probability {dist.hard_fraction:g} "
        f"uniform in {dist.hard_db[0]:g}-{dist.hard_db[1]:g} dB (harder than "
        f"real: below the real 5 % quantile, {p5:.1f} dB), otherwise uniform in "
        f"{dist.typical_db[0]:g}-{dist.typical_db[1]:g} dB (covers the real "
        f"range). Fraction of real fish inside [{lo:g}, {hi:g}] dB: "
        f"{np.mean((real >= lo) & (real <= hi)) if real.size else math.nan:.2f}.\n"
    )
    if implied.size:
        ref95 = float(np.percentile(diffs, 95)) if diffs.size else math.nan
        add(
            "Real fish are scaled with their background, so their SNR does not "
            "change and they end up `SNR_real - SNR_target` dB above the "
            "simulated fish. With the default SNRs on fish segments: "
            f"{_q(implied)} dB; {np.mean(implied > ref95):.0%} of draws exceed the "
            f"95 % quantile ({ref95:.1f} dB) of the level difference between two "
            "real fish in the same segment, "
            f"{np.mean(implied > 40):.0%} exceed 40 dB. Hard SNRs therefore pair a "
            "faint simulated fish with much louder real ones, a combination that "
            "is rare in the tanks. Options: condition the target SNR on the "
            "segment (e.g. `SNR_target >= real_snr_max_db - 30`), use fish-free "
            "segments for the hardest SNRs, or raise the floor with synthetic "
            "noise (`other_noise_level`), which lowers real and simulated fish "
            "together.\n"
        )
    add("![snr](figures/snr.png)\n")

    # demos
    add("## Demo: one simulated fish at three SNRs\n")
    snrs = (
        float(np.median(real)) if real.size else 45.0,
        float(dist.hard_db[1]),
        10.0,
    )
    add(
        f"High = real median ({snrs[0]:.0f} dB), typical-hard boundary "
        f"({snrs[1]:.0f} dB), hard (10 dB). Spectrograms are summed over "
        "electrodes (2048-point window at 20 kHz), first 10 s of the segment.\n"
    )
    for _, sub in manifest.groupby("dataset", sort=True):
        picks = rng.choice(
            len(sub), size=min(n_demo_per_dataset, len(sub)), replace=False
        )
        for k in sorted(picks):
            row = sub.iloc[int(k)].to_dict()
            name = f"demo_{row['segment_id']}.png"
            try:
                _demo_figure(library, row, figures / name, snrs)
            except Exception as error:  # report must not fail on one demo
                log.warning("demo %s failed: %s", row["segment_id"], error)
                continue
            add(f"![{row['segment_id']}](figures/{name})\n")

    out = root / "report.md"
    out.write_text("\n".join(lines))
    log.info("report: %s", out)
    return out
