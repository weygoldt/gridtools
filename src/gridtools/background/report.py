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


#: Candidate counts of a screened build, per recording (``recordings.csv``).
_COUNTS = (
    "n_grid",
    "n_outside_window",
    "n_gated",
    "n_rejected_other",
    "n_valid",
    "n_fish_segments",
)


def _screen_counts(
    recordings: pd.DataFrame, manifest: pd.DataFrame, screens: list[str]
) -> pd.DataFrame:
    """Per competition recording: why candidates were rejected, what was kept."""
    rows = []
    for r in recordings.to_dict("records"):
        if str(r["dataset"]).startswith("field_"):
            continue
        sub = manifest[
            (manifest["dataset"] == r["dataset"])
            & (manifest["recording"] == r["recording"])
        ]
        skipped = r.get("skipped")
        skipped = (
            ""
            if skipped is None or (isinstance(skipped, float) and math.isnan(skipped))
            else str(skipped)
        )
        status = (
            "used"
            if not skipped
            else "gated"
            if skipped.startswith("recording gate")
            else "skipped"
        )
        row: dict[str, Any] = {
            "dataset": r["dataset"],
            "recording": r["recording"],
            "split_group": f"{r['dataset']}/{r['recording']}",
            "status": status,
            "reason": skipped,
            "duration_h": float(r.get("duration_s", math.nan)) / 3600,
            "window_h": (
                float(r.get("window_stop_s", math.nan))
                - float(r.get("window_start_s", math.nan))
            )
            / 3600,
            "gate_rate_per_h": float(r.get("gate_rate_per_h", math.nan)),
        }
        for c in (
            *_COUNTS,
            *(f"n_rejected_{n}" for n in screens),
            *(f"n_rejected_{n}_any" for n in screens),
            "n_rejected_no_fish",
        ):
            row[c] = _int(r.get(c))
        classes = np.asarray(sub["segment_class"], dtype=str)
        row["segments_fish"] = int(np.sum(classes == "fish"))
        row["segments_fish_free"] = int(np.sum(classes == "fish_free"))
        rows.append(row)
    return pd.DataFrame(rows)


def _int(value: Any) -> int:
    """An integer count; 0 for a missing value."""
    if value is None:
        return 0
    v = float(value)
    return int(v) if math.isfinite(v) else 0


def _counts_table(df: pd.DataFrame, key: str, screens: list[str]) -> list[str]:
    head = (
        f"| {key} | recordings (used / gated / skipped) | candidates | outside window "
        "| gate | other | " + " | ".join(screens) + " | no fish | valid | segments |"
    )
    lines = [head, "|" + "---|" * (9 + len(screens))]
    for name, group in df.groupby(key, sort=True):
        g = pd.DataFrame(group)
        n = {
            str(c): int(np.nansum(_col(g, str(c))))
            for c in g.columns
            if str(c).startswith("n_")
        }
        status = list(np.asarray(g["status"], dtype=str))
        rej = " | ".join(f"{n.get(f'n_rejected_{s}', 0):,}" for s in screens)
        segments = int(
            np.nansum(_col(g, "segments_fish"))
            + np.nansum(_col(g, "segments_fish_free"))
        )
        lines.append(
            f"| {name} | {len(g)} ({status.count('used')} / "
            f"{status.count('gated')} / {status.count('skipped')}) | "
            f"{n.get('n_grid', 0):,} | {n.get('n_outside_window', 0):,} | "
            f"{n.get('n_gated', 0):,} | {n.get('n_rejected_other', 0):,} | {rej} | "
            f"{n.get('n_rejected_no_fish', 0):,} | {n.get('n_valid', 0):,} | "
            f"{segments} |"
        )
    return lines


def _screen_section(
    root: Path,
    manifest: pd.DataFrame,
    recordings: pd.DataFrame,
    build: dict[str, Any],
    splits: dict[str, list[str]],
) -> list[str]:
    """Report section of a build with chirp screens (time window, screens, gate)."""
    config = build["config"]
    screens = [s["name"] for s in config["screens"]]
    gate = config.get("gate")
    lines: list[str] = ["## Chirp screening\n"]
    lines.append(
        "Candidates come only from each recording's time window; a candidate "
        "is rejected if any screen box with at least the screen's score overlaps "
        "it widened by the margin, or if the screen did not cover that span. "
        "Recordings whose window has more gate-screen boxes per covered hour "
        "than the gate allows are dropped entirely. Definitions: "
        "`gridtools.background.screen`.\n"
    )
    lines.append("| dataset | window | segments per recording | grid step |")
    lines.append("|---|---|---|---|")
    base = config["selection"]
    for dataset, window in sorted(config.get("windows", {}).items()):
        sel = {**base, **config.get("selection_by_dataset", {}).get(dataset, {})}
        rule = []
        if window.get("last_s") is not None:
            rule.append(f"last {window['last_s'] / 3600:g} h")
        if window.get("min_start_s"):
            rule.append(f"not before {window['min_start_s'] / 3600:g} h")
        if window.get("min_duration_s"):
            rule.append(f"recordings >= {window['min_duration_s'] / 3600:g} h")
        lines.append(
            f"| {dataset} | {', '.join(rule)} | {sel['segments_per_recording']}"
            f"{' (fill)' if sel.get('fill') else ''} | {sel['step_s']:g} s, "
            f"edges {sel['edge_s']:g} s |"
        )
    lines.append("")
    lines.append(
        "| screen | detector | weights sha256 | stored from | threshold | margin |"
    )
    lines.append("|---|---|---|---|---|---|")
    from gridtools.background.screen import read_screen_meta

    for s in config["screens"]:
        try:
            meta = read_screen_meta(s["root"])
        except FileNotFoundError:
            meta = {}
        lines.append(
            f"| {s['name']} | {meta.get('detector', '?')} | "
            f"`{str(meta.get('weights_sha256', '?'))[:16]}` | "
            f"{meta.get('store_threshold', '?')} | {s['threshold']:g} | "
            f"{s['margin_s']:g} s |"
        )
    lines.append("")
    if gate is not None:
        lines.append(
            f"Gate: `{gate['screen']}` boxes with score >= {gate['threshold']:g}, at most "
            f"{gate['max_rate_per_h']:g} per hour of the window.\n"
        )
    counts = _screen_counts(recordings, manifest, screens)
    counts.to_csv(root / "screen_counts.csv", index=False, float_format="%.6g")
    lines.append(
        "Counts are candidates on each dataset's grid (starts every `step_s`, "
        "so neighbouring candidates overlap); rejections are attributed in "
        "order: window, gate, other rules (labels, vetoes, saved detections), "
        "then the screens in order, then windows without tracked fish. "
        "`segments` are the chosen, non-overlapping segments. Per recording: "
        "`screen_counts.csv` (also `n_rejected_<screen>_any`: every candidate "
        "the screen rejects on its own).\n"
    )
    lines.append("### Per dataset\n")
    lines += _counts_table(counts, "dataset", screens)
    lines.append("")
    for name in screens:
        col = f"n_rejected_{name}_any"
        if col in counts:
            in_window = (
                _col(counts, "n_grid")
                - _col(counts, "n_outside_window")
                - _col(counts, "n_gated")
            )
            lines.append(
                f"`{name}` alone would reject {int(np.nansum(_col(counts, col))):,} "
                f"of the {int(np.nansum(in_window)):,} in-window candidates of used "
                "recordings."
            )
    lines.append("")
    if splits:
        lines.append("### Per scene split\n")
        rows = []
        for split, groups in splits.items():
            sub = pd.DataFrame(
                counts.loc[counts["split_group"].isin(groups).to_numpy()]
            )
            sub["split"] = split
            rows.append(sub)
        lines += _counts_table(pd.concat(rows, ignore_index=True), "split", screens)
        lines.append("")
        lines.append("Split groups without any segment in this library:\n")
        per_group = {
            str(k): int(v)
            for k, v in zip(
                *np.unique(
                    np.asarray(manifest["split_group"], dtype=str), return_counts=True
                ),
                strict=True,
            )
        }
        any_empty = False
        for split, groups in splits.items():
            empty = [g for g in groups if per_group.get(g, 0) == 0]
            if empty:
                any_empty = True
                lines.append(
                    f"- {split} ({len(empty)} of {len(groups)}): "
                    + ", ".join(f"`{g}`" for g in empty)
                )
        if not any_empty:
            lines.append("- none")
        lines.append("")
        lines.append(
            "| split | groups | groups with segments | segments | competition segments |"
        )
        lines.append("|---|---|---|---|---|")
        for split, groups in splits.items():
            in_split = manifest["split_group"].isin(groups).to_numpy()
            datasets = np.asarray(manifest["dataset"], dtype=str)[in_split]
            n_groups = len(
                set(np.asarray(manifest["split_group"], dtype=str)[in_split])
            )
            n_comp = int(np.sum(~np.char.startswith(datasets, "field_")))
            lines.append(
                f"| {split} | {len(groups)} | {n_groups} | {int(in_split.sum())} | "
                f"{n_comp} |"
            )
        lines.append("")
    lines.append("### Per recording (split group)\n")
    lines.append(
        "| recording | status | duration (h) | window (h) | gate rate (/h) | "
        "candidates in window | " + " | ".join(screens) + " | valid | segments |"
    )
    lines.append("|---|---|---|---|---|---|" + "---|" * len(screens) + "---|---|")
    for r in counts.sort_values(["dataset", "recording"]).to_dict("records"):
        in_window = r["n_grid"] - r["n_outside_window"]
        rej = " | ".join(f"{r.get(f'n_rejected_{n}', 0):,}" for n in screens)
        status = r["status"] if r["status"] != "skipped" else f"skipped: {r['reason']}"
        lines.append(
            f"| {r['split_group']} | {status} | {r['duration_h']:.2f} | "
            f"{r['window_h']:.2f} | {r['gate_rate_per_h']:.1f} | {in_window:,} | "
            f"{rej} | {r['n_valid']:,} | "
            f"{r['segments_fish'] + r['segments_fish_free']} |"
        )
    lines.append("")
    gated = pd.DataFrame(
        counts.loc[(np.asarray(counts["status"], dtype=str) == "gated")]
    ).sort_values(by="gate_rate_per_h")
    if len(gated):
        lines.append(
            f"Dropped by the gate ({len(gated)}): "
            + ", ".join(
                f"{r['split_group']} ({r['gate_rate_per_h']:.1f}/h)"
                for r in gated.to_dict("records")
            )
            + ".\n"
        )
    diag = root / "figures" / "diagnostics"
    if (diag / "gate.png").exists():
        lines.append("![gate](figures/diagnostics/gate.png)\n")
    if (diag / "rates").exists():
        lines.append(
            "Screen boxes per 30 min over each recording: "
            "`figures/diagnostics/rates/` (rates per phase in "
            "`figures/diagnostics/rates.csv`); light-phase snippets: "
            "`figures/diagnostics/snippets/*.html`.\n"
        )
    return lines


def write_report(
    root: str | Path,
    *,
    seed: int = 0,
    n_demo_per_dataset: int = 2,
    splits: dict[str, list[str]] | None = None,
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
    splits : mapping of str to list of str, optional
        Named lists of split groups (e.g. a scene config's
        ``include_groups`` per train/val/test): the screening section counts
        per split and lists groups left without segments.

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

    screened_builds = [
        b for b in meta.get("builds", []) if b.get("config", {}).get("screens")
    ]
    if meta.get("carried_over"):
        for c in meta["carried_over"]:
            add(
                f"Carried over unchanged from `{c['source']}` ({c['created']}): "
                f"{', '.join(c['datasets'])} ({c['n_segments']} segments, hard links).\n"
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

    if screened_builds:
        lines.extend(
            _screen_section(
                root, manifest, recordings, screened_builds[-1], splits or {}
            )
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
        "The SNR refers to the fish's strongest electrode; in a spectrogram "
        "summed over electrodes its band stands about 10 log10(E / E_fish) "
        "lower (10-15 dB for a fish near one electrode of a 9-16 electrode "
        "grid; compare the demo figures).\n"
    )
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
            "is rare in the tanks. Options: truncate the target per segment "
            "(`SnrDistribution.sample(rng, min_db=real_snr_max_db - 30)`), use "
            "fish-free segments for the hardest SNRs, or raise the floor with "
            "synthetic noise (`other_noise_level`), which lowers real and "
            "simulated fish together.\n"
        )
        capped = np.concatenate(
            [np.asarray(dist.sample(rng, 50, min_db=r - 30.0)) for r in real_max]
        )
        add(
            "With the truncation at a 30-dB gap, targets on fish segments are "
            f"{_q(capped)} dB and {np.mean(capped < dist.hard_db[1]):.0%} of them "
            f"fall in the hard range (below {dist.hard_db[1]:g} dB), against "
            f"{dist.hard_fraction:.0%} without it; fish-free segments keep the "
            "full distribution.\n"
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
