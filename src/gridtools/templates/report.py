# ruff: noqa: RUF001
"""Quality report of a built template library (Markdown + PNG figures).

Needs the optional ``plot`` extra (matplotlib), imported lazily.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from gridtools.templates.features import detrended_rms
from gridtools.templates.library import TemplateLibrary, load_library

if TYPE_CHECKING:
    from matplotlib.axes import Axes

log = logging.getLogger(__name__)

#: pandas-stubs cannot type the column arithmetic below; the report treats frames as Any.
Frame = Any

FAMILY_COLORS = {"small": "#2a6f97", "big": "#c8553d"}
V1_DIP_RANGE = (0.3, 0.8)  # v1 simulator: amplitude dip ~ U(0.3, 0.8) for every chirp
LOWPASS_CUTOFFS_HZ = (25.0, 50.0, 100.0, 200.0)


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    """Checks on a random sample of source rows.

    Attributes
    ----------
    n_rows
        Rows examined.
    sigma_best
        Gaussian sigma (samples) that best maps ``orig_freq`` to
        ``smooth_freq``, per row.
    sigma_residual_hz
        Residual RMS at the best sigma, per row.
    jitter_ratio
        RMS of ``orig_freq - smooth_freq`` inside the chirp divided by the
        RMS outside it, per row.
    orig_peak_ratio
        Peak of ``orig_freq`` over peak of ``smooth_freq`` (both relative to
        baseline), per row.
    lowpass
        Per cutoff: median ratio (filtered / unfiltered) of the amplitude dip
        of clean rows and of the amplitude ripple of contaminated rows.
    """

    n_rows: int
    sigma_best: npt.NDArray[np.float64]
    sigma_residual_hz: npt.NDArray[np.float64]
    jitter_ratio: npt.NDArray[np.float64]
    orig_peak_ratio: npt.NDArray[np.float64]
    lowpass: dict[float, tuple[float, float]]


def source_evidence(
    h5_path: Path, meta: Frame, rng: np.random.Generator, n_rows: int = 600
) -> SourceEvidence:
    """Re-examine a random sample of source rows.

    Parameters
    ----------
    h5_path
        Source file used to build the library.
    meta
        Library metadata (for alignment, crossings and flags).
    rng
        Random generator for the row sample.
    n_rows
        Sample size.

    Returns
    -------
    SourceEvidence
    """
    import h5py
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import butter, sosfiltfilt

    from gridtools.templates.source import TRACE_NAMES, find_wrap_index, unroll_rows

    usable = meta.index[
        meta["valid"] & ~meta["step_flag"] & ~meta["truncated"]
    ].to_numpy()
    rows = np.sort(rng.choice(usable, size=min(n_rows, usable.size), replace=False))
    with h5py.File(h5_path, "r") as f:
        raw = {}
        for name in TRACE_NAMES:
            dataset = f[name]
            assert isinstance(dataset, h5py.Dataset)
            raw[name] = dataset[rows].astype(np.float64)
    fs = 20_000.0
    sigmas = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    best, resid, jitter_ratio, peak_ratio = [], [], [], []
    clean_dip: dict[float, list[float]] = {fc: [] for fc in LOWPASS_CUTOFFS_HZ}
    dirty_rip: dict[float, list[float]] = {fc: [] for fc in LOWPASS_CUTOFFS_HZ}
    filters = {
        fc: butter(2, fc, btype="lowpass", output="sos", fs=fs)
        for fc in LOWPASS_CUTOFFS_HZ
    }
    for k, row in enumerate(rows):
        traces = {name: raw[name][k] for name in TRACE_NAMES}
        wrap = find_wrap_index(traces["smooth_freq"])
        orig, smooth, amp = (
            unroll_rows(traces[name], wrap)
            for name in ("orig_freq", "smooth_freq", "smooth_amp")
        )
        inner = slice(200, orig.size - 200)
        errors = []
        for sigma in sigmas:
            diff = (smooth - gaussian_filter1d(orig, sigma))[inner]
            errors.append(float(np.std(diff - diff.mean())))
        best.append(sigmas[int(np.argmin(errors))])
        resid.append(min(errors))

        r = meta.loc[row]
        align = int(r["source_align_index"])
        on = int(align + r["onset_10_s"] * fs)
        off = int(align + r["offset_10_s"] * fs)
        chirp = np.zeros(orig.size, dtype=bool)
        chirp[max(on, 0) : max(off + 1, 0)] = True
        base = np.ones(orig.size, dtype=bool)
        base[max(on - 400, 0) : max(off + 1200, 0)] = False
        jitter = orig - smooth
        if chirp.sum() > 10 and base.sum() > 1000:
            jitter_ratio.append(
                float(
                    np.sqrt(np.mean(jitter[chirp] ** 2))
                    / np.sqrt(np.mean(jitter[base] ** 2))
                )
            )
            b_orig, b_smooth = np.median(orig[base]), np.median(smooth[base])
            peak_ratio.append(
                float(
                    np.max(orig[chirp] - b_orig)
                    / max(np.max(smooth[chirp] - b_smooth), 1e-9)
                )
            )
        if base.sum() < 1000 or chirp.sum() < 10:
            continue
        amp_rel = amp / np.median(amp[base])
        dip_window = np.zeros(orig.size, dtype=bool)
        dip_window[max(on - 100, 0) : off + 200] = True
        dip0 = 1.0 - np.min(amp_rel[dip_window])
        rip0, _ = detrended_rms(amp_rel, base, 400)
        for fc, sos in filters.items():
            filtered = sosfiltfilt(sos, amp_rel)
            if bool(r["amp_usable"]) and r["ripple_amp"] <= 0.01 and dip0 > 0.02:
                clean_dip[fc].append(float((1.0 - np.min(filtered[dip_window])) / dip0))
            if r["ripple_amp"] >= 0.05:
                rip, _ = detrended_rms(filtered, base, 400)
                dirty_rip[fc].append(float(rip / rip0))
    lowpass = {
        fc: (
            float(np.median(clean_dip[fc])) if clean_dip[fc] else float("nan"),
            float(np.median(dirty_rip[fc])) if dirty_rip[fc] else float("nan"),
        )
        for fc in LOWPASS_CUTOFFS_HZ
    }
    return SourceEvidence(
        n_rows=int(rows.size),
        sigma_best=np.asarray(best),
        sigma_residual_hz=np.asarray(resid),
        jitter_ratio=np.asarray(jitter_ratio),
        orig_peak_ratio=np.asarray(peak_ratio),
        lowpass=lowpass,
    )


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def _hist(
    ax: Axes, data: dict[str, npt.NDArray[np.float64]], bins: Any, **kwargs: Any
) -> None:
    for family, values in data.items():
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.4,
            color=FAMILY_COLORS.get(family, "0.3"),
            label=f"{family} (n={values.size})",
            **kwargs,
        )


def _fig_distributions(lib: TemplateLibrary, path: Path) -> None:
    import matplotlib.pyplot as plt

    meta: Frame = lib.metadata
    fam = {f: meta[meta["family"] == f] for f in ("small", "big")}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    _hist(
        axes[0, 0],
        {f: d["excursion_hz"].to_numpy() for f, d in fam.items()},
        bins=np.geomspace(20, 900, 70),
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("excursion (Hz)")
    _hist(
        axes[0, 1],
        {f: 1e3 * d["duration_10_s"].to_numpy() for f, d in fam.items()},
        bins=np.linspace(0, 80, 81),
    )
    axes[0, 1].set_xlabel("full width at 10% (ms)")
    _hist(
        axes[0, 2],
        {f: 1e3 * d["duration_50_s"].to_numpy() for f, d in fam.items()},
        bins=np.linspace(0, 60, 61),
    )
    axes[0, 2].set_xlabel("full width at 50% (ms)")
    _hist(
        axes[1, 0],
        {f: d["undershoot_hz"].to_numpy() for f, d in fam.items()},
        bins=np.linspace(0, 120, 61),
    )
    axes[1, 0].set_xlabel("undershoot (Hz below baseline)")
    _hist(
        axes[1, 1],
        {f: d.loc[d["amp_usable"], "amp_dip"].to_numpy() for f, d in fam.items()},
        bins=np.linspace(0, 1, 51),
    )
    axes[1, 1].axvspan(*V1_DIP_RANGE, color="0.85", zorder=0, label="v1 U(0.3, 0.8)")
    axes[1, 1].set_xlabel("amplitude dip (amp_usable only)")
    _hist(
        axes[1, 2],
        {f: d["asymmetry"].to_numpy() for f, d in fam.items()},
        bins=np.linspace(-1, 1, 51),
    )
    axes[1, 2].set_xlabel("asymmetry (fall - rise) / (fall + rise)")
    for ax in axes.flat:
        ax.set_ylabel("templates")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Default selection: {lib.n_templates} templates")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fig_examples(
    lib: TemplateLibrary,
    groups: dict[str, npt.NDArray[np.int64]],
    path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    t_ms = 1e3 * lib.time_s()
    names = [name for name, ids in groups.items() if ids.size]
    fig, axes = plt.subplots(
        len(names),
        2,
        figsize=(12, 2.1 * len(names)),
        squeeze=False,
        constrained_layout=True,
    )
    for i, name in enumerate(names):
        sample = lib.get(groups[name])
        for k in range(sample.template_ids.size):
            axes[i, 0].plot(t_ms, sample.delta_f_hz[k], lw=0.8, alpha=0.8)
            axes[i, 1].plot(t_ms, sample.amp_rel[k], lw=0.8, alpha=0.8)
        ids = ", ".join(str(t) for t in sample.template_ids[:6])
        axes[i, 0].set_title(f"{name}: ids {ids}", fontsize=8, loc="left")
        axes[i, 0].set_ylabel("Δf (Hz)")
        axes[i, 1].set_ylabel("amp_rel")
        for ax in axes[i]:
            ax.set_xlim(-60, 80)
    axes[-1, 0].set_xlabel("time from alignment point (ms)")
    axes[-1, 1].set_xlabel("time from alignment point (ms)")
    fig.suptitle(title)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _fig_family_overlay(
    lib: TemplateLibrary, rng: np.random.Generator, path: Path
) -> None:
    import matplotlib.pyplot as plt

    t_ms = 1e3 * lib.time_s()
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), constrained_layout=True)
    for j, family in enumerate(("small", "big")):
        rows = np.flatnonzero((lib.metadata["family"] == family).to_numpy())
        usable = rows[lib.metadata["amp_usable"].to_numpy()[rows]]
        pick = rng.choice(rows, size=min(40, rows.size), replace=False)
        for r in pick:
            axes[0, j].plot(
                t_ms, lib.delta_f_hz[r], lw=0.6, alpha=0.5, color=FAMILY_COLORS[family]
            )
        axes[0, j].plot(t_ms, np.median(lib.delta_f_hz[rows], axis=0), lw=2, color="k")
        pick_amp = rng.choice(usable, size=min(40, usable.size), replace=False)
        for r in pick_amp:
            axes[1, j].plot(
                t_ms, lib.amp_rel[r], lw=0.6, alpha=0.5, color=FAMILY_COLORS[family]
            )
        if usable.size:
            axes[1, j].plot(
                t_ms, np.median(lib.amp_rel[usable], axis=0), lw=2, color="k"
            )
        axes[0, j].set_title(
            f"{family}: 40 random templates, median in black (n={rows.size})"
        )
        axes[1, j].set_title(f"{family}: amplitude, amp_usable only (n={usable.size})")
        axes[0, j].set_ylabel("Δf (Hz)")
        axes[1, j].set_ylabel("amp_rel")
        for ax in axes[:, j]:
            ax.set_xlim(-60, 80)
            ax.set_xlabel("time from alignment point (ms)")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _fig_v1_comparison(meta: Frame, default: Frame, path: Path) -> None:
    import matplotlib.pyplot as plt

    v1 = meta[meta["in_v1_clustered"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    bins = np.geomspace(20, 900, 70)
    axes[0].hist(
        default["excursion_hz"], bins=bins, histtype="step", lw=1.4, label="v2 default"
    )
    axes[0].hist(
        v1["excursion_hz"],
        bins=bins,
        histtype="step",
        lw=1.4,
        label="v1 clustered rows",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("excursion (Hz)")
    bins_d = np.linspace(0, 60, 61)
    axes[1].hist(1e3 * default["duration_10_s"], bins=bins_d, histtype="step", lw=1.4)
    axes[1].hist(1e3 * v1["duration_10_s"], bins=bins_d, histtype="step", lw=1.4)
    axes[1].set_xlabel("full width at 10% (ms)")
    usable = default[default["amp_usable"]]
    bins_a = np.linspace(0, 1, 51)
    for family, color in FAMILY_COLORS.items():
        axes[2].hist(
            usable.loc[usable["family"] == family, "amp_dip"],
            bins=bins_a,
            histtype="step",
            lw=1.4,
            color=color,
            density=True,
            label=f"v2 {family}, amp_usable",
        )
    axes[2].hist(
        np.linspace(*V1_DIP_RANGE, 1000),
        bins=bins_a,
        density=True,
        color="0.8",
        label="v1 simulator U(0.3, 0.8)",
    )
    axes[2].set_xlabel("amplitude dip")
    axes[0].legend(fontsize=8, frameon=False)
    axes[2].legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fig_eodf(meta: Frame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    eodf = meta["baseline_hz"].dropna()
    axes[0].hist(eodf, bins=np.arange(300, 960, 5), color="0.3")
    axes[0].set_xlabel("baseline EODf (Hz), recovered from spline knots")
    axes[0].set_ylabel("templates")
    axes[1].hist(np.log10(meta["baseline_hz_mad"].dropna()), bins=60, color="0.3")
    axes[1].set_xlabel("log10 MAD of per-knot estimates (Hz)")
    groups = meta["inferred_group"].value_counts().to_numpy()
    axes[2].plot(
        np.arange(1, groups.size + 1), np.cumsum(groups) / groups.sum(), color="0.3"
    )
    axes[2].set_xscale("log")
    axes[2].set_xlabel("inferred groups, largest first")
    axes[2].set_ylabel("cumulative fraction of templates")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fig_amplitude_evidence(meta: Frame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
    sel = (meta["ripple_amp"] > 0.03) & (meta["beat_freq_hz"] > 20) & meta["valid"]
    ratio = meta.loc[sel, "ripple_freq_hz"] / meta.loc[sel, "ripple_amp"]
    axes[0].loglog(
        meta.loc[sel, "beat_freq_hz"], ratio, ".", ms=2, alpha=0.4, color="0.2"
    )
    lim = np.array([15, 400])
    axes[0].plot(lim, lim, color="r", lw=1, label="ratio = beat frequency")
    axes[0].set_xlabel("dominant amplitude-ripple frequency (Hz)")
    axes[0].set_ylabel("freq ripple (Hz) / amp ripple")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].hist(
        meta.loc[meta["ripple_amp"] > 0.03, "beat_freq_hz"],
        bins=np.geomspace(5, 400, 50),
        color="0.3",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("dominant ripple frequency (Hz), ripple_amp > 0.03")
    axes[2].loglog(
        meta["ripple_amp"], meta["amp_dip"].clip(lower=1e-3), ".", ms=1.5, alpha=0.3
    )
    axes[2].set_xlabel("amplitude ripple (RMS, baseline)")
    axes[2].set_ylabel("amplitude dip")
    axes[3].plot(meta["amp_dip_uncorrected"], meta["amp_dip"], ".", ms=1.5, alpha=0.3)
    axes[3].plot([0, 1], [0, 1], color="r", lw=1)
    axes[3].set_xlabel("dip before filter-gain correction")
    axes[3].set_ylabel("dip after correction")
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _q(
    values: pd.Series | npt.NDArray[Any], qs: tuple[float, ...] = (0.05, 0.5, 0.95)
) -> str:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "n/a"
    p = np.quantile(v, qs)
    return f"{p[1]:.3g} ({p[0]:.3g}–{p[2]:.3g})"


def _corr(x: Any, y: Any) -> float:
    """Pearson correlation of the finite pairs; NaN with fewer than 3 pairs."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _pick(
    meta: Frame,
    mask: npt.NDArray[np.bool_],
    rng: np.random.Generator,
    n: int = 4,
) -> npt.NDArray[np.int64]:
    ids = meta.loc[mask, "template_id"].to_numpy(dtype=np.int64)
    if ids.size == 0:
        return ids
    return np.sort(rng.choice(ids, size=min(n, ids.size), replace=False))


def write_report(
    library_dir: Path | str,
    *,
    h5_path: Path | str | None = None,
    seed: int = 0,
    n_evidence_rows: int = 600,
) -> Path:
    """Write ``report.md`` and figures for a built library.

    Parameters
    ----------
    library_dir
        Library directory written by :func:`gridtools.templates.build_library`.
    h5_path
        Source file, for checks on raw rows; defaults to the path in the
        manifest. Skipped if missing.
    seed
        Seed of the random examples and row sample.
    n_evidence_rows
        Source rows re-examined for the evidence section.

    Returns
    -------
    pathlib.Path
        Path of the report.
    """
    library_dir = Path(library_dir)
    fig_dir = library_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    lib = load_library(library_dir, mmap=True)
    meta: Frame = lib.metadata
    manifest = lib.manifest
    counts = manifest["counts"]
    default = lib.select()
    dmeta: Frame = default.metadata
    source = Path(h5_path if h5_path is not None else manifest["source"]["h5"]["path"])
    evidence = (
        source_evidence(source, meta, rng, n_evidence_rows) if source.exists() else None
    )

    _fig_distributions(default, fig_dir / "distributions.png")
    _fig_family_overlay(default, rng, fig_dir / "families.png")
    flag_groups = {
        "clean small": _pick(dmeta, (dmeta["family"] == "small").to_numpy(), rng),
        "clean big": _pick(dmeta, (dmeta["family"] == "big").to_numpy(), rng),
        "truncated": _pick(meta, meta["truncated"].to_numpy(), rng),
        "step artefact": _pick(meta, meta["step_flag"].to_numpy(), rng),
        "multi_peak": _pick(meta, meta["multi_peak"].to_numpy(), rng),
        "outlier": _pick(meta, meta["outlier"].to_numpy(), rng),
        "eodf_atypical": _pick(meta, meta["eodf_atypical"].to_numpy(), rng),
        "ripple_flag": _pick(meta, meta["ripple_flag"].to_numpy(), rng),
        "amp_ripple_flag only": _pick(
            meta,
            (meta["amp_ripple_flag"] & ~meta["ripple_flag"] & meta["valid"]).to_numpy(),
            rng,
        ),
    }
    _fig_examples(lib, flag_groups, fig_dir / "flags.png", "Examples per quality flag")
    _fig_v1_comparison(meta, dmeta, fig_dir / "v1_comparison.png")
    _fig_eodf(meta, fig_dir / "eodf.png")
    _fig_amplitude_evidence(meta, fig_dir / "amplitude_evidence.png")

    text = _report_text(lib, default, counts, evidence)
    path = library_dir / "report.md"
    path.write_text(text)
    log.info("report written to %s", path)
    return path


def _report_text(
    lib: TemplateLibrary,
    default: TemplateLibrary,
    counts: dict[str, Any],
    evidence: SourceEvidence | None,
) -> str:
    meta: Frame = lib.metadata
    dmeta: Frame = default.metadata
    manifest = lib.manifest
    cfg = manifest["config"]
    n = len(meta)
    small = dmeta[dmeta["family"] == "small"]
    big = dmeta[dmeta["family"] == "big"]
    usable = dmeta[dmeta["amp_usable"]]
    v1 = meta[meta["in_v1_clustered"]]
    wrap = v1["source_wrap_index"]
    crop_start = (8000 - 4838) // 2  # v1 crop of the 8000-sample rows
    v1_wrapped = int(((wrap >= crop_start) & (wrap < crop_start + 4838)).sum())
    eodf = meta["baseline_hz"]
    atyp = meta[meta["eodf_atypical"]]
    typical = meta[~meta["eodf_atypical"] & meta["baseline_hz"].notna()]
    groups = meta["inferred_group"].value_counts()
    beat_rows = (meta["ripple_amp"] > 0.03) & (meta["beat_freq_hz"] > 20)
    beat_ratio = (
        meta.loc[beat_rows, "ripple_freq_hz"] / meta.loc[beat_rows, "ripple_amp"]
    ) / (meta.loc[beat_rows, "beat_freq_hz"])
    beat_corr = _corr(
        np.log(
            meta.loc[beat_rows, "ripple_freq_hz"] / meta.loc[beat_rows, "ripple_amp"]
        ),
        np.log(meta.loc[beat_rows, "beat_freq_hz"]),
    )
    small_all = meta[(meta["family"] == "small") & meta["valid"]]
    dip_ripple_corr = _corr(small_all["amp_dip"], small_all["ripple_amp"])
    from gridtools.templates.build import _code_version

    report_code = _code_version()
    lines: list[str] = []
    add = lines.append

    add("# Chirp template library v1: build report\n")
    add(
        f"Generated {datetime.datetime.now(tz=datetime.UTC):%Y-%m-%d %H:%M} UTC from "
        f"`{manifest['source']['h5']['path']}` "
        f"(sha256 `{manifest['source']['h5'].get('sha256', 'n/a')[:16]}…`), "
        f"gridtools {manifest['code']['gridtools_version']} at "
        f"`{(manifest['code'].get('git_sha') or 'unknown')[:10]}`"
        f"{' (dirty)' if manifest['code'].get('git_dirty') else ''}; report written by "
        f"`{(report_code.get('git_sha') or 'unknown')[:10]}`"
        f"{' (dirty)' if report_code.get('git_dirty') else ''}. "
        f"Library: {n} templates × {manifest['n_samples']} samples at "
        f"{manifest['samplerate_hz']:.0f} Hz "
        f"({1e3 * manifest['n_samples'] / manifest['samplerate_hz']:.0f} ms), "
        f"alignment point at sample {manifest['centre_index']}. "
        "Per-chirp source recordings are lost; species, sex and temperature are not recorded "
        "per chirp.\n"
    )

    add("## 1. What the source arrays contain\n")
    sigma_line = ""
    if evidence is not None:
        exact = np.mean(evidence.sigma_residual_hz < 1e-3)
        sigma_line = (
            f" On {evidence.n_rows} random rows, `smooth_freq` equals "
            f"`gaussian_filter1d(orig_freq, σ)` with best σ = "
            f"{np.median(evidence.sigma_best):.0f} samples (1 ms) and zero residual in "
            f"{100 * exact:.0f}% of rows, confirming the thesis code version."
        )
    add(
        "- **Layout.** `extracted_chirps.h5` holds 13,015 rows × 8,000 samples (0.4 s at "
        "20 kHz) of four traces written by `extract_chirpchamber_and_livios_chirps.py`: "
        "`orig_freq` (instantaneous EOD frequency from linearly interpolated ascending zero "
        "crossings, one value per EOD cycle, cubic-spline interpolated to 20 kHz), "
        "`smooth_freq` (the same after a Gaussian with σ = 20 samples = 1 ms, applied before "
        "cutting), `orig_amp`/`smooth_amp` (per-cycle peak of the band-passed signal, "
        "divided by the window maximum, so the maximum is exactly 1 and the baseline is "
        "below 1). According to that script the recordings came from three sets "
        "(fishkurs2024, livio_jansubset, gp2024_chirps), all lost." + sigma_line
    )
    add(
        "- **Reference.** Both frequency traces are relative: the extraction subtracted the "
        "value at the first sample of the 0.4 s window (a single noisy sample, not a "
        "baseline). Every row contains exactly one sample that is exactly 0.0 in both "
        "frequency traces: that first sample."
    )
    add(
        "- **Alignment.** After cutting, chirps were centred with `np.roll` (a circular "
        "shift) on the midpoint of the region above 20 Hz. Rows are therefore not "
        "contiguous in time: the original window start sits at the exact-zero sample "
        f"(found in {int((meta['source_wrap_index'] >= 0).sum())} of {n} rows), with a "
        "jump to the window's end value right before it. The builder undoes the roll "
        "(`unroll_rows`) before measuring anything. In the unrolled windows the 1–99% range "
        f"of the chirp position is sample {np.quantile(meta['source_align_index'], 0.01):.0f}–"
        f"{np.quantile(meta['source_align_index'], 0.99):.0f}; a few chirps sit at the window "
        "edge (truncated)."
    )
    add(
        f"- **Units.** Relative frequency is in Hz. Check: every knot interval satisfies "
        f"1/Δt_k = c + x(t_k) with one constant c per row; the per-knot deviation is "
        f"{_q(meta['baseline_hz_mad'])} Hz (median, 5–95%) on the baseline and "
        f"{_q(meta['knot_check_chirp_mad_hz'])} Hz inside chirps (the large upper tail "
        "comes from missed knots where float32 precision at large Δf hides the knot "
        "signature in low-jitter recordings, not from a unit problem)."
    )
    add(
        f"- **Absolute baseline EODf is recoverable** ({counts['baseline_hz_recovered']} "
        f"of {n} rows). The cubic spline has a knot at every zero crossing, visible as "
        "isolated spikes in the fourth difference (~10³ above the floor); the knot "
        "spacing gives the absolute period. Stored as `baseline_hz` (median "
        f"{eodf.median():.0f} Hz, 5–95% {eodf.quantile(0.05):.0f}–{eodf.quantile(0.95):.0f} "
        "Hz). See `figures/eodf.png`. Temperature is unknown, so this is not "
        "temperature-corrected."
    )
    add(
        f"- **Two EODf populations.** {len(typical)} rows lie at 540–940 Hz (sharp peaks, "
        f"i.e. individual fish); {len(atyp)} rows lie at 320–440 Hz, separated by an empty "
        "gap at 440–540 Hz (`eodf_atypical`). Their chirps differ: FWHM "
        f"{_q(1e3 * atyp['duration_50_s'])} ms vs {_q(1e3 * typical['duration_50_s'])} ms, "
        f"excursion {_q(atyp['excursion_hz'])} Hz vs {_q(typical['excursion_hz'])} Hz, and "
        "almost no beat ripple. A wrong sampling rate (the thesis code hard-codes 40 kHz for "
        ".nix files) would scale EODf, excursion and 1/duration together; here EODf is lower "
        "while excursion is higher and duration shorter, so a rate error does not explain "
        "them. They may be another species or setup."
    )
    add(
        f"- **Few sources dominate.** Rows were appended file by file, so runs of rows with "
        f"continuous EODf approximate recordings (`inferred_group`, new group at a jump > "
        f"{cfg['group_eodf_jump_hz']:.0f} Hz): {groups.size} groups, the 10 largest hold "
        f"{100 * groups.iloc[:10].sum() / n:.0f}% of all rows, the largest "
        f"{groups.iloc[0]} rows."
    )
    add(
        "- **NaNs and truncation.** "
        f"{int(meta['has_nan'].sum())} rows contain NaN. {int(meta['truncated'].sum())} "
        "templates are truncated (support plus ramps does not fit into the source window)."
    )
    add(
        f"- **Step artefacts.** {int(meta['step_flag'].sum())} rows contain single-sample "
        "jumps in a smoothed trace, impossible after a 1 ms Gaussian. They come from the "
        "extraction's own in-place bug: `oc -= oc[0]` and `oa /= max(oa)` act on views of "
        "the full-recording arrays, so a window that overlaps earlier windows inherits their "
        "offsets/scalings piecewise (additive steps in frequency, multiplicative in "
        "amplitude) while still passing the thesis filters. Strong beat phase slips trigger "
        "the same detector. Flagged (`step_flag`)."
    )
    add(
        f"- **Duplicates.** {counts['flagged']['exact_duplicate']} rows are bit-identical to "
        "an earlier row (short runs of consecutive rows repeated a few to ~2,500 rows later, "
        "suggesting overlapping source files). No further re-extractions exist: the "
        "cycle-to-cycle jitter outside the chirp never correlates above "
        f"{np.nanmax(meta['max_fingerprint_corr']):.3f} between non-identical rows (threshold "
        f"{cfg['duplicate_fingerprint_corr_min']}). Aligned shape correlation alone reaches "
        f"≥ {cfg['duplicate_corr_min']} for {int((meta['max_shape_corr'] >= cfg['duplicate_corr_min']).sum())} "
        "rows; these are distinct, stereotyped chirps (independent jitter), so they are not "
        "flagged; `max_shape_corr` and its partner are stored."
    )
    add(
        f"- **Relation to the v1 set.** All 9,082 rows of `clustered_chirps.npy` equal "
        "`smooth_freq[j, 1581:6419]` of an h5 row (exact float32 match; 9,009 unique), and "
        "`clustered_orig_chirps.npy` equals the same crops of `orig_freq`. The h5 is "
        f"therefore the source of the v1 templates ({int(meta['in_v1_clustered'].sum())} "
        "rows marked `in_v1_clustered`, with `v1_cluster_label`), not a later, unrelated "
        f"extraction. {v1_wrapped} v1 rows contain the circular-shift jump inside their "
        "crop, and all v1 rows are relative to a single window sample rather than to a "
        "baseline.\n"
    )

    add("## 2. Template construction\n")
    add(
        f"1. Unroll; choose `{cfg['freq_source']}_freq` and `{cfg['amp_source']}_amp`.\n"
        "2. Baseline = median of samples outside [onset₁₀ − "
        f"{1e3 * cfg['baseline_exclusion_pre_s']:.0f} ms, offset₁₀ + "
        f"{1e3 * cfg['baseline_exclusion_post_s']:.0f} ms] (two passes); the 1 ms Gaussian "
        "shifts nothing in time.\n"
        "3. Alignment point = midpoint of the 50% crossings (FWHM centre), rounded to a "
        "sample. 50% crossings sit on the steep flanks, so they are less sensitive to "
        "baseline ripple than 10% crossings (10% of a 50 Hz chirp is 5 Hz, the size of "
        "typical ripple) and less sensitive than the peak to noisy plateaus of big chirps.\n"
        f"4. Crop {1e3 * cfg['template_duration_s']:.0f} ms centred on it.\n"
        f"5. Taper: weight 1 on [onset₁₀ − {1e3 * cfg['support_pad_pre_s']:.0f} ms, "
        f"offset₁₀ + {1e3 * cfg['support_pad_post_s']:.0f} ms] (includes the undershoot, "
        f"which recovers within ~20 ms), raised-cosine ramps of "
        f"{1e3 * cfg['taper_ramp_s']:.0f} ms, exactly 0 outside. `delta_f_hz = w·(f − "
        "baseline)` is exactly 0 and `amp_rel = 1 + w·(a/a₀/g − 1)` exactly 1 at both ends; "
        "outside the support the source's beat ripple is removed. New arrays only.\n"
        "6. Measure the final template (`measure_template`): excursion, widths at 10% and 50%, "
        "undershoot (largest dip below 0 within 50 ms after offset₁₀), amplitude dip "
        "(1 − min amp_rel), asymmetry ((fall − rise)/(fall + rise) with rise = peak − "
        "onset₁₀, fall = offset₁₀ − peak), `n_peaks` (prominent maxima inside the 10% "
        "crossings, structure of the chirp) and `n_extra_events` (outside them).\n"
        f"7. Family: `big` if excursion ≥ {cfg['big_chirp_threshold_hz']:.0f} Hz, at the "
        "trough of the excursion density (few templates between 160 and 250 Hz), otherwise "
        "`small`. Descriptive only; type 1/2 chirp classes would need more than excursion.\n"
    )
    add(
        "**Frequency trace: `smooth_freq`.** Inside chirps, `orig_freq` has cycle-level "
        "spikes"
        + (
            f": its jitter RMS is {np.median(evidence.jitter_ratio):.1f}× (median) the "
            f"baseline jitter, and its peak is {np.median(evidence.orig_peak_ratio):.2f}× "
            "(median) the smoothed peak"
            if evidence is not None
            else ""
        )
        + ". The spikes coincide with the amplitude drop, where the contaminating signal "
        "dominates the zero crossings, so they are measurement artefacts rather than "
        "pacemaker output. The cost of the 1 ms Gaussian: a Gaussian chirp with FWHM w is "
        "lowered by σ_c/√(σ_c² + σ²), i.e. 3% at 9 ms and 9% at 5 ms (the median FWHM is "
        f"{1e3 * dmeta['duration_50_s'].median():.1f} ms). `freq_source='orig'` is available.\n"
    )

    add("## 3. Amplitude traces: decision and evidence\n")
    add(
        f"**Decision:** use `smooth_amp`, correct it for the extraction filter, do **not** "
        f"low-pass it further, and mark only clean traces as usable (`amp_usable`: "
        f"baseline ripple RMS ≤ {cfg['amp_ripple_max']}, dip ≥ "
        f"{cfg['amp_dip_to_ripple_min']:.0f}× ripple, no step artefact, EODf known). "
        f"{int(meta['amp_usable'].sum())} of {n} templates "
        f"({100 * meta['amp_usable'].mean():.0f}%) qualify, {len(usable)} in the default "
        "selection.\n"
    )
    add(
        f"1. **The baseline ripple is a beat with a second EOD.** For two sinusoids with "
        "amplitude ratio m and frequency difference ΔF, the envelope ripples with depth m "
        "and the instantaneous frequency with amplitude m·ΔF, so freq ripple / amp ripple "
        f"≈ ΔF. In {int(beat_rows.sum())} rows with amp ripple > 0.03 and ripple frequency "
        f"> 20 Hz the ratio divided by the measured ripple frequency is "
        f"{beat_ratio.median():.2f} (median; log–log r = {beat_corr:.2f}); "
        "`figures/amplitude_evidence.png`, panel 1."
    )
    add(
        f"2. **It is common.** Amplitude ripple RMS: {_q(meta['ripple_amp'])}; frequency "
        f"ripple RMS {_q(meta['ripple_freq_hz'])} Hz. The measured dip of small chirps "
        f"correlates with the ripple (r = {dip_ripple_corr:.2f}), so raw dips are inflated "
        "by beat troughs."
    )
    beats = meta.loc[meta["ripple_amp"] > 0.03, "beat_freq_hz"]
    add(
        f"3. **Low-pass filtering cannot separate it.** The dominant ripple frequency "
        f"(ripple_amp > 0.03) is {_q(beats)} Hz, i.e. at or below the chirp's own AM band "
        "(10–20 ms dips need ~50–100 Hz)."
        + (
            " Zero-phase 2nd-order low-pass on source rows (median ratio filtered/raw): "
            + "; ".join(
                f"{fc:.0f} Hz: clean dips ×{v[0]:.2f}, contaminated ripple ×{v[1]:.2f}"
                for fc, v in evidence.lowpass.items()
            )
            + ". Any cutoff that removes a useful part of the ripple also shrinks genuine "
            "dips."
            if evidence is not None
            else ""
        )
    )
    add(
        f"4. **The extraction filter fakes AM.** The thesis band-pass (2nd-order "
        "Butterworth, EODf − 100 Hz to 1.9·EODf, `sosfiltfilt`, gain |H|²) has its lower "
        "edge just below the baseline, so the measured amplitude rises with frequency: +7.5% "
        "for a 30 Hz and +12.5% for a 60 Hz chirp at 800 Hz, comparable to the true dips of "
        "small chirps. Each template is divided by the quasi-static gain ratio "
        "G(f₀ + Δf(t))/G(f₀) with f₀ = `baseline_hz` (largest correction per template "
        f"`amp_filter_gain_max`: {_q(meta['amp_filter_gain_max'])}). Dip before → after "
        f"correction: {_q(meta['amp_dip_uncorrected'])} → {_q(meta['amp_dip'])}. The "
        "correction covers the fundamental only. The 2nd harmonic passes at ~0.44 relative "
        "gain and falls as frequency rises, so for EODs with a strong in-phase 2nd harmonic "
        "the correction is an upper bound (up to about twice the true effect). Uncorrected "
        "dips are kept as `amp_dip_uncorrected`."
    )
    add(
        f"5. **Result vs v1.** Clean (amp_usable) dips: small {_q(usable.loc[usable['family'] == 'small', 'amp_dip'])}, "
        f"big {_q(usable.loc[usable['family'] == 'big', 'amp_dip'])}. The v1 simulator drew "
        "U(0.3, 0.8) for every chirp, several times deeper than real small chirps "
        "(`figures/v1_comparison.png`, right).\n"
    )

    add("## 4. Quality control counts\n")
    add(
        "Flags are kept in `templates.parquet`; `TemplateLibrary.select()` drops the flags "
        "in `DEFAULT_EXCLUDE`. 'Flagged' counts each criterion alone; 'left' applies them "
        "in table order.\n"
    )
    add("| criterion | flagged | left |")
    add("|---|---:|---:|")
    add(f"| all rows | | {counts['cumulative']['all']} |")
    for name, flagged in counts["flagged"].items():
        add(f"| {name} | {flagged} | {counts['cumulative'][f'after_{name}']} |")
    add(
        f"\nDefault selection: {default.n_templates} templates "
        f"({counts['default_selection_per_family']['small']} small, "
        f"{counts['default_selection_per_family']['big']} big); "
        f"{counts['amp_usable_in_default_selection']} of them with usable amplitude. "
        f"Not in the default exclusion: `amp_ripple_flag` ({int(meta['amp_ripple_flag'].sum())} "
        "flagged; use `require_amp=True`).\n"
    )
    add(
        "Criteria: `invalid` (no positive excursion or no wrap marker); `has_nan`; "
        "`exact_duplicate` (identical raw bytes in all four traces); `near_duplicate` "
        f"(shape r ≥ {cfg['duplicate_corr_min']} and jitter r ≥ "
        f"{cfg['duplicate_fingerprint_corr_min']}); `truncated`; `step_artefact` "
        f"(single-sample jump > {cfg['step_score_max']:.0f}× local slope); `multi_peak` "
        "(another maximum ≥ max(0.3·excursion, 10 Hz) inside the template but outside the "
        "chirp's 10% crossings); `outlier` (log width at 10% or 50% more than 5 robust SD "
        "from the family median, or undershoot deeper than the excursion); "
        f"`eodf_atypical` (baseline < {cfg['eodf_atypical_below_hz']:.0f} Hz); `ripple_flag` "
        f"(frequency ripple RMS > {cfg['ripple_score_max']} × excursion; below that the "
        "chirp shape dominates visibly, `figures/flags.png`).\n"
    )

    add("## 5. Distributions\n")
    add("Default selection (`figures/distributions.png`, `figures/families.png`):\n")
    add("| | small | big |")
    add("|---|---|---|")
    add(f"| n | {len(small)} | {len(big)} |")
    for label, col, scale in [
        ("excursion (Hz)", "excursion_hz", 1.0),
        ("full width at 10% (ms)", "duration_10_s", 1e3),
        ("full width at 50% (ms)", "duration_50_s", 1e3),
        ("undershoot (Hz)", "undershoot_hz", 1.0),
        ("asymmetry", "asymmetry", 1.0),
        ("baseline EODf (Hz)", "baseline_hz", 1.0),
    ]:
        add(f"| {label} | {_q(scale * small[col])} | {_q(scale * big[col])} |")
    add(
        f"| amp dip (amp_usable) | {_q(usable.loc[usable['family'] == 'small', 'amp_dip'])} "
        f"| {_q(usable.loc[usable['family'] == 'big', 'amp_dip'])} |"
    )
    add("\nValues: median (5–95%).\n")
    add(
        f"Against the v1 clustered rows (same measurement code, `figures/v1_comparison.png`): "
        f"excursion {_q(v1['excursion_hz'])} Hz vs {_q(dmeta['excursion_hz'])} Hz; width at "
        f"10% {_q(1e3 * v1['duration_10_s'])} ms vs {_q(1e3 * dmeta['duration_10_s'])} ms. "
        f"The v1 filter (width > 9 ms at the thesis definition, height < 550 Hz) removed "
        f"{n - int(meta['in_v1_clustered'].sum())} rows, mostly brief chirps; "
        f"{int((dmeta['in_v1_clustered'] == False).sum())} templates of the default "  # noqa: E712
        "selection were not in v1.\n"
    )

    add("## 6. Decisions for the author\n")
    add(
        "1. **Atypical-EODf group** (320–440 Hz, brief large chirps): excluded by default. "
        "Keep out of an _A. leptorhynchus_ simulator, or include? Can you identify these "
        "recordings (species, setup)?"
    )
    long_big = meta[
        meta["outlier"] & (meta["family"] == "big") & (meta["duration_50_s"] > 0.04)
    ]
    add(
        f"2. **Long big chirps.** {len(long_big)} big chirps with FWHM "
        f"{_q(1e3 * long_big['duration_50_s'])} ms are flagged `outlier` (log-width z > 5). "
        "They are the end of a continuous tail of big-chirp durations (main mode 5–20 ms) "
        "spread over many inferred groups, not obvious artefacts. Keep them excluded?"
    )
    add(
        "3. **Source balance.** The 10 largest inferred groups hold most templates. Sample "
        "uniformly over templates (current) or first over groups? Groups are only inferred "
        "from EODf continuity."
    )
    add(
        "4. **Amplitude for templates without usable AM** (most of them): no AM, AM borrowed "
        "from a usable template of the same family (decoupled from its FM), or a measured "
        "per-family dip distribution? The library leaves this to the simulator."
    )
    add(
        "5. **Filter-gain correction** of the amplitude: keep (default, fundamental-only "
        "upper bound) or turn off (`amp_filter_correction=False`)? The truth lies between, "
        "depending on harmonic content, which is not recoverable."
    )
    add(
        "6. **Frequency trace**: `smooth_freq` (default; 3–9% lower peaks for brief chirps) "
        "or `orig_freq` (sharper but with beat-driven cycle spikes)."
    )
    add(
        f"7. **Family threshold** {cfg['big_chirp_threshold_hz']:.0f} Hz and **ripple "
        f"threshold** {cfg['ripple_score_max']} were set from the data; adjust if the "
        "simulator needs stricter templates."
    )
    add(
        "8. **Temperature and sex** are unknown per chirp; `baseline_hz` is the only "
        "conditioning variable. Conditioning on EODf assumes all recordings were made at "
        "similar temperatures.\n"
    )

    add("## Files\n")
    add(
        "- `delta_f_hz.npy`, `amp_rel.npy`: float32 (n, samples); row i ↔ `template_id` i "
        "↔ h5 row `source_row` i.\n"
        "- `templates.parquet`: metadata and flags; `near_duplicate_pairs.parquet`: scored "
        "candidate pairs.\n"
        "- `manifest.json`: config, source file size/mtime/sha256, code version, output "
        "hashes, counts.\n"
        "- `figures/`: distributions, families, flags, v1_comparison, eodf, "
        "amplitude_evidence.\n"
    )
    return "\n".join(lines) + "\n"
