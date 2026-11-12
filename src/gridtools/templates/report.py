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
        bins=np.geomspace(1, 400, 70),
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("full width at 10% (ms)")
    _hist(
        axes[0, 2],
        {f: 1e3 * d["duration_50_s"].to_numpy() for f, d in fam.items()},
        bins=np.geomspace(1, 400, 70),
    )
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_yscale("log")
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
        rows = lib.rows_of(groups[name])
        lo = min(
            -60.0,
            1e3 * float(lib.metadata["support_start_s"].to_numpy()[rows].min()) - 20,
        )
        hi = max(
            80.0,
            1e3 * float(lib.metadata["support_stop_s"].to_numpy()[rows].max()) + 20,
        )
        for ax in axes[i]:
            ax.set_xlim(lo, hi)
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
    bins_d = np.geomspace(1, 400, 70)
    axes[1].hist(1e3 * default["duration_10_s"], bins=bins_d, histtype="step", lw=1.4)
    axes[1].hist(1e3 * v1["duration_10_s"], bins=bins_d, histtype="step", lw=1.4)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
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


def crossfade_comparison(
    h5_path: Path, lib: TemplateLibrary, ramp_s: float, max_rows: int | None = None
) -> pd.DataFrame:
    """Compare the library taper with the thesis support detection per source row.

    For each valid row, the thesis heuristic (:func:`thesis_support`) finds
    chirp start and stop on the unrolled ``smooth_freq``; the library's taper
    weights are rebuilt in source coordinates. ``lost_hz`` is the largest
    part of the chirp-relative trace inside the thesis support that the
    taper attenuates.

    Parameters
    ----------
    h5_path
        Source file.
    lib
        Library built from it (full, unselected).
    ramp_s
        Taper ramp length of the build.
    max_rows
        Only compare the first rows (for tests).

    Returns
    -------
    pandas.DataFrame
        One row per library row: ``thesis_ok``, ``extend_left_ms``,
        ``extend_right_ms``, ``lost_hz``, ``excursion_hz``.
    """
    import h5py

    from gridtools.templates.source import find_wrap_index, unroll_rows
    from gridtools.templates.taper import taper_window
    from gridtools.templates.thesis import thesis_support

    meta: Frame = lib.metadata
    fs = lib.samplerate_hz
    centre = lib.n_samples // 2
    n_rows = len(meta) if max_rows is None else min(max_rows, len(meta))
    out = np.full((n_rows, 4), np.nan)
    ok = np.zeros(n_rows, dtype=bool)
    with h5py.File(h5_path, "r") as f:
        dataset = f["smooth_freq"]
        assert isinstance(dataset, h5py.Dataset)
        for start in range(0, n_rows, 1024):
            block = dataset[start : min(start + 1024, n_rows)].astype(np.float64)
            for k in range(block.shape[0]):
                i = start + k
                if not bool(meta["valid"].iloc[i]):
                    continue
                trace = unroll_rows(block[k], find_wrap_index(block[k]))
                align = int(meta["source_align_index"].iloc[i])
                baseline = trace[align] - float(lib.delta_f_hz[i, centre])
                support = thesis_support(trace)
                if support is None:
                    continue
                ok[i] = True
                lo = align + float(meta["support_start_s"].iloc[i]) * fs
                hi = align + float(meta["support_stop_s"].iloc[i]) * fs
                weights = taper_window(trace.size, lo, hi, ramp_s * fs).weights
                delta = trace - baseline
                seg = slice(support.start, support.stop + 1)
                out[i] = (
                    (lo - support.start) / fs * 1e3,
                    (support.stop - hi) / fs * 1e3,
                    float(np.max(np.abs((1.0 - weights[seg]) * delta[seg]))),
                    float(meta["excursion_hz"].iloc[i]),
                )
    frame = pd.DataFrame(
        out, columns=["extend_left_ms", "extend_right_ms", "lost_hz", "excursion_hz"]
    )
    frame.insert(0, "thesis_ok", ok)
    return frame


def _fig_equalization(
    lib: TemplateLibrary, rng: np.random.Generator, path: Path
) -> None:
    import matplotlib.pyplot as plt

    n = 100_000
    meta: Frame = lib.metadata
    natural = lib.rows_of(lib.cycler(rng, equalize=False).draw(n))
    equal = lib.rows_of(lib.cycler(rng, equalize=True).draw(n))
    eq = lib.equalization_bins()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
    exc = meta["excursion_hz"].to_numpy()
    dur = 1e3 * meta["duration_10_s"].to_numpy()
    for rows, label in (
        (natural, "natural"),
        (equal, f"equalized ({eq.spacing} bins)"),
    ):
        axes[0].hist(
            exc[rows], bins=eq.excursion_edges_hz, histtype="step", lw=1.5, label=label
        )
        axes[1].hist(
            dur[rows],
            bins=1e3 * eq.duration_edges_s,
            histtype="step",
            lw=1.5,
            label=label,
        )
    for ax in axes[:2]:
        ax.set_yscale("log")
        ax.set_ylabel(f"draws per {n:,}")
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_xlabel("excursion (Hz)")
    axes[1].set_xlabel("full width at 10% (ms)")
    image = axes[2].imshow(
        np.log10(np.where(eq.counts > 0, eq.counts, np.nan)),
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[2].set_xlabel("duration bin")
    axes[2].set_ylabel("excursion bin")
    axes[2].set_title("templates per bin (log10; empty bins blank)")
    fig.colorbar(image, ax=axes[2])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fig_crossfade(
    h5_path: Path,
    lib: TemplateLibrary,
    ids: npt.NDArray[np.int64],
    ramp_s: float,
    path: Path,
) -> None:
    import h5py
    import matplotlib.pyplot as plt

    from gridtools.templates.source import find_wrap_index, unroll_rows
    from gridtools.templates.thesis import thesis_crossfade, thesis_support

    meta: Frame = lib.metadata
    fs = lib.samplerate_hz
    centre = lib.n_samples // 2
    fig, axes = plt.subplots(len(ids), 1, figsize=(12, 2.2 * len(ids)), squeeze=False)
    with h5py.File(h5_path, "r") as f:
        dataset = f["smooth_freq"]
        assert isinstance(dataset, h5py.Dataset)
        for ax, template_id in zip(axes[:, 0], ids, strict=True):
            i = int(lib.rows_of([int(template_id)])[0])
            raw = dataset[i].astype(np.float64)
            trace = unroll_rows(raw, find_wrap_index(raw))
            align = int(meta["source_align_index"].iloc[i])
            baseline = trace[align] - float(lib.delta_f_hz[i, centre])
            t_ms = (np.arange(trace.size) - align) / fs * 1e3
            ax.plot(t_ms, trace - baseline, color="0.6", lw=0.8, label="source")
            thesis = thesis_crossfade(trace)
            support = thesis_support(trace)
            if thesis is not None and support is not None:
                ax.plot(
                    t_ms,
                    thesis - (baseline - support.baseline),
                    lw=1,
                    label="thesis cross-fade",
                )
            ax.plot(
                (np.arange(lib.n_samples) - centre) / fs * 1e3,
                lib.delta_f_hz[i],
                lw=1,
                label="v2 taper",
            )
            lo = 1e3 * float(meta["support_start_s"].iloc[i]) - 60
            hi = 1e3 * float(meta["support_stop_s"].iloc[i]) + 60
            ax.set_xlim(lo, hi)
            ax.set_title(f"template {template_id}", fontsize=8, loc="left")
            ax.set_ylabel("Δf (Hz)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[-1, 0].set_xlabel("time from alignment point (ms)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_report(
    library_dir: Path | str,
    *,
    h5_path: Path | str | None = None,
    compare_with: Path | str | None = None,
    seed: int = 0,
    n_evidence_rows: int = 600,
    max_crossfade_rows: int | None = None,
) -> Path:
    """Write ``report.md`` and figures for a built library.

    Parameters
    ----------
    library_dir
        Library directory written by :func:`gridtools.templates.build_library`.
    h5_path
        Source file, for checks on raw rows; defaults to the path in the
        manifest. Skipped if missing.
    compare_with
        Directory of an earlier library version (its ``templates.parquet``)
        to compare the selections with.
    seed
        Seed of the random examples, row sample and sampling demonstration.
    n_evidence_rows
        Source rows re-examined for the amplitude evidence.
    max_crossfade_rows
        Limit the taper/thesis comparison to the first rows (tests).

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
    default = lib.select()
    dmeta: Frame = default.metadata
    ramp_s = float(manifest["config"]["taper_ramp_s"])
    source = Path(h5_path if h5_path is not None else manifest["source"]["h5"]["path"])
    have_source = source.exists()
    evidence = (
        source_evidence(source, meta, rng, n_evidence_rows) if have_source else None
    )
    crossfade: Frame = (
        crossfade_comparison(source, lib, ramp_s, max_crossfade_rows)
        if have_source
        else None
    )
    previous: Frame = None
    if compare_with is not None:
        previous_path = Path(compare_with) / "templates.parquet"
        if previous_path.exists():
            previous = pd.read_parquet(previous_path)

    _fig_distributions(default, fig_dir / "distributions.png")
    _fig_family_overlay(default, rng, fig_dir / "families.png")
    long_mask = (dmeta["duration_10_s"] >= 0.05).to_numpy()
    flag_groups = {
        "small (default)": _pick(dmeta, (dmeta["family"] == "small").to_numpy(), rng),
        "big (default)": _pick(dmeta, (dmeta["family"] == "big").to_numpy(), rng),
        "long, width at 10% >= 50 ms (default)": _pick(dmeta, long_mask, rng),
        "truncated (excluded)": _pick(meta, meta["truncated"].to_numpy(), rng),
        "step artefact (excluded)": _pick(meta, meta["step_flag"].to_numpy(), rng),
        "eodf_atypical (kept)": _pick(meta, meta["eodf_atypical"].to_numpy(), rng),
        "multi_peak (kept)": _pick(meta, meta["multi_peak"].to_numpy(), rng),
        "ripple_flag (kept)": _pick(meta, meta["ripple_flag"].to_numpy(), rng),
    }
    _fig_examples(lib, flag_groups, fig_dir / "flags.png", "Examples per category")
    _fig_equalization(default, rng, fig_dir / "equalization.png")
    _fig_v1_comparison(meta, dmeta, fig_dir / "v1_comparison.png")
    _fig_eodf(meta, fig_dir / "eodf.png")
    _fig_amplitude_evidence(meta, fig_dir / "amplitude_evidence.png")
    crossfade_ids = np.empty(0, dtype=np.int64)
    if crossfade is not None and have_source:
        big_loss = crossfade.index[
            crossfade["thesis_ok"]
            & (crossfade["lost_hz"] > 5)
            & ~meta["step_flag"].to_numpy()
        ].to_numpy()
        typical = crossfade.index[
            crossfade["thesis_ok"] & (crossfade["lost_hz"] < 1)
        ].to_numpy()
        picks = [
            rng.choice(a, size=min(2, a.size), replace=False)
            for a in (typical, big_loss)
            if a.size
        ]
        crossfade_ids = (
            meta["template_id"].to_numpy()[np.concatenate(picks)]
            if picks
            else crossfade_ids
        )
        if crossfade_ids.size:
            _fig_crossfade(
                source, lib, crossfade_ids, ramp_s, fig_dir / "crossfade.png"
            )

    text = _report_text(lib, default, evidence, crossfade, previous, rng)
    path = library_dir / "report.md"
    path.write_text(text)
    log.info("report written to %s", path)
    return path


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    header = "| | " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|---|" + "---:|" * len(frame.columns)
    rows = [
        f"| {index} | " + " | ".join(str(v) if v else "·" for v in row) + " |"
        for index, row in zip(frame.index, frame.to_numpy().tolist(), strict=True)
    ]
    return [header, rule, *rows]


def _report_text(
    lib: TemplateLibrary,
    default: TemplateLibrary,
    evidence: SourceEvidence | None,
    crossfade: Frame,
    previous: Frame,
    rng: np.random.Generator,
) -> str:
    from gridtools.templates.amplitude import DEFAULT_DIP_RANGE
    from gridtools.templates.build import _code_version

    meta: Frame = lib.metadata
    dmeta: Frame = default.metadata
    manifest = lib.manifest
    counts = manifest["counts"]
    cfg = manifest["config"]
    report_code = _code_version()
    n = len(meta)
    fs = lib.samplerate_hz
    small = dmeta[dmeta["family"] == "small"]
    big = dmeta[dmeta["family"] == "big"]
    eq = default.equalization_bins((10, 10))
    expected = eq.expected_draws(100_000)
    lines: list[str] = []
    add = lines.append

    add("# Chirp template library v2: build report\n")
    add(
        f"Generated {datetime.datetime.now(tz=datetime.UTC):%Y-%m-%d %H:%M} UTC from "
        f"`{manifest['source']['h5']['path']}` (sha256 "
        f"`{manifest['source']['h5'].get('sha256', 'n/a')[:16]}…`); library built by gridtools "
        f"{manifest['code']['gridtools_version']} at "
        f"`{(manifest['code'].get('git_sha') or 'unknown')[:10]}`"
        f"{' (dirty)' if manifest['code'].get('git_dirty') else ''}, report by "
        f"`{(report_code.get('git_sha') or 'unknown')[:10]}`"
        f"{' (dirty)' if report_code.get('git_dirty') else ''}. "
        f"{n} templates × {lib.n_samples} samples at {fs:.0f} Hz "
        f"({1e3 * lib.n_samples / fs:.0f} ms, the full source window), alignment point "
        f"(midpoint of the 50% crossings) at sample {lib.n_samples // 2}. The v1 library "
        "and its report stay in `../v1` for comparison.\n"
    )

    add("## What changed since v1\n")
    add(
        "- **No length crop.** Templates are 0.4 s, the length of the source windows, so no "
        f"chirp is shortened by the template (`template_cut` in the default selection: "
        f"{int((dmeta['template_cut']).sum())}). `truncated` now means only that the chirp's "
        "support runs into the edge of its source window.\n"
        "- **Excursion and duration are no longer exclusion criteria.** v1's robust-z "
        "`outlier` flag is gone; the default selection drops only demonstrable artefacts "
        "(invalid, NaN, exact duplicates, truncated, step artefacts) plus the atypical-EODf "
        "group.\n"
        "- **Long chirps are measured as one event.** The 50% crossings bridge dips shorter "
        f"than {1e3 * cfg['crossing_bridge_s']:.0f} ms (phase slips on a plateau), and the "
        "10% crossings walk outward from that core without bridging, so baseline ripple is "
        "never merged into a chirp. The provisional baseline uses the thesis rule (median of "
        "samples within [min, |min|] of the window-start-relative trace), so it is not pulled "
        "up by chirps that fill most of the window; exclusion margins shrink when a long "
        "chirp leaves too little baseline.\n"
        "- **Histogram equalization** as sampling weights: `TemplateLibrary.sample(rng, n, "
        "equalize=True, bins=(10, 10))` (section 5).\n"
        "- **Amplitude** defaults to the thesis model: `amplitude_from_frequency(delta_f_hz, "
        f"dip)` with `dip ~ U{DEFAULT_DIP_RANGE}`; the measured `amp_rel` stays optional "
        "(section 6).\n"
        "- `baseline_hz` is metadata only; nothing in the library conditions on it.\n"
    )

    add("## 1. Consistency with the thesis\n")
    v1_rows = meta["in_v1_clustered"].to_numpy(dtype=bool)
    atyp_v1 = int((meta["eodf_atypical"] & meta["in_v1_clustered"]).sum())
    brief_kept = int((dmeta["duration_10_s"] < 0.009).sum())
    high_kept = int((dmeta["excursion_hz"] >= 550).sum())
    add(
        "| thesis | v2 library |\n|---|---|\n"
        "| Chirps made relative to the emitter's baseline so they fit fish of any EODf | "
        "`delta_f_hz` is baseline-relative (0 at baseline); `baseline_hz` is recovered and "
        "stored as metadata only |\n"
        "| Real chirps from the chirp-chamber extraction (9,082 after filtering and "
        "clustering) | the same extraction (`extracted_chirps.h5` is the exact source of "
        f"all {int(v1_rows.sum())} clustered chirps), all 13,015 rows considered |\n"
        "| Smooth baseline around each chirp by a cross-fade (Gaussian σ = 60 samples, "
        "120-sample linear blend) | raised-cosine taper to exactly 0 outside the support "
        "(compared below) |\n"
        "| Long and high chirps kept on purpose | no excursion or duration exclusion; "
        f"longest kept chirp {1e3 * counts['longest_chirp']['duration_10_s']:.0f} ms "
        f"(width at 10%), highest {counts['max_excursion_hz_default']:.0f} Hz |\n"
        "| Histogram equalization of heights and widths by duplicating chirps (121,380 rows) "
        "| sampling weights over a 10 × 10 grid of excursion × width at 10%, no duplicated "
        "rows |\n"
        "| Chirp AM: inverted frequency modulation, scaled so the EOD drops to 100–20% | "
        f"`amplitude_from_frequency`, dip ~ U{DEFAULT_DIP_RANGE} (simulator default) |\n"
    )
    add(
        "Differences that remain: (1) the thesis clustering step dropped chirps narrower "
        "than 9 ms (its width) and higher than 550 Hz; v2 keeps them "
        f"({brief_kept} templates of the default selection are narrower than 9 ms at 10%, "
        f"{high_kept} reach 550 Hz or more). (2) The thesis cross-fade blends the first and "
        "last 6 ms of the chirp itself with a smoothed copy; v2 leaves the chirp untouched "
        "and ramps outside it. (3) The atypical-EODf group is excluded by default "
        f"({atyp_v1} of its rows were in the thesis set). (4) Exact duplicates "
        f"({counts['flagged']['exact_duplicate']}) are dropped; equalization replaces them "
        "as the balancing mechanism.\n"
    )
    if crossfade is not None:
        cf = crossfade[crossfade["thesis_ok"]]
        lost = cf["lost_hz"]
        rel = lost / cf["excursion_hz"]
        step = meta["step_flag"].to_numpy(dtype=bool)
        big_loss = cf.index[cf["lost_hz"] > 5]
        add(
            "**Taper vs thesis cross-fade.** The thesis support detection (descend both "
            "flanks of the smoothed trace; on the right until the slope has changed sign "
            "twice, past the undershoot) was ported (`gridtools.templates.thesis`) and run "
            f"on every valid row. It fails on {int((~crossfade['thesis_ok'] & meta['valid'].to_numpy()).sum())} "
            "rows (more than two half-height crossings; the thesis discarded those). Where it "
            f"works, it starts {_q(cf['extend_left_ms'])} ms before and ends "
            f"{_q(cf['extend_right_ms'])} ms after the v2 support (median, 5–95%; negative = "
            "inside). The part of the chirp-relative trace inside the thesis support that the "
            f"v2 taper attenuates is {_q(lost)} Hz ({_q(100 * rel)}% of the excursion). "
            f"{len(big_loss)} rows lose more than 5 Hz; {int(step[big_loss].sum())} of them "
            "carry step artefacts. Inspection of these rows (`figures/crossfade.png`, last "
            "rows) shows the attenuated part is baseline ripple or a later bump that the slope "
            "descent runs into, not chirp structure; the v2 support (10% onset − "
            f"{1e3 * cfg['support_pad_pre_s']:.0f} ms to 10% offset + "
            f"{1e3 * cfg['support_pad_post_s']:.0f} ms, which contains the undershoot) "
            "therefore stays. In typical rows both give the same chirp and differ only in "
            "how much neighbouring baseline they keep: v2 starts a few ms earlier, the thesis "
            "often ends later; both baselines carry the source's ripple there "
            "(`figures/crossfade.png`, first rows). The two also estimate the baseline "
            "differently (median outside the chirp vs median within [min, |min|]), which "
            "can shift the whole trace by a few Hz.\n"
        )

    add("## 2. Source arrays (summary; details in the v1 report)\n")
    add(
        "13,015 rows of 0.4 s at 20 kHz written by the thesis extraction script. Frequencies "
        "are relative to the first sample of each window; rows were centred with a circular "
        "shift, which the builder undoes. `smooth_freq` = `orig_freq` smoothed with a 1 ms "
        "Gaussian (verified bit-exactly). The absolute baseline EODf is recovered from the "
        f"spline knots of `orig_freq` for {counts['baseline_hz_recovered']} rows (median "
        f"{meta['baseline_hz'].median():.0f} Hz). {int(meta['eodf_atypical'].sum())} rows "
        "form a separate 320–440 Hz group. Rows were appended per file; the 10 largest "
        "inferred groups hold "
        f"{100 * meta['inferred_group'].value_counts().iloc[:10].sum() / n:.0f}% of rows.\n"
    )

    add("## 3. Template length and the longest chirps\n")
    longest = dmeta.sort_values("duration_10_s", ascending=False).head(5)
    add(
        f"Longest chirp in the default selection: template "
        f"{counts['longest_chirp']['template_id']}, "
        f"{1e3 * counts['longest_chirp']['duration_10_s']:.0f} ms at 10% with "
        f"{counts['longest_chirp']['excursion_hz']:.0f} Hz excursion. The five longest: "
        + ", ".join(
            f"{1e3 * d:.0f} ms / {e:.0f} Hz"
            for d, e in zip(
                longest["duration_10_s"], longest["excursion_hz"], strict=True
            )
        )
        + ". Widths at 10% of the default selection: median "
        f"{1e3 * dmeta['duration_10_s'].median():.1f} ms, 99th percentile "
        f"{1e3 * dmeta['duration_10_s'].quantile(0.99):.1f} ms, maximum "
        f"{1e3 * dmeta['duration_10_s'].max():.1f} "
        "ms. No chirp in the source reaches 500 ms: the source windows are "
        "400 ms and the extraction discarded chirps whose >20 Hz region touched a window "
        "edge. Longer apparent widths (up to "
        f"{1e3 * counts['max_duration_10_s_all']:.0f} ms) occur only in excluded rows "
        "(truncated or step artefacts, where the trace never returns to baseline). Longest "
        f"support including undershoot: {1e3 * counts['max_support_s_all']:.0f} ms. "
        "`figures/flags.png` shows long chirps of the default selection.\n"
    )

    add("## 4. Exclusions\n")
    add(
        "Flags stay in `templates.parquet`; `TemplateLibrary.select()` drops these by default:\n"
    )
    add("| criterion | flagged | left |")
    add("|---|---:|---:|")
    add(f"| all rows | | {counts['cumulative']['all']} |")
    for name, flagged in counts["flagged"].items():
        add(f"| {name} | {flagged} | {counts['cumulative'][f'after_{name}']} |")
    add(
        f"\nDefault selection: **{default.n_templates}** templates "
        f"({counts['default_selection_per_family']['small']} small, "
        f"{counts['default_selection_per_family']['big']} big; family = excursion ≥ "
        f"{cfg['big_chirp_threshold_hz']:.0f} Hz, descriptive only). "
        "Why each is excluded: `truncated`, the chirp's support runs into the source window "
        "edge, so part of it is missing; `step_artefact`, single-sample jumps impossible "
        "after a 1 ms Gaussian, left by the extraction's in-place offsets on overlapping "
        "windows (the trace is broken); exact duplicates add nothing; the atypical-EODf "
        "group is an open question.\n"
    )
    info = counts["informational_flags"]
    add(
        "Flagged but kept: "
        + "; ".join(
            f"`{k}` {v['all']} ({v['in_default_selection']} in the default selection)"
            for k, v in info.items()
        )
        + ". `multi_peak` marks a second maximum outside the chirp but inside the "
        "template, `ripple_flag` a frequency ripple RMS above "
        f"{cfg['ripple_score_max']} × excursion (beats with another EOD); both are real "
        "chirps with contamination, not broken traces. Use `select(exclude=(..., "
        "'ripple_flag'))` for stricter sets.\n"
    )
    if previous is not None and "outlier" in previous.columns:
        old = previous["outlier"].to_numpy(dtype=bool)
        now_in = old & np.isin(
            meta["template_id"].to_numpy(), dmeta["template_id"].to_numpy()
        )
        reasons = {
            "eodf_atypical": int(
                (old & meta["eodf_atypical"].to_numpy(dtype=bool)).sum()
            ),
            "step_artefact": int((old & meta["step_flag"].to_numpy(dtype=bool)).sum()),
            "truncated": int((old & meta["truncated"].to_numpy(dtype=bool)).sum()),
            "duplicate": int((old & (meta["duplicate_of"].to_numpy() >= 0)).sum()),
        }
        old_long_big = (
            old
            & (previous["family"].to_numpy() == "big")
            & (previous["duration_50_s"].to_numpy() > 0.04)
        )
        add(
            f"**v1 outliers re-examined.** All {int(old.sum())} v1 outliers were inspected "
            "on their source traces. They were: the atypical-EODf group, long big chirps "
            "(300–450 Hz plateaus of 55–135 ms with the matching amplitude drop), small chirps "
            "with long low shoulders (30–40 ms at 10%), and brief events (FWHM 2–4 ms). None "
            f"is excluded for its size or length any more: {int(now_in.sum())} are in the v2 "
            "default selection (including "
            f"{int((old_long_big & now_in).sum())} of the {int(old_long_big.sum())} long big "
            "chirps); the others carry an artefact flag or the atypical-EODf flag "
            "(overlapping counts: "
            + ", ".join(f"{k} {v}" for k, v in reasons.items())
            + ").\n"
        )
    brief = dmeta[dmeta["duration_50_s"] < 0.003]
    add(
        f"**Brief events kept.** {len(brief)} templates of the default selection have a "
        "smoothed FWHM below 3 ms, barely wider than the 1 ms Gaussian itself (FWHM 2.35 "
        "ms), so they span one or two EOD cycles. Some coincide with beat troughs (phase "
        "slips), others look clean; a single-cycle event cannot be told from a zero-crossing "
        "error, so they are kept and listed as an open question.\n"
    )

    add("## 5. Histogram equalization\n")
    add(
        f"`sample(rng, n, equalize=True, bins=(10, 10))` bins the sampling pool (default "
        f"selection, one template per duplicate group: {eq.template_ids.size}) into 10 "
        "equal-width bins of excursion × 10 equal-width bins of **full width at 10% of the "
        "excursion** (`duration_10_s`). That is the thesis width (crossings descended from "
        "half height to 10% of the peak) and covers the whole chirp, while the width at 50% "
        "under-measures chirps with slow onsets or shoulders. Every non-empty bin is equally "
        "likely (bins in successive random permutations), and within a bin templates come "
        "in successive random permutations, so repeats are spread evenly and a cycler keeps "
        f"its balance across calls. {eq.n_nonempty} of 100 bins are occupied (rows: "
        "excursion, columns: width at 10%):\n"
    )
    lines.extend(_markdown_table(eq.table()))
    singles = int((eq.counts == 1).sum())
    add(
        f"\nPer 100,000 draws a template is drawn {np.median(expected):.2f} times "
        f"(median; min {expected.min():.2f}, max {expected.max():,.0f}). "
        "The rarest templates (alone in their bin, "
        f"{singles} bins) are drawn {expected.max():,.0f} times each; templates in the most "
        f"crowded bin ({int(eq.counts.max())} members) {expected.min():.2f} times; "
        f"{int((expected > 100).sum())} templates exceed 100 draws. With "
        "`equalize=False` every template is drawn 100,000 / "
        f"{eq.template_ids.size} = {1e5 / eq.template_ids.size:.1f} times. "
        "`figures/equalization.png` compares natural and equalized draws. Repeating a rare "
        "chirp often is what equalization means; the thesis accepted it (the same chirp "
        "appears against different backgrounds and fish).\n"
    )

    add("## 6. Amplitude\n")
    add(
        "**Default (thesis model):** `amplitude_from_frequency(delta_f_hz, dip)` = "
        "1 − dip · clip(Δf, 0, ∞) / max(Δf), with `dip ~ U(0, 0.8)` per chirp "
        "(`random_dips`): the EOD drops to between 100% and 20% at the excursion peak. The "
        "undershoot is clipped so the amplitude never exceeds 1 (a frequency undershoot does "
        "not imply a louder EOD). Templates are exactly 0 at both ends, so the envelope is "
        "exactly 1 there.\n"
    )
    usable = dmeta[dmeta["amp_usable"]]
    add(
        "**Optional (measured):** `amp_rel` with `amp_usable` "
        f"({int(meta['amp_usable'].sum())} templates; {len(usable)} in the default "
        "selection). The v1 evidence still holds: the baseline amplitude ripple is a beat "
        "with a second EOD (frequency ripple / amplitude ripple ≈ beat frequency), it is "
        "slow (median ≈ 13 Hz) so a low-pass cannot remove it without shrinking real dips, "
        "and the extraction band-pass fakes +7.5% to +12.5% amplitude for 30–60 Hz chirps, "
        "which `amp_rel` corrects (fundamental only, an upper bound). Clean measured dips: "
        f"small {_q(usable.loc[usable['family'] == 'small', 'amp_dip'])}, big "
        f"{_q(usable.loc[usable['family'] == 'big', 'amp_dip'])}; the default model's dips "
        "are uniform on 0–0.8 for every chirp. `figures/amplitude_evidence.png`.\n"
    )
    if evidence is not None:
        add(
            "Low-pass test on source rows (median ratio filtered/raw): "
            + "; ".join(
                f"{fc:.0f} Hz: clean dips ×{v[0]:.2f}, contaminated ripple ×{v[1]:.2f}"
                for fc, v in evidence.lowpass.items()
            )
            + ".\n"
        )

    add("## 7. Distributions of the default selection\n")
    add("| | small | big |")
    add("|---|---|---|")
    add(f"| n | {len(small)} | {len(big)} |")
    for label, col, scale in [
        ("excursion (Hz)", "excursion_hz", 1.0),
        ("full width at 10% (ms)", "duration_10_s", 1e3),
        ("full width at 50% (ms)", "duration_50_s", 1e3),
        ("undershoot (Hz)", "undershoot_hz", 1.0),
        ("asymmetry", "asymmetry", 1.0),
    ]:
        add(f"| {label} | {_q(scale * small[col])} | {_q(scale * big[col])} |")
    add(
        "\nMedian (5–95%). `figures/distributions.png` (log-scaled widths), "
        "`figures/families.png`, `figures/v1_comparison.png`.\n"
    )

    add("## 8. Open questions for the author\n")
    add(
        "1. Atypical-EODf group (320–440 Hz, 3–5 ms chirps of 60–220 Hz): still excluded by "
        "default. Which recordings are these?\n"
        f"2. Brief events ({len(brief)} templates with FWHM < 3 ms, one or two EOD "
        "cycles): kept. Exclude them, or the subset at beat troughs?\n"
        "3. Ripple-flagged chirps (beat contamination of the frequency trace) are kept by "
        "default; exclude them for cleaner templates?\n"
        "4. Equalization on width at 10% (thesis width) or at 50%? The bins are linear, so "
        "the long and high tails get few, heavily repeated templates; log-spaced bins would "
        "repeat them less.\n"
    )
    add("## Files\n")
    add(
        "- `delta_f_hz.npy`, `amp_rel.npy`: float32 (n, 8000); row i ↔ `template_id` i ↔ "
        "h5 row `source_row` i.\n"
        "- `templates.parquet` (metadata, flags), `near_duplicate_pairs.parquet`, "
        "`manifest.json` (config, source size/mtime/sha256, code version, hashes, counts).\n"
        "- `figures/`: distributions, families, flags, equalization, crossfade, "
        "v1_comparison, eodf, amplitude_evidence.\n"
    )
    del rng
    return "\n".join(lines) + "\n"
