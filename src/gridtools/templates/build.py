"""Build the chirp template library from ``extracted_chirps.h5``."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from gridtools.templates.config import BuildConfig
from gridtools.templates.features import (
    SnippetAnalysis,
    TemplateMetrics,
    analyse_snippet,
    extraction_filter_gain,
    measure_template,
)
from gridtools.templates.library import MIN_DURATION_S
from gridtools.templates.qc import exact_duplicates, near_duplicates
from gridtools.templates.source import (
    describe_source_file,
    find_wrap_index,
    iter_source_chunks,
    read_source_shape,
    unroll_rows,
)
from gridtools.templates.taper import (
    apply_taper_delta,
    apply_taper_ratio,
    crop_centred,
    taper_window,
)

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]

LIBRARY_FORMAT_VERSION = 1
DELTA_FILE = "delta_f_hz.npy"
AMP_FILE = "amp_rel.npy"
METADATA_FILE = "templates.parquet"
MANIFEST_FILE = "manifest.json"
PAIRS_FILE = "near_duplicate_pairs.parquet"


@dataclass(frozen=True, slots=True)
class ProcessedSnippet:
    """Result of turning one source row into a template.

    Attributes
    ----------
    wrap_index
        Position of the original window start in the stored row (-1 if not
        found).
    analysis
        Baseline, alignment and quality measures of the source snippet.
    delta_f_hz, amp_rel
        Final tapered template traces.
    amp_dip_uncorrected
        Amplitude dip before the extraction-filter gain correction.
    filter_gain_max
        Largest gain change of the extraction filter within the support
        (1 if no correction was applied).
    untapered_delta_hz
        Aligned, baseline-subtracted frequency trace before tapering
        (``duplicate_window_s`` long), used for near-duplicate detection.
    fingerprint_hz
        Aligned ``orig_freq - smooth_freq``: cycle-to-cycle jitter of the
        recording, used for near-duplicate detection.
    truncated
        The chirp support plus ramps runs into the edge of the source window.
    template_cut
        The template length cut source-backed support (never expected with
        the default length; counted in the manifest).
    support_start_s, support_stop_s
        Untapered support in template time (seconds from the centre).
    metrics
        Shape descriptors of the final template.
    """

    wrap_index: int
    analysis: SnippetAnalysis
    delta_f_hz: FloatArray
    amp_rel: FloatArray
    amp_dip_uncorrected: float
    filter_gain_max: float
    untapered_delta_hz: FloatArray
    fingerprint_hz: FloatArray
    truncated: bool
    template_cut: bool
    support_start_s: float
    support_stop_s: float
    metrics: TemplateMetrics


#: Gain of an extraction filter: ``gain(freq_hz, design_eodf_hz)``.
FilterGain = Callable[[FloatArray, float], FloatArray]


def process_snippet(
    traces: dict[str, FloatArray], samplerate_hz: float, config: BuildConfig
) -> ProcessedSnippet:
    """Turn one stored source row into an aligned, tapered template.

    Parameters
    ----------
    traces
        ``orig_freq``, ``smooth_freq``, ``orig_amp``, ``smooth_amp`` of one
        row exactly as stored (rolled). They are not modified.
    samplerate_hz
        Sampling rate of the traces.
    config
        Build configuration.

    Returns
    -------
    ProcessedSnippet
    """
    wrap = find_wrap_index(traces["smooth_freq"])
    if wrap != find_wrap_index(traces["orig_freq"]):
        wrap = -1
    unrolled = {name: unroll_rows(trace, wrap) for name, trace in traces.items()}
    return template_from_traces(unrolled, samplerate_hz, config, wrap_index=wrap)


def template_from_traces(
    traces: dict[str, FloatArray],
    samplerate_hz: float,
    config: BuildConfig,
    *,
    wrap_index: int = 0,
    absolute_freq: bool = False,
    peak_range: tuple[int, int] | None = None,
    baseline_band: tuple[float, float] | None = None,
    filter_gain: FilterGain = extraction_filter_gain,
) -> ProcessedSnippet:
    """Turn contiguous frequency and amplitude traces into a template.

    This is the library's template construction (baseline estimate,
    alignment on the 50% crossings, support detection, raised-cosine taper
    to exactly 0 / 1, measurements), shared by the chirp-chamber source rows
    (:func:`process_snippet`) and chirps harvested from grid recordings
    (:mod:`gridtools.templates.harvest`).

    Parameters
    ----------
    traces
        Contiguous ``orig_freq``, ``smooth_freq``, ``orig_amp`` and
        ``smooth_amp`` of equal length (not modified).
    samplerate_hz
        Sampling rate of the traces.
    config
        Build configuration.
    wrap_index
        Position of the window start in the stored row, recorded as
        ``ProcessedSnippet.wrap_index``: 0 for contiguous traces, -1 if a
        source row could not be unrolled (the template is then invalid).
    absolute_freq
        The frequency traces are absolute (Hz), not relative to the window
        start; the baseline is then the absolute EODf.
    peak_range
        Sample range searched for the excursion peak (None: all).
    baseline_band
        Frequency range of the samples that may count as baseline (None:
        all); see :func:`~gridtools.templates.features.analyse_snippet`.
    filter_gain
        Gain of the filter used to extract the amplitude, for
        ``amp_filter_correction`` (default: the thesis extraction filter).

    Returns
    -------
    ProcessedSnippet
    """
    fs = samplerate_hz
    n_out = config.n_samples(fs)
    wrap = wrap_index
    unrolled = traces
    freq = unrolled[f"{config.freq_source}_freq"]
    amp = unrolled[f"{config.amp_source}_amp"]
    analysis = analyse_snippet(
        freq,
        amp,
        unrolled["orig_freq"],
        unrolled["smooth_freq"],
        unrolled["smooth_amp"],
        fs,
        exclusion_pre_s=config.baseline_exclusion_pre_s,
        exclusion_post_s=config.baseline_exclusion_post_s,
        support_pad_pre_s=config.support_pad_pre_s,
        support_pad_post_s=config.support_pad_post_s,
        min_baseline_s=config.min_baseline_s,
        knot_threshold=config.knot_threshold,
        knot_mad_max_hz=config.knot_mad_max_hz,
        bridge_samples=round(config.crossing_bridge_s * fs),
        decay_level_rel=config.decay_level_rel,
        decay_level_hz=config.decay_level_hz,
        decay_smooth_s=config.decay_smooth_s,
        peak_range=peak_range,
        absolute_freq=absolute_freq,
        baseline_band=baseline_band,
        next_event_guard_s=config.taper_ramp_s,
    )
    centre = analysis.align_index
    n_dup = round(config.duplicate_window_s * fs)
    freq_crop, valid_start, valid_stop = crop_centred(
        np.nan_to_num(freq, nan=analysis.baseline_freq),
        centre,
        n_out,
        analysis.baseline_freq,
    )
    amp_crop, _, _ = crop_centred(
        np.nan_to_num(amp, nan=analysis.baseline_amp),
        centre,
        n_out,
        analysis.baseline_amp,
    )
    delta = freq_crop - analysis.baseline_freq
    shape, _, _ = crop_centred(
        np.nan_to_num(freq, nan=analysis.baseline_freq) - analysis.baseline_freq,
        centre,
        n_dup,
        0.0,
    )
    # cycle-to-cycle jitter outside the chirp (inside it, orig - smooth follows
    # the chirp's curvature, which is shared by similar chirps)
    jitter = np.nan_to_num(unrolled["orig_freq"] - unrolled["smooth_freq"])
    jitter = jitter - np.median(jitter)
    exclude_start = max(
        0, int(analysis.crossing_10.onset - config.baseline_exclusion_pre_s * fs)
    )
    exclude_stop = int(analysis.chirp_end + config.baseline_exclusion_post_s * fs) + 1
    jitter[exclude_start:exclude_stop] = 0.0
    fingerprint, _, _ = crop_centred(jitter, centre, n_dup, 0.0)
    ratio = (
        amp_crop / analysis.baseline_amp
        if analysis.baseline_amp > 0
        else np.ones(n_out, dtype=np.float64)
    )
    first = centre - n_out // 2
    window = taper_window(
        n_out,
        analysis.support_start - first,
        analysis.support_stop - first,
        config.taper_ramp_s * fs,
        valid_start=valid_start,
        valid_stop=valid_stop,
    )
    delta_f = apply_taper_delta(delta, window.weights)
    amp_rel_uncorrected = apply_taper_ratio(ratio, window.weights)
    gain = np.ones(n_out, dtype=np.float64)
    eodf = analysis.baseline_eodf_hz
    if config.amp_filter_correction and np.isfinite(eodf):
        gain = filter_gain(eodf + delta, eodf) / filter_gain(
            np.array([eodf], dtype=np.float64), eodf
        )
    amp_rel = apply_taper_ratio(ratio / gain, window.weights)
    # truncated: the support plus ramps runs into the edge of the source window
    ramp = config.taper_ramp_s * fs
    wanted_start = analysis.support_start - ramp
    wanted_stop = analysis.support_stop + ramp
    n_source = freq.size
    truncated = (
        wanted_start < 0
        or wanted_stop > n_source - 1
        or analysis.crossing_10.onset_clipped
        or analysis.crossing_10.offset_clipped
        or (analysis.decay is not None and analysis.decay.clipped)
    )
    # the template length cuts source-backed support (should never happen)
    template_cut = (
        max(wanted_start, 0.0) < first
        or min(wanted_stop, n_source - 1.0) > first + n_out - 1
    )
    metrics = measure_template(
        delta_f,
        amp_rel,
        fs,
        undershoot_window_s=config.undershoot_window_s,
        prominence_rel=config.multi_peak_prominence_rel,
        prominence_hz=config.multi_peak_prominence_hz,
        bridge_samples=round(config.crossing_bridge_s * fs),
    )
    in_support = window.weights > 0
    half = n_out // 2
    return ProcessedSnippet(
        wrap_index=wrap,
        analysis=analysis,
        delta_f_hz=delta_f,
        amp_rel=amp_rel,
        amp_dip_uncorrected=float(1.0 - np.min(amp_rel_uncorrected)),
        filter_gain_max=float(np.max(gain[in_support])) if in_support.any() else 1.0,
        untapered_delta_hz=shape,
        fingerprint_hz=fingerprint,
        truncated=bool(truncated),
        template_cut=bool(template_cut),
        support_start_s=(window.support_start - half) / fs,
        support_stop_s=(window.support_stop - half) / fs,
        metrics=metrics,
    )


def snippet_record(
    template_id: int, source_row: int, snippet: ProcessedSnippet, fs: float
) -> dict[str, Any]:
    """Metadata record of one template (measurements and source analysis)."""
    a = snippet.analysis
    m = snippet.metrics
    return {
        "template_id": template_id,
        "source_row": source_row,
        "excursion_hz": m.excursion_hz,
        "duration_10_s": m.duration_10_s,
        "duration_50_s": m.duration_50_s,
        "onset_10_s": m.onset_10_s,
        "offset_10_s": m.offset_10_s,
        "onset_50_s": m.onset_50_s,
        "offset_50_s": m.offset_50_s,
        "peak_time_s": m.peak_time_s,
        "undershoot_hz": m.undershoot_hz,
        "amp_dip": m.amp_dip,
        "amp_dip_uncorrected": snippet.amp_dip_uncorrected,
        "amp_filter_gain_max": snippet.filter_gain_max,
        "amp_peak": m.amp_peak,
        "asymmetry": m.asymmetry,
        "n_peaks": m.n_peaks,
        "n_extra_events": m.n_extra_events,
        "baseline_hz": a.baseline_eodf_hz,
        "baseline_hz_mad": a.eodf.mad_hz,
        "baseline_n_knots": a.eodf.n_knots,
        "knot_check_chirp_mad_hz": a.eodf.chirp_mad_hz,
        "ripple_freq_hz": a.ripple_freq_hz,
        "ripple_amp": a.ripple_amp,
        "beat_freq_hz": a.beat_freq_hz,
        "drift_hz": a.drift_hz,
        "step_score_freq": a.step_score_freq,
        "step_score_amp": a.step_score_amp,
        "baseline_duration_s": a.baseline_s,
        "support_start_s": snippet.support_start_s,
        "support_stop_s": snippet.support_stop_s,
        "source_wrap_index": snippet.wrap_index,
        "source_align_index": a.align_index,
        "valid": a.valid and snippet.wrap_index >= 0,
        "has_nan": a.has_nan,
        "truncated": snippet.truncated,
        "template_cut": snippet.template_cut,
        "clipped": m.clipped,
    }


@dataclass(frozen=True, slots=True)
class ProcessedSource:
    """All source rows turned into templates, before quality control.

    Attributes
    ----------
    meta
        One metadata record per row (no QC columns yet).
    delta_f_hz, amp_rel
        Final template arrays (float32).
    shape, fingerprint
        Untapered aligned frequency traces and jitter fingerprints (float32)
        for duplicate detection.
    hashes
        Raw-row digests for exact-duplicate detection.
    samplerate_hz
        Sampling rate.
    """

    meta: pd.DataFrame
    delta_f_hz: npt.NDArray[np.float32]
    amp_rel: npt.NDArray[np.float32]
    shape: npt.NDArray[np.float32]
    fingerprint: npt.NDArray[np.float32]
    hashes: list[bytes]
    samplerate_hz: float


def process_source(h5_path: Path | str, config: BuildConfig) -> ProcessedSource:
    """Turn every row of the source file into a template (no QC).

    Parameters
    ----------
    h5_path
        Path to ``extracted_chirps.h5``.
    config
        Build configuration.

    Returns
    -------
    ProcessedSource
    """
    h5_path = Path(h5_path)
    n_rows, _, fs = read_source_shape(h5_path)
    n_out = config.n_samples(fs)
    delta_all = np.zeros((n_rows, n_out), dtype=np.float32)
    amp_all = np.ones((n_rows, n_out), dtype=np.float32)
    n_dup = round(config.duplicate_window_s * fs)
    shape = np.zeros((n_rows, n_dup), dtype=np.float32)
    fingerprint = np.zeros((n_rows, n_dup), dtype=np.float32)
    hashes: list[bytes] = []
    records: list[dict[str, Any]] = []
    for chunk in iter_source_chunks(h5_path, config.chunk_rows):
        for k in range(chunk.n_rows):
            row = chunk.start_row + k
            traces = {name: chunk.traces[name][k] for name in chunk.traces}
            snippet = process_snippet(traces, fs, config)
            delta_all[row] = snippet.delta_f_hz
            amp_all[row] = snippet.amp_rel
            shape[row] = snippet.untapered_delta_hz
            fingerprint[row] = snippet.fingerprint_hz
            records.append(snippet_record(row, row, snippet, fs))
        hashes.extend(chunk.raw_hashes)
        log.info("processed %d / %d rows", chunk.start_row + chunk.n_rows, n_rows)
    return ProcessedSource(
        meta=pd.DataFrame.from_records(records),
        delta_f_hz=delta_all,
        amp_rel=amp_all,
        shape=shape,
        fingerprint=fingerprint,
        hashes=hashes,
        samplerate_hz=fs,
    )


def build_library(
    h5_path: Path | str, out_dir: Path | str, config: BuildConfig | None = None
) -> None:
    """Build the template library and write it to ``out_dir``.

    Writes ``delta_f_hz.npy`` and ``amp_rel.npy`` (float32, one row per
    template), ``templates.parquet`` (metadata and quality flags),
    ``near_duplicate_pairs.parquet`` (all scored candidate pairs) and
    ``manifest.json`` (configuration, source files, code version, counts).
    The build is deterministic: the same inputs and configuration give the
    same arrays and metadata.

    Parameters
    ----------
    h5_path
        Path to ``extracted_chirps.h5`` (read-only).
    out_dir
        Output directory; created if needed. Existing library files in it
        are overwritten.
    config
        Build configuration; defaults to ``BuildConfig()``.
    """
    config = BuildConfig() if config is None else config
    h5_path = Path(h5_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(tz=datetime.UTC)

    n_rows, n_source_samples, fs = read_source_shape(h5_path)
    n_out = config.n_samples(fs)
    log.info(
        "source %s: %d rows x %d samples at %g Hz",
        h5_path,
        n_rows,
        n_source_samples,
        fs,
    )

    processed = process_source(h5_path, config)
    meta, pairs = _quality_control(processed, config)
    meta = _add_v1_membership(meta, h5_path, n_source_samples)
    delta_all, amp_all = processed.delta_f_hz, processed.amp_rel
    del processed

    np.save(out_dir / DELTA_FILE, delta_all)
    np.save(out_dir / AMP_FILE, amp_all)
    meta.to_parquet(out_dir / METADATA_FILE, index=False)
    pairs.to_parquet(out_dir / PAIRS_FILE, index=False)

    manifest = {
        "library_format_version": LIBRARY_FORMAT_VERSION,
        "created_utc": started.isoformat(),
        "samplerate_hz": fs,
        "n_templates": int(n_rows),
        "n_samples": int(n_out),
        "centre_index": int(n_out // 2),
        "config": config.to_dict(),
        "source": {
            "h5": describe_source_file(h5_path),
            "recordings": "unknown: per-chirp source recordings were lost",
            "species": "Apteronotus leptorhynchus (assumed; not recorded per chirp)",
        },
        "code": _code_version(),
        "outputs": {
            name: _file_digest(out_dir / name)
            for name in (DELTA_FILE, AMP_FILE, METADATA_FILE, PAIRS_FILE)
        },
        "counts": qc_counts(meta),
    }
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, default=_json_default)
    )
    log.info("library written to %s", out_dir)


def add_row_flags(meta: pd.DataFrame, config: BuildConfig) -> pd.DataFrame:
    """Add the per-template family and quality flags to snippet records.

    Parameters
    ----------
    meta
        Records as written by :func:`snippet_record`.
    config
        Build configuration (thresholds).

    Returns
    -------
    pandas.DataFrame
        A copy with ``family``, ``multi_peak``, ``eodf_atypical``,
        ``too_short``, ``ripple_score``, ``ripple_flag``,
        ``amp_ripple_flag`` and ``step_flag``.
    """
    meta = meta.copy()
    meta["family"] = np.where(
        meta["excursion_hz"] >= config.big_chirp_threshold_hz, "big", "small"
    )
    meta["multi_peak"] = meta["n_extra_events"] > 0
    meta["eodf_atypical"] = meta["baseline_hz"] < config.eodf_atypical_below_hz
    meta["too_short"] = meta["duration_10_s"] < config.min_duration_10_s
    meta["ripple_score"] = meta["ripple_freq_hz"] / meta["excursion_hz"]
    meta["ripple_flag"] = ~(meta["ripple_score"] <= config.ripple_score_max)
    meta["amp_ripple_flag"] = ~(meta["ripple_amp"] <= config.amp_ripple_max) | ~(
        meta["amp_dip"] >= config.amp_dip_to_ripple_min * meta["ripple_amp"]
    )
    meta["step_flag"] = (meta["step_score_freq"] > config.step_score_max) | (
        meta["step_score_amp"] > config.step_score_max
    )
    return meta


def _quality_control(
    processed: ProcessedSource, config: BuildConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = add_row_flags(processed.meta, config)
    meta["inferred_group"] = inferred_groups(
        meta["baseline_hz"].to_numpy(dtype=float), config.group_eodf_jump_hz
    )

    exact = exact_duplicates(processed.hashes)
    invalid = ~meta["valid"].to_numpy()
    near = near_duplicates(
        processed.shape,
        processed.fingerprint,
        shape_corr_min=config.duplicate_corr_min,
        fingerprint_corr_min=config.duplicate_fingerprint_corr_min,
        candidate_corr_min=config.duplicate_candidate_corr_min,
        max_lag=config.duplicate_max_lag_samples,
        exclude=(exact >= 0) | invalid,
    )
    duplicate_of = np.where(exact >= 0, exact, near.duplicate_of)
    meta["duplicate_of"] = duplicate_of
    meta["duplicate_kind"] = np.where(
        exact >= 0, "exact", np.where(near.duplicate_of >= 0, "near", "")
    )
    meta["max_shape_corr"] = near.max_shape_corr
    meta["max_shape_corr_partner"] = near.shape_partner
    meta["max_fingerprint_corr"] = near.max_fingerprint_corr

    amp_usable = ~meta["amp_ripple_flag"] & ~meta["step_flag"] & meta["valid"]
    if config.amp_filter_correction:
        amp_usable &= meta["baseline_hz"].notna()
    meta["amp_usable"] = amp_usable

    pairs = pd.DataFrame(
        near.pairs,
        columns=[
            "template_a",
            "template_b",
            "fingerprint_corr",
            "shape_corr",
            "lag_samples",
        ],
    )
    return meta, pairs


def inferred_groups(
    baseline_hz: npt.NDArray[np.float64], jump_hz: float
) -> npt.NDArray[np.int64]:
    """Label runs of consecutive rows with continuous baseline EODf.

    The extraction script appended chirps file by file, in temporal order
    within a file, so consecutive rows with similar baseline EODf most
    likely come from the same recording. A new group starts where the EODf
    of consecutive rows differs by more than ``jump_hz`` or is unknown.
    These groups approximate recordings (or individuals); they are not
    recording identifiers.

    Parameters
    ----------
    baseline_hz
        Baseline EODf per row in source order (NaN if unknown).
    jump_hz
        EODf change that starts a new group.

    Returns
    -------
    numpy.ndarray
        Group label per row, counting from 0.
    """
    if baseline_hz.size == 0:
        return np.empty(0, dtype=np.int64)
    jump = np.abs(np.diff(baseline_hz))
    breaks = ~(jump <= jump_hz)  # NaN starts a new group
    return np.concatenate([[0], np.cumsum(breaks)]).astype(np.int64)


def _add_v1_membership(
    meta: pd.DataFrame, h5_path: Path, n_source_samples: int
) -> pd.DataFrame:
    """Mark rows whose centre crop is a row of the v1 ``clustered_chirps.npy``."""
    meta = meta.copy()
    meta["in_v1_clustered"] = False
    meta["v1_cluster_label"] = -1
    clustered_path = h5_path.parent / "clustered_chirps.npy"
    labels_path = h5_path.parent / "clustered_labels.npy"
    if not clustered_path.exists():
        return meta
    clustered = np.load(clustered_path, mmap_mode="r")
    labels = (
        np.load(labels_path)
        if labels_path.exists()
        else np.full(clustered.shape[0], -1)
    )
    width = clustered.shape[1]
    start = (n_source_samples - width) // 2
    lookup: dict[bytes, int] = {}
    for i in range(clustered.shape[0]):
        lookup.setdefault(_digest(clustered[i]), int(labels[i]))
    import h5py

    member = np.zeros(len(meta), dtype=bool)
    label = np.full(len(meta), -1, dtype=np.int64)
    with h5py.File(h5_path, "r") as f:
        dataset = f["smooth_freq"]
        assert isinstance(dataset, h5py.Dataset)
        for a in range(0, dataset.shape[0], 1024):
            block = dataset[a : a + 1024, start : start + width]
            for k in range(block.shape[0]):
                hit = lookup.get(_digest(block[k]))
                if hit is not None:
                    member[a + k] = True
                    label[a + k] = hit
    meta["in_v1_clustered"] = member
    meta["v1_cluster_label"] = label
    return meta


def _digest(row: npt.NDArray[Any]) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(row).tobytes(), digest_size=16).digest()


def qc_counts(meta: pd.DataFrame) -> dict[str, Any]:
    """Count flagged templates per criterion and the cumulative default selection.

    Parameters
    ----------
    meta
        Library metadata.

    Returns
    -------
    dict
        ``flagged`` (templates flagged by each criterion on its own) and
        ``cumulative`` (templates left after applying the criteria in order).
    """

    def col(name: str) -> npt.NDArray[np.bool_]:
        return meta[name].to_numpy(dtype=bool)

    kind = meta["duplicate_kind"].to_numpy()
    duration = meta["duration_10_s"].to_numpy(dtype=float)
    too_short = (
        col("too_short") if "too_short" in meta.columns else duration < MIN_DURATION_S
    )
    # the default exclusions of TemplateLibrary.select, in this order
    criteria: list[tuple[str, npt.NDArray[np.bool_]]] = [
        ("invalid", ~col("valid")),
        ("has_nan", col("has_nan")),
        ("exact_duplicate", kind == "exact"),
        ("near_duplicate", kind == "near"),
        ("truncated", col("truncated")),
        ("step_artefact", col("step_flag")),
        ("too_short", too_short),
    ]
    flagged = {name: int(mask.sum()) for name, mask in criteria}
    keep = np.ones(len(meta), dtype=bool)
    cumulative: dict[str, int] = {"all": len(meta)}
    for name, mask in criteria:
        keep &= ~mask
        cumulative[f"after_{name}"] = int(keep.sum())
    families = meta["family"].to_numpy()[keep]
    per_family = {
        str(family): int(np.sum(families == family)) for family in ("small", "big")
    }
    informational = {
        name: {
            "all": int(col(name).sum()),
            "in_default_selection": int((keep & col(name)).sum()),
        }
        for name in ("eodf_atypical", "multi_peak", "ripple_flag", "amp_ripple_flag")
    }
    excursion = meta["excursion_hz"].to_numpy(dtype=float)
    support = (meta["support_stop_s"] - meta["support_start_s"]).to_numpy(dtype=float)
    longest = int(np.nanargmax(np.where(keep, duration, np.nan)))
    return {
        "flagged": flagged,
        "cumulative": cumulative,
        "default_selection_per_family": per_family,
        "informational_flags": informational,
        "amp_usable_in_default_selection": int((keep & col("amp_usable")).sum()),
        "baseline_hz_recovered": int(
            np.isfinite(meta["baseline_hz"].to_numpy(dtype=float)).sum()
        ),
        "template_cut": int(col("template_cut").sum()),
        "longest_chirp": {
            "template_id": int(meta["template_id"].to_numpy()[longest]),
            "duration_10_s": float(duration[longest]),
            "excursion_hz": float(excursion[longest]),
        },
        "max_duration_10_s_all": float(np.nanmax(duration)),
        "max_support_s_all": float(np.nanmax(support)),
        "max_excursion_hz_default": float(np.nanmax(np.where(keep, excursion, np.nan))),
    }


def _code_version() -> dict[str, Any]:
    from gridtools import __version__

    info: dict[str, Any] = {
        "gridtools_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    package_dir = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=package_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", str(package_dir)],
            cwd=package_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        info["git_sha"] = sha
        info["git_dirty"] = bool(dirty)
    except (OSError, subprocess.CalledProcessError):
        info["git_sha"] = None
        info["git_dirty"] = None
    return info


def _file_digest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    msg = f"not JSON serialisable: {type(value)!r}"
    raise TypeError(msg)
