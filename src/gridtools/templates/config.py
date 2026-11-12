"""Configuration of the chirp template library build."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

TraceKind = Literal["smooth", "orig"]


@dataclass(frozen=True, slots=True)
class BuildConfig:
    """Parameters of :func:`gridtools.templates.build_library`.

    All durations are in seconds and all frequencies in Hz. The defaults are
    the values used for library version ``v1``; the report explains how each
    one was chosen.

    Parameters
    ----------
    template_duration_s
        Length of every template. The alignment point sits at sample
        ``n_samples // 2``. The default equals the source window (0.4 s), so
        no chirp is cut by the template: a support of up to 0.2 s on either
        side of the alignment point fits (the build reports any that do not).
    crossing_bridge_s
        Dips below a crossing level shorter than this are bridged, so long
        chirps with brief internal dips count as one event.
    freq_source
        Frequency trace used for ``delta_f_hz``: ``"smooth"`` (Gaussian,
        sigma = 1 ms, applied by the extraction code) or ``"orig"``
        (per-cycle instantaneous frequency).
    amp_source
        Amplitude trace used for ``amp_rel``.
    amp_filter_correction
        Divide the amplitude by the gain change of the extraction band-pass
        filter at the instantaneous frequency (quasi-static, fundamental
        only). Needs the recovered baseline EODf; templates without one get
        ``amp_usable = False``.
    baseline_exclusion_pre_s, baseline_exclusion_post_s
        Samples within ``[onset_10 - pre, offset_10 + post]`` are excluded
        from the baseline estimate; ``post`` is longer to exclude the
        undershoot.
    support_pad_pre_s, support_pad_post_s
        The template keeps the trace unchanged on
        ``[onset_10 - pad_pre, end + pad_post]``, where ``end`` is the 10%
        offset or, with the decay extension, the end of the slow decay.
    decay_level_rel, decay_level_hz, decay_smooth_s
        Decay extension for long chirps whose slow, exponential-like decay
        stays above baseline long after the 10% offset. If
        ``decay_level_rel`` is set, the chirp ends where the trace (smoothed
        with sigma ``decay_smooth_s``) has fallen to ``max(decay_level_rel *
        excursion, decay_level_hz)``, or at the trough before it rises again
        by that much (a following chirp). The baseline exclusion and the
        support pad then count from that point. None (library v1/v2) keeps
        the end at the 10% offset.
    taper_ramp_s
        Length of the raised-cosine ramps on both sides of the support.
        Outside support and ramps the template is exactly baseline.
    undershoot_window_s
        Window after ``offset_10`` in which the undershoot is measured.
    big_chirp_threshold_hz
        Excursion at and above which a template is labelled ``"big"``.
        Descriptive only.
    multi_peak_prominence_rel, multi_peak_prominence_hz
        A local maximum within the support counts as a separate peak if its
        prominence is at least ``max(rel * excursion, hz)``.
    eodf_atypical_below_hz
        ``eodf_atypical`` is set for baseline EODf below this value (the
        recovered EODf distribution has an empty gap at 440--540 Hz).
        Descriptive only since library v3 (the group is kept).
    min_duration_10_s
        ``too_short`` is set for templates whose full width at 10% of the
        excursion is below this value (excluded by default).
    max_brief_cycles
        ``brief_cycles`` is set when the full width at half maximum spans at
        most this many cycles of the baseline EODf (excluded by default).
    ripple_score_max
        ``ripple_flag`` is set when the RMS of the detrended frequency
        baseline exceeds this fraction of the excursion.
    amp_ripple_max
        ``amp_ripple_flag`` is set when the RMS of the detrended relative
        amplitude baseline exceeds this value.
    amp_dip_to_ripple_min
        ``amp_ripple_flag`` is also set when the amplitude dip is smaller
        than this multiple of the amplitude ripple.
    step_score_max
        ``step_flag`` is set when a single-sample jump in a smoothed trace
        exceeds this multiple of the local slope scale.
    duplicate_window_s
        Length of the aligned traces compared for duplicate detection.
    duplicate_corr_min
        Minimum correlation of the aligned, untapered frequency traces of a
        near-duplicate pair.
    duplicate_fingerprint_corr_min
        Minimum correlation of the cycle-to-cycle jitter
        (``orig_freq - smooth_freq``) of a near-duplicate pair; distinct
        chirps have independent jitter.
    duplicate_candidate_corr_min
        Zero-lag fingerprint correlation that makes a pair a candidate.
    duplicate_max_lag_samples
        Lags searched when comparing near-duplicate candidates.
    group_eodf_jump_hz
        Change in baseline EODf between consecutive source rows that starts
        a new ``inferred_group``.
    knot_threshold
        Detection threshold for spline knots (zero crossings) in the fourth
        difference of the ``orig_freq`` trace, relative to the local median.
    knot_mad_max_hz
        The baseline EODf is reported only when the per-knot estimates agree
        to within this median absolute deviation.
    min_baseline_s
        Baseline duration wanted; for long chirps the exclusion margins
        shrink until at least this much baseline remains.
    chunk_rows
        Rows read from the HDF5 file at a time.
    """

    template_duration_s: float = 0.4
    crossing_bridge_s: float = 0.005
    freq_source: TraceKind = "smooth"
    amp_source: TraceKind = "smooth"
    amp_filter_correction: bool = True
    baseline_exclusion_pre_s: float = 0.02
    baseline_exclusion_post_s: float = 0.06
    support_pad_pre_s: float = 0.005
    support_pad_post_s: float = 0.03
    decay_level_rel: float | None = None
    decay_level_hz: float = 0.0
    decay_smooth_s: float = 0.005
    taper_ramp_s: float = 0.01
    undershoot_window_s: float = 0.05
    big_chirp_threshold_hz: float = 200.0
    multi_peak_prominence_rel: float = 0.3
    multi_peak_prominence_hz: float = 10.0
    eodf_atypical_below_hz: float = 500.0
    min_duration_10_s: float = 0.005
    max_brief_cycles: float = 2.0
    ripple_score_max: float = 0.08
    amp_ripple_max: float = 0.02
    amp_dip_to_ripple_min: float = 3.0
    step_score_max: float = 20.0
    duplicate_window_s: float = 0.2
    duplicate_corr_min: float = 0.999
    duplicate_fingerprint_corr_min: float = 0.99
    duplicate_candidate_corr_min: float = 0.9
    duplicate_max_lag_samples: int = 20
    group_eodf_jump_hz: float = 10.0
    knot_threshold: float = 20.0
    knot_mad_max_hz: float = 1.0
    min_baseline_s: float = 0.05
    chunk_rows: int = 512

    def __post_init__(self) -> None:
        if self.template_duration_s <= 0:
            msg = "template_duration_s must be positive"
            raise ValueError(msg)
        if self.freq_source not in ("smooth", "orig"):
            msg = f"freq_source must be 'smooth' or 'orig', got {self.freq_source!r}"
            raise ValueError(msg)
        if self.amp_source not in ("smooth", "orig"):
            msg = f"amp_source must be 'smooth' or 'orig', got {self.amp_source!r}"
            raise ValueError(msg)
        if not 0.0 < self.duplicate_corr_min <= 1.0:
            msg = "duplicate_corr_min must be in (0, 1]"
            raise ValueError(msg)
        if self.chunk_rows < 1:
            msg = "chunk_rows must be >= 1"
            raise ValueError(msg)
        if self.decay_level_rel is not None and not 0.0 < self.decay_level_rel < 0.1:
            msg = "decay_level_rel must lie in (0, 0.1) (below the 10% crossing)"
            raise ValueError(msg)

    def n_samples(self, samplerate_hz: float) -> int:
        """Template length in samples at ``samplerate_hz``."""
        return round(self.template_duration_s * samplerate_hz)

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a JSON-serialisable dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> BuildConfig:
        """Create a configuration from :meth:`to_dict` output."""
        return cls(**values)
