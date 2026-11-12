"""Measurements on chirp snippets and templates.

Conventions
-----------
- ``delta`` traces are frequency excursions in Hz relative to the baseline
  EOD frequency (0 = baseline).
- Positions are fractional sample indices unless a name ends in ``_s``.
- Level crossings are found by walking outward from the excursion peak to the
  first sample at or below ``level * excursion`` and interpolating linearly
  between that sample and its neighbour above the level.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Crossing:
    """Two crossings of a level around a peak.

    Attributes
    ----------
    onset, offset
        Fractional sample positions where the trace crosses the level before
        and after the peak.
    onset_clipped, offset_clipped
        True if the trace never fell to the level before reaching the start
        (end) of the array; the crossing is then the array edge.
    """

    onset: float
    offset: float
    onset_clipped: bool
    offset_clipped: bool

    @property
    def width(self) -> float:
        """Distance between offset and onset in samples."""
        return self.offset - self.onset

    @property
    def midpoint(self) -> float:
        """Midpoint between onset and offset in samples."""
        return 0.5 * (self.onset + self.offset)


def level_crossings(
    delta: FloatArray, peak_index: int, level: float, max_gap: int = 0
) -> Crossing:
    """Find where ``delta`` crosses ``level * delta[peak_index]`` around the peak.

    Parameters
    ----------
    delta
        1-D trace with its baseline at 0.
    peak_index
        Index of the excursion peak; ``delta[peak_index]`` must be positive.
    level
        Fraction of the peak value, in (0, 1).
    max_gap
        Dips below the level that last fewer than ``max_gap`` samples and are
        bounded by samples above the level on both sides are bridged, so a
        long chirp with brief internal dips (phase slips on a plateau) is
        measured as one event.

    Returns
    -------
    Crossing
        Interpolated onset and offset positions.
    """
    peak_value = float(delta[peak_index])
    if not peak_value > 0:
        msg = "delta[peak_index] must be positive"
        raise ValueError(msg)
    threshold = level * peak_value
    above = delta > threshold
    if max_gap > 0:
        for start, stop in mask_segments(~above):
            if start > 0 and stop < delta.size and stop - start < max_gap:
                above[start:stop] = True
    below = ~above

    left_below = np.flatnonzero(below[:peak_index])
    if left_below.size == 0:
        onset, onset_clipped = 0.0, True
    else:
        i0 = int(left_below[-1])
        onset = i0 + _fraction(delta[i0], delta[i0 + 1], threshold)
        onset_clipped = False

    right_below = np.flatnonzero(below[peak_index + 1 :])
    if right_below.size == 0:
        offset, offset_clipped = float(delta.size - 1), True
    else:
        i1 = peak_index + 1 + int(right_below[0])
        offset = (i1 - 1) + _fraction(delta[i1 - 1], delta[i1], threshold)
        offset_clipped = False
    return Crossing(onset, offset, onset_clipped, offset_clipped)


def chirp_crossings(
    delta: FloatArray, peak_index: int, max_gap: int = 0
) -> tuple[Crossing, Crossing]:
    """Crossings of 10% and 50% of the excursion around the peak.

    The 50% crossings bridge dips shorter than ``max_gap`` samples (the core
    of a long chirp with brief internal dips). The 10% crossings are then
    found by walking outward from that core without bridging, so baseline
    ripple that pokes above 10% next to a chirp is never merged into it.

    Parameters
    ----------
    delta
        1-D trace with its baseline at 0.
    peak_index
        Index of the excursion peak.
    max_gap
        Longest dip (samples) bridged at the 50% level.

    Returns
    -------
    crossing_10, crossing_50 : Crossing
    """
    c50 = level_crossings(delta, peak_index, 0.5, max_gap)
    core = np.array(delta, dtype=np.float64, copy=True)
    lo = max(int(np.ceil(c50.onset)), 0)
    hi = min(int(np.floor(c50.offset)), core.size - 1)
    core[lo : hi + 1] = np.maximum(core[lo : hi + 1], delta[peak_index])
    c10 = level_crossings(core, peak_index, 0.1)
    return c10, c50


@dataclass(frozen=True, slots=True)
class DecayEnd:
    """End of a chirp's slow decay toward baseline.

    Attributes
    ----------
    position
        Fractional sample where the smoothed excursion has fallen to the
        level (or the trough where it started to rise again).
    interrupted
        The trace rose again before reaching the level (for example a
        following chirp); ``position`` is the trough before the rise.
    clipped
        The trace neither reached the level nor rose again before the end
        of the array; ``position`` is the last sample.
    """

    position: float
    interrupted: bool
    clipped: bool


def decay_end(
    delta: FloatArray, start: float, level_hz: float, smooth_samples: float = 0.0
) -> DecayEnd:
    """Follow a decay from ``start`` until it reaches ``level_hz``.

    Slow, exponential-like decays of long chirps stay above baseline long
    after their 10% crossing. This walks right from ``start`` along the
    (Gaussian-smoothed) trace until it first falls to ``level_hz`` or
    below. It stops early at a trough when the trace rises more than
    ``level_hz`` above its running minimum, so a following chirp is never
    merged into the decay.

    Parameters
    ----------
    delta
        1-D trace with its baseline at 0.
    start
        Fractional sample to start from (usually the 10% offset).
    level_hz
        Level that counts as back at baseline; also the rise tolerance.
    smooth_samples
        Sigma of the Gaussian applied before walking (0: none).

    Returns
    -------
    DecayEnd
    """
    trace = np.asarray(delta, dtype=np.float64)
    if smooth_samples > 0:
        trace = gaussian_filter1d(trace, smooth_samples, mode="nearest")
    i0 = min(max(int(np.ceil(start)), 0), trace.size - 1)
    segment = trace[i0:]
    below = np.flatnonzero(segment <= level_hz)
    rise = np.flatnonzero(segment - np.minimum.accumulate(segment) > level_hz)
    first_below = int(below[0]) if below.size else segment.size
    first_rise = int(rise[0]) if rise.size else segment.size
    if first_below < segment.size and first_below <= first_rise:
        return DecayEnd(float(i0 + first_below), False, False)
    if first_rise < segment.size:
        trough = int(np.argmin(segment[:first_rise]))
        return DecayEnd(float(i0 + trough), True, False)
    return DecayEnd(float(trace.size - 1), False, True)


def following_rise(
    delta: FloatArray, start: float, rise_hz: float, smooth_samples: float = 0.0
) -> float | None:
    """Start of the next rise after ``start``: the trough before the trace
    climbs ``rise_hz`` above its running minimum (None if it never does)."""
    trace = np.asarray(delta, dtype=np.float64)
    if smooth_samples > 0:
        trace = gaussian_filter1d(trace, smooth_samples, mode="nearest")
    i0 = int(np.ceil(start))
    if i0 >= trace.size:
        return None
    segment = trace[i0:]
    rises = np.flatnonzero(segment - np.minimum.accumulate(segment) > rise_hz)
    if rises.size == 0:
        return None
    return float(i0 + int(np.argmin(segment[: rises[0] + 1])))


def _fraction(a: float, b: float, threshold: float) -> float:
    """Position of ``threshold`` between samples with values ``a`` and ``b``."""
    if a == b:
        return 0.0
    return float(np.clip((threshold - a) / (b - a), 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class TemplateMetrics:
    """Shape descriptors of one template.

    Times are in seconds relative to the template centre (the alignment
    point).

    Attributes
    ----------
    excursion_hz
        Maximum of ``delta_f_hz``.
    peak_time_s
        Time of that maximum.
    onset_10_s, offset_10_s, onset_50_s, offset_50_s
        Crossings of 10% and 50% of the excursion.
    duration_10_s, duration_50_s
        Full width at 10% and 50% of the excursion.
    undershoot_hz
        ``max(0, -min(delta_f_hz))`` within ``undershoot_window_s`` after
        ``offset_10_s``; positive values mean the frequency fell below
        baseline after the chirp.
    amp_dip
        ``1 - min(amp_rel)`` over the template.
    amp_peak
        ``max(amp_rel) - 1`` over the template (amplitude increases).
    asymmetry
        ``(fall - rise) / (fall + rise)`` with ``rise = peak - onset_10`` and
        ``fall = offset_10 - peak``; 0 is symmetric, positive means a slower
        decay than rise.
    n_peaks
        Number of local maxima between the 10% crossings with prominence
        and height of at least ``max(prominence_rel * excursion,
        prominence_hz)``: internal structure of the chirp (at least 1).
    n_extra_events
        Number of such maxima outside the 10% crossings: a second chirp or
        a strong ripple peak inside the template.
    clipped
        A 10% crossing hit the template edge.
    """

    excursion_hz: float
    peak_time_s: float
    onset_10_s: float
    offset_10_s: float
    onset_50_s: float
    offset_50_s: float
    duration_10_s: float
    duration_50_s: float
    undershoot_hz: float
    amp_dip: float
    amp_peak: float
    asymmetry: float
    n_peaks: int
    n_extra_events: int
    clipped: bool


def measure_template(
    delta_f_hz: FloatArray,
    amp_rel: FloatArray,
    samplerate_hz: float,
    *,
    undershoot_window_s: float = 0.05,
    prominence_rel: float = 0.3,
    prominence_hz: float = 10.0,
    bridge_samples: int = 0,
) -> TemplateMetrics:
    """Compute shape descriptors of one template.

    Parameters
    ----------
    delta_f_hz
        Baseline-relative frequency excursion; the alignment point is sample
        ``len(delta_f_hz) // 2``.
    amp_rel
        Amplitude relative to baseline, same length.
    samplerate_hz
        Sampling rate of both traces.
    undershoot_window_s
        Window after the 10% offset for the undershoot.
    prominence_rel, prominence_hz
        Peak-counting thresholds (see :class:`TemplateMetrics`).
    bridge_samples
        Passed to :func:`level_crossings` as ``max_gap``.

    Returns
    -------
    TemplateMetrics
    """
    delta = np.asarray(delta_f_hz, dtype=np.float64)
    amp = np.asarray(amp_rel, dtype=np.float64)
    centre = delta.size // 2
    peak = int(np.argmax(delta))
    excursion = float(delta[peak])
    if not excursion > 0:
        nan = float("nan")
        return TemplateMetrics(
            excursion, nan, nan, nan, nan, nan, nan, nan, nan,
            float(1.0 - np.min(amp)), float(np.max(amp) - 1.0), nan, 0, 0, True,
        )  # fmt: skip
    c10, c50 = chirp_crossings(delta, peak, bridge_samples)

    def to_s(position: float) -> float:
        return (position - centre) / samplerate_hz

    post_start = int(np.ceil(c10.offset))
    post_stop = min(
        delta.size, post_start + round(undershoot_window_s * samplerate_hz) + 1
    )
    post = delta[post_start:post_stop]
    undershoot = float(max(0.0, -np.min(post))) if post.size else float("nan")

    rise = peak - c10.onset
    fall = c10.offset - peak
    asymmetry = (fall - rise) / (fall + rise) if (fall + rise) > 0 else float("nan")

    min_prominence = max(prominence_rel * excursion, prominence_hz)
    peaks, _ = find_peaks(delta, height=min_prominence, prominence=min_prominence)
    inside = (peaks >= c10.onset) & (peaks <= c10.offset)
    n_peaks = max(int(inside.sum()), 1)
    n_extra = int((~inside).sum())

    return TemplateMetrics(
        excursion_hz=excursion,
        peak_time_s=to_s(peak),
        onset_10_s=to_s(c10.onset),
        offset_10_s=to_s(c10.offset),
        onset_50_s=to_s(c50.onset),
        offset_50_s=to_s(c50.offset),
        duration_10_s=c10.width / samplerate_hz,
        duration_50_s=c50.width / samplerate_hz,
        undershoot_hz=undershoot,
        amp_dip=float(1.0 - np.min(amp)),
        amp_peak=float(np.max(amp) - 1.0),
        asymmetry=float(asymmetry),
        n_peaks=n_peaks,
        n_extra_events=n_extra,
        clipped=c10.onset_clipped or c10.offset_clipped,
    )


# --------------------------------------------------------------------------
# Gain of the extraction band-pass filter
# --------------------------------------------------------------------------


def extraction_filter_gain(freq_hz: npt.ArrayLike, design_eodf_hz: float) -> FloatArray:
    """Amplitude gain of the thesis extraction filter at ``freq_hz``.

    The extraction script band-passed each recording with a second-order
    Butterworth filter from ``EODf - 100 Hz`` to ``1.9 * EODf`` (EODf = power
    spectrum peak between 300 and 1200 Hz), applied forward and backward
    (``sosfiltfilt``), so the amplitude gain is ``|H(f)|**2`` and the phase is
    zero. The analog prototype is used; at the recordings' sampling rates
    (20--44.1 kHz) the digital response differs by less than 0.5%.

    Parameters
    ----------
    freq_hz
        Frequencies at which to evaluate the gain.
    design_eodf_hz
        EODf the filter was designed for (taken as the baseline EODf).

    Returns
    -------
    numpy.ndarray
        Gain at each frequency.
    """
    from scipy.signal import buttap, lp2bp_zpk

    low = 2 * np.pi * (design_eodf_hz - 100.0)
    high = 2 * np.pi * 1.9 * design_eodf_hz
    z0, p0, k0 = buttap(2)
    zeros, poles, k = lp2bp_zpk(z0, p0, k0, wo=np.sqrt(low * high), bw=high - low)
    f = np.asarray(freq_hz, dtype=np.float64)
    s = 1j * 2 * np.pi * f[..., None]
    h = k * np.prod(s - zeros, axis=-1) / np.prod(s - poles, axis=-1)
    return np.abs(h) ** 2


# --------------------------------------------------------------------------
# Absolute baseline EODf from the spline knots of orig_freq
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KnotEodf:
    """Absolute EOD frequency recovered from the ``orig_freq`` spline knots.

    ``orig_freq`` is a cubic spline through one value per EOD cycle, placed
    at each ascending zero crossing ``t_k`` with value
    ``1 / (t_{k+1} - t_k)``, minus an unknown constant ``c`` (the absolute
    frequency at the first window sample). The knots show up as isolated
    spikes in the fourth difference. Every knot interval gives one estimate
    ``c_k = 1 / (t_{k+1} - t_k) - x(t_k)``.

    Attributes
    ----------
    offset_hz
        Median of ``c_k`` over baseline knots: absolute frequency that
        corresponds to 0 in the relative trace.
    mad_hz
        Median absolute deviation of the baseline ``c_k`` around
        ``offset_hz``; small values (< 0.1 Hz) mean the identity holds.
    n_knots
        Number of baseline knot intervals used.
    chirp_mad_hz
        Same deviation over knot intervals inside the chirp, as an
        independent check that the relative trace is in Hz (NaN if none).
    """

    offset_hz: float
    mad_hz: float
    n_knots: int
    chirp_mad_hz: float


def find_knots(
    orig_freq: FloatArray, threshold: float = 20.0, window: int = 129
) -> FloatArray:
    """Locate spline knots (EOD zero crossings) in an ``orig_freq`` trace.

    Parameters
    ----------
    orig_freq
        Contiguous ``orig_freq`` trace.
    threshold
        Minimum ratio of a fourth-difference spike to the local median.
    window
        Length of the running median, in samples.

    Returns
    -------
    numpy.ndarray
        Fractional knot positions (centroid of the spike), in samples.
    """
    x = np.asarray(orig_freq, dtype=np.float64)
    d4 = np.abs(np.diff(x, 4))
    # float32 storage limits precision in proportion to |x|
    quantisation = 8.0 * np.finfo(np.float32).eps * np.abs(x[2:-2]) + 1e-12
    floor = median_filter(d4, size=window, mode="nearest") + quantisation
    peaks, _ = find_peaks(d4 / floor, height=threshold, distance=8)
    if peaks.size == 0:
        return np.empty(0)
    offsets = np.arange(-2, 3)
    idx = np.clip(peaks[:, None] + offsets[None, :], 0, d4.size - 1)
    weights = d4[idx]
    centroid = (weights * idx).sum(axis=1) / weights.sum(axis=1)
    return centroid + 2.0


def knot_eodf(
    orig_freq: FloatArray,
    baseline_mask: BoolArray,
    chirp_mask: BoolArray,
    samplerate_hz: float,
    threshold: float = 20.0,
) -> KnotEodf:
    """Estimate the absolute frequency offset of a relative ``orig_freq`` trace.

    Parameters
    ----------
    orig_freq
        Contiguous relative ``orig_freq`` trace.
    baseline_mask, chirp_mask
        Boolean masks of baseline and chirp samples.
    samplerate_hz
        Sampling rate.
    threshold
        Passed to :func:`find_knots`.

    Returns
    -------
    KnotEodf
    """
    x = np.asarray(orig_freq, dtype=np.float64)
    knots = find_knots(x, threshold=threshold)
    nan = float("nan")
    if knots.size < 3:
        return KnotEodf(nan, nan, 0, nan)
    period = np.diff(knots) / samplerate_hz
    x_at = np.interp(knots[:-1], np.arange(x.size), x)
    estimates = 1.0 / period - x_at
    start = np.clip(np.floor(knots[:-1]).astype(int), 0, x.size - 1)
    stop = np.clip(np.ceil(knots[1:]).astype(int), 0, x.size - 1)
    in_baseline = baseline_mask[start] & baseline_mask[stop]
    in_chirp = chirp_mask[start] | chirp_mask[stop]
    if in_baseline.sum() < 3:
        return KnotEodf(nan, nan, int(in_baseline.sum()), nan)
    offset = float(np.median(estimates[in_baseline]))
    mad = float(np.median(np.abs(estimates[in_baseline] - offset)))
    chirp_mad = (
        float(np.median(np.abs(estimates[in_chirp] - offset)))
        if in_chirp.any()
        else nan
    )
    return KnotEodf(offset, mad, int(in_baseline.sum()), chirp_mad)


# --------------------------------------------------------------------------
# Baseline segments: ripple, drift, beat frequency, step artefacts
# --------------------------------------------------------------------------


def mask_segments(mask: BoolArray) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` of every run of True values in ``mask``."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def detrended_rms(
    trace: FloatArray, mask: BoolArray, min_samples: int
) -> tuple[float, FloatArray | None]:
    """RMS of ``trace`` on the masked segments after removing a line from each.

    Parameters
    ----------
    trace
        1-D trace.
    mask
        Baseline samples.
    min_samples
        Segments shorter than this are ignored.

    Returns
    -------
    rms : float
        Root mean square of the residuals of all used segments (NaN if none).
    longest : numpy.ndarray or None
        Residual of the longest used segment, for spectral analysis.
    """
    residuals: list[FloatArray] = []
    for start, stop in mask_segments(mask):
        if stop - start < min_samples:
            continue
        segment = trace[start:stop]
        t = np.arange(segment.size, dtype=np.float64)
        slope, intercept = np.polyfit(t, segment, 1)
        residuals.append(segment - (slope * t + intercept))
    if not residuals:
        return float("nan"), None
    joined = np.concatenate(residuals)
    longest = max(residuals, key=lambda r: r.size)
    return float(np.sqrt(np.mean(joined**2))), longest


def dominant_frequency(
    residual: FloatArray | None,
    samplerate_hz: float,
    fmin_hz: float = 5.0,
    fmax_hz: float = 400.0,
    nfft: int = 1 << 15,
) -> float:
    """Frequency of the largest periodogram peak of ``residual`` in a band.

    Parameters
    ----------
    residual
        Detrended baseline segment (or None).
    samplerate_hz
        Sampling rate.
    fmin_hz, fmax_hz
        Search band. The amplitude trace has one value per EOD cycle, so
        beats above about half the EODf are not represented.
    nfft
        FFT length (zero padding).

    Returns
    -------
    float
        Peak frequency in Hz, NaN if the segment is missing or too short.
    """
    if residual is None or residual.size < 64:
        return float("nan")
    window = np.hanning(residual.size)
    spectrum = np.abs(np.fft.rfft(residual * window, n=max(nfft, residual.size))) ** 2
    freqs = np.fft.rfftfreq(max(nfft, residual.size), 1.0 / samplerate_hz)
    band = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    if not band.any():
        return float("nan")
    return float(freqs[band][np.argmax(spectrum[band])])


def step_score(trace: FloatArray, min_step: float, window: int = 201) -> float:
    """Largest single-sample jump in a smooth trace, in units of local slope.

    A Gaussian-smoothed trace (sigma = 20 samples) cannot change abruptly
    from one sample to the next, so a first difference that deviates from
    the mean of its neighbours is an artefact (for example a scaling step
    from the extraction code's in-place normalisation).

    Parameters
    ----------
    trace
        Contiguous smoothed trace.
    min_step
        Jumps smaller than this (in trace units) are ignored.
    window
        Running-median window for the local slope scale.

    Returns
    -------
    float
        ``max(|d1[i] - (d1[i-1] + d1[i+1]) / 2| / local_scale)`` over jumps
        of at least ``min_step``; 0 if there are none.
    """
    d1 = np.diff(np.asarray(trace, dtype=np.float64))
    if d1.size < 3:
        return 0.0
    jump = np.abs(d1[1:-1] - 0.5 * (d1[:-2] + d1[2:]))
    scale = median_filter(np.abs(d1), size=window, mode="nearest")[1:-1]
    scale = scale + 1e-3 * min_step
    big = jump >= min_step
    if not big.any():
        return 0.0
    return float(np.max(jump[big] / scale[big]))


# --------------------------------------------------------------------------
# Analysis of one contiguous source snippet
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SnippetAnalysis:
    """Baseline, alignment and quality measures of one contiguous snippet.

    Positions are sample indices in the unrolled source snippet.

    Attributes
    ----------
    valid
        False if no positive excursion was found.
    baseline_freq
        Baseline of the chosen frequency trace (source units, relative).
    baseline_amp
        Baseline of the chosen amplitude trace (source units).
    peak_index
        Excursion maximum of the chosen frequency trace.
    excursion_hz
        Peak minus baseline.
    crossing_10, crossing_50
        Crossings of 10% and 50% of the excursion.
    align_index
        Rounded midpoint of the 50% crossings: the template centre.
    support_start, support_stop
        Region kept unchanged in the template (before tapering).
    baseline_s
        Duration of baseline samples used.
    ripple_freq_hz
        RMS of the detrended frequency baseline.
    ripple_amp
        RMS of the detrended relative-amplitude baseline.
    beat_freq_hz
        Dominant frequency of the amplitude baseline ripple.
    drift_hz
        Median post-chirp minus median pre-chirp baseline frequency.
    step_score_freq, step_score_amp
        :func:`step_score` of the smoothed traces.
    eodf
        Knot-based absolute frequency estimate.
    baseline_eodf_hz
        Absolute baseline EODf (NaN if not recoverable).
    has_nan
        Any trace contained NaN (replaced by baseline before measuring).
    decay
        End of the slow decay after the 10% offset, or None when the decay
        extension is off (library v1/v2 behaviour).
    """

    valid: bool
    baseline_freq: float
    baseline_amp: float
    peak_index: int
    excursion_hz: float
    crossing_10: Crossing
    crossing_50: Crossing
    align_index: int
    support_start: float
    support_stop: float
    baseline_s: float
    ripple_freq_hz: float
    ripple_amp: float
    beat_freq_hz: float
    drift_hz: float
    step_score_freq: float
    step_score_amp: float
    eodf: KnotEodf
    baseline_eodf_hz: float
    has_nan: bool
    decay: DecayEnd | None = None

    @property
    def chirp_end(self) -> float:
        """End of the chirp: the 10% offset, or the decay end if later."""
        if self.decay is None:
            return self.crossing_10.offset
        return max(self.crossing_10.offset, self.decay.position)


def analyse_snippet(
    freq: FloatArray,
    amp: FloatArray,
    orig_freq: FloatArray,
    smooth_freq: FloatArray,
    smooth_amp: FloatArray,
    samplerate_hz: float,
    *,
    exclusion_pre_s: float,
    exclusion_post_s: float,
    support_pad_pre_s: float,
    support_pad_post_s: float,
    min_baseline_s: float,
    knot_threshold: float,
    knot_mad_max_hz: float,
    bridge_samples: int = 0,
    decay_level_rel: float | None = None,
    decay_level_hz: float = 0.0,
    decay_smooth_s: float = 0.0,
    peak_range: tuple[int, int] | None = None,
    absolute_freq: bool = False,
    baseline_band: tuple[float, float] | None = None,
    next_event_guard_s: float = 0.0,
) -> SnippetAnalysis:
    """Locate the chirp in a contiguous snippet and measure its context.

    Parameters
    ----------
    freq, amp
        The frequency and amplitude traces chosen for the templates.
    orig_freq
        Per-cycle frequency trace (for the knot-based EODf).
    smooth_freq, smooth_amp
        Smoothed traces (for step detection).
    samplerate_hz
        Sampling rate.
    exclusion_pre_s, exclusion_post_s, support_pad_pre_s, support_pad_post_s
        See :class:`gridtools.templates.config.BuildConfig`.
    min_baseline_s
        Baseline samples wanted; the exclusion margins shrink (to 50, 25, 10
        and 0%) until at least this much baseline remains.
    knot_threshold, knot_mad_max_hz
        Knot detection threshold and acceptance limit for the EODf.
    bridge_samples
        Brief internal dips shorter than this are bridged when finding the
        level crossings (see :func:`level_crossings`).
    decay_level_rel, decay_level_hz, decay_smooth_s
        If ``decay_level_rel`` is given, the chirp extends past its 10%
        offset over a slow decay until the trace (smoothed with sigma
        ``decay_smooth_s``) falls to ``max(decay_level_rel * excursion,
        decay_level_hz)`` (:func:`decay_end`). The baseline exclusion and the
        support then start from that point, and the support pad stops
        ``next_event_guard_s`` before a following chirp. None (default) is
        the library v1/v2 behaviour.
    peak_range
        Sample range ``[start, stop)`` searched for the excursion peak (the
        chirp of interest when the snippet holds several); None: all.
    absolute_freq
        ``freq`` and ``orig_freq`` are absolute frequencies (Hz), so the
        baseline itself is the absolute EODf and the spline-knot recovery
        is skipped.
    baseline_band
        Only samples of ``freq`` inside ``[low, high]`` count as baseline
        (provisional and final). Use it when the snippet holds other chirps
        whose tails would bias the median, and the baseline is known
        independently (for example from the power spectrum). None: all.

    Returns
    -------
    SnippetAnalysis
    """
    fs = samplerate_hz
    n = freq.size

    def find_peak(trace: FloatArray) -> int:
        if peak_range is None:
            return int(np.argmax(trace))
        lo = min(max(peak_range[0], 0), n - 1)
        hi = min(max(peak_range[1], lo + 1), n)
        return lo + int(np.argmax(trace[lo:hi]))

    def chirp_extent(trace: FloatArray, peak: int) -> tuple[Crossing, DecayEnd | None]:
        c10, _ = chirp_crossings(trace, peak, bridge_samples)
        if decay_level_rel is None:
            return c10, None
        level = max(decay_level_rel * float(trace[peak]), decay_level_hz)
        decay = decay_end(trace, c10.offset, level, decay_smooth_s * fs)
        return c10, decay

    def extended(c10: Crossing, decay: DecayEnd | None) -> Crossing:
        if decay is None or decay.position <= c10.offset:
            return c10
        return dataclasses.replace(c10, offset=decay.position)

    has_nan = bool(
        np.isnan(freq).any() or np.isnan(amp).any() or np.isnan(orig_freq).any()
    )
    freq = _fill_nan(freq)
    amp = _fill_nan(amp)
    orig_freq = _fill_nan(orig_freq)
    allowed = (
        None
        if baseline_band is None
        else (freq >= baseline_band[0]) & (freq <= baseline_band[1])
    )

    # pass 1: provisional baseline by the thesis rule, the median of the samples
    # within [min, |min|] of the trace relative to its first sample (the window
    # start); unlike the whole-window median it is not pulled up by long chirps
    if allowed is not None and allowed.any():
        baseline = float(np.median(freq[allowed]))
    else:
        baseline = provisional_baseline(freq)
    delta = freq - baseline
    peak = find_peak(delta)
    nan = float("nan")
    if not delta[peak] > 0:
        empty = Crossing(nan, nan, True, True)
        return SnippetAnalysis(
            False, baseline, float(np.median(amp)), peak, 0.0, empty, empty, peak,
            nan, nan, 0.0, nan, nan, nan, nan, 0.0, 0.0,
            KnotEodf(nan, nan, 0, nan), nan, has_nan,
        )  # fmt: skip
    c10, decay = chirp_extent(delta, peak)

    # pass 2 (twice): baseline from samples away from the chirp; the exclusion
    # margins shrink when a long chirp leaves too little baseline
    mask = np.zeros(n, dtype=bool)
    for _ in range(2):
        mask = _shrinking_baseline_mask(
            n,
            extended(c10, decay),
            exclusion_pre_s * fs,
            exclusion_post_s * fs,
            min_baseline_s * fs,
            allowed,
        )
        if not mask.any():
            break
        baseline = float(np.median(freq[mask]))
        delta = freq - baseline
        peak = find_peak(delta)
        c10, decay = chirp_extent(delta, peak)
    mask = _shrinking_baseline_mask(
        n,
        extended(c10, decay),
        exclusion_pre_s * fs,
        exclusion_post_s * fs,
        min_baseline_s * fs,
        allowed,
    )
    baseline_s = mask.sum() / fs
    if mask.any():
        baseline = float(np.median(freq[mask]))
        baseline_amp = float(np.median(amp[mask]))
    else:
        baseline_amp = float(np.median(amp))
    delta = freq - baseline
    peak = find_peak(delta)
    excursion = float(delta[peak])
    c10, c50 = chirp_crossings(delta, peak, bridge_samples)
    _, decay = chirp_extent(delta, peak)
    chirp_end = extended(c10, decay).offset
    support_stop = chirp_end + support_pad_post_s * fs
    if decay is not None:
        # never pad into a following chirp: stop ``next_event_guard_s`` before
        # the trace starts rising by 10% of this excursion again
        next_onset = following_rise(
            delta,
            chirp_end,
            max(0.1 * excursion, decay_level_hz),
            decay_smooth_s * fs,
        )
        if next_onset is not None:
            support_stop = max(
                chirp_end, min(support_stop, next_onset - next_event_guard_s * fs)
            )
    align = round(c50.midpoint)

    min_segment = round(0.02 * fs)
    ripple_freq, _ = detrended_rms(delta, mask, min_segment)
    amp_rel = amp / baseline_amp if baseline_amp > 0 else amp * nan
    ripple_amp, amp_residual = detrended_rms(amp_rel, mask, min_segment)
    beat = dominant_frequency(amp_residual, fs)

    segments = [s for s in mask_segments(mask) if s[1] - s[0] >= min_segment]
    pre = [s for s in segments if s[1] <= c10.onset]
    post = [s for s in segments if s[0] >= chirp_end]
    if pre and post:
        drift = float(
            np.median(freq[post[0][0] : post[0][1]])
            - np.median(freq[pre[-1][0] : pre[-1][1]])
        )
    else:
        drift = nan

    chirp_mask = np.zeros(n, dtype=bool)
    chirp_mask[max(0, int(c10.onset)) : min(n, int(np.ceil(c10.offset)) + 1)] = True
    if absolute_freq:
        eodf = KnotEodf(nan, nan, 0, nan)
        baseline_eodf = baseline
    else:
        eodf = knot_eodf(orig_freq, mask, chirp_mask, fs, threshold=knot_threshold)
        if eodf.n_knots >= 20 and eodf.mad_hz <= knot_mad_max_hz:
            orig_baseline = float(np.median(orig_freq[mask])) if mask.any() else nan
            baseline_eodf = eodf.offset_hz + orig_baseline
        else:
            baseline_eodf = nan

    return SnippetAnalysis(
        valid=True,
        baseline_freq=baseline,
        baseline_amp=baseline_amp,
        peak_index=peak,
        excursion_hz=excursion,
        crossing_10=c10,
        crossing_50=c50,
        align_index=align,
        support_start=c10.onset - support_pad_pre_s * fs,
        support_stop=support_stop,
        baseline_s=float(baseline_s),
        ripple_freq_hz=ripple_freq,
        ripple_amp=ripple_amp,
        beat_freq_hz=beat,
        drift_hz=drift,
        step_score_freq=step_score(_fill_nan(smooth_freq), min_step=0.5),
        step_score_amp=step_score(_fill_nan(smooth_amp), min_step=0.005),
        eodf=eodf,
        baseline_eodf_hz=float(baseline_eodf),
        has_nan=has_nan,
        decay=decay,
    )


def provisional_baseline(trace: FloatArray) -> float:
    """Median of the samples within ``[min, |min|]`` relative to the first sample.

    The extraction made every trace relative to the first sample of its window,
    which lies on the baseline. Samples close to that value (bounded by the
    deepest dip) are baseline, however long the chirp is (thesis rule).
    """
    relative = trace - trace[0]
    low = float(np.min(relative))
    band = relative[(relative >= low) & (relative <= abs(low))]
    if band.size == 0:
        return float(np.median(trace))
    return float(np.median(band) + trace[0])


def _shrinking_baseline_mask(
    n: int,
    crossing: Crossing,
    pre: float,
    post: float,
    min_samples: float,
    allowed: BoolArray | None = None,
) -> BoolArray:
    """Baseline mask whose exclusion margins shrink until enough samples remain."""
    mask = np.zeros(n, dtype=bool)
    for factor in (1.0, 0.5, 0.25, 0.1, 0.0):
        mask = _baseline_mask(n, crossing, factor * pre, factor * post)
        if allowed is not None:
            mask &= allowed
        if mask.sum() >= min_samples:
            return mask
    return mask


def _baseline_mask(
    n: int, crossing: Crossing, pre_samples: float, post_samples: float
) -> BoolArray:
    mask = np.ones(n, dtype=bool)
    start = max(0, int(np.floor(crossing.onset - pre_samples)))
    stop = min(n, int(np.ceil(crossing.offset + post_samples)) + 1)
    mask[start:stop] = False
    return mask


def _fill_nan(trace: FloatArray) -> FloatArray:
    """Return a copy with NaNs replaced by the median of the finite samples."""
    out = np.array(trace, dtype=np.float64, copy=True)
    bad = ~np.isfinite(out)
    if bad.any():
        fill = float(np.median(out[~bad])) if (~bad).any() else 0.0
        out[bad] = fill
    return out
