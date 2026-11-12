"""Harvest long chirps from multi-electrode grid recordings.

The chirp-chamber extraction behind library v2 cut 0.4 s windows, so its
longest chirp lasts 151 ms. Chirps of 300--450 ms exist: abrupt rises of
400--550 Hz, a 100--250 ms plateau and a slow decay to baseline. This module
finds them in the staged-competition grid recordings and turns them into
templates with the library's own construction
(:func:`gridtools.templates.build.template_from_traces`).

Pipeline per candidate
----------------------
1. **Candidates**: chirpdetector v1 boxes at least ``min_box_s`` wide and
   ``min_height_hz`` high (box width is about the chirp duration plus 160 ms
   of spectrogram smear).
2. **Emitter**: every track alive around the box is a suspect; v1's
   ``track_id`` is only a hint. The emitter is the fish whose baseline band
   loses the most power during the chirp (its energy moves up), measured on
   the electrode where that fish is strongest. Per-electrode powers come
   from the raw data, so recordings whose ``sign_v`` has fewer columns than
   channels (the 2019 recordings: 8 of 9) are handled too; ``sign_v`` is
   only cross-checked where the columns match.
3. **Extraction**: all electrodes are band-passed to ``[baseline - margin,
   chirp top + margin]`` and combined with weights matched to the emitter's
   field pattern before the chirp, with the other fish's patterns
   (fundamental and second harmonic) projected out: a quasi-static dipole
   field is in phase or antiphase across electrodes, so each line has one
   real pattern. Other fish's lines outside the chirp's span are notched,
   and the upper band edge keeps a guard below them. Without the null and
   the notches, a fish just below the band or a harmonic just above it
   leaves a frequency ripple or a bias.

   The instantaneous frequency is the phase advance of the analytic signal
   averaged with a Gaussian window (lag-one autocorrelation, "pulse-pair"
   estimator). Averaging the complex products weights every sample by the
   signal power, so the near-zero EOD amplitude on the plateau of large
   chirps (often 5--20% of baseline) does not produce the +-300 Hz jumps
   that per-cycle zero crossings (the thesis extraction, made for the
   high-SNR chirp chamber) give there. The window is 1 ms (the thesis
   smoothing) at baseline amplitude and widens as ``(A0 / A)**2`` up to
   ``freq_sigma_max_s`` where the amplitude drops, which keeps the
   frequency error about constant. On such plateaus noise still pulls the
   estimate toward the pass band's centre; iterative demodulation
   (:func:`demodulated_frequency`) removes most of that bias.
4. **Templates**: :func:`~gridtools.templates.build.template_from_traces`
   with the decay extension switched on, the peak searched inside the box,
   and the amplitude corrected for the extraction filter's gain.
5. **Rejections**: no emitter, another fish's EODf or its second harmonic
   inside the chirp's frequency span (the extraction would be
   contaminated), extraction failures, duplicates (two boxes on one
   chirp).
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import (
    butter,
    find_peaks,
    hilbert,
    iirnotch,
    sosfiltfilt,
    sosfreqz,
    stft,
    tf2sos,
    welch,
)

from gridtools.io import (
    Recording,
    Tracks,
    load_tracks,
    open_recording,
    read_v1_detections,
)
from gridtools.templates.build import ProcessedSnippet, template_from_traces
from gridtools.templates.config import BuildConfig

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
#: pandas-stubs cannot type the row access below; frames and rows are Any here.
Frame = Any


def _missing(value: object) -> bool:
    """True for None, pandas NA and NaN."""
    if value is None or value is pd.NA:
        return True
    return isinstance(value, float) and np.isnan(value)


def default_template_config() -> BuildConfig:
    """Template construction for harvested chirps.

    The library v2 defaults, plus the decay extension (support until the
    slow decay has fallen to 2% of the excursion, at least 3 Hz) and a
    1.2 s template so that no harvested chirp is cut; the merged library
    crops to its own length.
    """
    return BuildConfig(
        template_duration_s=1.2,
        decay_level_rel=0.02,
        decay_level_hz=3.0,
        decay_smooth_s=0.005,
        min_baseline_s=0.2,
    )


@dataclass(frozen=True, slots=True)
class HarvestConfig:
    """Parameters of the long-chirp harvest. Seconds and Hz throughout.

    Parameters
    ----------
    min_box_s, min_height_hz
        Candidate v1 boxes are at least this wide and high.
    context_s
        Data read before and after the box (baseline context).
    track_window_s
        Tracks with at least ``min_track_detections`` detections within
        this distance of the box are suspects.
    min_track_detections
        See ``track_window_s``.
    refine_hz
        A suspect's EODf is refined to the highest local maximum of the
        median power spectrum within this distance of its track frequency
        (the track median can lag a fish that raised its EODf).
    hint_window_s
        Wider window used when ``track_window_s`` finds fewer than two
        tracks or misses v1's track (the tracker can lose a fish during
        intense chirping).
    same_fish_hz
        Suspects whose refined EODf lie closer than this are fragments of
        one fish's track and are merged.
    probe_gap_s
        Gap between the baseline probe (the context before the box) and the
        box start.
    emitter_band_hz
        Half width of the band whose power drop identifies the emitter.
    emitter_min_drop_db
        Smallest power drop of the emitter's baseline band during the
        chirp core.
    margin_low_hz, margin_high_hz
        Band-pass margins below the baseline and above the chirp top.
    gap_fraction
        A margin never exceeds this fraction of the distance to another
        fish's EODf or second harmonic.
    filter_order
        Butterworth order of the band-pass (applied forward and backward).
    electrode_min_rel
        Electrodes whose weight is at least this fraction of the largest are
        listed as ``electrodes`` (all electrodes are combined).
    min_null_keep
        Another fish's pattern (fundamental or second harmonic) is projected
        out of the electrode weights only if the emitter keeps at least this
        fraction of its matched-filter gain (fish with nearly the same field
        pattern cannot be separated); strongest patterns first.
    line_guard_hz
        In the second pass the upper band edge stays this far below any
        other fish's line above the chirp (a Butterworth edge attenuates
        little just beyond it).
    notch_width_hz, notch_reach_hz
        Other fish's lines outside the chirp's span and within
        ``notch_reach_hz`` of the band are notched (bandwidth
        ``notch_width_hz``).
    baseline_band_hz
        Only frequency samples within this distance of the emitter's
        spectral peak count as baseline (neighbouring chirps' tails would
        bias the median).
    freq_sigma_min_s, freq_sigma_max_s
        Gaussian window of the frequency estimator at full and at low
        amplitude.
    demod_bandwidths_hz
        Low-pass cutoffs of the iterative demodulation that removes the
        noise bias on low-amplitude plateaus (:func:`demodulated_frequency`).
    amp_sigma_s
        Smoothing of the amplitude envelope (``smooth_amp``).
    contamination_margin_hz
        Another fish counts as inside the chirp's span if its EODf (or second
        harmonic) lies in ``(baseline, top)`` with ``top`` the larger of
        ``baseline + excursion + margin`` and the v1 box top (the box
        overestimates the chirp top by its smear, median 86 Hz, but a line
        near the top can hide the top from the extraction).
    reject_harmonics
        Also reject when another fish's second harmonic lies in the span.
    baseline_tol_hz
        The extracted baseline must match the departure frequency (median
        over ``departure_window_s`` ending ``departure_gap_s`` before the 10%
        onset) to within this.
    departure_window_s, departure_gap_s
        See ``baseline_tol_hz``; the same window after the chirp end plus the
        support pad gives ``post_level_hz``.
    drift_max_hz
        ``baseline_drift`` is flagged when the post-chirp level differs from
        the baseline by more than this (chirps riding on a rise-like EODf
        elevation in dense trains; the template is still tapered to 0).
    min_excursion_hz
        Smaller extracted excursions are rejected (no chirp found).
    duplicate_window_s
        Templates of one emitter whose 10% onsets are closer than this are
        the same chirp (overlapping v1 boxes).
    low_amp_rel
        ``low_snr`` is flagged when the amplitude falls below this fraction
        of baseline for more than ``low_amp_frac_max`` of the support.
    low_amp_frac_max
        See ``low_amp_rel``.
    line_gap_hz, line_max_db
        ``line_in_span`` is flagged when a spectral line before the chirp,
        between ``baseline + line_gap_hz`` and the chirp top, is stronger
        than ``line_max_db`` relative to the emitter's baseline peak.
    template
        Template construction (:func:`default_template_config`).
    """

    min_box_s: float = 0.25
    min_height_hz: float = 150.0
    context_s: float = 0.6
    track_window_s: float = 10.0
    hint_window_s: float = 120.0
    same_fish_hz: float = 5.0
    min_track_detections: int = 3
    refine_hz: float = 40.0
    probe_gap_s: float = 0.02
    emitter_band_hz: float = 12.0
    emitter_min_drop_db: float = 6.0
    margin_low_hz: float = 50.0
    margin_high_hz: float = 80.0
    gap_fraction: float = 0.5
    filter_order: int = 4
    electrode_min_rel: float = 0.1
    min_null_keep: float = 0.3
    line_guard_hz: float = 40.0
    notch_width_hz: float = 20.0
    notch_reach_hz: float = 150.0
    baseline_band_hz: float = 10.0
    freq_sigma_min_s: float = 0.001
    freq_sigma_max_s: float = 0.008
    demod_bandwidths_hz: tuple[float, ...] = (150.0, 80.0, 50.0)
    amp_sigma_s: float = 0.001
    contamination_margin_hz: float = 20.0
    reject_harmonics: bool = True
    baseline_tol_hz: float = 5.0
    departure_window_s: float = 0.08
    departure_gap_s: float = 0.01
    drift_max_hz: float = 10.0
    min_excursion_hz: float = 100.0
    duplicate_window_s: float = 0.05
    low_amp_rel: float = 0.1
    low_amp_frac_max: float = 0.3
    line_gap_hz: float = 50.0
    line_max_db: float = -35.0
    template: BuildConfig = field(default_factory=default_template_config)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary (the template config nested)."""
        out = dataclasses.asdict(self)
        out["template"] = self.template.to_dict()
        return out


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------


def open_recording_lenient(path: Path | str) -> tuple[Recording, int]:
    """Open a recording and report a dropped partial last frame.

    Four 2020 recordings end with an incomplete frame (the writer stopped
    mid-frame). :func:`gridtools.io.open_recording` memory-maps the complete
    frames and drops the remainder with a warning; this wrapper also returns
    how many bytes were dropped, for the harvest's provenance.

    Returns
    -------
    recording : Recording
    dropped_bytes : int
        Bytes of the partial frame left out (0 if none, or for wav files).
    """
    path = Path(path)
    recording = open_recording(path)
    raw = path / "traces-grid1.raw" if path.is_dir() else path
    dropped = 0
    if raw.suffix == ".raw" and raw.exists():
        dropped = raw.stat().st_size % (4 * recording.n_channels)
    return recording, int(dropped)


def read_detections_lenient(path: Path) -> pd.DataFrame:
    """v1 detections of a recording; an empty or missing file gives no rows."""
    csv = path / "chirpdetector_bboxes.csv"
    if not csv.exists() or csv.stat().st_size < 8:
        return pd.DataFrame(
            {
                "t_start_s": pd.Series(dtype=float),
                "t_end_s": pd.Series(dtype=float),
                "f_low_hz": pd.Series(dtype=float),
                "f_high_hz": pd.Series(dtype=float),
                "score": pd.Series(dtype=float),
                "emitter_hz": pd.Series(dtype=float),
                "track_id": pd.Series(dtype="Int64"),
            }
        )
    return read_v1_detections(csv)


def find_candidates(detections: pd.DataFrame, config: HarvestConfig) -> pd.DataFrame:
    """v1 boxes that may hold a long chirp.

    Parameters
    ----------
    detections
        Output of :func:`gridtools.io.read_v1_detections`.
    config
        Harvest configuration (``min_box_s``, ``min_height_hz``).

    Returns
    -------
    pandas.DataFrame
        The selected boxes (original index kept as ``box_index``) with
        ``box_s`` and ``height_hz`` columns.
    """
    width = detections["t_end_s"] - detections["t_start_s"]
    height = detections["f_high_hz"] - detections["f_low_hz"]
    keep = (width >= config.min_box_s) & (height >= config.min_height_hz)
    out = detections.loc[keep].copy()
    out.insert(0, "box_index", out.index.to_numpy())
    out["box_s"] = width[keep]
    out["height_hz"] = height[keep]
    return out.reset_index(drop=True)


_RECORDING_KEY = re.compile(r"[^0-9A-Za-z_-]")


def read_individuals(path: Path) -> dict[tuple[str, int], str]:
    """Map ``(recording, track id)`` to an individual (``order_meta.csv``).

    The competition metadata lists, per recording, the two fish (group and
    fish number) and their wavetracker track ids (``rec_id1``, ``rec_id2``).

    Returns
    -------
    dict
        ``(recording directory name, track id) -> "g<group>f<fish>"``; empty
        if the file is missing.
    """
    if not path.exists():
        return {}
    table: Frame = pd.read_csv(path, dtype=str)
    out: dict[tuple[str, int], str] = {}
    for row in table.to_dict("records"):
        name = _RECORDING_KEY.sub("", str(row["recording"]))
        for fish, rec in (("fish1", "rec_id1"), ("fish2", "rec_id2")):
            track, number = row.get(rec), row.get(fish)
            if _missing(track) or _missing(number) or not str(track).strip():
                continue
            try:
                out[(name, int(float(track)))] = f"g{row['group']}f{int(float(number))}"
            except ValueError:
                continue
    return out


def suspects(
    tracks: Tracks,
    t0: float,
    t1: float,
    config: HarvestConfig,
    hint: int | None = None,
) -> dict[int, float]:
    """EODf of every track that may be the emitter.

    Tracks with detections within ``track_window_s`` of the box count; if
    the ``hint`` track (v1's assignment) is not among them, or no other track
    is, tracks with detections within ``hint_window_s`` count too: the
    tracker can lose a fish for minutes during intense chirping. The
    frequency is the median of the 50 detections nearest to the box.
    """

    def collect(window: float) -> dict[int, float]:
        out: dict[int, float] = {}
        lo, hi = t0 - window, t1 + window
        for track in tracks:
            if track.stop_s < lo or track.start_s > hi:
                continue
            sel = (track.times_s >= lo) & (track.times_s <= hi)
            if np.count_nonzero(sel) >= config.min_track_detections:
                nearest = np.argsort(np.abs(track.times_s[sel] - t0))[:50]
                out[track.ident] = float(np.median(track.freqs_hz[sel][nearest]))
        return out

    near = collect(config.track_window_s)
    if len(near) >= 2 and (hint is None or hint in near):
        return near
    wide = collect(config.hint_window_s)
    return {**wide, **near}


# --------------------------------------------------------------------------
# Signal processing
# --------------------------------------------------------------------------


def pulse_pair_frequency(
    analytic: npt.ArrayLike,
    samplerate_hz: float,
    sigma_samples: float | FloatArray,
    ladder: int = 7,
) -> FloatArray:
    """Instantaneous frequency from the averaged lag-one autocorrelation.

    ``f = angle(G * (z[n+1] conj(z[n]))) * fs / (2 pi)`` with a Gaussian
    ``G``. A constant ``sigma_samples`` is a plain Gaussian; an array gives
    a per-sample width, realised by interpolating (in log sigma) between
    ``ladder`` geometrically spaced widths.

    Parameters
    ----------
    analytic
        Analytic signal of a band-passed EOD.
    samplerate_hz
        Sampling rate.
    sigma_samples
        Gaussian sigma in samples, scalar or one per sample.
    ladder
        Number of widths for the per-sample case.

    Returns
    -------
    numpy.ndarray
        Frequency in Hz, one per sample.
    """
    z = np.asarray(analytic, dtype=np.complex128)
    lag = z[1:] * np.conj(z[:-1])
    lag = np.concatenate([lag, lag[-1:]])

    def smooth(sigma: float) -> npt.NDArray[np.complex128]:
        re = gaussian_filter1d(lag.real, sigma, mode="nearest")
        im = gaussian_filter1d(lag.imag, sigma, mode="nearest")
        return np.asarray(re + 1j * im, dtype=np.complex128)

    sigma = np.asarray(sigma_samples, dtype=np.float64)
    if sigma.ndim == 0:
        smoothed = smooth(float(sigma))
    else:
        lo, hi = float(sigma.min()), float(sigma.max())
        if np.isclose(lo, hi):
            smoothed = smooth(lo)
        else:
            levels = np.geomspace(lo, hi, ladder)
            stack = np.stack([smooth(s) for s in levels])
            position = np.interp(np.log(sigma), np.log(levels), np.arange(ladder))
            below = np.clip(np.floor(position).astype(int), 0, ladder - 2)
            weight = position - below
            columns = np.arange(z.size)
            smoothed = (1 - weight) * stack[below, columns] + weight * stack[
                below + 1, columns
            ]
    return np.angle(smoothed) * samplerate_hz / (2 * np.pi)


def demodulated_frequency(
    analytic: npt.ArrayLike,
    samplerate_hz: float,
    initial_hz: FloatArray,
    sigma_samples: float | FloatArray,
    bandwidths_hz: Sequence[float],
) -> FloatArray:
    """Refine a frequency estimate by iterative demodulation.

    Where the EOD amplitude nearly vanishes (plateaus of large chirps), the
    noise in the pass band pulls the power-weighted estimate toward the
    band's centre, below the true plateau. Each iteration shifts the signal
    by the current estimate's phase to 0 Hz, low-passes it to
    ``+-bandwidth`` (so the remaining noise is symmetric around the estimate
    and no longer biases it) and adds the residual frequency of the result.

    Parameters
    ----------
    analytic
        Analytic signal.
    samplerate_hz
        Sampling rate.
    initial_hz
        First estimate (e.g. :func:`pulse_pair_frequency`).
    sigma_samples
        Window of the residual estimate (scalar or per sample).
    bandwidths_hz
        Low-pass cutoffs of the iterations, usually decreasing; empty: no
        refinement.

    Returns
    -------
    numpy.ndarray
        Refined frequency (Hz).
    """
    z = np.asarray(analytic, dtype=np.complex128)
    freq = np.array(initial_hz, dtype=np.float64, copy=True)
    for bandwidth in bandwidths_hz:
        phase = 2 * np.pi * np.cumsum(freq) / samplerate_hz
        base = z * np.exp(-1j * phase)
        sos = butter(4, bandwidth, fs=samplerate_hz, output="sos")
        low = sosfiltfilt(sos, base.real) + 1j * sosfiltfilt(sos, base.imag)
        freq = freq + pulse_pair_frequency(low, samplerate_hz, sigma_samples)
    return freq


def adaptive_sigma(
    amplitude: FloatArray,
    baseline_amplitude: float,
    sigma_min: float,
    sigma_max: float,
) -> FloatArray:
    """Window width that keeps the frequency error about constant.

    The phase error of one sample scales as ``noise / A``, so the error of
    an average over ``N`` samples as ``1 / (A sqrt(N))``; keeping it at the
    baseline value needs ``N ∝ (A0 / A)**2``.
    """
    ratio = baseline_amplitude / np.maximum(amplitude, 1e-12 * baseline_amplitude)
    return np.clip(sigma_min * ratio**2, sigma_min, sigma_max)


def field_pattern(
    probe: FloatArray, samplerate_hz: float, freq_hz: float
) -> FloatArray:
    """Real field pattern of a fish across electrodes.

    Demodulates every electrode at ``freq_hz`` (Hann window) and projects the
    complex amplitudes onto the phase of the strongest electrode: a
    quasi-static field is in phase or antiphase everywhere.

    Parameters
    ----------
    probe
        Raw samples (n_samples, n_electrodes) without chirps of this fish.
    samplerate_hz
        Sampling rate.
    freq_hz
        The fish's EODf.

    Returns
    -------
    numpy.ndarray
        Signed amplitude per electrode.
    """
    t = np.arange(probe.shape[0]) / samplerate_hz
    carrier = np.exp(-2j * np.pi * freq_hz * t) * np.hanning(t.size)
    amplitude = probe.T @ carrier
    ref = int(np.argmax(np.abs(amplitude)))
    return np.real(amplitude * np.conj(amplitude[ref])) / max(
        float(np.abs(amplitude[ref])), 1e-30
    )


def greedy_null_weights(
    emitter: FloatArray, others: Sequence[FloatArray], min_keep: float
) -> tuple[FloatArray, float, int]:
    """Project out as many interferer patterns as the emitter can afford.

    Patterns are added strongest first; one that would leave the emitter
    less than ``min_keep`` of its matched-filter gain is skipped.

    Returns
    -------
    weights : numpy.ndarray
        Unit-norm electrode weights.
    keep : float
        Fraction of the matched-filter gain kept.
    n_nulled : int
        Number of patterns projected out.
    """
    order = sorted(range(len(others)), key=lambda i: -float(np.linalg.norm(others[i])))
    chosen: list[FloatArray] = []
    weights, keep = null_weights(emitter, [])
    for i in order:
        trial_w, trial_keep = null_weights(emitter, [*chosen, others[i]])
        if trial_keep >= min_keep:
            chosen.append(others[i])
            weights, keep = trial_w, trial_keep
    return weights, keep, len(chosen)


def null_weights(
    emitter: FloatArray, others: Sequence[FloatArray]
) -> tuple[FloatArray, float]:
    """Electrode weights matched to ``emitter`` with ``others`` projected out.

    Returns
    -------
    weights : numpy.ndarray
        Unit-norm weights.
    keep : float
        Cosine between the weights and the emitter pattern: the fraction of
        the matched-filter gain the emitter keeps (1 without others).
    """
    g = np.asarray(emitter, dtype=np.float64)
    w = g.copy()
    if len(others):
        basis = np.stack([np.asarray(o, dtype=np.float64) for o in others], axis=1)
        coef, *_ = np.linalg.lstsq(basis, g, rcond=None)
        w = g - basis @ coef
    norm_w, norm_g = np.linalg.norm(w), np.linalg.norm(g)
    if norm_w == 0 or norm_g == 0:
        return np.zeros_like(g), 0.0
    w = w / norm_w
    return w, float(w @ g / norm_g)


def band_power_drop(
    signal: FloatArray,
    samplerate_hz: float,
    freq_hz: float,
    half_band_hz: float,
    references: Sequence[tuple[float, float]],
    core: tuple[float, float],
) -> float:
    """Power drop (dB) in ``freq_hz +- half_band_hz`` during ``core``.

    Time ranges are in seconds from the signal start. The median short-time
    band power over ``core`` is compared with the largest median over the
    ``references`` (before and after the chirp; in a chirp train one of
    them may fall into a neighbouring chirp).
    """
    f, t, spec = stft(signal, fs=samplerate_hz, nperseg=1024, noverlap=896)
    band = np.abs(f - freq_hz) <= half_band_hz
    power = np.sum(np.abs(spec[band]) ** 2, axis=0)
    levels = [
        float(np.median(power[(t >= lo) & (t <= hi)]))
        for lo, hi in references
        if np.any((t >= lo) & (t <= hi))
    ]
    in_core = (t >= core[0]) & (t <= core[1])
    if not levels or not in_core.any():
        return float("nan")
    p_ref = max(levels)
    p_core = float(np.median(power[in_core]))
    return 10.0 * np.log10(max(p_ref, 1e-30) / max(p_core, 1e-30))


def strongest_line(
    signal: FloatArray,
    samplerate_hz: float,
    span_hz: tuple[float, float],
    baseline_hz: float,
) -> tuple[float, float]:
    """Strongest stationary spectral line inside ``span_hz``.

    Untracked sources (a fish in a neighbouring tank, mains harmonics) can
    sit inside the chirp's span. The spectrum is the median over 0.2 s
    segments, so chirps (transient) drop out and stationary lines stay.
    Their level is reported relative to the emitter's baseline peak (dB);
    the chirp itself is weak on large-chirp plateaus (often 5--20% of
    baseline), so lines within about 30 dB of the baseline can bias the
    frequency there.

    Returns
    -------
    freq_hz, level_db
        Frequency and relative level of the line (NaN, -inf if none).
    """
    nperseg = min(4096, signal.size)
    f, p = welch(signal, fs=samplerate_hz, nperseg=nperseg, average="median")
    near = np.abs(f - baseline_hz) <= 10.0
    if not near.any():
        return float("nan"), float("-inf")
    level = 10 * np.log10(p + 1e-300)
    # narrow lines only: 10 dB above their surroundings within +-50 Hz
    wlen = 2 * round(50.0 / (f[1] - f[0])) + 1
    peaks, _ = find_peaks(level, prominence=10.0, wlen=wlen)
    peaks = peaks[(f[peaks] > span_hz[0]) & (f[peaks] < span_hz[1])]
    if peaks.size == 0:
        return float("nan"), float("-inf")
    k = int(peaks[np.argmax(p[peaks])])
    return float(f[k]), float(10 * np.log10(p[k] / np.max(p[near])))


def band_snr(
    signal: FloatArray,
    samplerate_hz: float,
    baseline_hz: float,
    band: tuple[float, float],
    line_half_width_hz: float = 10.0,
    guard_hz: float = 30.0,
) -> float:
    """Emitter line power over the rest of the pass band (dB).

    The frequency estimator averages the phase advance weighted by power,
    so broadband noise in the pass band pulls the estimate toward the
    band's centre by roughly ``noise / (signal + noise)`` times the distance;
    at 20 dB that is 1% of it. Median spectrum over 0.2 s segments, so
    chirps do not count as noise.
    """
    nperseg = min(4096, signal.size)
    f, p = welch(signal, fs=samplerate_hz, nperseg=nperseg, average="median")
    line = np.abs(f - baseline_hz) <= line_half_width_hz
    rest = (f >= band[0]) & (f <= band[1]) & (np.abs(f - baseline_hz) > guard_hz)
    if not line.any() or not rest.any():
        return float("nan")
    return float(10 * np.log10(np.sum(p[line]) / max(np.sum(p[rest]), 1e-300)))


def contaminants(
    baseline_hz: float, top_hz: float, others_hz: Sequence[float], *, harmonics: bool
) -> tuple[list[float], list[float]]:
    """Other fish whose EODf or second harmonic lies in ``(baseline, top)``.

    Returns
    -------
    fundamentals, second_harmonics : list of float
        Offending frequencies.
    """
    fundamentals = [f for f in others_hz if baseline_hz < f < top_hz]
    second = (
        [2 * f for f in others_hz if baseline_hz < 2 * f < top_hz] if harmonics else []
    )
    return fundamentals, second


# --------------------------------------------------------------------------
# One candidate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Extraction:
    """Traces of one harvested chirp.

    Attributes
    ----------
    window_start_s
        Recording time of the first sample.
    samplerate_hz
        Sampling rate.
    signal
        Combined, band-passed electrode signal (float32).
    traces
        ``orig_freq`` (1 ms window), ``smooth_freq`` (adaptive window),
        ``orig_amp``, ``smooth_amp``: absolute frequency (Hz) and envelope.
    sigma_s
        Per-sample window of ``smooth_freq``.
    snippet
        Template built from the traces.
    """

    window_start_s: float
    samplerate_hz: float
    signal: npt.NDArray[np.float32]
    traces: dict[str, FloatArray]
    sigma_s: FloatArray
    snippet: ProcessedSnippet


@dataclass(frozen=True)
class CandidateResult:
    """Outcome of one candidate.

    Attributes
    ----------
    record
        Provenance, measurements and status (``status`` is ``"harvested"``
        or ``"rejected"``, ``reason`` says why).
    extraction
        Traces and template, if extraction ran.
    """

    record: dict[str, Any]
    extraction: Extraction | None


def harvest_candidate(
    recording: Recording,
    tracks: Tracks,
    box: Frame,
    config: HarvestConfig,
    *,
    recording_name: str = "",
    individuals: dict[tuple[str, int], str] | None = None,
) -> CandidateResult:
    """Identify the emitter, extract and build the template of one candidate.

    Parameters
    ----------
    recording
        The grid recording (only a short span is read).
    tracks
        Its wavetracker tracks.
    box
        One row of :func:`find_candidates`.
    config
        Harvest configuration.
    recording_name
        Name stored in the provenance.
    individuals
        Optional ``(recording, track) -> individual`` map.

    Returns
    -------
    CandidateResult
    """
    fs = recording.samplerate_hz
    t0, t1 = float(box["t_start_s"]), float(box["t_end_s"])
    hint = box["track_id"]
    record: dict[str, Any] = {
        "recording": recording_name,
        "box_index": int(box["box_index"]),
        "box_t_start_s": t0,
        "box_t_end_s": t1,
        "box_f_low_hz": float(box["f_low_hz"]),
        "box_f_high_hz": float(box["f_high_hz"]),
        "box_score": float(box["score"]),
        "v1_track_id": -1 if _missing(hint) else int(hint),
        "v1_emitter_hz": float(box["emitter_hz"]),
        "status": "rejected",
        "reason": "",
    }

    def reject(reason: str, extraction: Extraction | None = None) -> CandidateResult:
        record["reason"] = reason
        return CandidateResult(record, extraction)

    start_s = max(t0 - config.context_s, 0.0)
    stop_s = min(t1 + config.context_s, recording.duration_s)
    if t0 - start_s < 0.5 * config.context_s or stop_s - t1 < 0.5 * config.context_s:
        return reject("recording_edge")
    live = suspects(tracks, t0, t1, config, None if _missing(hint) else int(hint))
    record["n_suspects"] = len(live)
    if not live:
        return reject("no_track")
    raw = recording.read_time(start_s, stop_s).astype(np.float64)
    start_s = recording.time_to_sample(start_s) / fs
    n = raw.shape[0]

    def rel(t: float) -> float:
        return t - start_s

    probe = (0.0, rel(t0) - config.probe_gap_s)
    after = (rel(t1) + config.probe_gap_s, n / fs)
    probe_slice = slice(0, int(probe[1] * fs))
    # the whole window: the steady baseline is the strongest narrow line near
    # a track, also in chirp trains whose decays sweep through the probe
    freqs, psd = welch(raw, fs=fs, nperseg=4096, axis=0, average="median")
    total = psd.sum(axis=1)
    refined: dict[int, float] = {}
    channel_db: dict[int, FloatArray] = {}
    maxima, _ = find_peaks(np.log(total + 1e-300))
    for ident, f_track in live.items():
        near = maxima[np.abs(freqs[maxima] - f_track) <= config.refine_hz]
        if near.size == 0:
            continue
        k = int(near[np.argmax(total[near])])
        refined[ident] = float(_parabolic_peak(freqs, total, k))
        lo, hi = max(k - 1, 0), min(k + 2, freqs.size)
        channel_db[ident] = 10 * np.log10(psd[lo:hi].max(axis=0) + 1e-30)
    if not refined:
        return reject("no_track")
    # fragments of one fish's track refine to the same line: keep one (the
    # hint, else the lowest id), so a fish is never its own contaminant
    hint_id = None if _missing(hint) else int(hint)
    for ident in sorted(refined, key=lambda i: (i != hint_id, i)):
        if ident not in refined:
            continue
        for other in list(refined):
            if other != ident and abs(refined[other] - refined[ident]) < (
                config.same_fish_hz
            ):
                del refined[other]
                del channel_db[other]
    record["n_suspects"] = len(refined)

    # emitter: the suspect whose baseline band loses most power during the chirp
    width = t1 - t0
    core = (rel(t0) + 0.1, rel(t0) + 0.1 + 0.5 * max(width - 0.2, 0.05))
    drops = {
        ident: band_power_drop(
            raw[:, int(np.argmax(channel_db[ident]))],
            fs,
            f,
            config.emitter_band_hz,
            (probe, after),
            core,
        )
        for ident, f in refined.items()
    }
    emitter = max(drops, key=lambda i: -np.inf if np.isnan(drops[i]) else drops[i])
    f_e = refined[emitter]
    others = [f for i, f in refined.items() if i != emitter]
    record.update(
        emitter_track_id=int(emitter),
        emitter_hz=f_e,
        emitter_drop_db=float(drops[emitter]),
        other_drops_db=";".join(
            f"{i}:{d:.1f}" for i, d in drops.items() if i != emitter
        ),
        other_fish_hz=";".join(f"{f:.1f}" for f in others),
        hint_agrees=None if _missing(hint) else bool(int(hint) == emitter),
        individual=(individuals or {}).get((recording_name, int(emitter)), ""),
    )
    if not drops[emitter] >= config.emitter_min_drop_db:
        return reject("no_emitter")

    # electrodes, cross-checked against the tracker's powers
    emitter_db = channel_db[emitter]
    best = int(np.argmax(emitter_db))
    record["best_channel"] = best
    record["sign_v_best_channel"] = _sign_v_best(
        tracks, emitter, t0, recording.n_channels
    )

    # spatial filter: matched to the emitter, other fish projected out (their
    # fundamental and second harmonic: the waveform, and so the harmonic
    # content, differs along the body, so the patterns differ)
    probe_raw = raw[probe_slice]
    pattern_e = field_pattern(probe_raw, fs, f_e)
    pattern_o = [
        field_pattern(probe_raw, fs, h * f)
        for f in others
        for h in (1, 2)
        if h * f < 0.45 * fs
    ]
    weights, keep, n_nulled = greedy_null_weights(
        pattern_e, pattern_o, config.min_null_keep
    )
    nulled = n_nulled > 0
    largest = np.max(np.abs(weights))
    use = np.flatnonzero(np.abs(weights) >= config.electrode_min_rel * largest)
    record.update(
        electrodes=",".join(str(int(c)) for c in use),
        electrode_weights=",".join(f"{w:.3f}" for w in weights),
        null_applied=bool(nulled),
        n_nulled=int(n_nulled),
        null_keep=float(keep),
    )

    # band: from below the baseline to above the chirp, clear of other fish
    lines = sorted([*others, *(2 * f for f in others)])
    below = [f_e - f for f in lines if f < f_e]
    margin_low = min([config.margin_low_hz, *(config.gap_fraction * g for g in below)])
    top = max(float(box["f_high_hz"]), f_e + config.min_excursion_hz)
    above = [f - top for f in lines if f > top]
    margin_high = min(
        [config.margin_high_hz, *(config.gap_fraction * g for g in above)]
    )
    band = (f_e - margin_low, top + margin_high)
    box_range = (round(rel(t0) * fs), round(rel(t1) * fs))
    baseline_band = (f_e - config.baseline_band_hz, f_e + config.baseline_band_hz)
    # pass 1 notches every other line near the band, so the chirp top is
    # measured without interference (even where a line crosses the chirp)
    reach = (band[0] - config.notch_reach_hz, band[1] + config.notch_reach_hz)
    extracted = _extract(
        raw,
        band,
        [f for f in lines if reach[0] < f < reach[1]],
        weights,
        fs,
        probe_slice,
        config,
        box_range,
        baseline_band,
    )
    # pass 2, now that the chirp top is known (the box overestimates it by its
    # smear): the upper edge keeps a guard below the lines above the chirp,
    # and lines outside the chirp's span are notched out
    peak_hz = (
        extracted.snippet.analysis.baseline_freq
        + extracted.snippet.metrics.excursion_hz
    )
    floor_hz = peak_hz + config.contamination_margin_hz
    upper = peak_hz + config.margin_high_hz
    for line in lines:
        if line > floor_hz:
            upper = min(upper, max(floor_hz, line - config.line_guard_hz))
    band = (band[0], upper)
    notches = [
        f
        for f in lines
        if (f > floor_hz or f < band[0])
        and band[0] - config.notch_reach_hz < f < band[1] + config.notch_reach_hz
    ]
    extracted = _extract(
        raw, band, notches, weights, fs, probe_slice, config, box_range, baseline_band
    )
    record.update(
        band_low_hz=band[0],
        band_high_hz=band[1],
        notches_hz=";".join(f"{f:.1f}" for f in notches),
    )

    # baseline = the frequency the chirp departs from (in chirp trains the EODf
    # between chirps can sit well above its level elsewhere in the window)
    snippet = extracted.snippet
    freq = extracted.traces["smooth_freq"]
    departure = _level(
        freq,
        snippet.analysis.crossing_10.onset - config.departure_window_s * fs,
        snippet.analysis.crossing_10.onset - config.departure_gap_s * fs,
    )
    if np.isfinite(departure) and abs(snippet.analysis.baseline_freq - departure) > (
        0.5 * config.baseline_tol_hz
    ):
        baseline_band = (
            departure - config.baseline_band_hz,
            departure + config.baseline_band_hz,
        )
        snippet = template_from_traces(
            extracted.traces,
            fs,
            config.template,
            absolute_freq=True,
            peak_range=box_range,
            baseline_band=baseline_band,
            filter_gain=extracted.gain,
        )
    extraction = Extraction(
        window_start_s=start_s,
        samplerate_hz=fs,
        signal=extracted.signal.astype(np.float32),
        traces=extracted.traces,
        sigma_s=extracted.sigma / fs,
        snippet=snippet,
    )
    a = snippet.analysis
    m = snippet.metrics
    record.update(_provenance(snippet, start_s, fs))
    post = _level(
        freq,
        a.chirp_end
        + config.departure_gap_s * fs
        + config.template.support_pad_post_s * fs,
        a.chirp_end
        + (config.departure_gap_s + config.departure_window_s) * fs
        + config.template.support_pad_post_s * fs,
        quantile=0.25,
    )
    record.update(
        departure_hz=departure,
        post_level_hz=post,
        baseline_drift_hz=post - a.baseline_freq,
        baseline_drift=bool(abs(post - a.baseline_freq) > config.drift_max_hz),
    )
    support = slice(
        max(int(a.support_start), 0), min(int(np.ceil(a.support_stop)) + 1, n)
    )
    envelope = np.abs(extracted.analytic)
    low = envelope[support] < config.low_amp_rel * extracted.a0
    sigma = extracted.sigma
    record.update(
        sigma_max_frac=float(
            np.mean(sigma[support] >= 0.99 * config.freq_sigma_max_s * fs)
        ),
        low_amp_frac=float(np.mean(low)) if low.size else float("nan"),
        min_amp_rel=float(np.min(extracted.amp_smooth[support]) / extracted.a0)
        if low.size
        else np.nan,
        band_clipped=bool(
            a.baseline_freq + m.excursion_hz > band[1] - 0.5 * margin_high
        ),
    )
    record["low_snr"] = bool(record["low_amp_frac"] > config.low_amp_frac_max)
    line_hz, line_db = strongest_line(
        extracted.signal,
        fs,
        (
            a.baseline_freq + config.line_gap_hz,
            a.baseline_freq + m.excursion_hz + config.contamination_margin_hz,
        ),
        a.baseline_freq,
    )
    record.update(
        span_line_hz=line_hz,
        span_line_db=line_db,
        line_in_span=bool(line_db > config.line_max_db),
        band_snr_db=band_snr(extracted.signal, fs, f_e, band),
        baseline_jitter_hz=_jitter(
            extracted.traces["orig_freq"],
            fs,
            a.crossing_10.onset
            - (config.departure_window_s + config.departure_gap_s) * fs,
            a.crossing_10.onset - config.departure_gap_s * fs,
        ),
    )

    if not a.valid:
        return reject("no_chirp", extraction)
    reference = departure if np.isfinite(departure) else f_e
    if abs(a.baseline_freq - reference) > config.baseline_tol_hz:
        return reject("baseline_mismatch", extraction)
    if not m.excursion_hz >= config.min_excursion_hz:
        return reject("no_chirp", extraction)
    onset_s = record["onset_10_abs_s"]
    if not (t0 - 0.05 <= onset_s <= t1):
        return reject("onset_outside_box", extraction)
    # the span ends at the measured top plus a margin, or at the v1 box top
    # if higher: a line near the top can hide the top from the extraction
    fundamentals, harmonics = contaminants(
        a.baseline_freq,
        max(
            a.baseline_freq + m.excursion_hz + config.contamination_margin_hz,
            float(box["f_high_hz"]),
        ),
        others,
        harmonics=config.reject_harmonics,
    )
    record["contaminants_hz"] = ";".join(
        f"{f:.0f}" for f in [*fundamentals, *harmonics]
    )
    if fundamentals:
        return reject("fish_in_span", extraction)
    if harmonics:
        return reject("harmonic_in_span", extraction)
    record["status"] = "harvested"
    return CandidateResult(record, extraction)


@dataclass(frozen=True)
class _Extracted:
    signal: FloatArray
    analytic: npt.NDArray[np.complex128]
    a0: float
    amp_smooth: FloatArray
    sigma: FloatArray
    traces: dict[str, FloatArray]
    gain: Any
    snippet: ProcessedSnippet


def _extract(
    raw: FloatArray,
    band: tuple[float, float],
    notches: Sequence[float],
    weights: FloatArray,
    fs: float,
    probe_slice: slice,
    config: HarvestConfig,
    box_range: tuple[int, int],
    baseline_band: tuple[float, float],
) -> _Extracted:
    """Band-pass, notch, combine, estimate frequency and amplitude, build the template."""
    sos = butter(config.filter_order, band, btype="bandpass", fs=fs, output="sos")
    for f in notches:
        b, a = iirnotch(f, f / config.notch_width_hz, fs=fs)
        sos = np.vstack([np.asarray(sos), np.asarray(tf2sos(b, a))])
    combined = sosfiltfilt(sos, raw, axis=0) @ weights
    analytic = np.asarray(hilbert(combined), dtype=np.complex128)
    envelope = np.abs(analytic)
    a0 = float(np.median(envelope[probe_slice]))
    amp_smooth = gaussian_filter1d(envelope, 2e-3 * fs)
    sigma = adaptive_sigma(
        amp_smooth, a0, config.freq_sigma_min_s * fs, config.freq_sigma_max_s * fs
    )
    sigma = np.exp(gaussian_filter1d(np.log(sigma), 2e-3 * fs))
    traces = {
        "orig_freq": pulse_pair_frequency(analytic, fs, config.freq_sigma_min_s * fs),
        "smooth_freq": demodulated_frequency(
            analytic,
            fs,
            pulse_pair_frequency(analytic, fs, sigma),
            sigma,
            config.demod_bandwidths_hz,
        ),
        "orig_amp": envelope,
        "smooth_amp": gaussian_filter1d(envelope, config.amp_sigma_s * fs),
    }

    def gain(freq_hz: FloatArray, _eodf: float) -> FloatArray:
        f = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
        _, h = sosfreqz(sos, worN=np.clip(f, 1.0, fs / 2 - 1.0), fs=fs)
        return np.abs(h) ** 2

    snippet = template_from_traces(
        traces,
        fs,
        config.template,
        absolute_freq=True,
        peak_range=box_range,
        baseline_band=baseline_band,
        filter_gain=gain,
    )
    return _Extracted(combined, analytic, a0, amp_smooth, sigma, traces, gain, snippet)


def _jitter(trace: FloatArray, fs: float, start: float, stop: float) -> float:
    """Robust SD of the fast fluctuation (relative to a 5 ms Gaussian) on a span.

    On the baseline, the 1 ms frequency estimate fluctuates in proportion to
    noise / amplitude; a large value means a weak emitter, whose frequency
    estimate is also biased toward the pass band's centre.
    """
    lo = max(int(np.floor(start)), 0)
    hi = min(int(np.ceil(stop)), trace.size)
    if hi - lo < 10:
        return float("nan")
    fast = trace - gaussian_filter1d(trace, 5e-3 * fs, mode="nearest")
    segment = fast[lo:hi]
    return float(1.4826 * np.median(np.abs(segment - np.median(segment))))


def _level(
    trace: FloatArray, start: float, stop: float, *, quantile: float = 0.5
) -> float:
    """Quantile of ``trace`` on samples ``[start, stop)`` (NaN if empty).

    The post-chirp level uses the lower quartile, so a following chirp that
    starts inside the window does not count as an elevated baseline.
    """
    lo = max(int(np.floor(start)), 0)
    hi = min(int(np.ceil(stop)), trace.size)
    if hi - lo < 10:
        return float("nan")
    return float(np.quantile(trace[lo:hi], quantile))


def _provenance(snippet: ProcessedSnippet, start_s: float, fs: float) -> dict[str, Any]:
    a = snippet.analysis
    decay = a.decay
    return {
        "window_start_s": start_s,
        "onset_10_abs_s": start_s + a.crossing_10.onset / fs,
        "offset_10_abs_s": start_s + a.crossing_10.offset / fs,
        "align_abs_s": start_s + a.align_index / fs,
        "chirp_end_abs_s": start_s + a.chirp_end / fs,
        "decay_interrupted": bool(decay.interrupted) if decay is not None else False,
        "decay_clipped": bool(decay.clipped) if decay is not None else False,
        "decay_extension_s": (a.chirp_end - a.crossing_10.offset) / fs,
    }


def _sign_v_best(tracks: Tracks, ident: int, t_s: float, n_channels: int) -> int:
    """Strongest electrode by the tracker's powers, or -1 if they do not map."""
    if tracks.n_electrodes != n_channels:
        return -1
    track = tracks.track(ident)
    if track.powers_db is None:
        return -1
    powers = track.powers_at([t_s], max_gap_s=30.0)[0]
    if not np.all(np.isfinite(powers)):
        return -1
    return int(np.argmax(powers))


def _parabolic_peak(x: FloatArray, y: FloatArray, k: int) -> float:
    """Peak position by a parabola through ``y[k-1:k+2]`` (on a log scale)."""
    if k <= 0 or k >= y.size - 1:
        return float(x[k])
    a, b, c = np.log(y[k - 1 : k + 2] + 1e-300)
    denom = a - 2 * b + c
    shift = 0.0 if denom == 0 else 0.5 * (a - c) / denom
    return float(x[k] + np.clip(shift, -0.5, 0.5) * (x[1] - x[0]))


# --------------------------------------------------------------------------
# Whole recordings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordingHarvest:
    """Harvest of one recording.

    Attributes
    ----------
    name
        Recording directory name.
    summary
        Counts and recording-level provenance.
    results
        One :class:`CandidateResult` per candidate, in box order.
    """

    name: str
    summary: dict[str, Any]
    results: list[CandidateResult]


def harvest_recording(
    path: Path | str,
    config: HarvestConfig,
    *,
    individuals: dict[tuple[str, int], str] | None = None,
    max_candidates: int | None = None,
    candidate_range: tuple[int, int] | None = None,
) -> RecordingHarvest:
    """Harvest every candidate box of one recording.

    Parameters
    ----------
    path
        Recording directory (fishgrid raw, tracks, ``chirpdetector_bboxes.csv``).
    config
        Harvest configuration.
    individuals
        Optional ``(recording, track) -> individual`` map.
    max_candidates
        Process only the first candidates (tests).
    candidate_range
        Process only candidates ``[start, stop)`` (parallel chunks); the
        summary still counts all of them.

    Returns
    -------
    RecordingHarvest
    """
    path = Path(path)
    name = path.name
    detections = read_detections_lenient(path)
    candidates = find_candidates(detections, config)
    if max_candidates is not None:
        candidates = candidates.head(max_candidates)
    summary: dict[str, Any] = {
        "recording": name,
        "n_detections": len(detections),
        "n_candidates": len(candidates),
    }
    if candidate_range is not None:
        candidates = candidates.iloc[candidate_range[0] : candidate_range[1]]
    if candidates.empty:
        return RecordingHarvest(name, summary, [])
    recording, dropped = open_recording_lenient(path)
    tracks = load_tracks(path)
    summary.update(
        samplerate_hz=recording.samplerate_hz,
        n_channels=recording.n_channels,
        sign_v_columns=tracks.n_electrodes,
        sign_v_matches_channels=tracks.n_electrodes == recording.n_channels,
        raw_partial_frame_bytes=dropped,
        n_tracks=len(tracks.ids),
    )
    results = []
    rows: Frame = candidates
    for box in rows.to_dict("records"):
        try:
            result = harvest_candidate(
                recording,
                tracks,
                box,
                config,
                recording_name=name,
                individuals=individuals,
            )
        except (ValueError, IndexError) as err:
            log.warning("%s box %s failed: %s", name, box["box_index"], err)
            result = CandidateResult(
                {
                    "recording": name,
                    "box_index": int(box["box_index"]),
                    "box_t_start_s": float(box["t_start_s"]),
                    "box_t_end_s": float(box["t_end_s"]),
                    "status": "rejected",
                    "reason": f"error: {err}",
                },
                None,
            )
        results.append(result)
    log.info(
        "%s: %d candidates, %d harvested",
        name,
        len(results),
        sum(r.record["status"] == "harvested" for r in results),
    )
    return RecordingHarvest(name, summary, results)


def mark_duplicates(records: Frame, window_s: float) -> pd.Series:
    """Harvested templates that repeat an earlier one (same emitter, same chirp).

    Overlapping v1 boxes can cover one chirp twice. Within a recording and
    emitter, templates whose 10% onsets are less than ``window_s`` apart are
    one chirp; the one from the higher-scoring box is kept.

    Parameters
    ----------
    records
        Candidate records with ``status``, ``recording``, ``emitter_track_id``,
        ``onset_10_abs_s`` and ``box_score``.
    window_s
        Onset distance below which two templates are the same chirp.

    Returns
    -------
    pandas.Series of bool
        True for the duplicates to drop (aligned with ``records``).
    """
    duplicate = pd.Series(False, index=records.index)
    harvested: Frame = records[records["status"] == "harvested"]
    for _, group in harvested.groupby(["recording", "emitter_track_id"], sort=True):
        order = group.sort_values(
            ["box_score", "box_index"], ascending=[False, True], kind="stable"
        )
        kept: list[float] = []
        for index, row in order.iterrows():
            onset = float(row["onset_10_abs_s"])
            if any(abs(onset - k) < window_s for k in kept):
                duplicate[index] = True
            else:
                kept.append(onset)
    return duplicate


# --------------------------------------------------------------------------
# Running the harvest and writing its outputs
# --------------------------------------------------------------------------

HARVEST_DELTA_FILE = "delta_f_hz.npy"
HARVEST_AMP_FILE = "amp_rel.npy"
HARVEST_METADATA_FILE = "templates.parquet"
HARVEST_TRACES_FILE = "traces.npz"
HARVEST_MANIFEST_FILE = "manifest.json"
CANDIDATES_FILE = "candidates.csv"
RECORDINGS_FILE = "recordings.csv"
REVIEW_FILE = "review.csv"

#: Quality flags shown in the review sheet (none of them rejects a template).
REVIEW_FLAGS: tuple[str, ...] = (
    "step_flag",
    "truncated",
    "decay_interrupted",
    "low_snr",
    "line_in_span",
    "band_clipped",
    "multi_peak",
    "ripple_flag",
    "too_short",
)


def harvest_key(recording: str, box_index: int) -> str:
    """Stable identifier of a harvested chirp: recording and v1 box index."""
    return f"{recording}#{box_index}"


#: Candidates per parallel job (one recording can hold hundreds).
CHUNK_CANDIDATES = 25


def _harvest_one(
    args: tuple[Path, HarvestConfig, dict[tuple[str, int], str], tuple[int, int]],
) -> RecordingHarvest:
    path, config, individuals, chunk = args
    return harvest_recording(
        path, config, individuals=individuals, candidate_range=chunk
    )


def _plan(path: Path, config: HarvestConfig) -> list[tuple[int, int]]:
    """Candidate chunks of one recording (at least one, possibly empty)."""
    n = len(find_candidates(read_detections_lenient(path), config))
    return [
        (a, min(a + CHUNK_CANDIDATES, n)) for a in range(0, n, CHUNK_CANDIDATES)
    ] or [(0, 0)]


def run_harvest(
    recordings: Sequence[Path | str],
    out_dir: Path | str,
    config: HarvestConfig | None = None,
    *,
    individuals_csv: Path | str | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Harvest long chirps from several recordings and write the results.

    Writes to ``out_dir``: ``candidates.csv`` (every candidate with status,
    rejection reason and provenance), ``recordings.csv`` (per-recording
    counts), ``harvest/`` (templates of the harvested chirps:
    ``delta_f_hz.npy``, ``amp_rel.npy``, ``templates.parquet``,
    ``traces.npz`` for the contact sheets, ``manifest.json``) and
    ``review.csv`` (one row per harvested chirp with an ``accept`` column,
    True by default). An existing ``review.csv`` keeps its ``accept`` and
    ``note`` values. Deterministic for fixed inputs and configuration.

    Parameters
    ----------
    recordings
        Recording directories.
    out_dir
        Output directory (created if needed).
    config
        Harvest configuration (default ``HarvestConfig()``).
    individuals_csv
        ``order_meta.csv`` mapping tracks to individuals (optional).
    workers
        Worker processes (one recording each).

    Returns
    -------
    pandas.DataFrame
        The candidate table.
    """
    import datetime
    import json
    from concurrent.futures import ProcessPoolExecutor

    from gridtools.templates.build import (
        _code_version,
        _file_digest,
        _json_default,
        add_row_flags,
        snippet_record,
    )

    config = HarvestConfig() if config is None else config
    out_dir = Path(out_dir)
    harvest_dir = out_dir / "harvest"
    harvest_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(tz=datetime.UTC)
    individuals = read_individuals(Path(individuals_csv)) if individuals_csv else {}
    paths = sorted(map(Path, recordings))
    jobs = [
        (p, config, individuals, chunk) for p in paths for chunk in _plan(p, config)
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_harvest_one, jobs))
    else:
        parts = [_harvest_one(job) for job in jobs]
    harvests = []
    for path in paths:
        mine = [h for h in parts if h.name == path.name]
        harvests.append(
            RecordingHarvest(
                path.name, mine[0].summary, [r for h in mine for r in h.results]
            )
        )

    records = pd.DataFrame(
        [r.record for h in harvests for r in h.results],
    )
    extractions = [r.extraction for h in harvests for r in h.results]
    if records.empty:
        records = pd.DataFrame(columns=["recording", "box_index", "status", "reason"])
    else:
        duplicate = mark_duplicates(records, config.duplicate_window_s)
        records.loc[duplicate, "status"] = "rejected"
        records.loc[duplicate, "reason"] = "duplicate"
    records.insert(
        0,
        "harvest_key",
        [
            harvest_key(r, int(b))
            for r, b in zip(records["recording"], records["box_index"], strict=True)
        ],
    )
    records["harvest_id"] = -1
    harvested = np.flatnonzero(records["status"].to_numpy() == "harvested")
    records.loc[records.index[harvested], "harvest_id"] = np.arange(harvested.size)

    fs = 0.0
    rows: list[dict[str, Any]] = []
    deltas, amps = [], []
    trace_parts: dict[str, list[npt.NDArray[Any]]] = {
        "signal": [],
        "smooth_freq": [],
        "orig_freq": [],
        "sigma_s": [],
    }
    starts, lengths = [], []
    for harvest_id, i in enumerate(harvested):
        extraction = extractions[i]
        assert extraction is not None
        fs = extraction.samplerate_hz
        snippet = extraction.snippet
        row = snippet_record(harvest_id, -1, snippet, fs)
        row.update(records.iloc[i].to_dict())
        rows.append(row)
        deltas.append(snippet.delta_f_hz.astype(np.float32))
        amps.append(snippet.amp_rel.astype(np.float32))
        trace_parts["signal"].append(extraction.signal)
        trace_parts["smooth_freq"].append(
            extraction.traces["smooth_freq"].astype(np.float32)
        )
        trace_parts["orig_freq"].append(
            extraction.traces["orig_freq"].astype(np.float32)
        )
        trace_parts["sigma_s"].append(extraction.sigma_s.astype(np.float32))
        starts.append(extraction.window_start_s)
        lengths.append(extraction.signal.size)

    n_out = config.template.n_samples(fs) if fs else 0
    meta = pd.DataFrame(rows)
    if not meta.empty:
        meta = add_row_flags(meta, config.template)
        meta["template_id"] = meta["harvest_id"]
        meta["source"] = "competition"
        meta["duplicate_of"] = -1
        meta["duplicate_kind"] = ""
        meta["amp_usable"] = (
            ~meta["amp_ripple_flag"] & ~meta["step_flag"] & meta["valid"]
        )
        groups = {
            key: k
            for k, key in enumerate(
                sorted(
                    set(zip(meta["recording"], meta["emitter_track_id"], strict=True))
                )
            )
        }
        meta["inferred_group"] = [
            groups[(r, t)]
            for r, t in zip(meta["recording"], meta["emitter_track_id"], strict=True)
        ]
        meta["group_label"] = [
            f"{r}/track{int(t)}"
            for r, t in zip(meta["recording"], meta["emitter_track_id"], strict=True)
        ]
    np.save(
        harvest_dir / HARVEST_DELTA_FILE,
        np.stack(deltas) if deltas else np.zeros((0, n_out), np.float32),
    )
    np.save(
        harvest_dir / HARVEST_AMP_FILE,
        np.stack(amps) if amps else np.ones((0, n_out), np.float32),
    )
    meta.to_parquet(harvest_dir / HARVEST_METADATA_FILE, index=False)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    arrays: dict[str, Any] = {
        name: np.concatenate(parts) if parts else np.zeros(0, np.float32)
        for name, parts in trace_parts.items()
    }
    arrays["offsets"] = offsets
    arrays["window_start_s"] = np.asarray(starts, dtype=np.float64)
    arrays["samplerate_hz"] = np.float64(fs)
    np.savez(harvest_dir / HARVEST_TRACES_FILE, **arrays)

    records.to_csv(out_dir / CANDIDATES_FILE, index=False)
    summary = recording_summary(harvests, records)
    summary.to_csv(out_dir / RECORDINGS_FILE, index=False)
    write_review(meta, out_dir / REVIEW_FILE)

    manifest = {
        "created_utc": started.isoformat(),
        "samplerate_hz": fs,
        "n_samples": n_out,
        "centre_index": n_out // 2,
        "n_templates": len(meta),
        "config": config.to_dict(),
        "recordings": [str(p) for p in paths],
        "individuals_csv": str(individuals_csv) if individuals_csv else None,
        "code": _code_version(),
        "counts": {
            "candidates": len(records),
            "harvested": int(harvested.size),
            "rejected_by_reason": records.loc[records["status"] == "rejected", "reason"]
            .value_counts()
            .to_dict(),
        },
        "outputs": {
            name: _file_digest(harvest_dir / name)
            for name in (HARVEST_DELTA_FILE, HARVEST_AMP_FILE, HARVEST_METADATA_FILE)
        },
        "species": "Apteronotus leptorhynchus",
    }
    (harvest_dir / HARVEST_MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, default=_json_default)
    )
    log.info(
        "harvest: %d candidates, %d harvested -> %s",
        len(records),
        harvested.size,
        out_dir,
    )
    return records


def recording_summary(
    harvests: Sequence[RecordingHarvest], records: Frame
) -> pd.DataFrame:
    """Per-recording counts: detections, candidates, harvested, rejections."""
    rows = []
    for h in harvests:
        mine: Frame = (
            records[records["recording"] == h.name] if len(records) else records
        )
        row = dict(h.summary)
        row["n_harvested"] = (
            int((mine["status"] == "harvested").sum()) if len(mine) else 0
        )
        if len(mine):
            for reason, count in (
                mine.loc[mine["status"] == "rejected", "reason"].value_counts().items()
            ):
                row[f"rejected_{reason}"] = int(count)
            individuals = mine.loc[mine["status"] == "harvested", "individual"]
            row["individuals"] = ";".join(sorted({str(i) for i in individuals if i}))
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def write_review(meta: Frame, path: Path) -> pd.DataFrame:
    """Write the review sheet, keeping earlier ``accept`` and ``note`` values.

    One row per harvested chirp, ``accept`` True by default. If ``path``
    exists, rows with the same ``harvest_key`` keep their values, so the
    author's vetoes survive a re-run.

    Returns
    -------
    pandas.DataFrame
        The written sheet.
    """
    columns = [
        "harvest_id",
        "harvest_key",
        "recording",
        "emitter_track_id",
        "individual",
        "onset_10_abs_s",
        "duration_10_s",
        "excursion_hz",
        "baseline_hz",
    ]
    sheet: Frame
    if meta.empty:
        sheet = pd.DataFrame(columns=[*columns, "flags", "accept", "note"])
    else:
        sheet = meta[columns].copy()
        sheet["duration_10_ms"] = (1e3 * sheet.pop("duration_10_s")).round(1)
        sheet["excursion_hz"] = sheet["excursion_hz"].round(1)
        sheet["baseline_hz"] = sheet["baseline_hz"].round(1)
        sheet["onset_10_abs_s"] = sheet["onset_10_abs_s"].round(3)
        flags = [f for f in REVIEW_FLAGS if f in meta.columns]
        sheet["flags"] = [
            " ".join(f for f in flags if bool(meta[f].iloc[k]))
            for k in range(len(meta))
        ]
        sheet["accept"] = True
        sheet["note"] = ""
    if path.exists():
        old: Frame = pd.read_csv(path, dtype={"harvest_key": str, "note": str})
        old = old.set_index("harvest_key")
        known = sheet["harvest_key"].isin(old.index)
        if known.any():
            keys = sheet.loc[known, "harvest_key"]
            sheet.loc[known, "accept"] = old.loc[keys, "accept"].astype(bool).to_numpy()
            sheet.loc[known, "note"] = old.loc[keys, "note"].fillna("").to_numpy()
    sheet.to_csv(path, index=False)
    return sheet


def read_review(path: Path | str) -> pd.DataFrame:
    """Read a review sheet; ``accept`` is parsed as bool (True/False/1/0)."""
    sheet: Frame = pd.read_csv(path, dtype={"harvest_key": str, "note": str})
    accept = sheet["accept"].astype(str).str.strip().str.lower()
    unknown = ~accept.isin(["true", "false", "1", "0", "yes", "no"])
    if unknown.any():
        msg = (
            f"{path}: accept must be True or False, got {sorted(set(accept[unknown]))}"
        )
        raise ValueError(msg)
    sheet["accept"] = accept.isin(["true", "1", "yes"])
    return sheet
