"""Noise floor, power-line hum, fish amplitude and clicks from raw segments.

Power spectral densities are one-sided Welch estimates with a periodic Hann
window and mean removal, scaled to unit**2/Hz. This is the scaling of
``matplotlib.mlab.specgram(mode="psd")`` that produced the wavetracker
``sign_v`` values, so a noise density measured here and a ``sign_v`` entry
are directly comparable: for a sinusoid of amplitude :math:`A` centred in a
bin of an ``nfft``-point Hann spectrum the PSD peak is

.. math::  P = \\frac{A^2/2}{\\mathrm{ENBW}}, \\qquad
           \\mathrm{ENBW} = 1.5\\,\\frac{f_s}{n_\\mathrm{fft}},

so the carrier-to-noise-density ratio of a fish is
``sign_v + 10 log10(ENBW) - N0`` in dB-Hz, independent of the FFT length
(:func:`carrier_to_noise_db_hz`).

Amplitudes of lines (hum harmonics, a fish's fundamental) are measured by
summing the PSD above the local floor over the main lobe of the line
(Parseval), which is insensitive to where the line falls between bins.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import welch

__all__ = [
    "HANN_ENBW_BINS",
    "ClickParams",
    "HumLines",
    "carrier_to_noise_db_hz",
    "count_clicks",
    "floor_spectrum",
    "hum_lines",
    "line_amplitude",
    "noise_floor",
    "power_law_fit",
    "psd",
    "segment_starts",
]

FloatArray = npt.NDArray[np.float64]

#: Equivalent noise bandwidth of a Hann window, in bins.
HANN_ENBW_BINS = 1.5


def segment_starts(
    rng: np.random.Generator,
    n_samples: int,
    segment_samples: int,
    n_segments: int,
    *,
    margin_samples: int = 0,
) -> npt.NDArray[np.int64]:
    """Draw sorted, non-overlapping segment start indices.

    The recording is cut into ``n_segments`` equal strata and one segment is
    drawn uniformly inside each stratum, so segments cover the whole
    recording (dark and light phase) and never overlap.

    Parameters
    ----------
    rng : numpy.random.Generator
    n_samples : int
        Length of the recording.
    segment_samples : int
        Length of each segment.
    n_segments : int
        Number of segments (>= 1).
    margin_samples : int
        Samples to keep free at both ends.

    Returns
    -------
    numpy.ndarray of int64, shape (n_segments,)
    """
    usable = n_samples - 2 * margin_samples
    stratum = usable // n_segments
    if n_segments < 1 or stratum < segment_samples:
        msg = (
            f"cannot place {n_segments} segments of {segment_samples} samples "
            f"in {n_samples} samples"
        )
        raise ValueError(msg)
    offsets = rng.integers(0, stratum - segment_samples + 1, n_segments)
    return (margin_samples + np.arange(n_segments) * stratum + offsets).astype(np.int64)


def psd(
    x: npt.ArrayLike,
    samplerate_hz: float,
    nfft: int,
    overlap: float = 0.5,
) -> tuple[FloatArray, FloatArray]:
    """Welch PSD of each column, mlab/thunderfish scaling.

    Parameters
    ----------
    x : array_like, shape (n, E)
        Samples (one column per electrode).
    samplerate_hz : float
    nfft : int
        Segment and FFT length.
    overlap : float
        Fractional overlap of the segments (default 0.5).

    Returns
    -------
    freqs_hz : numpy.ndarray, shape (nfft // 2 + 1,)
    power : numpy.ndarray, shape (nfft // 2 + 1, E)
        One-sided PSD in unit**2/Hz, averaged over segments.
    """
    data = np.asarray(x, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    freqs, power = welch(
        data,
        fs=samplerate_hz,
        window="hann",
        nperseg=nfft,
        noverlap=int(nfft * overlap),
        detrend="constant",
        scaling="density",
        axis=0,
        average="mean",
    )
    return np.asarray(freqs, dtype=np.float64), np.asarray(power, dtype=np.float64)


def noise_floor(
    freqs_hz: npt.ArrayLike,
    power: npt.ArrayLike,
    band_hz: tuple[float, float],
) -> FloatArray:
    """Median PSD in a band, per electrode.

    Lines (fish, hum harmonics) occupy a small fraction of the bins of a
    band of a few hundred Hz, so the median is the floor between them. With
    a Welch average over many segments the median of the averaged noise is
    within a few percent of its mean.

    Parameters
    ----------
    freqs_hz : array_like, shape (F,)
    power : array_like, shape (F, E)
    band_hz : (low, high)

    Returns
    -------
    numpy.ndarray, shape (E,)
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    p = np.asarray(power, dtype=np.float64)
    sel = (f >= band_hz[0]) & (f < band_hz[1])
    return np.median(p[sel], axis=0)


def floor_spectrum(
    freqs_hz: npt.ArrayLike,
    power: npt.ArrayLike,
    edges_hz: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Noise floor in consecutive bands (median per band and electrode).

    Returns
    -------
    centres_hz : numpy.ndarray, shape (B,)
        Geometric band centres.
    floor : numpy.ndarray, shape (B, E)
    """
    edges = np.asarray(edges_hz, dtype=np.float64)
    floors = [
        noise_floor(freqs_hz, power, (float(lo), float(hi)))
        for lo, hi in itertools.pairwise(edges)
    ]
    return np.sqrt(edges[:-1] * edges[1:]), np.vstack(floors)


def power_law_fit(freqs_hz: npt.ArrayLike, floor: npt.ArrayLike) -> tuple[float, float]:
    """Least-squares fit of ``log10 floor = c - alpha log10 f``.

    Parameters
    ----------
    freqs_hz : array_like, shape (B,)
    floor : array_like, shape (B,)
        Positive PSD values.

    Returns
    -------
    alpha : float
        Spectral exponent (0 = white, 1 = pink).
    level_at_1khz : float
        Fitted PSD at 1 kHz.
    """
    lf = np.log10(np.asarray(freqs_hz, dtype=np.float64))
    lp = np.log10(np.asarray(floor, dtype=np.float64))
    slope, intercept = np.polyfit(lf - 3.0, lp, 1)
    return float(-slope), float(10.0**intercept)


def line_amplitude(
    freqs_hz: npt.ArrayLike,
    power: npt.ArrayLike,
    line_hz: float,
    *,
    lobe_bins: int = 3,
    floor_hz: tuple[float, float] = (8.0, 20.0),
) -> FloatArray:
    """Amplitude of a sinusoid from the PSD around its frequency.

    :math:`A = \\sqrt{2 \\sum (P - P_\\mathrm{floor}) \\Delta f}` over
    ``lobe_bins`` bins on each side of the bin nearest ``line_hz``, with the
    floor taken as the median PSD at ``floor_hz`` distance on both sides.

    Parameters
    ----------
    freqs_hz : array_like, shape (F,)
    power : array_like, shape (F, E)
    line_hz : float
    lobe_bins : int
        Half-width of the summed lobe in bins (default 3 covers the Hann
        main lobe of a line that drifts by a fraction of a bin).
    floor_hz : (near, far)
        Distance range of the local floor from the line, in Hz.

    Returns
    -------
    numpy.ndarray, shape (E,)
        Amplitudes in the unit of the data; 0 where the lobe does not exceed
        the floor.
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    p = np.asarray(power, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    df = f[1] - f[0]
    k = int(np.argmin(np.abs(f - line_hz)))
    lo, hi = max(0, k - lobe_bins), min(f.size, k + lobe_bins + 1)
    distance = np.abs(f - line_hz)
    near = (distance >= floor_hz[0]) & (distance <= floor_hz[1])
    floor = np.median(p[near], axis=0) if np.any(near) else np.zeros(p.shape[1])
    excess = np.sum(p[lo:hi] - floor[None, :], axis=0) * df
    return np.sqrt(2.0 * np.maximum(excess, 0.0))


@dataclass(frozen=True)
class HumLines:
    """Power-line hum measured on one segment.

    Attributes
    ----------
    frequencies_hz : numpy.ndarray, shape (H,)
        Nominal harmonic frequencies.
    amplitudes : numpy.ndarray, shape (H, E)
        Amplitude of each harmonic on each electrode (data unit).
    detected : numpy.ndarray of bool, shape (H, E)
        Whether the harmonic's peak PSD exceeds the local floor by
        ``min_snr_db``.
    """

    frequencies_hz: FloatArray
    amplitudes: FloatArray
    detected: npt.NDArray[np.bool_]


def hum_lines(
    freqs_hz: npt.ArrayLike,
    power: npt.ArrayLike,
    *,
    line_hz: float = 50.0,
    n_harmonics: int = 40,
    min_snr_db: float = 6.0,
) -> HumLines:
    """Amplitudes of the mains frequency and its harmonics.

    Parameters
    ----------
    freqs_hz : array_like, shape (F,)
    power : array_like, shape (F, E)
    line_hz : float
        Mains frequency (default 50 Hz).
    n_harmonics : int
        Number of components including the fundamental (default 40, up to
        2 kHz).
    min_snr_db : float
        Detection threshold of the peak over the local floor.

    Returns
    -------
    HumLines
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    p = np.asarray(power, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    harmonics = line_hz * np.arange(1, n_harmonics + 1, dtype=np.float64)
    amps = np.zeros((n_harmonics, p.shape[1]))
    detected = np.zeros((n_harmonics, p.shape[1]), dtype=bool)
    for i, h in enumerate(harmonics):
        amps[i] = line_amplitude(f, p, float(h))
        k = int(np.argmin(np.abs(f - h)))
        distance = np.abs(f - h)
        near = (distance >= 8.0) & (distance <= 20.0)
        floor = np.median(p[near], axis=0)
        peak = p[max(0, k - 1) : k + 2].max(axis=0)
        detected[i] = peak > floor * 10 ** (min_snr_db / 10.0)
    return HumLines(harmonics, amps, detected)


def carrier_to_noise_db_hz(
    peak_psd_db: npt.ArrayLike,
    noise_psd_db: npt.ArrayLike,
    samplerate_hz: float,
    nfft: int,
) -> FloatArray:
    """Carrier-to-noise-density ratio from a Hann-PSD peak and a noise density.

    ``C/N0 = (A^2/2) / N0`` in dB-Hz; the SNR in any spectrogram with an
    equivalent noise bandwidth ``B`` follows as ``C/N0 - 10 log10(B)``.

    Parameters
    ----------
    peak_psd_db : array_like
        PSD at the line's bin in dB (e.g. ``sign_v``).
    noise_psd_db : array_like
        Noise density in dB (same reference).
    samplerate_hz : float
    nfft : int
        FFT length of the spectrum behind ``peak_psd_db``.

    Returns
    -------
    numpy.ndarray
        C/N0 in dB-Hz (a line between two bins loses up to 1.4 dB).
    """
    enbw_hz = HANN_ENBW_BINS * samplerate_hz / nfft
    return (
        np.asarray(peak_psd_db, dtype=np.float64)
        + 10.0 * math.log10(enbw_hz)
        - np.asarray(noise_psd_db, dtype=np.float64)
    )


@dataclass(frozen=True)
class ClickParams:
    """Click detection, after :class:`audian.clicks.ClickRemoval`.

    A click is an impulsive event in the high band (above ``high_band``
    times the sampling rate), where narrowband fish signals do not reach:
    its 1-ms envelope reaches ``threshold`` robust standard deviations of
    the channel's high band, it lasts as long as it stays above
    ``extent_threshold``, and it is shorter than ``max_duration_s``.

    Attributes
    ----------
    threshold : float
    extent_threshold : float
    high_band : float
    high_band_edge : float
    resolution_s : float
    max_duration_s : float
    line_margin_db : float
        Spectral lines this far above the median spectrum (in 10-Hz chunks)
        are not part of the high band.
    """

    threshold: float = 12.0
    extent_threshold: float = 8.0
    high_band: float = 0.2
    high_band_edge: float = 0.05
    resolution_s: float = 1e-3
    max_duration_s: float = 0.5
    line_margin_db: float = 15.0


def _runs(mask: npt.NDArray[np.bool_]) -> tuple[npt.NDArray[np.int64], ...]:
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def count_clicks(
    x: npt.ArrayLike,
    samplerate_hz: float,
    params: ClickParams | None = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Detect clicks on each channel of a segment.

    Parameters
    ----------
    x : array_like, shape (n, E)
    samplerate_hz : float
    params : ClickParams, optional

    Returns
    -------
    counts : numpy.ndarray of int, shape (E,)
        Clicks per channel.
    times_s : numpy.ndarray of float
        Onset times (s from segment start) of clicks on any channel, merged
        across channels when they overlap.
    """
    params = ClickParams() if params is None else params
    data = np.asarray(x, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    n, n_channels = data.shape
    spectrum = np.fft.rfft(data - data.mean(axis=0), axis=0)
    freqs = np.fft.rfftfreq(n, 1.0 / samplerate_hz)
    power = np.abs(spectrum) ** 2
    df = freqs[1] - freqs[0]
    chunk = min(max(8, round(10.0 / df)), power.shape[0])
    n_chunks = power.shape[0] // chunk
    medians = np.median(
        power[: n_chunks * chunk].reshape(n_chunks, chunk, n_channels), axis=1
    )
    smooth = np.repeat(medians, chunk, axis=0)
    smooth = np.concatenate(
        [smooth, np.repeat(medians[-1:], power.shape[0] - smooth.shape[0], axis=0)]
    )
    narrow = power > smooth * 10 ** (params.line_margin_db / 10.0)
    position = np.clip(
        (freqs / samplerate_hz - params.high_band) / params.high_band_edge, 0.0, 1.0
    )
    gain = np.sin(0.5 * np.pi * position) ** 2
    high = np.fft.irfft(np.where(narrow, 0.0, spectrum * gain[:, None]), n=n, axis=0)

    size = max(1, round(params.resolution_s * samplerate_hz))
    n_bins = n // size
    longest = max(1, round(params.max_duration_s / params.resolution_s))
    counts = np.zeros(n_channels, dtype=np.int64)
    any_hot = np.zeros(n_bins, dtype=bool)
    edge = 5
    for c in range(n_channels):
        magnitude = np.abs(high[:, c])
        sample = magnitude[::16]
        sample = sample[sample > 0]
        if sample.size == 0:
            continue
        sigma = float(np.median(sample)) / 0.6745
        envelope = magnitude[: n_bins * size].reshape(n_bins, size).max(axis=1)
        trigger = envelope > params.threshold * sigma
        extent = envelope > params.extent_threshold * sigma
        starts, stops = _runs(extent)
        hot = np.zeros(n_bins, dtype=bool)
        for s0, s1 in zip(starts, stops, strict=True):
            if s1 - s0 > longest or s0 < edge or s1 > n_bins - edge:
                continue
            if trigger[s0:s1].any():
                hot[s0:s1] = True
                counts[c] += 1
        any_hot |= hot
    onsets, _ = _runs(any_hot)
    return counts, onsets * size / samplerate_hz
