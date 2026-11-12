"""Levels of real backgrounds and simulated fish, and SNR scaling.

Definitions
-----------
All spectra are one-sided Welch power spectral densities (Hann window, 50 %
overlap, mean removed, ``unit**2/Hz``) computed on each electrode separately,
the scaling of ``matplotlib.mlab.specgram(mode="psd")`` that wavetracker used
for ``sign_v``. Electrodes are never summed as signals: fish fields have
opposite polarity on different electrodes and partly cancel in a sum, while
common-mode interference adds coherently, so a level of the summed signal
would not be comparable to a per-electrode fish level. (The detector front
end also aggregates electrodes in power, not in amplitude.)

Background level
    The noise floor :math:`N_0` (unit**2/Hz) is the median over electrodes of
    each electrode's median PSD in ``LEVEL_BAND_HZ`` (500-1500 Hz, the band
    the detector sees), after excluding tonal bins: bins more than
    ``tonal_threshold_db`` above a running median of the electrode-median
    spectrum (fish fundamentals and harmonics, other lines) and bins within
    2 Hz of a mains harmonic. The background *level* is the noise power in
    a reference bandwidth,

    .. math::  B = N_0 \\, b_\\mathrm{ref}, \\qquad
               b_\\mathrm{ref} = 1.5\\,\\frac{20\\,\\mathrm{kHz}}{4096}
               = 7.32\\,\\mathrm{Hz},

    the equivalent noise bandwidth of one bin of the v1 detector's
    4096-point Hann spectrogram at 20 kHz (:data:`REFERENCE_BANDWIDTH_HZ`).

Signal level
    The power of a fish is the power of its EOD fundamental,
    :math:`a^2/2` for fundamental amplitude :math:`a`, on its strongest
    electrode, as a median over time. For simulated fish it is computed from
    the known amplitudes and field gains (:func:`fish_signal_level`); for
    real fish it is measured from the recording along the fish's track
    (:func:`tracked_line_power`), by summing the PSD above the floor over the
    main lobe of the line (Parseval; insensitive to where the line falls
    between bins). The scene's signal level :math:`S` is the median over its
    simulated fish.

SNR
    :math:`\\mathrm{SNR} = 10 \\log_{10}(S / B)` in dB: how far the fish's
    band stands above the noise floor in one bin of the reference
    spectrogram on the fish's strongest electrode. Add
    :math:`10\\log_{10} b_\\mathrm{ref} = 8.65` dB to get the
    carrier-to-noise density in dB-Hz (the unit of the calibration report),
    and for white noise of standard deviation :math:`\\sigma` at 20 kHz,
    ``SNR = 20 log10(a / sigma) + 28.3 dB`` (the scene config's broadband
    ``noise.snr_db`` plus 28.3 dB).

Scaling
    A background scaled by :math:`s` has level :math:`s^2 B`, so a scene with
    signal level :math:`S` reaches the target SNR for

    .. math::  s = \\sqrt{\\frac{S}{B \\cdot 10^{\\mathrm{SNR}/10}}}

    (:func:`scale_for_snr`). Real fish inside the background are scaled with
    it, so their own SNR does not change; their level relative to the
    simulated fish is ``SNR_real - SNR_target`` dB.

Clicks are counted with the same algorithm as the calibration
(audian's click removal: impulsive events in the band above 0.2 times the
sampling rate, where fish signals do not reach).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_dilation, median_filter
from scipy.signal import welch

__all__ = [
    "HANN_ENBW_BINS",
    "LEVEL_BAND_HZ",
    "MAINS_HZ",
    "REFERENCE_BANDWIDTH_HZ",
    "BackgroundLevel",
    "ClickParams",
    "SnrDistribution",
    "background_level",
    "count_clicks",
    "fish_signal_level",
    "fish_signal_powers",
    "level_from_psd",
    "line_amplitude",
    "psd_nfft",
    "scale_for_snr",
    "snr_db",
    "tonal_mask",
    "tonal_peaks",
    "tracked_line_power",
    "welch_psd",
]

FloatArray = npt.NDArray[np.float64]

#: Equivalent noise bandwidth of a Hann window, in bins.
HANN_ENBW_BINS = 1.5

#: Band in which the background level is measured (the detector's band).
LEVEL_BAND_HZ: tuple[float, float] = (500.0, 1500.0)

#: Reference bandwidth of the SNR: ENBW of a 4096-point Hann window at 20 kHz.
REFERENCE_BANDWIDTH_HZ = HANN_ENBW_BINS * 20_000.0 / 4096

#: Mains frequency of the recordings (Germany, Brazil uses 60 Hz).
MAINS_HZ = 50.0

_PSD_WINDOW_S = 0.4096  # 8192 points at 20 kHz: 2.44 Hz bins


def psd_nfft(samplerate_hz: float) -> int:
    """Default Welch segment length: the power of two nearest 0.41 s.

    8192 points at 20 kHz (2.44 Hz resolution), enough to separate fish
    fundamentals from neighbouring mains harmonics.
    """
    return int(2 ** round(math.log2(_PSD_WINDOW_S * samplerate_hz)))


def welch_psd(
    x: npt.ArrayLike,
    samplerate_hz: float,
    nfft: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """One-sided Welch PSD of each electrode (Hann, 50 % overlap, mean removed).

    Parameters
    ----------
    x : array_like, shape (n, E) or (n,)
        Samples, one column per electrode.
    samplerate_hz : float
        Sampling rate.
    nfft : int, optional
        Segment and FFT length; default :func:`psd_nfft`.

    Returns
    -------
    freqs_hz : numpy.ndarray, shape (F,)
    psd : numpy.ndarray, shape (F, E)
        Power spectral density in ``unit**2/Hz``.
    """
    data = np.asarray(x)
    if data.ndim == 1:
        data = data[:, None]
    n = psd_nfft(samplerate_hz) if nfft is None else int(nfft)
    n = min(n, data.shape[0])
    freqs, power = welch(
        data,
        fs=samplerate_hz,
        window="hann",
        nperseg=n,
        noverlap=n // 2,
        detrend="constant",
        scaling="density",
        axis=0,
    )
    return np.asarray(freqs, dtype=np.float64), np.asarray(power, dtype=np.float64)


def _mains_mask(
    freqs_hz: FloatArray, mains_hz: float | None, halfwidth_hz: float
) -> npt.NDArray[np.bool_]:
    if mains_hz is None:
        return np.zeros(freqs_hz.shape, dtype=bool)
    harmonic = np.round(freqs_hz / mains_hz)
    return (harmonic >= 1) & (np.abs(freqs_hz - harmonic * mains_hz) <= halfwidth_hz)


def _running_median_db(
    spectrum: FloatArray, df_hz: float, smooth_hz: float
) -> FloatArray:
    size = max(3, round(smooth_hz / df_hz) | 1)
    log = 10.0 * np.log10(np.maximum(spectrum, 1e-300))
    return median_filter(log, size=size, mode="nearest")


def tonal_mask(
    freqs_hz: npt.ArrayLike,
    psd: npt.ArrayLike,
    *,
    threshold_db: float = 6.0,
    smooth_hz: float = 100.0,
    mains_hz: float | None = MAINS_HZ,
    mains_halfwidth_hz: float = 2.0,
    dilate_bins: int = 1,
) -> npt.NDArray[np.bool_]:
    """Bins occupied by spectral lines (fish, hum harmonics, other tones).

    A bin is tonal if the electrode-median PSD exceeds its running median
    (over ``smooth_hz``) by more than ``threshold_db``, or if it lies within
    ``mains_halfwidth_hz`` of a mains harmonic. Tonal runs are widened by
    ``dilate_bins`` on each side to cover the skirts of the lines.

    Parameters
    ----------
    freqs_hz : array_like, shape (F,)
    psd : array_like, shape (F, E) or (F,)
    threshold_db, smooth_hz, mains_hz, mains_halfwidth_hz, dilate_bins
        See above; ``mains_hz=None`` disables the mains mask.

    Returns
    -------
    numpy.ndarray of bool, shape (F,)
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    p = np.asarray(psd, dtype=np.float64)
    spectrum = np.median(p, axis=1) if p.ndim == 2 else p
    df = float(f[1] - f[0])
    baseline = _running_median_db(spectrum, df, smooth_hz)
    excess = 10.0 * np.log10(np.maximum(spectrum, 1e-300)) - baseline
    mask = np.asarray(excess > threshold_db, dtype=bool)
    if dilate_bins > 0:
        mask = np.asarray(binary_dilation(mask, iterations=dilate_bins), dtype=bool)
    return mask | _mains_mask(f, mains_hz, mains_halfwidth_hz)


@dataclass(frozen=True)
class BackgroundLevel:
    """Noise floor of a background and its level in the reference bandwidth.

    Attributes
    ----------
    noise_psd : float
        Floor :math:`N_0` in ``unit**2/Hz``: median over electrodes of the
        per-electrode floors.
    noise_psd_per_electrode : numpy.ndarray, shape (E,)
        Median non-tonal PSD in the band, per electrode.
    reference_bandwidth_hz : float
        Bandwidth that turns the density into a level.
    band_hz : (float, float)
        Band of the measurement.
    tonal_fraction : float
        Fraction of the band's bins excluded as tonal.
    """

    noise_psd: float
    noise_psd_per_electrode: FloatArray
    reference_bandwidth_hz: float = REFERENCE_BANDWIDTH_HZ
    band_hz: tuple[float, float] = LEVEL_BAND_HZ
    tonal_fraction: float = 0.0

    @property
    def level(self) -> float:
        """Noise power in the reference bandwidth (``unit**2``)."""
        return self.noise_psd * self.reference_bandwidth_hz

    @property
    def level_db(self) -> float:
        """:attr:`level` in dB re 1 ``unit**2``."""
        return 10.0 * math.log10(self.level) if self.level > 0 else -math.inf

    @property
    def noise_psd_db(self) -> float:
        """:attr:`noise_psd` in dB re 1 ``unit**2/Hz``."""
        return 10.0 * math.log10(self.noise_psd) if self.noise_psd > 0 else -math.inf


def level_from_psd(
    freqs_hz: npt.ArrayLike,
    psd: npt.ArrayLike,
    *,
    band_hz: tuple[float, float] = LEVEL_BAND_HZ,
    reference_bandwidth_hz: float = REFERENCE_BANDWIDTH_HZ,
    tonal_threshold_db: float = 6.0,
    mains_hz: float | None = MAINS_HZ,
) -> BackgroundLevel:
    """Background level from a precomputed Welch PSD (see module docstring).

    Parameters
    ----------
    freqs_hz : array_like, shape (F,)
    psd : array_like, shape (F, E)
    band_hz : (low, high)
        Band of the floor (default 500-1500 Hz).
    reference_bandwidth_hz : float
        Bandwidth of the level (default :data:`REFERENCE_BANDWIDTH_HZ`).
    tonal_threshold_db : float
        Lines more than this above the running median are excluded.
    mains_hz : float or None
        Mains frequency whose harmonics are excluded.

    Returns
    -------
    BackgroundLevel

    Raises
    ------
    ValueError
        If fewer than 10 non-tonal bins remain in the band.
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    p = np.asarray(psd, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    in_band = (f >= band_hz[0]) & (f < band_hz[1])
    tonal = tonal_mask(f, p, threshold_db=tonal_threshold_db, mains_hz=mains_hz)
    use = in_band & ~tonal
    if use.sum() < 10:
        msg = f"only {int(use.sum())} non-tonal bins in {band_hz} Hz"
        raise ValueError(msg)
    per_electrode = np.median(p[use], axis=0)
    return BackgroundLevel(
        noise_psd=float(np.median(per_electrode)),
        noise_psd_per_electrode=per_electrode,
        reference_bandwidth_hz=float(reference_bandwidth_hz),
        band_hz=(float(band_hz[0]), float(band_hz[1])),
        tonal_fraction=float(1.0 - use.sum() / max(1, in_band.sum())),
    )


def background_level(
    x: npt.ArrayLike,
    samplerate_hz: float,
    *,
    band_hz: tuple[float, float] = LEVEL_BAND_HZ,
    reference_bandwidth_hz: float = REFERENCE_BANDWIDTH_HZ,
    nfft: int | None = None,
    tonal_threshold_db: float = 6.0,
    mains_hz: float | None = MAINS_HZ,
) -> BackgroundLevel:
    """Background level of a multichannel recording (see module docstring).

    Parameters
    ----------
    x : array_like, shape (n, E)
        Samples.
    samplerate_hz : float
    band_hz, reference_bandwidth_hz, tonal_threshold_db, mains_hz
        See :func:`level_from_psd`.
    nfft : int, optional
        Welch segment length (default :func:`psd_nfft`).

    Returns
    -------
    BackgroundLevel
    """
    freqs, power = welch_psd(x, samplerate_hz, nfft)
    return level_from_psd(
        freqs,
        power,
        band_hz=band_hz,
        reference_bandwidth_hz=reference_bandwidth_hz,
        tonal_threshold_db=tonal_threshold_db,
        mains_hz=mains_hz,
    )


def tonal_peaks(
    freqs_hz: npt.ArrayLike,
    psd: npt.ArrayLike,
    *,
    band_hz: tuple[float, float] = (200.0, 2000.0),
    threshold_db: float = 10.0,
    smooth_hz: float = 100.0,
    mains_hz: float | None = MAINS_HZ,
    mains_halfwidth_hz: float = 2.0,
) -> tuple[FloatArray, FloatArray]:
    """Spectral lines of the electrode-summed power, outside mains harmonics.

    The PSDs of all electrodes are summed (power, not signals), so a fish
    seen on any electrode shows up.

    Returns
    -------
    freqs_hz : numpy.ndarray
        Frequencies of local maxima more than ``threshold_db`` above the
        running median, within ``band_hz``.
    excess_db : numpy.ndarray
        Their height above the running median in dB.
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    p = np.asarray(psd, dtype=np.float64)
    spectrum = p.sum(axis=1) if p.ndim == 2 else p
    df = float(f[1] - f[0])
    baseline = _running_median_db(spectrum, df, smooth_hz)
    excess = 10.0 * np.log10(np.maximum(spectrum, 1e-300)) - baseline
    interior = np.zeros(f.shape, dtype=bool)
    interior[1:-1] = (excess[1:-1] >= excess[:-2]) & (excess[1:-1] > excess[2:])
    keep = (
        interior
        & (excess > threshold_db)
        & (f >= band_hz[0])
        & (f <= band_hz[1])
        & ~_mains_mask(f, mains_hz, mains_halfwidth_hz)
    )
    return f[keep], excess[keep]


def line_amplitude(
    freqs_hz: npt.ArrayLike,
    psd: npt.ArrayLike,
    line_hz: float,
    *,
    lobe_bins: int = 3,
    floor_hz: tuple[float, float] = (8.0, 20.0),
) -> FloatArray:
    """Amplitude of a sinusoid from the PSD around its frequency.

    :math:`A = \\sqrt{2 \\sum (P - P_\\mathrm{floor}) \\Delta f}` over
    ``lobe_bins`` bins on each side of the bin nearest ``line_hz``, with the
    floor taken as the median PSD ``floor_hz`` away on both sides (same
    definition as the calibration).

    Returns
    -------
    numpy.ndarray, shape (E,)
        Amplitudes in the unit of the data; 0 where the lobe does not exceed
        the floor.
    """
    f = np.asarray(freqs_hz, dtype=np.float64)
    p = np.asarray(psd, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    df = float(f[1] - f[0])
    k = int(np.argmin(np.abs(f - line_hz)))
    lo, hi = max(0, k - lobe_bins), min(f.size, k + lobe_bins + 1)
    distance = np.abs(f - line_hz)
    near = (distance >= floor_hz[0]) & (distance <= floor_hz[1])
    floor = np.median(p[near], axis=0) if np.any(near) else np.zeros(p.shape[1])
    excess = np.sum(p[lo:hi] - floor[None, :], axis=0) * df
    return np.sqrt(2.0 * np.maximum(excess, 0.0))


def tracked_line_power(
    x: npt.ArrayLike,
    samplerate_hz: float,
    frequency_hz: Callable[[FloatArray], FloatArray],
    noise_psd: npt.ArrayLike,
    *,
    nfft: int | None = None,
    lobe_bins: int = 3,
    others_hz: Sequence[Callable[[FloatArray], FloatArray]] = (),
    min_separation_hz: float = 15.0,
) -> tuple[FloatArray, FloatArray]:
    """Power of a line that follows a frequency track, per frame and electrode.

    Each frame (``nfft`` points, half overlap, Hann) gives
    :math:`\\sum_k (P_k - N_0) \\Delta f` over ``lobe_bins`` bins on each
    side of the bin nearest the track frequency, the power of the line
    without the noise in the lobe.

    Parameters
    ----------
    x : array_like, shape (n, E)
        Samples.
    samplerate_hz : float
    frequency_hz : callable
        Maps frame-centre times (s from the first sample) to the line's
        frequency; NaN where unknown.
    noise_psd : array_like, shape (E,)
        Per-electrode noise floor (``unit**2/Hz``) to subtract.
    nfft : int, optional
        Frame length (default :func:`psd_nfft`).
    lobe_bins : int
        Half-width of the summed lobe in bins (default 3).
    others_hz : sequence of callable
        Tracks of other lines (other fish); frames where one of them, or a
        harmonic or subharmonic of one of them, comes closer than
        ``min_separation_hz`` are set to NaN.
    min_separation_hz : float
        See ``others_hz``.

    Returns
    -------
    times_s : numpy.ndarray, shape (T,)
        Frame centres.
    power : numpy.ndarray, shape (T, E)
        Line power (``unit**2``, the :math:`a^2/2` of a sinusoid), NaN for
        frames without a usable frequency.
    """
    data = np.asarray(x)
    if data.ndim == 1:
        data = data[:, None]
    n = psd_nfft(samplerate_hz) if nfft is None else int(nfft)
    hop = n // 2
    n_frames = 1 + (data.shape[0] - n) // hop if data.shape[0] >= n else 0
    starts = np.arange(n_frames) * hop
    times: FloatArray = np.asarray((starts + n / 2) / samplerate_hz, dtype=np.float64)
    freq = np.asarray(frequency_hz(times), dtype=np.float64)
    valid = np.isfinite(freq)
    for other in others_hz:
        g = np.asarray(other(times), dtype=np.float64)
        for ratio in (0.5, 1.0, 2.0, 3.0):
            close = np.abs(freq - ratio * g) < min_separation_hz
            valid &= ~(close & np.isfinite(g))
    window = np.hanning(n + 1)[:-1]  # periodic Hann, as scipy's welch
    scale = 2.0 / (samplerate_hz * np.sum(window**2))
    df = samplerate_hz / n
    floor = np.asarray(noise_psd, dtype=np.float64)
    power = np.full((n_frames, data.shape[1]), np.nan)
    for i in np.flatnonzero(valid):
        k = round(freq[i] / df)
        lo, hi = k - lobe_bins, k + lobe_bins + 1
        if lo < 1 or hi > n // 2:
            continue
        frame = np.asarray(data[starts[i] : starts[i] + n], dtype=np.float64)
        frame = (frame - frame.mean(axis=0)) * window[:, None]
        bins = np.arange(lo, hi)
        basis = np.exp(-2j * np.pi * np.outer(bins, np.arange(n)) / n)
        spectrum = basis @ frame
        psd = scale * np.abs(spectrum) ** 2
        power[i] = np.sum(psd - floor[None, :], axis=0) * df
    return times, np.asarray(power, dtype=np.float64)


def fish_signal_powers(fundamental_amplitudes: Sequence[npt.ArrayLike]) -> FloatArray:
    """Per-fish signal power: median over time of the strongest electrode's power.

    Parameters
    ----------
    fundamental_amplitudes : sequence of array_like, each shape (T, E)
        Amplitude of each fish's EOD fundamental on every electrode at
        ``T`` times (EOD amplitude x relative amplitude of the fundamental x
        field gain x channel gain; sign ignored). ``T`` may differ between
        fish; NaN rows are ignored.

    Returns
    -------
    numpy.ndarray, shape (n_fish,)
        :math:`\\mathrm{median}_t \\max_e a_{te}^2 / 2` per fish
        (``unit**2``).
    """
    powers = []
    for amplitude in fundamental_amplitudes:
        a = np.asarray(amplitude, dtype=np.float64)
        if a.ndim == 1:
            a = a[None, :]
        strongest = np.nanmax(0.5 * a**2, axis=1)
        powers.append(float(np.nanmedian(strongest)))
    return np.asarray(powers, dtype=np.float64)


def fish_signal_level(fundamental_amplitudes: Sequence[npt.ArrayLike]) -> float:
    """Signal level of a scene: median over fish of :func:`fish_signal_powers`.

    Raises
    ------
    ValueError
        If no fish is given.
    """
    powers = fish_signal_powers(fundamental_amplitudes)
    if powers.size == 0:
        msg = "a signal level needs at least one fish"
        raise ValueError(msg)
    return float(np.median(powers))


def snr_db(signal_level: float, noise_level: float) -> float:
    """SNR in dB from a signal level and a background level (both ``unit**2``)."""
    if signal_level <= 0 or noise_level <= 0:
        msg = f"levels must be > 0, got {signal_level!r}, {noise_level!r}"
        raise ValueError(msg)
    return 10.0 * math.log10(signal_level / noise_level)


def scale_for_snr(
    segment_level: float,
    signal_level: float,
    snr_db: float,
    *,
    other_noise_level: float = 0.0,
) -> float:
    """Factor on a background that puts a scene's fish at ``snr_db``.

    .. math::  s = \\sqrt{\\frac{S / 10^{\\mathrm{SNR}/10} - B_\\mathrm{other}}{B}}

    Parameters
    ----------
    segment_level : float
        Background level :math:`B` (:attr:`BackgroundLevel.level`, noise
        power in the reference bandwidth, in the background's unit**2).
    signal_level : float
        Scene signal level :math:`S` (:func:`fish_signal_level`, in the
        scene's unit**2).
    snr_db : float
        Target SNR in dB.
    other_noise_level : float
        Level of noise the scene adds on top of the background (e.g. its own
        coloured noise), in the same reference bandwidth; default 0.

    Returns
    -------
    float
        The scale :math:`s`. The result converts the background's unit into
        the scene's, so it is not dimensionless unless both share a unit.

    Raises
    ------
    ValueError
        If a level is not positive, or the other noise alone already exceeds
        the noise the target SNR allows.
    """
    if not (segment_level > 0 and math.isfinite(segment_level)):
        msg = f"segment_level must be finite and > 0, got {segment_level!r}"
        raise ValueError(msg)
    if not (signal_level > 0 and math.isfinite(signal_level)):
        msg = f"signal_level must be finite and > 0, got {signal_level!r}"
        raise ValueError(msg)
    if not math.isfinite(snr_db):
        msg = f"snr_db must be finite, got {snr_db!r}"
        raise ValueError(msg)
    if other_noise_level < 0:
        msg = f"other_noise_level must be >= 0, got {other_noise_level!r}"
        raise ValueError(msg)
    allowed = signal_level / 10.0 ** (snr_db / 10.0) - other_noise_level
    if allowed <= 0:
        msg = (
            f"other noise ({other_noise_level:.3g}) already exceeds the noise "
            f"level allowed at {snr_db} dB SNR"
        )
        raise ValueError(msg)
    return math.sqrt(allowed / segment_level)


@dataclass(frozen=True)
class SnrDistribution:
    """Distribution of target SNRs (dB, :data:`REFERENCE_BANDWIDTH_HZ`).

    A mixture of two uniform ranges: with probability ``hard_fraction`` the
    SNR is drawn from ``hard_db`` (harder than real recordings), otherwise
    from ``typical_db`` (the range of real fish). The defaults follow the real
    fish of the 2019/2020 competitions at their strongest electrode, with
    the same definitions: median 46 dB, 5-95 % 29.5-62 dB in the calibration
    (``gridtools.calibrate``, 33 trials); in the library v1 fish segments
    48.5 (34-64) dB for 2019/2020 and 57 (44-80) dB for 2022, whose floor is
    lower. ``typical_db`` covers that range; ``hard_db`` lies below the real
    5 % quantile and reaches down to where a fish's band barely stands out
    of the floor (0 dB).

    Attributes
    ----------
    hard_db : (float, float)
        Range of hard SNRs (default 0-30 dB).
    typical_db : (float, float)
        Range of realistic SNRs (default 30-80 dB).
    hard_fraction : float
        Probability of a hard SNR (default 0.4).
    """

    hard_db: tuple[float, float] = (0.0, 30.0)
    typical_db: tuple[float, float] = (30.0, 80.0)
    hard_fraction: float = 0.4

    def __post_init__(self) -> None:
        for name in ("hard_db", "typical_db"):
            low, high = getattr(self, name)
            if not (math.isfinite(low) and math.isfinite(high) and low <= high):
                msg = f"{name} must be a finite (low, high) range, got {(low, high)}"
                raise ValueError(msg)
        if not 0.0 <= self.hard_fraction <= 1.0:
            msg = f"hard_fraction must lie in [0, 1], got {self.hard_fraction!r}"
            raise ValueError(msg)

    @property
    def range_db(self) -> tuple[float, float]:
        """Smallest range containing every possible draw."""
        return (
            min(self.hard_db[0], self.typical_db[0]),
            max(self.hard_db[1], self.typical_db[1]),
        )

    def sample(
        self,
        rng: np.random.Generator,
        size: int | None = None,
        *,
        min_db: float | None = None,
    ) -> float | FloatArray:
        """Draw target SNRs.

        Parameters
        ----------
        rng : numpy.random.Generator
        size : int, optional
            Number of draws; one float if omitted.
        min_db : float, optional
            Draw from the distribution truncated below ``min_db`` (each
            uniform component cut at ``min_db``, the mixture weights scaled
            by the probability mass left in each). Use it to cap how much
            louder a background's real fish may be than the simulated ones:
            ``min_db = segment.info["real_snr_max_db"] - max_gap_db``. If no
            mass is left, every draw is ``min_db``.

        Returns
        -------
        float or numpy.ndarray
        """
        n = 1 if size is None else int(size)
        parts = [
            (self.hard_fraction, *self.hard_db),
            (1.0 - self.hard_fraction, *self.typical_db),
        ]
        weights, lows, highs = [], [], []
        for weight, low, high in parts:
            if min_db is not None:
                kept = (
                    1.0
                    if high <= low and low >= min_db
                    else (max(0.0, high - max(low, min_db)) / (high - low))
                    if high > low
                    else 0.0
                )
                low = max(low, min_db)
                weight *= kept
            weights.append(weight)
            lows.append(low)
            highs.append(max(high, low))
        total = sum(weights)
        if total <= 0:
            values = np.full(n, float(min_db) if min_db is not None else math.nan)
        else:
            first = rng.random(n) < weights[0] / total
            u = rng.random(n)
            low = np.where(first, lows[0], lows[1])
            high = np.where(first, highs[0], highs[1])
            values = low + u * (high - low)
        return float(values[0]) if size is None else values


@dataclass(frozen=True)
class ClickParams:
    """Click detection, after :class:`audian.clicks.ClickRemoval`.

    Same algorithm and defaults as the calibration (``gridtools.calibrate``):
    a click is an impulsive event in the high band (above ``high_band``
    times the sampling rate) whose 1-ms envelope reaches ``threshold``
    robust standard deviations, lasting while it stays above
    ``extent_threshold`` and shorter than ``max_duration_s``. Spectral lines
    ``line_margin_db`` above the median spectrum are removed first.
    """

    threshold: float = 12.0
    extent_threshold: float = 8.0
    high_band: float = 0.2
    high_band_edge: float = 0.05
    resolution_s: float = 1e-3
    max_duration_s: float = 0.5
    line_margin_db: float = 15.0


def _runs(mask: npt.NDArray[np.bool_]) -> tuple[npt.NDArray[np.intp], ...]:
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def count_clicks(
    x: npt.ArrayLike,
    samplerate_hz: float,
    params: ClickParams | None = None,
) -> tuple[npt.NDArray[np.int64], FloatArray]:
    """Detect clicks on each channel.

    Parameters
    ----------
    x : array_like, shape (n, E)
    samplerate_hz : float
    params : ClickParams, optional

    Returns
    -------
    counts : numpy.ndarray of int, shape (E,)
        Clicks per channel.
    times_s : numpy.ndarray
        Onsets (s from the first sample) of clicks on any channel, merged
        where they overlap across channels.
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
    del spectrum, power, smooth, narrow

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
