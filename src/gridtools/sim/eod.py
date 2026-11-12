"""Electric organ discharge (EOD) of wave-type fish.

The EOD of a wave-type fish is modelled as a harmonic series locked to one
instantaneous phase,

.. math::

    \\theta(t) = \\theta_0 + 2\\pi \\int_0^t f(t')\\,dt', \\qquad
    e(t) = a(t) \\sum_{k=1}^{K} A_k \\cos(k\\,\\theta(t) + \\varphi_k),

where :math:`f(t)` is the instantaneous EOD frequency (baseline, drift and,
later, chirps and rises), :math:`a(t)` an amplitude envelope (chirp amplitude
modulation), and :math:`A_k, \\varphi_k` the relative amplitudes and phases of
the fundamental and its harmonics. Because frequency enters only through the
integrated phase, any frequency excursion leaves the waveform continuous.

Slow changes of the baseline frequency are generated on a coarse control grid
and interpolated with a cubic spline:

- an individual drift (Hz) per fish, and
- a shared, relative drift common to all fish in a tank (water temperature
  changes shift every fish's EODf by roughly the same fraction).

Both are Gaussian-smoothed white noise: stationary, zero-mean Gaussian
processes with autocorrelation :math:`\\rho(\\Delta t) =
\\exp(-\\Delta t^2 / (4\\tau^2))`, where :math:`\\tau` is ``timescale_s``.
Their standard deviation is the ensemble value set by the parameters, not
rescaled to any sample statistic.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

from gridtools.sim._common import (
    Float32Array,
    FloatArray,
    as_1d_float64,
    require_n_samples,
    require_nonnegative,
    require_positive,
)

__all__ = [
    "APTERONOTUS_LEPTORHYNCHUS",
    "DriftParams",
    "HarmonicJitterParams",
    "Harmonics",
    "SharedDriftParams",
    "baseline_frequency",
    "eod_phase",
    "harmonic_waveform",
    "jitter_harmonics",
    "shared_frequency_drift",
    "synthesize_eod",
]

_CHUNK_SAMPLES = 1 << 20


@dataclass(frozen=True)
class Harmonics:
    """Relative Fourier amplitudes and phases of a wave-type EOD.

    The waveform at phase :math:`\\theta` is
    :math:`\\sum_k A_k \\cos(k\\theta + \\varphi_k)` for :math:`k = 1..K`.

    Attributes
    ----------
    amplitudes : tuple of float
        Relative amplitudes :math:`A_k` of the fundamental (first entry) and
        the higher harmonics. Dimensionless; the overall EOD amplitude is set
        separately.
    phases_rad : tuple of float
        Phases :math:`\\varphi_k` in radians, same length as ``amplitudes``.
    """

    amplitudes: tuple[float, ...]
    phases_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        amplitudes = tuple(float(a) for a in self.amplitudes)
        phases = tuple(float(p) for p in self.phases_rad)
        if len(amplitudes) == 0 or len(amplitudes) != len(phases):
            msg = (
                "amplitudes and phases_rad must be non-empty and equally long, "
                f"got {len(amplitudes)} and {len(phases)}"
            )
            raise ValueError(msg)
        if not all(math.isfinite(a) and a >= 0 for a in amplitudes):
            msg = "harmonic amplitudes must be finite and >= 0"
            raise ValueError(msg)
        if not all(math.isfinite(p) for p in phases):
            msg = "harmonic phases must be finite"
            raise ValueError(msg)
        object.__setattr__(self, "amplitudes", amplitudes)
        object.__setattr__(self, "phases_rad", phases)

    @property
    def n_harmonics(self) -> int:
        """Number of components including the fundamental."""
        return len(self.amplitudes)


# Relative amplitudes and phases of the fundamental and four harmonics of an
# Apteronotus leptorhynchus EOD, copied from thunderfish
# (https://github.com/janscience/thunderfish, src/thunderfish/fakefish.py,
# ``Apteronotus_leptorhynchus_harmonics``, commit 997132c, 2026-04-22).
# thunderfish synthesises sum_k A_k cos(k * 2 pi f t + phi_k), the same
# convention as here. The fundamental has amplitude 0.90, so an EOD with
# amplitude 1 has a fundamental of amplitude 0.90 and a peak of about 1.
APTERONOTUS_LEPTORHYNCHUS = Harmonics(
    amplitudes=(0.90062, 0.15311, 0.072049, 0.012609, 0.011708),
    phases_rad=(1.3623, 2.3246, 0.9869, 2.6492, -2.6885),
)


@dataclass(frozen=True)
class HarmonicJitterParams:
    """Per-fish variation of the EOD waveform.

    The fundamental is left unchanged (its amplitude is the overall scale and
    its phase only shifts the time origin). Higher harmonics get independent
    log-normal amplitude factors and Gaussian phase offsets.

    The defaults are small, plausible values chosen to keep the waveform
    recognisably *A. leptorhynchus*; they are not fitted to data. They could be
    calibrated from harmonic analyses of isolated EODs in the grid recordings.

    Attributes
    ----------
    amplitude_log_sd : float
        Standard deviation of the natural log of the amplitude factor
        (default 0.2, i.e. roughly +/-20 %).
    phase_sd_rad : float
        Standard deviation of the phase offset in radians (default 0.2,
        about 11 degrees).
    """

    amplitude_log_sd: float = 0.2
    phase_sd_rad: float = 0.2

    def __post_init__(self) -> None:
        require_nonnegative(self.amplitude_log_sd, "amplitude_log_sd")
        require_nonnegative(self.phase_sd_rad, "phase_sd_rad")


def jitter_harmonics(
    rng: np.random.Generator,
    harmonics: Harmonics = APTERONOTUS_LEPTORHYNCHUS,
    params: HarmonicJitterParams | None = None,
) -> Harmonics:
    """Draw an individual waveform by perturbing the higher harmonics.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of all random draws.
    harmonics : Harmonics
        Species template (default *A. leptorhynchus*).
    params : HarmonicJitterParams, optional
        Size of the perturbation; defaults to ``HarmonicJitterParams()``.

    Returns
    -------
    Harmonics
        Same number of components; fundamental unchanged.
    """
    params = HarmonicJitterParams() if params is None else params
    n_higher = harmonics.n_harmonics - 1
    log_factors = rng.normal(0.0, params.amplitude_log_sd, n_higher)
    phase_offsets = rng.normal(0.0, params.phase_sd_rad, n_higher)
    amplitudes = np.asarray(harmonics.amplitudes)
    phases = np.asarray(harmonics.phases_rad)
    amplitudes[1:] *= np.exp(log_factors)
    phases[1:] += phase_offsets
    phases = np.angle(np.exp(1j * phases))
    return Harmonics(
        amplitudes=tuple(amplitudes.tolist()),
        phases_rad=tuple(phases.tolist()),
    )


def eod_phase(
    freq_hz: npt.ArrayLike,
    samplerate_hz: float,
    phase0_rad: float = 0.0,
) -> FloatArray:
    """Integrate an instantaneous frequency into an unwrapped phase.

    The phase is accumulated in float64 with the trapezoidal rule,
    ``phase[0] = phase0_rad`` and
    ``phase[i] = phase0_rad + 2 pi / fs * sum_{j<i} (f[j] + f[j+1]) / 2``.
    For a constant frequency this equals ``phase0_rad + 2 pi f i / fs``
    exactly. To keep rounding errors small over long recordings, the sum is
    split into an exact ramp at the mean frequency and a cumulative sum of the
    (small) residual.

    Parameters
    ----------
    freq_hz : array_like, shape (n,)
        Instantaneous EOD frequency per sample in Hz.
    samplerate_hz : float
        Sampling rate in Hz.
    phase0_rad : float
        Phase of the fundamental at the first sample, in radians.

    Returns
    -------
    numpy.ndarray of float64, shape (n,)
        Unwrapped phase of the fundamental in radians.
    """
    freq = as_1d_float64(freq_hz, "freq_hz")
    require_positive(samplerate_hz, "samplerate_hz")
    if not math.isfinite(phase0_rad):
        msg = "phase0_rad must be finite"
        raise ValueError(msg)
    n = freq.size
    f_ref = float(np.mean(freq))
    cycles = np.empty(n, dtype=np.float64)
    cycles[0] = 0.0
    if n > 1:
        residual = freq - f_ref
        np.add(residual[:-1], residual[1:], out=cycles[1:])
        del residual
        cycles[1:] *= 0.5
        np.cumsum(cycles, out=cycles)
        ramp = np.arange(n, dtype=np.float64)
        ramp *= f_ref
        cycles += ramp
        del ramp
    cycles *= 2.0 * np.pi / samplerate_hz
    cycles += phase0_rad
    return cycles


def harmonic_waveform(
    phase_rad: npt.ArrayLike,
    harmonics: Harmonics,
    amplitude: float | npt.ArrayLike = 1.0,
) -> Float32Array:
    """Evaluate the harmonic series at given phases.

    Parameters
    ----------
    phase_rad : array_like, shape (n,)
        Phase of the fundamental in radians (e.g. from :func:`eod_phase`).
    harmonics : Harmonics
        Relative amplitudes and phases of the components.
    amplitude : float or array_like of shape (n,)
        Amplitude envelope (arbitrary units), multiplied into the waveform.

    Returns
    -------
    numpy.ndarray of float32, shape (n,)
        ``amplitude * sum_k A_k cos(k * phase + phi_k)``.
    """
    phase = as_1d_float64(phase_rad, "phase_rad")
    n = phase.size
    envelope = np.asarray(amplitude, dtype=np.float64)
    if envelope.ndim == 0:
        envelope = np.full(1, float(envelope))
        scalar_envelope = True
    elif envelope.shape == (n,):
        scalar_envelope = False
    else:
        msg = f"amplitude must be a scalar or have shape ({n},), got {envelope.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(envelope)):
        msg = "amplitude contains non-finite values"
        raise ValueError(msg)

    out = np.empty(n, dtype=np.float32)
    components = list(
        zip(harmonics.amplitudes, harmonics.phases_rad, strict=True),
    )
    for start in range(0, n, _CHUNK_SAMPLES):
        stop = min(n, start + _CHUNK_SAMPLES)
        theta = phase[start:stop]
        acc = np.zeros(stop - start, dtype=np.float64)
        for k, (a_k, phi_k) in enumerate(components, start=1):
            if a_k == 0.0:
                continue
            acc += a_k * np.cos(k * theta + phi_k)
        acc *= envelope[0] if scalar_envelope else envelope[start:stop]
        out[start:stop] = acc
    return out


def synthesize_eod(
    freq_hz: npt.ArrayLike,
    amplitude: float | npt.ArrayLike,
    samplerate_hz: float,
    harmonics: Harmonics = APTERONOTUS_LEPTORHYNCHUS,
    phase0_rad: float = 0.0,
) -> Float32Array:
    """Synthesize a phase-continuous wave-type EOD.

    Parameters
    ----------
    freq_hz : array_like, shape (n,)
        Instantaneous EOD frequency per sample in Hz. Chirps and rises are
        added here; they pass through without discontinuities because only
        the integrated phase enters the waveform.
    amplitude : float or array_like of shape (n,)
        Amplitude envelope (arbitrary units; chirp amplitude modulation is
        multiplied in here).
    samplerate_hz : float
        Sampling rate in Hz.
    harmonics : Harmonics
        Waveform shape (default *A. leptorhynchus*).
    phase0_rad : float
        Phase of the fundamental at the first sample in radians.

    Returns
    -------
    numpy.ndarray of float32, shape (n,)
        The EOD waveform.

    See Also
    --------
    eod_phase, harmonic_waveform : the two steps, for rendering sub-spans of a
        longer signal from one globally integrated phase.
    """
    phase = eod_phase(freq_hz, samplerate_hz, phase0_rad)
    return harmonic_waveform(phase, harmonics, amplitude)


@dataclass(frozen=True)
class DriftParams:
    """Individual slow drift of a fish's baseline EOD frequency.

    Defaults are rough values, not fitted: in the 20-s test subset of the
    2020-03-16 recording the tracked EODf of two fish has a standard
    deviation of about 0.25 Hz (including tracking jitter). Calibrate against
    the slow component of the tracks of the 33 trials (e.g. after removing
    chirps and rises with a running median) before relying on them.

    Attributes
    ----------
    sd_hz : float
        Stationary standard deviation of the drift in Hz (default 0.5).
    timescale_s : float
        Correlation time-scale :math:`\\tau` in seconds (default 30); the
        autocorrelation is :math:`\\exp(-\\Delta t^2 / (4\\tau^2))`.
    control_rate_hz : float
        Rate of the coarse grid on which the drift is generated before cubic
        interpolation (default 1). ``timescale_s * control_rate_hz`` must be
        >= 2 so the drift is resolved by the grid.
    """

    sd_hz: float = 0.5
    timescale_s: float = 30.0
    control_rate_hz: float = 1.0

    def __post_init__(self) -> None:
        require_nonnegative(self.sd_hz, "sd_hz")
        _check_drift_grid(self.timescale_s, self.control_rate_hz)


@dataclass(frozen=True)
class SharedDriftParams:
    """Relative EODf drift shared by all fish in a tank (water temperature).

    EODf of *A. leptorhynchus* rises with temperature, so a common temperature
    change shifts every fish by about the same *fraction* of its EODf. The
    water temperature logged during the 2019-2020 competition recordings
    changed by 1-2 degC over 6-h trials (about 0.1 degC per 10 min). The
    default standard deviation of 0.1 % (0.8 Hz at 800 Hz) is a rough guess
    in that range, not a fitted value; calibrate it from the common-mode
    component of the tracked EODf of fish recorded together.

    Attributes
    ----------
    sd_rel : float
        Stationary standard deviation of the relative frequency change
        (dimensionless; default 1e-3).
    timescale_s : float
        Correlation time-scale in seconds (default 300).
    control_rate_hz : float
        Rate of the coarse generation grid (default 0.1);
        ``timescale_s * control_rate_hz`` must be >= 2.
    """

    sd_rel: float = 1e-3
    timescale_s: float = 300.0
    control_rate_hz: float = 0.1

    def __post_init__(self) -> None:
        require_nonnegative(self.sd_rel, "sd_rel")
        if self.sd_rel >= 0.5:
            msg = f"sd_rel must be < 0.5, got {self.sd_rel!r}"
            raise ValueError(msg)
        _check_drift_grid(self.timescale_s, self.control_rate_hz)


def baseline_frequency(
    rng: np.random.Generator,
    n_samples: int,
    samplerate_hz: float,
    baseline_hz: float,
    params: DriftParams | None = None,
    shared_drift_rel: npt.ArrayLike | None = None,
) -> FloatArray:
    """Baseline EOD frequency with slow individual and shared drift.

    ``f(t) = (baseline_hz + d_i(t)) * (1 + d_shared(t))``, where ``d_i`` is
    the individual drift in Hz and ``d_shared`` the relative drift common to
    all fish (from :func:`shared_frequency_drift`).

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of the individual drift.
    n_samples : int
        Number of samples.
    samplerate_hz : float
        Sampling rate in Hz.
    baseline_hz : float
        Mean baseline EOD frequency in Hz.
    params : DriftParams, optional
        Individual drift parameters; defaults to ``DriftParams()``.
    shared_drift_rel : array_like of shape (n_samples,), optional
        Relative drift shared across fish (dimensionless). Omit for none.

    Returns
    -------
    numpy.ndarray of float64, shape (n_samples,)
        Instantaneous baseline frequency in Hz.
    """
    params = DriftParams() if params is None else params
    require_positive(baseline_hz, "baseline_hz")
    freq = _smooth_drift(
        rng,
        n_samples,
        samplerate_hz,
        sd=params.sd_hz,
        timescale_s=params.timescale_s,
        control_rate_hz=params.control_rate_hz,
    )
    freq += baseline_hz
    if shared_drift_rel is not None:
        shared = as_1d_float64(shared_drift_rel, "shared_drift_rel")
        if shared.size != n_samples:
            msg = f"shared_drift_rel must have shape ({n_samples},), got {shared.shape}"
            raise ValueError(msg)
        freq *= 1.0 + shared
    return freq


def shared_frequency_drift(
    rng: np.random.Generator,
    n_samples: int,
    samplerate_hz: float,
    params: SharedDriftParams | None = None,
) -> FloatArray:
    """Relative EODf drift common to all fish in a recording.

    Generate it once per scene and pass it to :func:`baseline_frequency` of
    every fish.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of all random draws.
    n_samples : int
        Number of samples.
    samplerate_hz : float
        Sampling rate in Hz.
    params : SharedDriftParams, optional
        Defaults to ``SharedDriftParams()``.

    Returns
    -------
    numpy.ndarray of float64, shape (n_samples,)
        Relative frequency change (dimensionless, zero-mean).
    """
    params = SharedDriftParams() if params is None else params
    return _smooth_drift(
        rng,
        n_samples,
        samplerate_hz,
        sd=params.sd_rel,
        timescale_s=params.timescale_s,
        control_rate_hz=params.control_rate_hz,
    )


def _check_drift_grid(timescale_s: float, control_rate_hz: float) -> None:
    require_positive(timescale_s, "timescale_s")
    require_positive(control_rate_hz, "control_rate_hz")
    if timescale_s * control_rate_hz < 2.0:
        msg = (
            "timescale_s * control_rate_hz must be >= 2 so the drift is "
            f"resolved by the control grid, got {timescale_s * control_rate_hz:g}"
        )
        raise ValueError(msg)


def _smooth_noise(
    rng: np.random.Generator,
    n_points: int,
    sd: float,
    timescale_points: float,
) -> FloatArray:
    """Stationary Gaussian noise smoothed by a Gaussian kernel.

    The kernel (SD ``timescale_points``) has unit L2 norm, so the output has
    ensemble standard deviation ``sd`` exactly.
    """
    half = math.ceil(4.0 * timescale_points)
    lags = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (lags / timescale_points) ** 2)
    kernel /= np.sqrt(np.sum(kernel**2))
    white = rng.standard_normal(n_points + 2 * half)
    smooth = fftconvolve(white, kernel, mode="valid")
    return sd * smooth


def _smooth_drift(
    rng: np.random.Generator,
    n_samples: int,
    samplerate_hz: float,
    *,
    sd: float,
    timescale_s: float,
    control_rate_hz: float,
) -> FloatArray:
    """Generate smooth drift on a coarse grid and spline it to the signal rate."""
    require_n_samples(n_samples)
    require_positive(samplerate_hz, "samplerate_hz")
    pad = 2
    duration_s = (n_samples - 1) / samplerate_hz
    n_control = math.ceil(duration_s * control_rate_hz) + 1 + 2 * pad
    control_times_s = (np.arange(n_control, dtype=np.float64) - pad) / control_rate_hz
    values = _smooth_noise(rng, n_control, sd, timescale_s * control_rate_hz)
    spline = CubicSpline(control_times_s, values)
    out = np.empty(n_samples, dtype=np.float64)
    for start in range(0, n_samples, _CHUNK_SAMPLES):
        stop = min(n_samples, start + _CHUNK_SAMPLES)
        times_s = np.arange(start, stop, dtype=np.float64) / samplerate_hz
        out[start:stop] = spline(times_s)
    return out
