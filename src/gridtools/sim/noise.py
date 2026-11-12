"""Recording-chain imperfections: noise, power-line hum, faults, clipping.

The recording of electrode ``e`` is modelled as

.. math::

    x_e(t) = \\mathrm{clip}\\Big(c_e\\,\\big[\\textstyle\\sum_i g_{ie}(t)\\,
    \\mathrm{EOD}_i(t) + h_e(t) + s_e\\,\\eta_e(t)\\big]\\Big),

with per-electrode channel gain :math:`c_e` (gain mismatch, dead electrodes),
power-line hum :math:`h_e` (one shared source, per-electrode coupling) and
independent coloured noise :math:`\\eta_e` scaled by :math:`s_e` (noisy
electrodes). This module provides each ingredient; :mod:`gridtools.sim.mix`
combines them.

Levels are in recording units, the same arbitrary units as EOD amplitude
times field gain (see :mod:`gridtools.sim.field`). The defaults are
placeholders: noise and hum levels have to be set per scene relative to fish
amplitudes, ideally calibrated against the noise floor and hum peaks of real
grid recordings.

Not modelled: the amplifier's band-pass (the real grids have a 100-Hz
high-pass and a 10-kHz low-pass), which attenuates hum and low-frequency
noise; quantisation; time-varying faults.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.fft

from gridtools.sim._common import (
    BoolArray,
    Float32Array,
    require_n_samples,
    require_nonnegative,
    require_positive,
    require_probability,
)

__all__ = [
    "ClipParams",
    "ColoredNoiseParams",
    "ElectrodeFaultParams",
    "ElectrodeFaults",
    "GainMismatchParams",
    "Hum",
    "HumParams",
    "clip",
    "colored_noise",
    "electrode_faults",
    "gain_mismatch",
    "powerline_hum",
]

_CHUNK_SAMPLES = 1 << 20


@dataclass(frozen=True)
class ColoredNoiseParams:
    """Independent power-law noise per electrode.

    The power spectral density is proportional to :math:`f^{-\\alpha}` above
    ``f_min_hz`` and flat below it (so it stays finite at DC); the mean is
    zero.

    Attributes
    ----------
    sd : float
        Ensemble standard deviation per electrode in recording units
        (default 1e-3; placeholder, set relative to fish amplitudes).
    exponent : float
        Spectral exponent :math:`\\alpha` (default 1, pink noise; 0 is white).
    f_min_hz : float
        Corner below which the spectrum is flat, in Hz (default 1).
    """

    sd: float = 1e-3
    exponent: float = 1.0
    f_min_hz: float = 1.0

    def __post_init__(self) -> None:
        require_nonnegative(self.sd, "sd")
        require_nonnegative(self.exponent, "exponent")
        require_positive(self.f_min_hz, "f_min_hz")


def colored_noise(
    rng: np.random.Generator,
    n_samples: int,
    n_electrodes: int,
    samplerate_hz: float,
    params: ColoredNoiseParams | None = None,
    *,
    sd_scale: npt.ArrayLike | None = None,
    workers: int = 1,
) -> Float32Array:
    """Independent coloured noise on each electrode, generated via FFT.

    Each electrode gets complex Gaussian Fourier coefficients with variance
    proportional to the target spectrum, transformed back with an inverse
    real FFT. The result is scaled by the analytically expected (ensemble)
    standard deviation of that construction, not by the sample's own.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of all random draws.
    n_samples : int
        Number of samples.
    n_electrodes : int
        Number of electrodes.
    samplerate_hz : float
        Sampling rate in Hz.
    params : ColoredNoiseParams, optional
        Defaults to ``ColoredNoiseParams()``.
    sd_scale : array_like of shape (n_electrodes,), optional
        Per-electrode factor on the standard deviation (e.g.
        :attr:`ElectrodeFaults.noise_scale`).
    workers : int
        Electrodes transformed in parallel by the FFT (default 1). Random
        numbers are always drawn electrode by electrode in the same order,
        so the result does not depend on it.

    Returns
    -------
    numpy.ndarray of float32, shape (n_samples, n_electrodes)
    """
    params = ColoredNoiseParams() if params is None else params
    require_n_samples(n_samples)
    require_n_samples(n_electrodes, "n_electrodes")
    require_positive(samplerate_hz, "samplerate_hz")
    scale = _per_electrode(sd_scale, n_electrodes, "sd_scale")

    fast_len = scipy.fft.next_fast_len(n_samples, real=True)
    n_fft = n_samples if fast_len is None else int(fast_len)
    freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / samplerate_hz)
    psd = np.maximum(freqs_hz, params.f_min_hz) ** (-params.exponent)
    psd[0] = 0.0
    weights = np.full(freqs_hz.size, 2.0)  # bins counted twice in the full FFT
    if n_fft % 2 == 0:
        weights[-1] = 1.0  # Nyquist bin is real and counted once
    expected_var = float(np.sum(weights * psd)) / n_fft**2
    amplitude = np.sqrt(psd / 2.0)
    if n_fft % 2 == 0:
        amplitude[-1] = math.sqrt(psd[-1])
    amplitude *= params.sd / math.sqrt(expected_var)
    amplitude32 = amplitude.astype(np.float32)

    if isinstance(workers, bool) or workers < 1:
        msg = f"workers must be >= 1, got {workers!r}"
        raise ValueError(msg)

    out = np.empty((n_samples, n_electrodes), dtype=np.float32)
    n_bins = freqs_hz.size
    for first in range(0, n_electrodes, workers):
        batch = range(first, min(n_electrodes, first + workers))
        spectra = np.empty((len(batch), n_bins), dtype=np.complex64)
        for row, e in enumerate(batch):
            spectra[row].real = rng.standard_normal(n_bins, dtype=np.float32)
            spectra[row].imag = rng.standard_normal(n_bins, dtype=np.float32)
            if n_fft % 2 == 0:
                spectra[row, -1] = spectra[row, -1].real
            spectra[row] *= amplitude32 * np.float32(scale[e])
        signals = np.asarray(
            scipy.fft.irfft(spectra, n=n_fft, axis=-1, workers=workers),
            dtype=np.float32,
        )
        out[:, first : first + len(batch)] = signals[:, :n_samples].T
    return out


@dataclass(frozen=True)
class HumParams:
    """Power-line hum: one shared source with per-electrode coupling.

    Attributes
    ----------
    line_freq_hz : float
        Nominal mains frequency in Hz (default 50).
    line_freq_sd_hz : float
        SD of a constant random offset of the mains frequency in Hz
        (default 0.02; grid frequency deviations are of this order).
    n_harmonics : int
        Number of components including the fundamental (default 5:
        50-250 Hz).
    amplitude_range : tuple of float
        Range of the fundamental's amplitude in recording units, drawn
        log-uniformly (default 1e-4 to 1e-2; placeholder).
    harmonic_decay : float
        Expected amplitude ratio between successive harmonics (default 0.5).
    harmonic_log_sd : float
        Log-normal jitter of each harmonic's amplitude (default 0.5).
    coupling_log_sd : float
        Log-normal spread of the per-electrode coupling around 1
        (default 0.5).
    """

    line_freq_hz: float = 50.0
    line_freq_sd_hz: float = 0.02
    n_harmonics: int = 5
    amplitude_range: tuple[float, float] = (1e-4, 1e-2)
    harmonic_decay: float = 0.5
    harmonic_log_sd: float = 0.5
    coupling_log_sd: float = 0.5

    def __post_init__(self) -> None:
        require_positive(self.line_freq_hz, "line_freq_hz")
        require_nonnegative(self.line_freq_sd_hz, "line_freq_sd_hz")
        require_n_samples(self.n_harmonics, "n_harmonics")
        low, high = self.amplitude_range
        require_nonnegative(low, "amplitude_range[0]")
        if not (math.isfinite(high) and high >= low):
            msg = f"amplitude_range must be (low, high) with high >= low, got {self.amplitude_range}"
            raise ValueError(msg)
        require_nonnegative(self.harmonic_decay, "harmonic_decay")
        require_nonnegative(self.harmonic_log_sd, "harmonic_log_sd")
        require_nonnegative(self.coupling_log_sd, "coupling_log_sd")


@dataclass(frozen=True, eq=False)
class Hum:
    """Rendered hum: a shared source waveform and per-electrode coupling.

    Stored as a rank-1 factorisation (``source[:, None] * coupling``) so a
    full (n, E) array never needs to exist; :meth:`render` returns any span.

    Attributes
    ----------
    source : numpy.ndarray of float32, shape (n,)
        Hum waveform at unit coupling.
    coupling : numpy.ndarray of float32, shape (E,)
        Per-electrode coupling factors.
    frequencies_hz : tuple of float
        Frequencies of the components.
    amplitudes : tuple of float
        Amplitudes of the components at unit coupling.
    """

    source: Float32Array
    coupling: Float32Array
    frequencies_hz: tuple[float, ...]
    amplitudes: tuple[float, ...]

    @property
    def shape(self) -> tuple[int, int]:
        """Shape (n_samples, n_electrodes) of the rendered hum."""
        return (self.source.shape[0], self.coupling.shape[0])

    def render(self, start: int, stop: int) -> Float32Array:
        """Hum on all electrodes for samples ``start:stop``, shape (k, E)."""
        return self.source[start:stop, None] * self.coupling[None, :]


def powerline_hum(
    rng: np.random.Generator,
    n_samples: int,
    n_electrodes: int,
    samplerate_hz: float,
    params: HumParams | None = None,
) -> Hum:
    """Draw power-line hum with harmonics, shared across electrodes.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of all random draws.
    n_samples : int
        Number of samples.
    n_electrodes : int
        Number of electrodes.
    samplerate_hz : float
        Sampling rate in Hz.
    params : HumParams, optional
        Defaults to ``HumParams()``.

    Returns
    -------
    Hum
        Components above the Nyquist frequency are dropped.
    """
    params = HumParams() if params is None else params
    require_n_samples(n_samples)
    require_n_samples(n_electrodes, "n_electrodes")
    require_positive(samplerate_hz, "samplerate_hz")
    harmonics = np.arange(1, params.n_harmonics + 1)
    line_hz = params.line_freq_hz + rng.normal(0.0, params.line_freq_sd_hz)
    low, high = params.amplitude_range
    if low > 0:
        fundamental = math.exp(rng.uniform(math.log(low), math.log(high)))
    else:
        fundamental = rng.uniform(low, high)
    jitter = np.exp(rng.normal(0.0, params.harmonic_log_sd, params.n_harmonics))
    jitter[0] = 1.0
    amplitudes = fundamental * params.harmonic_decay ** (harmonics - 1) * jitter
    phases = rng.uniform(0.0, 2.0 * np.pi, params.n_harmonics)
    coupling = np.exp(rng.normal(0.0, params.coupling_log_sd, n_electrodes))

    frequencies = harmonics * line_hz
    keep = frequencies < samplerate_hz / 2.0
    frequencies, amplitudes, phases = frequencies[keep], amplitudes[keep], phases[keep]

    source = np.empty(n_samples, dtype=np.float32)
    for start in range(0, n_samples, _CHUNK_SAMPLES):
        stop = min(n_samples, start + _CHUNK_SAMPLES)
        times_s = np.arange(start, stop, dtype=np.float64) / samplerate_hz
        acc = np.zeros(stop - start, dtype=np.float64)
        for f_hz, a, phi in zip(frequencies, amplitudes, phases, strict=True):
            acc += a * np.cos(2.0 * np.pi * f_hz * times_s + phi)
        source[start:stop] = acc
    return Hum(
        source=source,
        coupling=coupling.astype(np.float32),
        frequencies_hz=tuple(frequencies.tolist()),
        amplitudes=tuple(amplitudes.tolist()),
    )


@dataclass(frozen=True)
class GainMismatchParams:
    """Per-electrode gain mismatch (electrode impedance, amplifier channels).

    Attributes
    ----------
    log_sd : float
        SD of the natural log of the channel gain (default 0.1, about
        +/-10 %; placeholder, could be estimated from real recordings by
        comparing electrode pairs equidistant from a resting fish).
    """

    log_sd: float = 0.1

    def __post_init__(self) -> None:
        require_nonnegative(self.log_sd, "log_sd")


def gain_mismatch(
    rng: np.random.Generator,
    n_electrodes: int,
    params: GainMismatchParams | None = None,
) -> Float32Array:
    """Draw log-normal channel gains around 1.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of all random draws.
    n_electrodes : int
        Number of electrodes.
    params : GainMismatchParams, optional
        Defaults to ``GainMismatchParams()``.

    Returns
    -------
    numpy.ndarray of float32, shape (n_electrodes,)
    """
    params = GainMismatchParams() if params is None else params
    require_n_samples(n_electrodes, "n_electrodes")
    return np.exp(rng.normal(0.0, params.log_sd, n_electrodes)).astype(np.float32)


@dataclass(frozen=True)
class ElectrodeFaultParams:
    """Dead and noisy electrodes.

    Attributes
    ----------
    dead_prob : float
        Probability that an electrode is dead (default 0.05).
    noisy_prob : float
        Probability that a live electrode is noisy (default 0.05).
    dead_gain : float
        Channel gain of a dead electrode (default 1e-3: near-zero output).
    noisy_factor : float
        Factor on the coloured-noise SD of a noisy electrode (default 5).
    """

    dead_prob: float = 0.05
    noisy_prob: float = 0.05
    dead_gain: float = 1e-3
    noisy_factor: float = 5.0

    def __post_init__(self) -> None:
        require_probability(self.dead_prob, "dead_prob")
        require_probability(self.noisy_prob, "noisy_prob")
        require_nonnegative(self.dead_gain, "dead_gain")
        require_nonnegative(self.noisy_factor, "noisy_factor")


@dataclass(frozen=True, eq=False)
class ElectrodeFaults:
    """Fault flags and the factors they imply.

    Attributes
    ----------
    dead : numpy.ndarray of bool, shape (E,)
    noisy : numpy.ndarray of bool, shape (E,)
        Never set on dead electrodes.
    signal_gain : numpy.ndarray of float32, shape (E,)
        Channel-gain factor (``dead_gain`` on dead electrodes, else 1);
        multiply into the channel gain.
    noise_scale : numpy.ndarray of float32, shape (E,)
        Noise SD factor (``noisy_factor`` on noisy electrodes, else 1); pass
        as ``sd_scale`` to :func:`colored_noise`.
    """

    dead: BoolArray
    noisy: BoolArray
    signal_gain: Float32Array
    noise_scale: Float32Array


def electrode_faults(
    rng: np.random.Generator,
    n_electrodes: int,
    params: ElectrodeFaultParams | None = None,
) -> ElectrodeFaults:
    """Draw which electrodes are dead or noisy.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of all random draws.
    n_electrodes : int
        Number of electrodes.
    params : ElectrodeFaultParams, optional
        Defaults to ``ElectrodeFaultParams()``.

    Returns
    -------
    ElectrodeFaults
    """
    params = ElectrodeFaultParams() if params is None else params
    require_n_samples(n_electrodes, "n_electrodes")
    draws = rng.random((2, n_electrodes))
    dead = draws[0] < params.dead_prob
    noisy = (draws[1] < params.noisy_prob) & ~dead
    return ElectrodeFaults(
        dead=dead,
        noisy=noisy,
        signal_gain=np.where(dead, params.dead_gain, 1.0).astype(np.float32),
        noise_scale=np.where(noisy, params.noisy_factor, 1.0).astype(np.float32),
    )


@dataclass(frozen=True)
class ClipParams:
    """Symmetric clipping at the input range of the acquisition system.

    Attributes
    ----------
    limit : float
        Clipping level in recording units (default 20; the real grids have an
        input range of +/-20 mV, so this matches if recording units are mV).
    """

    limit: float = 20.0

    def __post_init__(self) -> None:
        require_positive(self.limit, "limit")


def clip(x: npt.ArrayLike, params: ClipParams | None = None) -> Float32Array:
    """Clip a recording to ``[-limit, limit]``; returns a new float32 array.

    Parameters
    ----------
    x : array_like
        Recording (any shape).
    params : ClipParams, optional
        Defaults to ``ClipParams()``.

    Returns
    -------
    numpy.ndarray of float32
    """
    params = ClipParams() if params is None else params
    data = np.asarray(x, dtype=np.float32)
    return np.clip(data, -params.limit, params.limit).astype(np.float32, copy=False)


def _per_electrode(
    values: npt.ArrayLike | None, n_electrodes: int, name: str
) -> Float32Array:
    if values is None:
        return np.ones(n_electrodes, dtype=np.float32)
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (n_electrodes,) or not np.all(np.isfinite(array)):
        msg = f"{name} must be finite with shape ({n_electrodes},), got {array.shape}"
        raise ValueError(msg)
    return array
