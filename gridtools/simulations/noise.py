"""Noise generation functions."""

import numpy as np


def fftnoise(f: np.ndarray) -> np.ndarray:
    """Generate noise with a given power spectrum.

    Parameters
    ----------
    - `f` : `numpy.ndarray`
        The power spectrum of the noise.

    Returns
    -------
    - `numpy.ndarray`
        The noise with the given power spectrum.
    """
    f = np.array(f, dtype="complex")
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1 : Np + 1] *= phases
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
    return np.fft.ifft(f).real


def band_limited_noise(
    min_freq: float,
    max_freq: float,
    samples: int = 1024,
    samplerate: int = 1,
    std: float = 1,
) -> np.ndarray:
    """Generate band limited noise.

    Parameters
    ----------
    min_freq : float
        The minimum frequency of the band.
    max_freq : float
        The maximum frequency of the band.
    samples : int, Optional
        The number of samples to generate. Default is 1024.
    samplerate : int, Optional
        The samplerate of the signal. Default is 1.
    std : float, Optional
        The standard deviation of the noise. Default is 1.

    Returns
    -------
    numpy.ndarray
        An array of band limited noise.
    """
    # Check for nyquist frequency
    if max_freq >= samplerate / 2:
        msg = "max_freq must be less than samplerate / 2"
        raise ValueError(msg)

    # Check for min_freq > max_freq
    if min_freq >= max_freq:
        msg = "min_freq must be less than max_freq"
        raise ValueError(msg)

    # Check for min_freq < 0
    if (min_freq < 0) or (max_freq < 0):
        msg = "min_freq and max_freq must be greater than 0"
        raise ValueError(msg)

    # Check for samples < 0
    if samples < 0:
        msg = "samples must be greater than 0"
        raise ValueError(msg)

    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    f[idx] = 1
    noise = fftnoise(f)

    # shift to 0 and make std 1
    noise = (noise - np.mean(noise)) / np.std(noise)
    noise *= std

    return noise
