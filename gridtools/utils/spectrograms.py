#! /usr/bin/env python3

"""
Helper functions for spectrograms.
"""

import math

import numpy as np
import torch
from torchaudio.transforms import AmplitudeToDB, Spectrogram


def check_device():
    """
    Check if a CUDA-enabled GPU is available, and return the appropriate device.

    Returns
    -------
    device : torch.device
        The device to use for tensor operations. If a CUDA-enabled GPU is available, returns a device object for that
        GPU. If an Apple M1 GPU is available, returns a device object for that GPU. Otherwise, returns a device object
        for the CPU.
    """
    if torch.cuda.is_available() is True:
        device = torch.device("cuda")  # nvidia / amd gpu
    elif torch.backends.mps.is_available() is True:
        device = torch.device("mps")  # apple m1 gpu
    else:
        device = torch.device("cpu")  # no gpu
    return device


def next_power_of_two(num):
    """Computes the next power of two for a given number.

    Parameters
    ----------
    num : float
        The input number.

    Returns
    -------
    int
        The next power of two.
    """
    if math.log2(num).is_integer():
        return num
    next_pow = math.ceil(math.log2(num))
    return 2**next_pow


def freqres_to_nfft(freq_res, samplingrate):
    """Convert the frequency resolution of a spectrogram to
    the number of FFT bins.

    Parameters
    ----------
    freq_res : float
        Frequency resolution of the spectrogram.
    samplingrate : int
        The sampling rate of the signal.

    Returns
    -------
    int
        The number of FFT bins.
    """
    return int(next_power_of_two(samplingrate / freq_res))


def nfft_to_freqres(nfft, samplingrate):
    """Convert the number of FFT bins of a spectrogram to
    the frequency resolution.

    Parameters
    ----------
    nfft : int
        Number of FFT bins.
    samplingrate : int
        The sampling rate of the signal.

    Returns
    -------
    float
        The frequency resolution of the spectrogram.
    """
    return samplingrate / nfft


def overlap_to_hoplen(overlap, nfft):
    """Convert the overlap of a spectrogram to the hop length.

    Parameters
    ----------
    overlap : float
        Overlap of the spectrogram. Must be between 0 and 1.
    nfft : int
        Number of FFT bins.

    Returns
    -------
    int
        The hop length on the spectrogram.
    """
    return int(np.floor(nfft * (1 - overlap)))


def sint(num):
    """Convert a float to an int without rounding.

    Parameters
    ----------
    num : float
        The input number.

    Returns
    -------
    int
        The input number as an integer.

    Raises
    ------
    ValueError
        Fails if the input number is not an integer.
    """
    if isinstance(num, int):
        return num
    elif num.is_integer():
        return int(num)
    else:
        raise ValueError("Number is not an integer.")


def specshow(spec, time, freq, ax, **kwargs):
    """Plot a spectrogram.

    Parameters
    ----------
    spec : np.ndarray
        The spectrogram matrix.
    time : np.ndarray
        The time axis of the spectrogram.
    freq : np.ndarray
        The frequency axis of the spectrogram.
    ax : matplotlib.axes.Axes
        The axes to plot the spectrogram on.

    Returns
    -------
    matplotlib.image.AxesImage
        The image object of the spectrogram.
    """
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()

    im = ax.imshow(
        spec, extent=[time[0], time[-1], freq[0], freq[-1]], **kwargs
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    return im


def compute_spectrogram(data, samplingrate, nfft, hop_length, trycuda=True):
    """Compute the spectrogram of a signal.

    Parameters
    ----------
    data : np.ndarray
        The 1D signal.
    samplingrate : float
        The sampling rate of the signal.
    frequency_resolution : float
        The frequency resolution of the spectrogram.
    overlap : float
        The overlap of the spectrogram. Must be between 0 and 1.

    Returns
    -------
    torch.Tensor
        The spectrogram matrix.
    """
    if trycuda:
        device = check_device()
    else:
        device = torch.device("cpu")

    data = torch.from_numpy(data).to(device)
    spectrogram_of = Spectrogram(
        n_fft=nfft,
        hop_length=hop_length,
        power=2,
        normalized=True,
        window_fn=torch.hann_window,
    ).to(device)
    spec = spectrogram_of(data)
    time = np.arange(0, spec.shape[1]) * hop_length / samplingrate
    freq = np.arange(0, spec.shape[0]) * samplingrate / nfft
    return spec, time, freq


def to_decibel(spec, trycuda=True):
    """Convert a spectrogram to decibel scale.

    Parameters
    ----------
    spec : np.ndarray
        The spectrogram matrix.

    Returns
    -------
    torch.Tensor
        The spectrogram matrix in decibel scale.
    """
    if trycuda:
        device = check_device()
    else:
        device = torch.device("cpu")

    decibel_of = AmplitudeToDB(stype="power", top_db=60).to(device)
    return decibel_of(spec)
