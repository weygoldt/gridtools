"""Helper functions for spectrograms."""

import math
from typing import Tuple, Union

import numpy as np
import torch
from torchaudio.transforms import AmplitudeToDB, Spectrogram


def check_device() -> torch.device:
    """Check if a CUDA-enabled GPU is available, and return the device.

    Returns
    -------
    device : torch.device
        The device to use for tensor operations. If a CUDA-enabled GPU is
        available, returns a device object for that GPU. If an Apple M1
        GPU is available, returns a device object for that GPU. Otherwise,
        returns a device object for the CPU.
    """
    if torch.cuda.is_available() is True:
        device = torch.device("cuda")  # nvidia / amd gpu
    elif torch.backends.mps.is_available() is True:
        device = torch.device("mps")  # apple m1 gpu
    else:
        device = torch.device("cpu")  # no gpu
    return device


def next_power_of_two(num: float) -> int:
    """Compute the next power of two for a given number.

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
        return int(num)
    next_pow = math.ceil(math.log2(num))
    return int(2**next_pow)


def freqres_to_nfft(freq_res: float, samplingrate: float) -> int:
    """Convert the frequency resolution to the number of FFT bins.

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
    return next_power_of_two(samplingrate / freq_res)


def nfft_to_freqres(nfft: int, samplingrate: float) -> float:
    """Convert the number of FFT bins to the frequency resolution.

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


def overlap_to_hoplen(overlap: float, nfft: int) -> int:
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
    overlap = int(np.floor(nfft * (1 - overlap)))
    if overlap % 2 == 0:
        return overlap
    return overlap + 1


def sint(num: float) -> int:
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
    if num.is_integer():
        return int(num)
    msg = "Number is not an integer."
    raise ValueError(msg)


def specshow(
    spec: np.ndarray,
    time: np.ndarray,
    freq: np.ndarray,
    ax: "matplotlib.axes.Axes",
    **kwargs: dict,
) -> np.ndarray:
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


def compute_spectrogram(
    data: Union[np.ndarray, torch.Tensor],
    samplingrate: float,
    nfft: int,
    hop_length: int,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
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
    device = check_device()
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).to(device)
    elif isinstance(data, torch.Tensor):
        pass
    else:
        msg = "data must be a numpy array or a torch tensor."
        raise TypeError(msg)

    data = data.to(device)

    pad = nfft
    spectrogram_of = Spectrogram(
        n_fft=nfft,
        hop_length=hop_length,
        power=2,
        # normalized=True,
        normalized=False,
        window_fn=torch.hann_window,
        pad=pad,  # <--- this is the important part
    ).to(device)
    spec = spectrogram_of(data)
    pad_samples = int(np.ceil(pad / hop_length))

    # check how many dimensions the spectrogram has
    if len(spec.shape) == 3:
        spec = spec[:, :, pad_samples:-pad_samples]
    elif len(spec.shape) == 2:
        spec = spec[:, pad_samples:-pad_samples]
    else:
        msg = (
            "Spectrogram has strange dimensions. Please check the input data."
        )
        raise ValueError(msg)

    time = np.arange(0, spec.shape[1]) * hop_length / samplingrate
    freq = np.arange(0, spec.shape[0]) * samplingrate / nfft

    print(
        f"Spectrogram params: nfft={nfft}, hop_length={hop_length}, power=2, window_fn=torch.hann_window, normalized=True"
    )
    return spec, time, freq


def to_decibel(spec: torch.Tensor) -> torch.Tensor:
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
    device = check_device()
    if isinstance(spec, np.ndarray):
        spec = torch.from_numpy(spec).to(device)
    decibel_of = AmplitudeToDB(stype="power", top_db=60).to(device)
    return decibel_of(spec)
