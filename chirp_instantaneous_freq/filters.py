from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d
import numpy as np


def instantaneous_frequency(
    signal: np.ndarray,
    samplerate: int,
    smoothing_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the instantaneous frequency of a signal that is approximately
    sinusoidal and symmetric around 0.

    Parameters
    ----------
    signal : np.ndarray
        Signal to compute the instantaneous frequency from.
    samplerate : int
        Samplerate of the signal.
    smoothing_window : int
        Window size for the gaussian filter.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]

    """
    # calculate instantaneous frequency with zero crossings
    roll_signal = np.roll(signal, shift=1)
    time_signal = np.arange(len(signal)) / samplerate
    period_index = np.arange(len(signal))[(roll_signal < 0) & (signal >= 0)][
        1:-1
    ]

    upper_bound = np.abs(signal[period_index])
    lower_bound = np.abs(signal[period_index - 1])
    upper_time = np.abs(time_signal[period_index])
    lower_time = np.abs(time_signal[period_index - 1])

    # create ratio
    lower_ratio = lower_bound / (lower_bound + upper_bound)

    # appy to time delta
    time_delta = upper_time - lower_time
    true_zero = lower_time + lower_ratio * time_delta

    # create new time array
    instantaneous_frequency_time = true_zero[:-1] + 0.5 * np.diff(true_zero)

    # compute frequency
    instantaneous_frequency = gaussian_filter1d(
        1 / np.diff(true_zero), smoothing_window
    )

    return instantaneous_frequency_time, instantaneous_frequency


def inst_freq(signal, fs):
    """
    Computes the instantaneous frequency of a periodic signal using zero-crossings.
    
    Parameters:
    -----------
    signal : array-like
        The input signal.
    fs : float
        The sampling frequency of the input signal.
    
    Returns:
    --------
    freq : array-like
        The instantaneous frequency of the input signal.
    """
    # Compute the sign of the signal
    sign = np.sign(signal)
    
    # Compute the crossings of the sign signal with a zero line
    crossings = np.where(np.diff(sign))[0]
    
    # Compute the time differences between zero crossings
    dt = np.diff(crossings) / fs
    
    # Compute the instantaneous frequency as the reciprocal of the time differences
    freq = 1 / dt

    # Gaussian filter the signal 
    freq = gaussian_filter1d(freq, 10)
    
    # Pad the frequency vector with zeros to match the length of the input signal
    freq = np.pad(freq, (0, len(signal) - len(freq)))
    
    return freq

def bandpass_filter(
        signal: np.ndarray,
        samplerate: float,
        lowf: float,
        highf: float,
) -> np.ndarray:
    """Bandpass filter a signal.

    Parameters
    ----------
    signal : np.ndarray
        The data to be filtered
    rate : float
        The sampling rate
    lowf : float
        The low cutoff frequency
    highf : float
        The high cutoff frequency

    Returns
    -------
    np.ndarray
        The filtered data
    """
    sos = butter(2, (lowf, highf), "bandpass", fs=samplerate, output="sos")
    filtered_signal = sosfiltfilt(sos, signal)

    return filtered_signal


def highpass_filter(
    signal: np.ndarray,
    samplerate: float,
    cutoff: float,
) -> np.ndarray:
    """Highpass filter a signal.

    Parameters
    ----------
    signal : np.ndarray
        The data to be filtered
    rate : float
        The sampling rate
    cutoff : float
        The cutoff frequency

    Returns
    -------
    np.ndarray
        The filtered data
    """
    sos = butter(2, cutoff, "highpass", fs=samplerate, output="sos")
    filtered_signal = sosfiltfilt(sos, signal)

    return filtered_signal


def lowpass_filter(
    signal: np.ndarray,
    samplerate: float,
    cutoff: float
) -> np.ndarray:
    """Lowpass filter a signal.

    Parameters
    ----------
    data : np.ndarray
        The data to be filtered
    rate : float
        The sampling rate
    cutoff : float
        The cutoff frequency

    Returns
    -------
    np.ndarray
        The filtered data
    """
    sos = butter(2, cutoff, "lowpass", fs=samplerate, output="sos")
    filtered_signal = sosfiltfilt(sos, signal)

    return filtered_signal


def envelope(signal: np.ndarray,
             samplerate: float,
             cutoff_frequency: float
             ) -> np.ndarray:
    """Calculate the envelope of a signal using a lowpass filter.

    Parameters
    ----------
    signal : np.ndarray
        The signal to calculate the envelope of
    samplingrate : float
        The sampling rate of the signal
    cutoff_frequency : float
        The cutoff frequency of the lowpass filter

    Returns
    -------
    np.ndarray
        The envelope of the signal
    """
    sos = butter(2, cutoff_frequency, "lowpass", fs=samplerate, output="sos")
    envelope = np.sqrt(2) * sosfiltfilt(sos, np.abs(signal))

    return envelope
