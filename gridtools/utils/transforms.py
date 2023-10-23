#!/usr/bin/env python3

"""
Some common transforms for use with time series data.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt


def instantaneous_frequency(
    signal: np.ndarray,
    samplerate: int,
    smoothing_window: int,
) -> np.ndarray:
    """
    Compute the instantaneous frequency of a signal that is approximately
    sinusoidal and symmetric around 0.

    Parameters
    ----------
    signal : np.ndarray
        Signal to compute the instantaneous frequency from.
    samplerate : int
        Samplerate of the signal.
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

    upper_time = np.abs(time_signal[period_index])
    lower_time = np.abs(time_signal[period_index - 1])

    # create ratio
    lower_ratio = np.abs(signal[period_index - 1]) / (
        np.abs(signal[period_index - 1]) + np.abs(signal[period_index])
    )

    # apply to time delta
    true_zero = lower_time + lower_ratio * (upper_time - lower_time)

    # compute frequency
    inst_freq = gaussian_filter1d(1 / np.diff(true_zero), smoothing_window)

    # Resample the frequency using specified interpolation method to match the
    # dimensions of the input array
    old_x = np.arange(0, len(inst_freq))
    new_x = np.linspace(0, len(inst_freq), len(signal))
    freq = np.interp(new_x, old_x, inst_freq)

    return freq


def envelope(
    signal: np.ndarray, samplerate: float, cutoff_frequency: float
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
    env = np.sqrt(2) * sosfiltfilt(sos, np.abs(signal))

    old_x = np.arange(0, len(env))
    new_x = np.linspace(0, len(env), len(signal))
    env = np.interp(new_x, old_x, env)

    return env
