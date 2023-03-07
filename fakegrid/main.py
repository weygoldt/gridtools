import matplotlib.pyplot as plt
import numpy as np
from numpy.random import sample
import thunderfish.fakefish as ff
from tqdm import tqdm
import thunderfish.powerspectrum as ps
from scipy.signal import butter, sosfiltfilt
from IPython import embed

samplerate = 20000
duration = 20.0
time = np.arange(0, duration, 1/samplerate)

# generate frequency trace with chirps
chirp_trace, _ = ff.chirps(eodf=0.0,
                           samplerate=samplerate,
                           duration=duration,
                           chirp_freq=0.1,
                           chirp_size=100.0,
                           chirp_width=0.01,
                           chirp_kurtosis=1.0,
                           chirp_contrast=0.05
                           )

# generate frequency trace with rises
rise_trace = ff.rises(eodf=0.0,
                      samplerate=samplerate,
                      duration=duration,
                      rise_freq=0.01,
                      rise_size=40.0,
                      rise_tau=1.0,
                      decay_tau=10.0
                      )

# combine traces to one
# full_trace = rise_trace + chirp_trace + 500.0
full_trace = chirp_trace + 500.0

# make the EOD from the frequency trace
fish = ff.wavefish_eods(
    fish='Alepto',
    frequency=full_trace,
    samplerate=samplerate,
    duration=duration,
    phase0=0.0,
    noise_std=0.05
)

# comptute a spectrogram of the resutling EOD
spec, freqs, spectime = ps.spectrogram(
    data=fish,
    ratetime=samplerate,
    freq_resolution=0.5,
    overlap_frac=0.5,
)

# build the rolling bandpass filter


def bandpass_filter(
        signal: np.ndarray,
        samplerate: float,
        lowf: float,
        highf: float,
        order: int = 2,
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
    sos = butter(order, (lowf, highf), "bandpass", fs=samplerate, output="sos")
    filtered_signal = sosfiltfilt(sos, signal)

    return filtered_signal


# test the bandpass filter
filtered_signal = bandpass_filter(fish, samplerate, 495, 505, order=5)
filt_spec, filt_freqs, filt_spectime = ps.spectrogram(
    data=filtered_signal,
    ratetime=samplerate,
    freq_resolution=0.5,
    overlap_frac=0.5,
)


def rolling_bandpass(
        signal: np.ndarray,
        time: np.ndarray,
        samplerate: float,
        freq_trace: np.ndarray,
        freq_padding: float,
        filter_order: int = 2,
        window_size: int = 101,
        step_size: int = 1,
):

    # compute the bandpass filter for each
    # frequency in the trace

    window_radius = int((window_size-1) / 2)
    center_indices = np.arange(window_radius+1, len(freq_trace), step_size)
    filtered_signal = np.zeros_like(signal)

    for i in tqdm(center_indices, total=len(center_indices)):
        freq = freq_trace[i]
        filtered_signal[i] = bandpass_filter(
            signal[i-window_radius:i+window_radius],
            samplerate,
            freq - freq_padding,
            freq + freq_padding,
            order=filter_order,
            )[window_radius+1]
    return filtered_signal


rolling_filtered_signal, rol_time = rolling_bandpass(
    fish, time, samplerate, np.ones_like(fish)*500.0, 5.0)

fig, ax = plt.subplots(7, 1, sharex=True)
ax[0].plot(time, chirp_trace)
ax[0].plot(time, rise_trace)
ax[1].plot(time, full_trace)
ax[2].plot(time, fish)
ax[3].imshow(X=ps.decibel(spec),
             aspect='auto',
             origin='lower',
             extent=(spectime.min(), spectime.max(), freqs.min(), freqs.max())
             )
ax[3].set_ylim(450, 700)
ax[4].plot(time, filtered_signal)
ax[5].imshow(X=ps.decibel(filt_spec),
             aspect='auto',
             origin='lower',
             extent=(filt_spectime.min(), filt_spectime.max(),
                     filt_freqs.min(), filt_freqs.max())
             )
ax[5].set_ylim(450, 700)
ax[6].plot(time, rolling_filtered_signal)
plt.show()
