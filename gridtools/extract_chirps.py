#!/usr/bin/env python3

"""
This file extracts the evolution of the instantaneous frequency and amplitude
of chirps from a real labeled dataset. The extracted parameters can then 
be used to simulate chirps with the same characteristics.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from rich import print as rprint
from rich.progress import track
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal.windows import tukey

from .datasets import Dataset, load
from .simulations import chirp_model_v4
from .utils.filters import bandpass_filter
from .utils.transforms import instantaneous_frequency

model = chirp_model_v4


def get_upper_fish(dataset):
    """Return the fish with the highest frequency.

    Parameters
    ----------
    dataset : Dataset
        A dataset object as defined in datasets.py.

    Returns
    -------
    int
        The id of the fish with the highest frequency.
    """
    min_fs = []
    track_ids = np.unique(dataset.track.idents[~np.isnan(dataset.track.idents)])
    for track_id in track_ids:
        f = dataset.track.freqs[dataset.track.idents == track_id]
        min_fs.append(np.min(f))
    return track_ids[np.argmax(min_fs)]


def get_next_lower_fish(dataset, upper_fish):
    """Return the fish that, in the frequency domain, is directly below the
    upper fish.

    Parameters
    ----------
    dataset : Dataset
        Dataset as defined in datasets.py.
    upper_fish : int
        Id of the upper fish.

    Returns
    -------
    int
        Id of the lower fish.
    """
    min_fs = []
    track_ids = np.unique(dataset.track.idents[~np.isnan(dataset.track.idents)])
    assert upper_fish in track_ids

    for track_id in track_ids:
        if track_id == upper_fish:
            min_fs.append(np.inf)
        else:
            f = dataset.track.freqs[dataset.track.idents == track_id]
            min_fs.append(np.min(f))
    return track_ids[np.argmin(min_fs)]


def get_fish_freq(dataset, time, fish_id):
    """Get the baseline EODf as estimated by the wavetracker
    of a fish at a given time.

    Parameters
    ----------
    dataset : Dataset
        Dataset as defined in datasets.py.
    time : float
        Time in seconds.
    fish_id : int
        Id of the fish.

    Returns
    -------
    float
        Frequency in Hz.
    """
    track_freqs = dataset.track.freqs[dataset.track.idents == fish_id]
    track_times = dataset.track.times[
        dataset.track.indices[dataset.track.idents == fish_id]
    ]
    track_index = np.argmin(np.abs(track_times - time))
    return track_freqs[track_index]


def extract_features(data: "Dataset") -> np.ndarray:
    # get the fish with the highest baseline frequency
    time_window = 1
    upper_fish = get_upper_fish(data)
    lower_fish = get_next_lower_fish(data, upper_fish)

    # get data for the upper fish
    chirp_times = data.com.chirp.times[data.com.chirp.idents == upper_fish]
    track_freqs = data.track.freqs[data.track.idents == upper_fish]
    track_times = data.track.times[
        data.track.indices[data.track.idents == upper_fish]
    ]
    track_powers = data.track.powers[
        data.track.indices[data.track.idents == upper_fish]
    ]

    # store chip instantaneous frequencies here
    freqs = []
    for chirp in chirp_times:
        # Find the best electrode
        track_index = np.argmin(np.abs(track_times - chirp))
        track_power = track_powers[track_index, :]
        best_electrode = np.argmax(track_power)

        start_index = int(
            np.round((chirp - time_window / 2) * data.grid.rec.samplerate)
        )
        stop_index = int(
            np.round((chirp + time_window / 2) * data.grid.rec.samplerate)
        )
        index_array = np.arange(start_index, stop_index)
        raw = data.grid.rec[index_array, best_electrode]

        # apply tukey window to avoid edge effects of the bandpass filter
        tuk = tukey(raw.size, alpha=0.1)
        raw *= tuk

        # extract frequencies to make limits for bandpass filter
        lower_fish_freq = get_fish_freq(data, chirp, lower_fish)
        track_freq = track_freqs[track_index]

        # make limits for bandpass filter: Cut right between the two fish
        # and below the first harmonic of the lower fish
        lowcut = lower_fish_freq + 0.8 * (track_freq - lower_fish_freq)
        highcut = 0.9 * (2 * lower_fish_freq)

        # apply bandpass filter
        freq = bandpass_filter(
            signal=raw,
            samplerate=data.grid.rec.samplerate,
            lowf=lowcut,
            highf=highcut,
        )

        # normalize
        freq = freq / np.max(np.abs(freq))

        # extract the instantaneous frequency
        freq = instantaneous_frequency(
            signal=freq,
            samplerate=data.grid.rec.samplerate,
            smoothing_window=5,
        )

        # extract the mode of the instantaneous frequency
        dist = np.histogram(freq, bins=100)
        mode = dist[1][np.argmax(dist[0])]

        # center the frequency at 0
        freq = freq - mode

        # apply a tukey window to smoothe edges to 0
        tuk = tukey(freq.size, alpha=0.3)
        freq *= tuk

        # build a time axis for the extracted frequency
        time = (np.arange(len(freq)) - len(freq) / 2) / data.grid.rec.samplerate

        # search for the main peak of the frequency excursion during chirp
        peak_height = np.percentile(freq, 99)
        peaks, _ = find_peaks(freq, height=peak_height)
        peak = peaks[np.argmax(np.abs(freq[peaks]))]

        # check if peak is too close to the edge
        if abs(time[peak]) > 0.25 * time_window:
            continue

        # check if there are multiple large peaks and simply skip the chirp
        # if there are
        height = np.max(freq)
        cp, _ = find_peaks(freq, prominence=0.3 * height)
        if len(cp) > 1:
            continue

        # center the chirp on the time axis:
        # descend the peak in both directions until it drops below 10 %
        # of the peak height
        left, right = peak, peak
        while freq[left] > 0.1 * peak_height:
            left -= 1
        while freq[right] > 0.1 * peak_height:
            right += 1

        # find the center between the flanks of the peaks
        center = (left + right) // 2

        # roll the peak to the center
        roll = len(freq) // 2 - center
        freq = np.roll(freq, roll.astype(int))

        # apply tukey window again because edges are now not smooth anymore
        tuk = tukey(freq.size, alpha=0.3)
        freq *= tuk

        # save the frequency
        freqs.append(freq)

    return freqs


def estimate_params(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # compute mean and standard deviation of signal for peak detection
    y_mean = np.mean(y)
    y_std = np.std(y)

    # find peaks and troughs
    peaks, _ = find_peaks(y, height=y_mean + 2 * y_std)
    troughs, _ = find_peaks(-y, height=-y_mean + 2 * y_std)

    # check if there are enough peaks and troughs
    if len(peaks) == 0 or len(troughs) == 0:
        return None

    # sort the peaks and troughts by their height
    peaks = peaks[np.argsort(y[peaks])]
    troughs = troughs[np.argsort(y[troughs])]

    # initial estimate for center is position of highest peak
    m = x[peaks[-1]]
    # get the height of the highest peak as the amplitude
    a = y[peaks[-1]]
    # estimate standard deviation of largest peak
    s = (y[peaks[-1]] - y[troughs[-1]]) / 2
    # initial kurtosis estimate
    k = 1.5
    return np.array([m, a, s, k])


def fit_model(freqs: np.ndarray, model: object) -> np.ndarray:
    fits = []

    for iter, freq in track(
        enumerate(freqs), description="Fitting chirps", total=len(freqs)
    ):
        # make a time axis
        x = (np.arange(len(freq)) / 20000) - (len(freq) / 20000 / 2)

        # estimate initial parameters
        p0_c = estimate_params(x, freq)

        # set bounds for parameters
        bounds = (
            [-0.1, 20, -np.inf, 0.9],
            [0.1, 600, np.inf, 10],
        )

        # try to fit the model
        try:
            popt, _ = curve_fit(
                model,
                x,
                freq,
                p0=p0_c,
                bounds=bounds,
            )
            fits.append(popt)
            rprint(f"[green]Fit {iter} successful[/green]")
        except:
            rprint(f"[red]Fit {iter} failed[/red]")
            continue
    return np.array(fits)


def extract_chirp_params(
    input_dir: pathlib.Path, output_dir: pathlib.Path
) -> None:
    # load a dataset
    data = load(input_dir, grid=True)

    # extract instantaneous frequencies at each chirp of the fish with
    # the highest baseline frequency
    freqs = extract_features(data)

    # try to fit a gaussian to the extracted frequencies
    fits = fit_model(freqs, model)

    fits[:, 0] = 0

    # plot the results
    _, ax = plt.subplots(2, 1, sharex=True, figsize=(20, 20))

    for freq, fit in zip(freqs, fits):
        x = (np.arange(len(freq)) / 20000) - (len(freq) / 20000 / 2)
        ax[0].plot(x, freq, color="black", alpha=0.1)
        ax[1].plot(x, model(x, *fit), color="black", alpha=0.1)

    ax[0].axvline(0, color="gray", linestyle="--", lw=1)
    ax[0].axhline(0, color="gray", linestyle="--", lw=1)
    ax[1].axvline(0, color="gray", linestyle="--", lw=1)
    ax[1].axhline(0, color="gray", linestyle="--", lw=1)

    ax[0].set_ylabel("Frequency (Hz)")
    ax[1].set_xlabel("Time (s)")

    ax[0].set_title("Instantaneous Frequency")
    ax[1].set_title("Fitted instantaneous frequency")

    plt.savefig(output_dir / f"{input_dir.name}_chirps.png")
    plt.show()

    np.save(output_dir / f"{input_dir.name}_chirp_freqs.npy", freqs)
    np.save(output_dir / f"{input_dir.name}_chirp_fits.npy", fits)


def load_chirp_fits(path: pathlib.Path) -> np.ndarray:
    files = list(path.glob("*_chirp_fits.npy"))
    rprint(f"[green]Found {len(files)} chirp fit datasets[/green]")

    fits = []
    for file in files:
        fits.append(np.load(file))
    return np.concatenate(fits)


def resample_chirp_fits(input: pathlib.Path, output: pathlib.Path) -> None:
    # load chirp fits
    fits = load_chirp_fits(input)

    # make standard deviations positive
    fits[:, 2] = np.abs(fits[:, 2])

    # sort fits by standard deviation
    fits = fits[np.argsort(fits[:, 2])]

    # interpolate the fits based on their standard deviation
    oldx = np.arange(len(fits))
    newx = np.linspace(0, len(fits), 1000)
    newfits = np.zeros((len(newx), 4))
    for i in range(4):
        newfits[:, i] = np.interp(newx, oldx, fits[:, i])
        plt.plot(oldx, fits[:, i], ".", color="black", alpha=0.1)
        plt.plot(newx, newfits[:, i], ".", color="red", alpha=0.1)
        plt.title(f"Parameter {i}")
        plt.show()

    newfits = np.asarray(newfits).T

    # plot the distributions of the parameters
    _, ax = plt.subplots(2, 2, figsize=(20, 20))
    ax[0, 0].hist(newfits[0, :], bins=100)
    ax[0, 0].hist(fits[:, 0], bins=100, alpha=0.5)
    ax[0, 0].set_title("Center")
    ax[0, 1].hist(newfits[1, :], bins=100)
    ax[0, 1].hist(fits[:, 1], bins=100, alpha=0.5)
    ax[0, 1].set_title("Amplitude")
    ax[1, 0].hist(newfits[2, :], bins=100)
    ax[1, 0].hist(fits[:, 2], bins=100, alpha=0.5)
    ax[1, 0].set_title("Standard Deviation")
    ax[1, 1].hist(newfits[3, :], bins=100)
    ax[1, 1].hist(fits[:, 3], bins=100, alpha=0.5)
    ax[1, 1].set_title("Kurtosis")
    plt.savefig(output / "chirp_fit_distributions.png")
    plt.show()

    # plot the resulting chirps
    _, ax = plt.subplots(1, 1, figsize=(20, 20))
    t = np.linspace(0, 1, 20000) - 0.5

    for fit in newfits.T:
        ax.plot(t, model(t, *fit), color="black", alpha=0.1)
    for fit in fits:
        ax.plot(t, model(t, *fit), color="red", alpha=0.1)

    ax.set_title("Simulated chirps")
    ax.axvline(0, color="k", linestyle="--", lw=0.5)
    ax.axhline(0, color="k", linestyle="--", lw=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.savefig(output / "chirp_simulations.png")
    plt.show()

    np.save(output / "interpolation.npy", newfits)


def extract_interface():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=pathlib.Path, help="Path to dataset."
    )
    parser.add_argument(
        "--output", "-o", type=pathlib.Path, help="Path to output."
    )
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    return args


def main_extract():
    args = extract_interface()
    extract_chirp_params(args.input, args.output)


def main_resample():
    args = extract_interface()
    resample_chirp_fits(args.input, args.output)
