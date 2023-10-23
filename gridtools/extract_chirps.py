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
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal.windows import tukey

from .datasets import ChirpData, Dataset, RawData, WavetrackerData
from .simulations import chirp_model
from .utils.filters import bandpass_filter
from .utils.transforms import envelope, instantaneous_frequency


def initial_params(x, y):
    """
    Compute initial parameter estimates for the chirp model.

    Parameters
    ----------
    x : array-like
        Time axis.
    y : array-like
        Signal.

    Returns
    -------
    tuple
        Initial parameter estimates and boundaries.
    """
    # Compute the mean and standard deviation of the signal
    y_mean = np.mean(y)
    y_std = np.std(y)

    # Find the indices of the peaks and troughs
    peaks, _ = find_peaks(y, height=y_mean + 2 * y_std)
    troughs, _ = find_peaks(-y, height=-y_mean - 2 * y_std)

    # Check if any peaks or troughs were found
    if len(peaks) == 0 or len(troughs) == 0:
        return [
            y_mean,
            y_mean / 2,
            1,
            2,
            x[0],
            y_mean,
            y_mean / 2,
            1,
            2,
            x[-1],
            y_mean,
            y_mean / 2,
        ]

    # Sort the peaks and troughs by their x values
    peak_x = x[peaks]
    peak_y = y[peaks]
    trough_x = x[troughs]
    trough_y = y[troughs]
    peak_order = np.argsort(peak_y)
    trough_order = np.argsort(trough_x)
    peak_x = peak_x[peak_order]
    peak_y = peak_y[peak_order]
    trough_x = trough_x[trough_order]
    trough_y = trough_y[trough_order]

    # Compute the initial parameter estimates
    m1 = 0  # Center of the first Gaussian
    h1 = peak_y[-1] - trough_y[-1]  # Amplitude of the first Gaussian
    w1 = (
        peak_x[-1] - trough_x[-1]
    ) / 5  # Standard deviation of the first Gaussian
    k1 = 1  # Kurtosis of the first Gaussian
    m2 = 0 + abs(w1) * 0.6  # Center of the second Gaussian
    h2 = -h1 / 5  # Amplitude of the second Gaussian
    w2 = w1 / 2  # Standard deviation of the second Gaussian
    k2 = 1  # Kurtosis of the second Gaussian
    m3 = (
        peak_x[-1] + (peak_x[-1] - trough_x[-1]) / 10
    )  # Center of the third Gaussian
    h3 = np.abs(h2)  # Amplitude of the third Gaussian
    w3 = w1 / 2  # Standard deviation of the third Gaussian
    k3 = 1  # Kurtosis of the third Gaussian

    return [m1, h1, w1, k1, m2, h2, w2, k2, m3, h3, w3, k3]


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


def extract_features(data):
    """Extract instantaneous frequency and amplitude of chirps of the
    upper fish in a dataset.

    Parameters
    ----------
    data : Dataset
        Dataset as defined in datasets.py.

    Returns
    -------
        _description_
    """
    time_window = 0.5

    upper_fish = get_upper_fish(data)
    lower_fish = get_next_lower_fish(data, upper_fish)
    chirp_times = data.chirp.times[data.chirp.idents == upper_fish]
    track_freqs = data.track.freqs[data.track.idents == upper_fish]
    track_times = data.track.times[
        data.track.indices[data.track.idents == upper_fish]
    ]
    track_powers = data.track.powers[
        data.track.indices[data.track.idents == upper_fish], :
    ]

    freqs = []
    envs = []
    widths = []
    for chirp in chirp_times:
        track_index = np.argmin(np.abs(track_times - chirp))
        track_freq = track_freqs[track_index]
        lower_fish_freq = get_fish_freq(data, chirp, lower_fish)

        lower_bound = track_freq - lower_fish_freq - 50
        track_power = track_powers[track_index, :]
        best_electrode = np.argmax(track_power)

        start_index = int(
            np.round((chirp - time_window / 2) * data.rec.samplerate)
        )
        stop_index = int(
            np.round((chirp + time_window / 2) * data.rec.samplerate)
        )
        raw_index = np.arange(start_index, stop_index)

        raw = data.rec.raw[raw_index, best_electrode]
        tuk = tukey(len(raw), alpha=0.1)
        raw = raw * tuk

        # WARNING: Youll have to play around with the bandpass filter cutoffs
        # depending on the reocrding. The further away the baseline EODf of two
        # fish, the lower can be lowf but highf must also be decreased to remove
        # the harmonic of the upper fish.
        raw = bandpass_filter(
            signal=raw,
            samplerate=data.rec.samplerate,
            lowf=track_freq - 40,
            highf=track_freq + 280,
        )

        raw = raw / np.max(np.abs(raw))
        rawtime = (np.arange(len(raw)) - len(raw) / 2) / data.rec.samplerate

        freq = instantaneous_frequency(
            signal=raw,
            samplerate=data.rec.samplerate,
            smoothing_window=5,
        )

        dist = np.histogram(freq, bins=100)
        mode = dist[1][np.argmax(dist[0])]

        freq = freq - mode
        freq = freq[1000:-1000]
        tuk = tukey(len(freq), alpha=0.3)
        freq = freq * tuk
        time = (np.arange(len(freq)) - len(freq) / 2) / data.rec.samplerate

        peak_height = np.percentile(freq, 95)
        peaks, _ = find_peaks(freq, height=peak_height)

        peak_index = np.argmin(np.abs(time[peaks] - chirp))
        peak = peaks[peak_index]

        # skip if peak it too close to edge
        if abs(time[peak]) > time_window / 4:
            continue

        if np.min(freq) < -80:
            continue

        # compute envelope as well
        renv = envelope(
            raw, samplerate=data.rec.samplerate, cutoff_frequency=100
        )

        # remove low frequency modulation
        env = bandpass_filter(
            signal=renv,
            samplerate=data.rec.samplerate,
            lowf=0.1,
            highf=100,
        )

        # cut off the edges of the envelope to remove tukey window
        env = env[1000:-1000]

        mode = np.histogram(env, bins=100)[1][
            np.argmax(np.histogram(env, bins=100)[0])
        ]
        env = (env - mode) * tuk
        envtime = (np.arange(len(env)) - len(env) / 2) / data.rec.samplerate

        # detect anolaies (peaks and troughs) in the envelope
        absenv = np.abs(env)
        env_peaks, _ = find_peaks(absenv, height=np.percentile(absenv, 99))

        # get the peak closest to the chirp
        env_peak_index = np.argmin(np.abs(envtime[env_peaks] - chirp))

        # skip if peak it too close to edge
        if abs(envtime[env_peaks[env_peak_index]]) > time_window / 4:
            continue

        # check if the peak is close to the frequency peak
        if abs(envtime[env_peaks[env_peak_index]] - time[peak]) > 0.05:
            continue

        # center the chirp on the center index using the peak on the frequency

        # descend the peak in both directions until the frequency drops below 10
        left, right = peak, peak
        while freq[left] > 10:
            left += 1
        while freq[right] > 10:
            right -= 1

        # find the center between the flanks of the peak
        center = (right - left) // 2 + left
        width = (right - left) * 20000

        roll = len(freq) // 2 - center
        freq = np.roll(freq, roll)

        # center the env on the freq peak as well
        env = np.roll(env, roll)

        tuk = tukey(len(freq), alpha=1)
        freq = freq * tuk
        env = (env * tuk) + 1

        height = np.max(freq)

        # check if there are multiple large peaks
        cp, _ = find_peaks(freq, prominence=height * 0.5)
        if len(cp) > 1:
            continue

        # fig, axs = plt.subplots(3, 1, sharex=True)
        # axs[0].plot(rawtime, raw)
        # axs[0].plot(rawtime, renv, color="red")
        # axs[1].plot(time, freq)
        # axs[1].plot(time[peak], freq[peak], "o")
        # axs[1].axhline(0, color="gray")
        # axs[2].plot(envtime, env)
        # axs[2].plot(envtime[env_peaks], env[env_peaks], "o")
        # axs[2].axhline(1, color="gray")
        # plt.show()

        # fig, ax = plt.subplots(2, 1, sharex=True)
        # ax[0].plot(freq)
        # ax[0].plot(cp, freq[cp], "o")
        # ax[1].plot(env)
        # ax[0].axvline(len(freq) // 2, color="gray", linestyle="--", lw=1)
        # ax[1].axvline(len(freq) // 2, color="gray", linestyle="--", lw=1)
        # ax[0].axhline(0, color="gray", linestyle="--", lw=1)
        # ax[1].axhline(1, color="gray", linestyle="--", lw=1)
        # plt.show()

        freqs.append(freq)
        envs.append(env)
        widths.append(width)

    return freqs, envs, width


def env_to_gauss(env):
    return -(env - 1)


def gauss_to_env(env):
    return -env + 1


def fit_model(freqs, envs):
    freq_fit = []
    env_contrasts = []
    for iter, (freq, env) in track(
        enumerate(zip(freqs, envs)),
        description="Fitting chirps",
        total=len(freqs),
    ):
        # fig, ax = plt.subplots(2, 1, sharex=True)

        x = (np.arange(len(freq)) / 20000) - (len(freq) / 20000 / 2)
        p0_c = initial_params(x, freq)

        ec = 1 - np.min(env)
        env_contrasts.append(ec)

        try:
            popt_c, pcov_c = curve_fit(
                f=chirp_model,
                xdata=x,
                ydata=freq,
                maxfev=100000,
                p0=p0_c,
            )
            freq_fit.append(popt_c)
            rprint(f"[bold green]Iter: {iter} SUCCESS![/bold green]")
        except:
            # ax[0].plot(x, freq, color="black")
            # ax[0].plot(x, chirp_model(x, *p0_c), color="blue", alpha=0.5)
            # ax[1].plot(x, env, color="black")
            # plt.show()
            freq_fit.append(np.full_like(p0_c, np.nan))
            rprint(f"[bold red]Iter: {iter} FITTING FAILED![/bold red]")
            continue

        # ax[0].plot(x, freq, color="black", label="data")
        # ax[0].plot(x, chirp_model(x, *popt_c), color="red", label="fit")
        # ax[0].plot(x, chirp_model(x, *p0_c), color="blue", label="initial", alpha=0.5)
        # ax[1].plot(x, env, color="black", label="data")
        # ax[0].legend()
        # plt.show()

    return freq_fit


def save_arrays(freqs, envs, freq_fit, input_dir, output_dir):
    filename = input_dir.name
    path = output_dir
    np.save(path / f"{filename}_freqs.npy", freqs)
    np.save(path / f"{filename}_envs.npy", envs)
    np.save(path / f"{filename}_freq_fit.npy", freq_fit)


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


def extract_chirp_params(input_dir, output_dir):
    raw = RawData(input_dir)
    chirps = ChirpData(input_dir, detector="gt")
    wavetracker = WavetrackerData(input_dir)
    dataset = Dataset(
        path=input_dir,
        track=wavetracker,
        rec=raw,
        chirp=chirps,
    )

    freqs, envs, widths = extract_features(dataset)
    freq_fit = fit_model(freqs, envs)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(20, 20))
    for freq, env, fit in zip(freqs, envs, freq_fit):
        x = (np.arange(len(freq)) / 20000) - (len(freq) / 20000 / 2)
        ax[0].plot(x, freq, color="black", alpha=0.1)
        ax[1].plot(x, chirp_model(x, *fit), color="black", alpha=0.1)
        ax[2].plot(x, env, color="black", alpha=0.1)

    ax[0].axvline(0, color="gray", linestyle="--", lw=1)
    ax[0].axhline(0, color="gray", linestyle="--", lw=1)
    ax[1].axvline(0, color="gray", linestyle="--", lw=1)
    ax[1].axhline(0, color="gray", linestyle="--", lw=1)
    ax[2].axvline(0, color="gray", linestyle="--", lw=1)
    ax[2].axhline(1, color="gray", linestyle="--", lw=1)

    ax[0].set_ylabel("Frequency (Hz)")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Amplitude")

    ax[0].set_title("Instantaneous Frequency")
    ax[1].set_title("Fitted instantaneous frequency")
    ax[2].set_title("Envelope")

    plt.savefig(output_dir / f"{input_dir.name}_chirps.png")
    plt.show()

    save_arrays(freqs, envs, freq_fit, input_dir, output_dir)


def main_extract():
    args = extract_interface()
    extract_chirp_params(args.input, args.output)


def load_dataset(path: pathlib.Path):
    freq_fits = list(path.glob("*freq_fit.npy"))

    rprint(f"Found {len(freq_fits)} files.")

    freq_params = []
    for freq in freq_fits:
        fps = np.load(freq)
        freq_params.append(fps)
    freq_params = np.concatenate(freq_params)
    return freq_params


def sort_dataset(dataset):
    """
    Sort the Gaussians in each chirp by their amplitude.
    """

    sorted_dataset = np.zeros_like(dataset)

    for i in range(len(dataset)):
        curr_fit = dataset[i, :]

        # split fits into 3 gaussians
        split_fit = np.split(curr_fit, 3)

        # sort by height, which is the second parameter of each gaussian
        sorted_fit = sorted(split_fit, key=lambda x: x[1])

        # put it back into the matrix
        sorted_dataset[i, :] = np.concatenate(sorted_fit)

    return sorted_dataset


def remove_outliers(dataset):
    """
    Remove all parameter arrays that have large outliers.
    """

    # TODO: Switch to quantiles to remove outliers because some of the distributions
    # are definetely not normal.

    # get indices of rows that contain outliers in any parameter
    for i in range(dataset.shape[1]):
        param = dataset[:, i]
        outliers = np.abs(param - np.mean(param)) > 2 * np.std(param)
        dataset = dataset[~outliers, :]

    return dataset


def resample_chirp_params(path: pathlib.Path):
    freq_fits = load_dataset(path)
    # freq_fits = sort_dataset(freq_fits)

    # remove the nans from the dataset
    freq_fits = freq_fits[~np.isnan(freq_fits).any(axis=1)]

    freq_fits = remove_outliers(freq_fits)

    params = [
        "m1",
        "h1",
        "w1",
        "k1",
        "m2",
        "h2",
        "w2",
        "k2",
        "m3",
        "h3",
        "w3",
        "k3",
    ]

    # interpolate to make more
    old_x = np.arange(freq_fits.shape[0])
    new_x = np.linspace(0, freq_fits.shape[0], 1000)

    new_freq_fits = []
    new_env_fits = []

    for i in range(freq_fits.shape[1]):
        rprint(i)
        ff = interp1d(
            old_x, freq_fits[:, i], kind="linear", fill_value="extrapolate"
        )

        new_freq_fits.append(ff(new_x))

        plt.plot(new_x, ff(new_x), ".", label="freq")
        plt.plot(old_x, freq_fits[:, i], ".", label="old freq")

        plt.title(params[i])
        plt.legend()
        plt.show()

    new_freq_fits = np.array(new_freq_fits).T

    # plot the new distributions and overlay the old ones
    param_names = [
        "Center",
        "Height",
        "Width",
        "Kurtosis",
    ]

    fig, ax = plt.subplots(3, 4, figsize=(20, 20), sharex="col", sharey="row")
    for i, param in enumerate(params):
        ax[i // 4, i % 4].hist(
            freq_fits[:, i], bins=100, alpha=0.5, label="old"
        )
        ax[i // 4, i % 4].hist(
            new_freq_fits[:, i], bins=100, alpha=0.5, label="new"
        )
        ax[i // 4, i % 4].axvline(0, color="k", ls="--", lw=0.5)

        if i == 0:
            ax[i // 4, i % 4].legend()

        # add row labels
        if i % 4 == 0:
            ax[i // 4, i % 4].set_ylabel("Density")

        # add column labels
        if i // 4 == 2:
            ax[i // 4, i % 4].set_xlabel("Parameter value")

        # add column titles
        if i < 4:
            ax[i // 4, i % 4].set_title(param_names[i % 4])

    fig.suptitle("Frequency fits")
    plt.savefig(path / "interpolation_distributions.png")
    plt.show()

    # plot the resulting chirps
    fig, ax = plt.subplots()
    t = np.linspace(0, 0.5, 20000) - 0.25
    tuk = tukey(len(t), alpha=0.4)
    for i in range(len(new_freq_fits)):
        popt = new_freq_fits[i, :]
        chirp = chirp_model(t, *popt)
        mode = np.histogram(chirp, bins=100)[1][
            np.argmax(np.histogram(chirp, bins=100)[0])
        ]
        chirp -= mode
        chirp *= tuk
        ax.plot(
            t,
            chirp,
            alpha=0.1,
            color="k",
            lw=0.5,
        )

    ax.axvline(0, color="k", ls="--", lw=0.5)
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-50, 400)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Interpolated chirp simulations")
    plt.savefig(path / "interpolation.png")
    plt.show()

    np.save(path / "interpolation.npy", new_freq_fits)


def resample_interface():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", "-p", type=pathlib.Path, help="Path to the dataset."
    )
    args = parser.parse_args()
    return args


def main_resample():
    args = resample_interface()
    resample_chirp_params(args.path)
