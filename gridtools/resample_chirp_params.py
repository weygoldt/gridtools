#!/usr/bin/env python

"""
Use the fitted chirp parameters to resample from a real parameter space.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from rich import print as rprint
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey
from simulations import chirp_model


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


def interface():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", "-p", type=pathlib.Path, help="Path to the dataset."
    )
    args = parser.parse_args()
    return args


def resample_chirps():
    args = interface()
    resample_chirp_params(args.path)


if __name__ == "__main__":
    main()
