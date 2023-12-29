"""Models of chirps for fitting."""

import numpy as np
from numba import jit
import matplotlib.pyplot as plt


@jit(nopython=True, parallel=True)
def rise(
    t: np.ndarray,
    start: float,
    height: float,
    rise_tau: float,
    decay_tau: float
) -> np.ndarray:
    """Model of a rise as a combination of two exponentials.

    Parameters
    ----------
    t : np.ndarray
        The points at which to evaluate the rise.
    start : float
        Where the rise starts.
    height : float
        The height of the rise.
    rise_tau : float
        The time constant of the rising exponential.
    decay_tau : float
        The time constant of the decaying exponential.

    Returns
    -------
    np.ndarray
        The value of the rise at the given points.
    """
    t_shifted = t - start
    t_shifted[t_shifted < 0] = 0
    increase = 1.0 - np.exp(-t_shifted / rise_tau)
    decrease = np.exp(-t_shifted / decay_tau)
    return height * increase * decrease


@jit(nopython=True, parallel=True)
def monophasic_chirp(
    x: np.ndarray, mu: float, height: float, width: float, kurt: float
) -> np.ndarray:
    """
    Compute the value of a Gaussian function at the given points.

    Parameters
    ----------
    x : np.ndarray
        The points at which to evaluate the Gaussian function.
    mu : float
        The mean of the Gaussian function.
    height : float
        The height of the Gaussian function.
    width : float
        The width of the Gaussian function.
    kurt : float
        The kurtosis of the Gaussian function.

    Returns
    -------
    np.ndarray
        The value of the Gaussian function at the given points.
    """
    sigma = 0.5 * width / (2.0 * np.log(10.0)) ** (0.5 / kurt)
    return height * np.exp(-0.5 * (((x - mu) / sigma) ** 2.0) ** kurt)


@jit(nopython=True, parallel=True)
def biphasic_chirp( # noqa
    x: np.ndarray,
    mu: float,
    height: float,
    width: float,
    kurt: float,
    undershoot: float,
) -> np.ndarray:
    """Model of a chirp as a combination of two Gaussians to add an undershoot.

    Parameters
    ----------
    x : np.ndarray
        The points at which to evaluate the Gaussian function.
    mu : float
        The mean of the Gaussian function.
    height : float
        The height of the Gaussian function.
    width : float
        The width of the Gaussian function.
    kurt : float
        The kurtosis of the Gaussian function.
    undershoot : float
        The undershoot as a fraction of the height of the first Gaussian.

    Returns
    -------
    np.ndarray
        The value of the Gaussian function at the given points.
    """
    main_gauss = monophasic_chirp(x, mu, height, width, kurt)


    # lock undershoot mu to half of descending slope of main_gauss
    max_slope = np.argmin(np.diff(main_gauss))
    new_mu = (x[max_slope] + width / 2) / 2

    # lock undershoot width below 30 ms
    new_width = width / 2

    # Create undershoot Gaussian
    undershoot_gauss = monophasic_chirp(
        x, new_mu, height * undershoot, new_width, 1
    )
    return main_gauss - undershoot_gauss


def rise_demo() -> None:
    """Plot a rise."""
    fs = 44100
    t = np.arange(0, 120, 1.0 / fs)
    mus = [10, 20, 60]
    heigths = [20, 40, 60]
    rise_tau = 0.001
    decay_taus = [5, 10, 15]

    fig, ax = plt.subplots(
        len(decay_taus), len(mus), figsize=(10, 10), sharex=True, sharey=True
    )
    for i, decay_tau in enumerate(decay_taus):
        for j, height in enumerate(heigths):
            rises = []
            for mu in mus:
                rises.append(rise(t, mu, height, rise_tau, decay_tau))
            rises = np.sum(rises, axis=0)

            ax[i, j].plot(t, rises)
            ax[i, j].set_xlim(0, 120)
            ax[i, j].set_title(f"height: {height}, decay_tau: {decay_tau}")
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.show()


def chirp_demo() -> None:
    """Plot a chirp."""
    fs = 44100
    t = np.arange(-3.0, 3.0, 1.0 / fs)
    mu = 0
    height = 200
    widths = [0.02, 0.05, 0.2, 0.4]
    kurts = [1, 2, 3, 4]
    undershoot = 0.2

    fig, ax = plt.subplots(len(widths), len(kurts), figsize=(10, 10))

    for i, width in enumerate(widths):
        for j, kurt in enumerate(kurts):
            ax[i, j].plot(
                t, biphasic_chirp(t, mu, height, width, kurt, undershoot)
            )
            ax[i, j].plot(t, monophasic_chirp(t, mu, height, width, kurt))
            ax[i, j].set_xlim(-0.3, 0.3)
            ax[i, j].set_title(f"width: {width}, kurt: {kurt}")
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    fig.legend(["Biphasic", "Monophasic"], fancybox=False, framealpha=0)
    plt.show()


if __name__ == "__main__":
    rise_demo()
    chirp_demo()
