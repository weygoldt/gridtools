"""
Simulations to generate electrode grids and simulate fish movement 
and electric organ discharge (EOD) waveforms, including chirps and rises.
Most of the code concerning EOD generation is just slightly modified from 
the original code by Jan Benda et al. in the thunderfish package. The original 
code can be found here: https://github.com/janscience/thunderfish
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)
from scipy.interpolate import interp1d
from scipy.stats import gamma, norm

np.random.seed(42)


@dataclass
class MovementParams:
    """
    All parameters for simulating the movement of a fish.
    """

    duration: int = 1200
    origin: Tuple[float, float] = (0, 0)
    boundaries: Tuple[float, float, float, float] = (-5, 5, -5, 5)
    forward_s: float = 0.2
    backward_s: float = 0.1
    backward_h: float = 0.1
    mode_veloc: float = 0.2
    max_veloc: float = 1
    measurement_fs: float = 30
    target_fs: int = 30


@dataclass
class ChirpParams:
    """
    All parameters for simulating the frequency trace of chirps.
    """

    eodf: float = 0
    samplerate: float = 44100.0
    duration: float = 0.2
    chirp_times: List[float] = field(default_factory=lambda: np.array([0.1]))
    chirp_sizes: List[float] = field(default_factory=lambda: np.array([100.0]))
    chirp_widths: List[float] = field(default_factory=lambda: np.array([0.01]))
    chirp_undershoots: List[float] = field(
        default_factory=lambda: np.array([0.1])
    )
    chirp_kurtosis: List[float] = field(default_factory=lambda: np.array([1.0]))
    chirp_contrasts: List[float] = field(
        default_factory=lambda: np.array([0.05])
    )


@dataclass
class RiseParams:
    """
    All parameters for simulating the frequency trace of rises.
    """

    eodf: float = 0
    samplerate: float = 44100.0
    duration: float = 5.0
    rise_times: List[float] = field(default_factory=lambda: np.array([0.5]))
    rise_sizes: List[float] = field(default_factory=lambda: np.array([80.0]))
    rise_taus: List[float] = field(default_factory=lambda: np.array([0.01]))
    decay_taus: List[float] = field(default_factory=lambda: np.array([0.1]))


def direction_pdf(
    forward_s: float,
    backward_s: float,
    backward_h: float,
    measurement_fs: int = 30,
    target_fs: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Probability density function of the directions a fish can take.

    This is currently a mere approximation of the real probability density
    function. In the future, we should fit a probability density function
    to the real data.

    Parameters
    ----------
    forward_s : _type_
        Standard deviation of the forward direction.
    backward_s : _type_
        Standard deviation of the backward direction.
    backward_h : _type_
        Height of the curve for the backward direction.
    measurement_fs : int, optional
        Samplerate to simulate steps, by default 30
    target_fs : int, optional
        Samplerate to interpolate results to, by default 3

    Returns
    -------
    np.ndarray
        Directions and probabilities for each direction.
    """
    forward_s = np.sqrt((1 / target_fs) / (1 / measurement_fs)) * forward_s
    backward_s = np.sqrt((1 / target_fs) / (1 / measurement_fs)) * backward_s

    directions = np.arange(0, 2 * np.pi, 0.0001)
    p_forward1 = norm.pdf(directions, 0, forward_s)
    p_forward2 = norm.pdf(directions, np.max(directions), forward_s)
    p_backward = norm.pdf(directions, np.pi, backward_s) * backward_h
    probabilities = (p_forward1 + p_forward2 + p_backward) / np.sum(
        p_forward1 + p_forward2 + p_backward
    )

    return directions, probabilities


def step_pdf(max_veloc: float, duration: int, target_fs: int = 3) -> np.ndarray:
    """
    Generate a sequence of steps representing the steps of a random walker.
    Step lengths are drawn from a gamma distribution.

    Parameters
    ----------
    peak_veloc : float
        Peak velocity of the step function (mode of the gamma distribution).
    max_veloc : float
        Maximum velocity of the step function (maximum value of the gamma distribution).
    duration : int
        Duration of the step function in seconds.
    target_fs : int, optional
        Sampling frequency of the step function (default is 3 Hz).

    Returns
    -------
    np.ndarray
        Array of step values.

    """

    # this is just a random gamma distribution
    # in the future, fit one to the real data and use that one
    g = gamma.rvs(a=5, scale=1, size=(duration * target_fs) - 1)

    # scale the gamma distribution to the desired peak velocity
    g = g * (max_veloc / np.max(g))

    # scale the meters per second to the time step specified by the
    # target sampling frequency
    g = g * (1 / target_fs)

    return g


def make_steps(params: MovementParams):
    """
    Makes steps (distance) and directions (trajectories) of a random walker
    in a 2D space.

    Parameters
    ----------
    duration : int, optional
        Total duration of simulation, by default 600
    fs : int, optional
        Sampling rate, by default 30
    species : str, optional
        The species key, by default "Alepto"
    plot : bool, optional
        Enable or disable plotting the probability distributions
        underlying the parameters, by default False

    Returns
    -------
    Tuple(np.ndarray, np.ndarray)
        Tuple of trajectories and steps.

    """
    # get the probability distribution of directions
    directions, probabilities = direction_pdf(
        params.forward_s,
        params.backward_s,
        params.backward_h,
        params.measurement_fs,
        params.measurement_fs,
    )

    # make random step lengths according to a gamma distribution
    steps = step_pdf(
        params.max_veloc,
        params.duration,
        params.measurement_fs,
    )

    # draw random directions according to the probability distribution
    trajectories = np.random.choice(
        directions,
        size=(params.duration * params.measurement_fs) - 1,
        p=probabilities,
    )

    return trajectories, steps


def make_positions(
    trajectories: np.ndarray,
    steps: np.ndarray,
    params: MovementParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates a random walk with a given set of trajectories and step sizes.
    Given an origin position, boundaries, a set of trajectories, and a set of
    step sizes, this function computes the final x and y positions of the
    trajectories after taking steps and folding back to the boundaries.

    Parameters
    ----------
    origin : Tuple[float, float]
        The (x, y) starting position of the agent.

    boundaries : Tuple[float, float, float, float]
        The minimum and maximum x and y positions allowed for the positions.
        Everything outside these boundaries will be folded back to the boundaries.

    trajectories : np.ndarray
        A 1D array of angle values in radians specifying the direction of each
        step in the trajectory.

    steps : np.ndarray
        A 1D array of step sizes for each step in the trajectory.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two 1D arrays representing the final x and y
        positions of the trajectories.
    """
    origin = params.origin
    boundaries = params.boundaries

    x = np.full(len(trajectories) + 1, np.nan)
    y = np.full(len(trajectories) + 1, np.nan)
    x[0] = origin[0]
    y[0] = origin[1]

    for i, _ in enumerate(trajectories):
        # use the first trajectory as is
        if i == 0:
            converted_trajectory = trajectories[i]

        # make all other trajectories relative to the previous one
        else:
            converted_trajectory = trajectories[i - 1] - trajectories[i]

            # make sure the trajectory is between 0 and 2pi
            if converted_trajectory > 2 * np.pi:
                converted_trajectory = converted_trajectory - 2 * np.pi
            if converted_trajectory < 0:
                converted_trajectory = converted_trajectory + 2 * np.pi

            # write current trajectory to trajectories to correct
            # future trajectories relative to the current one
            trajectories[i] = converted_trajectory

        # use trigonometric identities to calculate the x and y positions
        y[i + 1] = np.sin(converted_trajectory) * steps[i]
        x[i + 1] = np.cos(converted_trajectory) * steps[i]

    # cumulatively add the steps to the positions
    x = np.cumsum(x)
    y = np.cumsum(y)

    # interpolate the position to the target fs
    time_sim = np.arange(0, params.duration, 1 / params.measurement_fs)
    time_target = np.arange(0, params.duration, 1 / params.target_fs)
    xinterper = interp1d(time_sim, x, kind="cubic", fill_value="extrapolate")
    yinterper = interp1d(time_sim, y, kind="cubic", fill_value="extrapolate")

    x = xinterper(time_target)
    y = yinterper(time_target)

    # fold back the positions if they are outside the boundaries
    boundaries = np.ravel(boundaries)
    while (
        np.any(x < boundaries[0])
        or np.any(x > boundaries[1])
        or np.any(y < boundaries[2])
        or np.any(y > boundaries[3])
    ):
        x[x < boundaries[0]] = boundaries[0] + (
            boundaries[0] - x[x < boundaries[0]]
        )
        x[x > boundaries[1]] = boundaries[1] - (
            x[x > boundaries[1]] - boundaries[1]
        )
        y[y < boundaries[2]] = boundaries[2] + (
            boundaries[2] - y[y < boundaries[2]]
        )
        y[y > boundaries[3]] = boundaries[3] - (
            y[y > boundaries[3]] - boundaries[3]
        )

    return x, y


def gaussian(
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
    curve = height * np.exp(-0.5 * (((x - mu) / sigma) ** 2.0) ** kurt)
    return curve


def chirp_model_v2(x, m1, h1, w1, k1, m2, h2, w2, k2, m3, h3, w3, k3):
    """Model a chirp as a combination of 3 Gaussians.

    This model can be fit to the extracted instantaneous frequencies of real
    chirps and is not supposed to be tuned "by hand".

    This is the most complex model and is not that useful because of the
    12 dimensonal parameter space. What motivated this model is the fact that
    while the averaged chirp is most of the time a single Gaussian with a small
    undershoot, the instantaneous frequency of real chirps rarely looks like
    that. Instead, the instantaneous frequency of real chirps come in a variety
    of shapes, to which a more complex model is better conform to. And when the
    aim is to model realistic chirps, this variation in shape is important.

    Parameters
    ----------
    x : np.array
        The time axis.
    m1 : float
        The mean of the first Gaussian
    h1 : float
        The height of the first Gaussian
    w1 : float
        The width of the first Gaussian
    k1 : float
        The kurtosis of the first Gaussian
    m2 : float
        The mean of the second Gaussian
    h2 : float
        The height of the second Gaussian
    w2 : float
        The width of the second Gaussian
    k2 : float
        The kurtosis of the second Gaussian
    m3 : float
        The mean of the third Gaussian
    h3 : float
        The height of the third Gaussian
    w3 : float
        The width of the third Gaussian
    k3 : float
        The kurtosis of the third Gaussian

    Returns
    -------
    np.array
        The modeled chirp
    """
    g1 = gaussian(x, m1, h1, w1, k1)
    g2 = gaussian(x, m2, h2, w2, k2)
    g3 = gaussian(x, m3, h3, w3, k3)
    return g1 + g2 + g3


def chirp_model_v3(x, m1, h1, w1, k1, m2, h2, w2, k2):
    """Model a chirp as a combination of 2 Gaussians. This model
    can be fit to the extracted instantaneous frequencies of real chirps and
    is not supposed to be tuned "by hand".

    Parameters
    ----------
    x : np.array
        The time axis.
    m1 : float
        The mean of the first Gaussian
    h1 : float
        The height of the first Gaussian
    w1 : float
        The width of the first Gaussian
    k1 : float
        The kurtosis of the first Gaussian
    m2 : float
        The mean of the second Gaussian
    h2 : float
        The height of the second Gaussian
    w2 : float
        The width of the second Gaussian
    k2 : float
        The kurtosis of the second Gaussian

    Returns
    -------
    np.array
        The modeled chirp
    """
    g1 = gaussian(x, m1, h1, w1, k1)
    g2 = gaussian(x, m2, h2, w2, k2)
    return g1 + g2


def chirp_model_v4(x, m1, h1, w1, k1):
    """Model a chirp as a combination of 1 Gaussians. This model
    can be fit to the extracted instantaneous frequencies of real chirps and
    is not supposed to be tuned "by hand".

    It is overly simplistic and does not capture the variation in shape of
    real chirps. But on the other hand, it is fast and easy to fit
    and controlling the parameters is intuitive.

    Parameters
    ----------
    x : np.array
        The time axis.
    m1 : float
        The mean of the first Gaussian
    h1 : float
        The height of the first Gaussian
    w1 : float
        The width of the first Gaussian
    k1 : float
        The kurtosis of the first Gaussian

    Returns
    -------
    np.array
        The modeled chirp
    """
    return gaussian(x, m1, h1, w1, k1)


def make_chirps(
    eodf: float,
    samplerate: float,
    duration: float,
    chirp_times: np.ndarray,
    chirp_contrasts: np.ndarray,
    chirp_params: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a chirp frequency trace from a fitted chirp model.

    This function essentially does the same as make_simple_chirps(), but
    the model that generates the chirps cannot be fine tuned by hand. Instead,
    the model is fitted to the extracted instantaneous frequencies of real
    chirps. Use this function if this kind of data is available.

    Parameters
    ----------
    eodf : float
        Baseline EOD frequency of the fish.
    samplerate : float
        Samplerate of the frequency trace.
    duration : float
        Duration of the frequency trace.
    chirp_times : np.ndarray
        Times where chirps should be added to the frequency trace.
    chirp_contrasts : np.ndarray
        Contrasts of the chirps.
    chirp_params : np.ndarray
        Array of tuples, each containing the parameters for the chirp model.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Chirp frequency trace and amplitude modulation.
    """

    n = int(duration * samplerate)
    f = eodf * np.ones(n)
    a = np.ones(n)

    for time, contrasts, params in zip(
        chirp_times, chirp_contrasts, chirp_params
    ):
        chirp_t = np.arange(-0.5, 0.5, 1.0 / samplerate)
        chirp_f = chirp_model(chirp_t, *params)

        index = int(time * samplerate)
        i0 = index - len(chirp_f) // 2
        i1 = i0 + len(chirp_f)
        gi0 = 0
        gi1 = len(chirp_f)
        if i0 < 0:
            gi0 -= i0
            i0 = 0
        if i1 >= len(f):
            gi1 -= i1 - len(f)
            i1 = len(f)
        f[i0:i1] += chirp_f[gi0:gi1]
        a[i0:i1] -= contrasts * chirp_f[gi0:gi1] / np.max(chirp_f)

    return f, a


def make_simple_chirps(params: ChirpParams) -> tuple[np.ndarray, np.ndarray]:
    """Simulate frequency trace with chirps. Original code by Jan Benda et al.

    A chirp is modeled as a combination of 2 Gaussians. This model is used to
    simulate chirps "by hand". The first Gaussian is
    centered at the chirp time and has a width of chirp_width. The second Gaussian
    is centered at chirp_time + chirp_width / 2 and has the same width but a
    smaller amlitude determined by chirp_undershoot, which is a factor that
    is multiplied with the amplitude of the first Gaussian.

    Parameters
    ----------
    params : ChirpParams
        Parameters for simulating the frequency trace with chirps.

    Returns
    -------
    frequency : np.ndarray
        Generated frequency trace that can be passed on to wavefish_eods().
    amplitude : np.ndarray
        Generated amplitude modulation that can be used to multiply the trace
        generated by wavefish_eods().
    """
    n = int(params.duration * params.samplerate)
    frequency = params.eodf * np.ones(n)
    amplitude = np.ones(n)

    for time, width, undershoot, size, kurtosis, contrast in zip(
        params.chirp_times,
        params.chirp_widths,
        params.chirp_undershoots,
        params.chirp_sizes,
        params.chirp_kurtosis,
        params.chirp_contrasts,
    ):
        chirp_t = np.arange(-3.0 * width, 3.0 * width, 1.0 / params.samplerate)
        g1 = gaussian(chirp_t, mu=0, height=size, width=width, kurt=kurtosis)
        g2 = gaussian(
            chirp_t, mu=width / 2, height=size * undershoot, width=width, kurt=1
        )
        gauss = g1 - g2

        index = int(time * params.samplerate)
        i0 = index - len(gauss) // 2
        i1 = i0 + len(gauss)
        gi0 = 0
        gi1 = len(gauss)
        if i0 < 0:
            gi0 -= i0
            i0 = 0
        if i1 >= len(frequency):
            gi1 -= i1 - len(frequency)
            i1 = len(frequency)
        frequency[i0:i1] += gauss[gi0:gi1]
        amplitude[i0:i1] -= contrast * gauss[gi0:gi1] / size

    return frequency, amplitude


def make_rises(params: RiseParams) -> np.ndarray:
    """
    Simulate frequency trace with rises. Original code by Jan Benda et al.

    A rise is modeled as a double exponential frequency modulation.

    Parameters
    ----------
    params : RisesParams
        A dataclass containing the parameters for simulating the frequency trace
        with rises.

    Returns
    -------
    numpy.ndarray
        Generate frequency trace that can be passed on to wavefish_eods().
    """

    n = int(params.duration * params.samplerate)

    # baseline eod frequency:
    frequency = params.eodf * np.ones(n)

    for time, size, riset, decayt in zip(
        params.rise_times,
        params.rise_sizes,
        params.rise_taus,
        params.decay_taus,
    ):
        # rise frequency waveform:
        rise_t = np.arange(0.0, 10.0 * decayt, 1.0 / params.samplerate)
        rise = size * (1.0 - np.exp(-rise_t / riset)) * np.exp(-rise_t / decayt)

        # add rises on baseline eodf:
        index = int(time * params.samplerate)
        if index + len(rise) > len(frequency):
            rise_index = len(frequency) - index
            frequency[index : index + rise_index] += rise[:rise_index]
            break
        else:
            frequency[index : index + len(rise)] += rise
    return frequency


def make_grid(origin, shape, spacing, style="hex"):
    """
    Simulate a grid of electrodes as points in space. The grid can be either
    hexagonal or square. The grid is centered around the origin. Currently,
    only symmetric grids with an even number of electrodes are supported.

    Parameters
    ----------
    origin : tuple
        Origin of the grid.
    shape : tuple
        The number of electrodes for the x and y demensions.
    spacing : float
        Distance between electrodes.
    type : str, optional
        Simulate a grid with a square or hexagonal electrode arrangement,
        by default "hex"

    Returns
    -------
    np.ndarray
        x coordinates of the electrodes and y coordinates of the electrodes.
    """

    assert style in ["hex", "square"], "type must be 'hex' or 'square'"
    assert (shape[0] % 2 == 0) or (style == "square"), "shape must be even"

    # grid parameters
    electrode_number = shape[0] * shape[1]
    electrode_index = np.arange(0, electrode_number)
    electrode_x = np.mod(electrode_index, shape[0]) * spacing
    electrode_y = np.floor(electrode_index / shape[0]) * spacing

    if style == "hex":
        # shift every second row to make a hexagonal grid
        electrode_y[1::2] += spacing / 2

    # shift the grid to the specified origin
    electrode_x += origin[0] - np.mean(electrode_x)
    electrode_y += origin[1] - np.mean(electrode_y)

    return electrode_x, electrode_y


def movement_demo():
    """
    Plot some simulations of moving fish.
    """
    n_fish = 3
    fig, ax = plt.subplots(constrained_layout=True)
    params = MovementParams()

    for _ in range(n_fish):
        trajectories, s = make_steps(params)
        x, y = make_positions(trajectories, s, params)
        ax.plot(x, y, marker=".")

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    fig.suptitle("Fish movement simulations")
    plt.show()


def communication_demo():
    """
    Demo of chirp and rise simulations.
    """
    cp = ChirpParams()
    rp = RiseParams()
    cf, ca = make_simple_chirps(cp)
    rf = make_rises(rp)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True)
    ax1.plot(cf)
    ax1.set_title("Frequency trace of chirp")
    ax2.plot(ca)
    ax2.set_title("Amplitude modulation of chirp")
    ax3.plot(rf)
    ax3.set_title("Frequency trace of rise")
    fig.suptitle("Chirp and rise simulations")
    plt.show()


def grid_demo():
    """
    Plot some simulations of electrode grid arrangements.
    """
    grid1 = make_grid((0, 0), (6, 6), 1.2, style="square")
    grid2 = make_grid((0, 0), (6, 6), 1.2, style="hex")
    grid4 = make_grid((0, 0), (16, 16), 0.5, style="square")
    grid3 = make_grid((0, 0), (16, 16), 0.5, style="hex")

    fig, ax = plt.subplots(2, 2, constrained_layout=True)
    ax[0, 0].scatter(grid1[0], grid1[1], marker=".")
    ax[1, 0].scatter(grid2[0], grid2[1], marker=".")
    ax[1, 1].scatter(grid3[0], grid3[1], marker=".")
    ax[0, 1].scatter(grid4[0], grid4[1], marker=".")

    for a in ax.ravel():
        a.set_xlim(-5, 5)
        a.set_ylim(-5, 5)
        a.set_aspect("equal")
    fig.suptitle("Electrode grid arrangements")

    plt.show()


def main():
    """
    Show some demos.
    """
    movement_demo()
    communication_demo()
    grid_demo()


if __name__ == "__main__":
    main()
