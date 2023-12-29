"""Simulate movement of fish."""

from typing import Tuple
from dataclasses import dataclass

import numpy as np
from numba import jit
from scipy.interpolate import interp1d
from scipy.stats import gamma, norm
from rich.console import Console
import matplotlib.pyplot as plt

from gridtools.utils.logger import Timer


con = Console()


@dataclass
class MovementParams:
    """All parameters for simulating the movement of a fish."""

    duration: float = 30
    origin: Tuple[float, float] = (0, 0)
    boundaries: Tuple[float, float, float, float] = (-5, -5, 5, 5)
    forward_s: float = 0.2
    backward_s: float = 0.1
    backward_h: float = 0.2
    mode_veloc: float = 0.2
    max_veloc: float = 1
    measurement_fs: float = 30
    target_fs: int = 30


def direction_pdf(
    forward_s: float,
    backward_s: float,
    backward_h: float,
    fs: int = 10000,  # * 2 * np.pi = Number of possible directions
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
        Height of the curve for the backward direction relative to the forward
        direction, e.g. 0.5 when the backward direction is half as likely as
        the forward direction.
    measurement_fs : int, optional
        Samplerate to simulate steps, by default 30
    target_fs : int, optional
        Samplerate to interpolate results to, by default 3

    Returns
    -------
    np.ndarray
        Directions and probabilities for each direction.
    """
    assert forward_s > 0, "forward_s must be greater than 0"
    assert backward_s > 0, "backward_s must be greater than 0"
    assert backward_h >= 0, "backward_h must be greater than or equal to 0"

    directions = np.arange(0, 2 * np.pi, 1 / fs)
    p_forward1 = norm.pdf(directions, 0, forward_s)
    p_forward2 = norm.pdf(directions, np.max(directions), forward_s)
    p_backward = norm.pdf(directions, np.pi, backward_s) * backward_h
    probabilities = (p_forward1 + p_forward2 + p_backward) / np.sum(
        p_forward1 + p_forward2 + p_backward
    )

    return directions, probabilities


def step_pdf(
        max_veloc: float, duration: int, target_fs: int = 30
    ) -> np.ndarray:
    """Generate a distribution of steps lengths of a random walker.

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
    assert max_veloc > 0, "max_veloc must be greater than 0"
    assert duration > 0, "duration must be greater than 0"
    assert target_fs > 0, "target_fs must be greater than 0"
    assert isinstance(target_fs, int), "target_fs must be an integer"

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
    """Simulate a random walked based on step distance and trajectory.

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
    origin: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert sequences of step lengths and trajectories to positions.

    Simulates a random walk with a given set of trajectories and step sizes.
    Given an origin position, boundaries, a set of trajectories, and a set of
    step sizes, this function computes the final x and y positions of the
    trajectories after taking steps and folding back to the boundaries.

    Parameters
    ----------
    origin : Tuple[float, float]
        The (x, y) starting position of the agent.

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
    assert len(trajectories) == len(
        steps
    ), "trajectories and steps must be of equal length"

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

    return x, y


def interpolate_positions(
    x: np.ndarray, y: np.ndarray, t_tot: float, fss: int, fst: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate positions to a different sampling frequency.

    This is particularly useful to modulate a simulated EOD amplitude
    on multiple electrodes because the positions should then ideally be
    sampled in the same way as the EOD amplitude.

    Parameters
    ----------
    - `x` : `np.ndarray`
        The x positions.
    - `y` : `np.ndarray`
        The y positions.
    - `t_tot` : `float`
        The total duration of the simulation.
    - `fss` : `int`
        The sampling frequency of the simulation.
    - `fst` : `int`
        The sampling frequency to interpolate to.

    Returns
    -------
    - `Tuple[np.ndarray, np.ndarray]`
        The interpolated x and y positions.
    """
    assert len(x) == len(y), "x and y must be of equal length"

    time_sim = np.arange(0, t_tot, 1 / fss)
    time_target = np.arange(0, t_tot, 1 / fst)

    xinterper = interp1d(time_sim, x, kind="cubic", fill_value="extrapolate")
    yinterper = interp1d(time_sim, y, kind="cubic", fill_value="extrapolate")

    x = xinterper(time_target)
    y = yinterper(time_target)
    return x, y


@jit(nopython=True, parallel=True)
def fold_space(
    x: np.ndarray, y: np.ndarray, boundaries: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Fold back positions that are outside the boundaries.

    This function folds the space in which x and y are defined back to the
    boundaries. Imagine a 2D space with the x and y axes. If a point is outside
    the boundaries, it is reflected back to the boundaries. For example, if
    the boundaries are (0, 1, 0, 1) and a point is at (1.5, 0.5), it is
    reflected back to (0.5, 0.5). If a point is outside the boundaries in
    multiple dimensions, it is reflected back in all dimensions.

    Parameters
    ----------
    - `x` : `np.ndarray`
        The x positions.
    - `y` : `np.ndarray`
        The y positions.
    - `boundaries` : `Tuple[float, float, float, float]`
        The boundaries of the space in which x and y are defined.

    Returns
    -------
    - `Tuple[np.ndarray, np.ndarray]`
        The folded back x and y positions.
    """
    assert len(x) == len(y), "x and y must be of equal length"

    # Check how narrow the boundaries are because this might take a while then
    limit = 1000
    minx, maxx = np.min(x), np.max(x)
    miny, maxy = np.min(y), np.max(y)
    if (
        minx < boundaries[0] * limit
        or miny < boundaries[1] * limit
        or maxx > boundaries[2] * limit
        or maxy > boundaries[3] * limit
    ):
        msg = (
            "The boundaries are very narrow. "
            "This might take a while to compute."
        )
        raise ValueError(msg)

    # fold back the positions if they are outside the boundaries
    boundaries = np.ravel(boundaries)
    while (
        np.any(x < boundaries[0])
        or np.any(y < boundaries[1])
        or np.any(x > boundaries[2])
        or np.any(y > boundaries[3])
    ):
        x[x < boundaries[0]] = boundaries[0] + (
            boundaries[0] - x[x < boundaries[0]]
        )
        x[x > boundaries[2]] = boundaries[2] - (
            x[x > boundaries[2]] - boundaries[2]
        )
        y[y < boundaries[1]] = boundaries[1] + (
            boundaries[1] - y[y < boundaries[1]]
        )
        y[y > boundaries[3]] = boundaries[3] - (
            y[y > boundaries[3]] - boundaries[3]
        )

    return x, y


def make_grid(
    origin: Tuple[float, float],
    shape: Tuple[int, int],
    spacing: float,
    style: str = "hex",
) -> Tuple[np.ndarray, np.ndarray]:
    """Make x and y coordinates for a grid of electrodes.

    Simulate a grid of electrodes as points in space, each point consisting of
    an x and a y coordinate.

    The grid can be either be hexagonal or square. In a hexagonal grid, the
    electrodes are arranged in triangles. In a square grid, the electrodes are
    arranged in squares. This means, that all electrodes in a hexagonal grid
    have 6 neighbors, while all electrodes in a square grid have 4 neighbors.

    The grid is centered around the origin.

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
    Tuple[np.ndarray, np.ndarray]
        x coordinates of the electrodes and y coordinates of the electrodes.
    """
    assert style in ["hex", "square"], "type must be 'hex' or 'square'"

    if style == "hex":
        x = np.arange(shape[0]) * spacing
        y = np.arange(shape[1]) * spacing
        x, y = np.meshgrid(x, y)
        x[::2] += spacing / 2  # shift every other row
    else:  # square grid
        x = np.arange(shape[0]) * spacing
        y = np.arange(shape[1]) * spacing
        x, y = np.meshgrid(x, y)

    # center the grid around the origin
    x -= origin[0]
    y -= origin[1]
    grid = np.dstack([x, y])
    grid = grid.reshape(-1, 2).T
    xcoords = grid[0]
    ycoords = grid[1]

    return xcoords, ycoords


def movement_demo() -> None:
    """Plot some simulations of moving fish."""
    n_fish = 3
    fig, ax = plt.subplots(constrained_layout=True)
    params = MovementParams()

    for _ in range(n_fish):
        trajectories, s = make_steps(params)
        x, y = make_positions(trajectories, s, params.origin)
        ax.plot(x, y, marker=".")

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    fig.suptitle("Fish movement simulations")
    plt.show()


def grid_demo() -> None:
    """Plot some simulations of electrode grid arrangements."""
    ngrids = 4
    styles = ["square", "hex", "square", "hex"]
    origins = np.array([(0, 0), (0, 0), (0, 0), (0, 0)])
    shapes = np.array([(6, 6), (6, 6), (16, 16), (16, 16)])
    spacings = np.array([1.2, 1.2, 0.5, 0.5])
    grids = []

    for i in range(ngrids):
        grids.append(make_grid(origins[i], shapes[i], spacings[i], styles[i]))

    fig, ax = plt.subplots(1, ngrids, constrained_layout=True)
    for i in range(ngrids):
        ax[i].scatter(grids[i][0], grids[i][1])
        ax[i].set_title(f"{styles[i]} grid")
        ax[i].set_xlim(-5, 5)
        ax[i].set_ylim(-5, 5)
        ax[i].set_aspect("equal")

    maxdim = np.max([np.max(spacings * shape) for shape in shapes.T])
    for a in ax.ravel():
        a.set_xlim(np.min(origins[:, 0]) - 1, maxdim + 1)
        a.set_ylim(np.min(origins[:, 1]) - 1, maxdim + 1)
        a.set_aspect("equal")
    fig.suptitle("Electrode grid arrangements")
    plt.show()


if __name__ == "__main__":
    movement_demo()
    grid_demo()
