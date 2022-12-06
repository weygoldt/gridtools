import numpy as np

from .datahandling import nanPad


def velocity1d(t, d):
    """
    Compute velocity with padding at ends.

    Parameters
    ----------
    t : array-like
        array with time stamps, e.g. in seconds
    d : array-like
        array with distances

    Returns
    -------
    velocity: numpy array
        velocities at time points
    """

    times = t
    dist = d

    # make times
    dt = np.array([x - x0 for x0, x in zip(times, times[2:])])

    # compute distances
    dx = np.array(
        [(x2 - x1) + (x1 - x0) for x0, x1, x2 in zip(dist, dist[1:], dist[2:])]
    )

    # compute velocity, i.e. distance over time
    v = dx / dt

    # add nans to make same dimension as input
    v = nanPad(v, position="center", padlen=1)

    return v


def velocity2d(t, x, y):
    """
    Compute the velocity of an object in 2D space from x and y coordinates over time.

    Parameters
    ----------
    t : array-like
        time axis for coordinates
    x : array-like
        x coordinates of object
    y : array-like
        y coordinates of object

    Returns
    -------
    velocity : numpy array
        velocity of object with same dimension as input (padded with nans).
    """

    # delta t
    dt = np.array([x - x0 for x0, x in zip(t, t[2:])])

    # delta d x and y
    dx = np.array([(x2 - x1) + (x1 - x0)
                  for x0, x1, x2 in zip(x, x[1:], x[2:])])
    dy = np.array([(x2 - x1) + (x1 - x0)
                  for x0, x1, x2 in zip(y, y[1:], y[2:])])

    # delta d tot. (pythagoras)
    dd = np.sqrt(dx**2 + dy**2)

    # velocity & correcsponding time
    v = dd / dt

    # pad to len of original time array with nans
    v = nanPad(v, position="center", padlen=1)

    return v
