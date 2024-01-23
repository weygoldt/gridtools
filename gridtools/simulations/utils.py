"""Utility functions for grid simulation."""

import numpy as np

rng = np.random.default_rng(42)


def get_random_timestamps(
    start_t: float, stop_t: float, n_timestamps: int, min_dt: float
) -> np.ndarray:
    """Generate an array of random timestamps between start_t and stop_t.

    ... with a minimum time difference of min_dt.

    Parameters
    ----------
    start_t : float
        The start time for the timestamps.
    stop_t : float
        The stop time for the timestamps.
    n_timestamps : int
        The number of timestamps to generate.
    min_dt : float
        The minimum time difference between timestamps.

    Returns
    -------
    numpy.ndarray
        An array of random timestamps between start_t and stop_t with a
        minimum time difference of min_dt.
    """
    # Check for start_t > stop_t
    if start_t >= stop_t:
        msg = f"start_t must be less than stop_t: {start_t} >= {stop_t}"
        raise ValueError(msg)

    # Check for n_timestamps < 0
    if n_timestamps < 0:
        msg = f"n_timestamps must be greater than 0: {n_timestamps} < 0"
        raise ValueError(msg)

    # Check for min_dt < 0
    if min_dt < 0:
        msg = f"min_dt must be greater than 0: {min_dt} < 0"
        raise ValueError(msg)

    # Check if min_dt is larger than the time difference between start_t and
    # stop_t
    if min_dt > (stop_t - start_t):
        msg = (
            f"min_dt must be less than stop_t - start_t:"
            f"{min_dt} > {stop_t - start_t}"
        )
        raise ValueError(msg)

    # Generate all timestamps at once
    timestamps = np.sort(rng.uniform(start_t, stop_t, n_timestamps))

    # Adjust timestamps to meet the minimum time difference criteria
    time_diffs = np.diff(timestamps)
    invalid_indices = np.where(time_diffs < min_dt)[0]

    while len(invalid_indices) > 0:
        num_invalid = len(invalid_indices)
        new_timestamps = rng.uniform(start_t, stop_t, num_invalid)
        timestamps[invalid_indices] = new_timestamps
        timestamps.sort()
        time_diffs = np.diff(timestamps)
        invalid_indices = np.where(time_diffs < min_dt)[0]

    return timestamps


def get_width_heigth(time: np.ndarray, chirp: np.ndarray) -> tuple:
    """Estimate the heigth and width of a chirp.

    Parameters
    ----------
    time : numpy.ndarray
        The time array for the chirp.
    chirp : numpy.ndarray
        The chirp signal.

    Returns
    -------
    height : float
        The height of the chirp.
    duration : float
        The duration of the chirp.
    indices : tuple
        The indices where the width was measured at.
    """
    # height is easy
    height = np.max(chirp)
    # width as the duration at 20% of the height
    cutoff_chirp = chirp.copy()
    cutoff_chirp[np.where(cutoff_chirp < 0.5 * height)] = 0
    cutoff_chirp[np.where(cutoff_chirp > 0)] = 1
    # get transitions from 0 to 1 and from 1 to 0
    transitions = np.diff(cutoff_chirp)
    # get the indices of the transitions
    indices = np.where(transitions != 0)[0]
    if len(indices) < 2:
        return height, np.nan, np.nan
    # now from the indices, descend to the median of the signal
    md = np.median(chirp)
    # descend down the left side to the median
    for i in range(indices[0], 0, -1):
        if chirp[i] < md:
            indices[0] = i
            break
    # descend down the right side to the median
    for i in range(indices[-1], len(chirp)):
        if chirp[i] < md:
            indices[-1] = i
            break
    # duration is the difference between the two indices on time
    duration = (time[indices[-1]] - time[indices[0]])

    # fig, ax = plt.subplots()
    # ax.plot(time, chirp, c="C0")
    # ax.axhline(md, c="C1", linestyle="--")
    # ax.axvline(time[indices[0]], c="C1")
    # ax.axvline(time[indices[-1]], c="C1")
    # ax.set_xlabel("Time [s]")
    # ax.set_ylabel("Height [Hz]")
    # ax.set_title(f"Width: {duration:.3f} s")
    # plt.show()
    # print(f"Width: {duration:.3f} s")
    # print(f"Height: {height:.3f} Hz")
    # print(f"Indices: {indices[0]}, {indices[-1]}")
    return height, duration, indices
