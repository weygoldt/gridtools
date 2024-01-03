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
