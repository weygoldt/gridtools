"""Order-free statistics of per-electrode power profiles (``sign_v``).

The mapping from recording channel to electrode position is not known
reliably for the tube competition recordings (see :mod:`gridtools.io.geometry`).
Everything here is therefore invariant to permutations of the electrodes:

- *sorted relative profiles*: each detection's powers sorted in descending
  order, minus the maximum (rank 1 is 0 dB, rank 2 is the drop to the second
  strongest electrode, and so on);
- *strongest-electrode sequences*: which electrode is strongest, compared
  only for equality over time (switches and dwell times), as a movement
  proxy that needs no positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "DwellStats",
    "dwell_stats",
    "profile_quantiles",
    "relative_sorted_profiles",
    "strongest_electrode",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def relative_sorted_profiles(powers_db: npt.ArrayLike) -> FloatArray:
    """Sort each row in descending order and subtract its maximum.

    Parameters
    ----------
    powers_db : array_like, shape (n, E)
        Per-electrode powers in dB (rows with non-finite values are kept;
        NaN sorts last).

    Returns
    -------
    numpy.ndarray, shape (n, E)
        Column 0 is 0; column k is the power of the (k+1)-th strongest
        electrode relative to the strongest, in dB (<= 0).
    """
    p = np.asarray(powers_db, dtype=np.float64)
    if p.ndim != 2:
        msg = f"powers_db must be 2-D, got shape {p.shape}"
        raise ValueError(msg)
    ordered = -np.sort(-p, axis=1)
    return ordered - ordered[:, :1]


def profile_quantiles(profiles: npt.ArrayLike, quantiles: npt.ArrayLike) -> FloatArray:
    """Quantiles of each rank of sorted relative profiles.

    Parameters
    ----------
    profiles : array_like, shape (n, E)
        Output of :func:`relative_sorted_profiles`.
    quantiles : array_like, shape (Q,)

    Returns
    -------
    numpy.ndarray, shape (Q, E)
    """
    return np.nanquantile(
        np.asarray(profiles, dtype=np.float64),
        np.asarray(quantiles, dtype=np.float64),
        axis=0,
    )


def strongest_electrode(
    powers_db: npt.ArrayLike, hysteresis_db: float = 0.0
) -> IntArray:
    """Index of the strongest electrode per row, with hysteresis.

    The current electrode is kept until another one exceeds it by more than
    ``hysteresis_db``, which suppresses flicker between two electrodes of
    nearly equal power (a fish midway between them).

    Parameters
    ----------
    powers_db : array_like, shape (n, E)
    hysteresis_db : float
        Margin in dB (0 = plain argmax).

    Returns
    -------
    numpy.ndarray of int64, shape (n,)
    """
    p = np.asarray(powers_db, dtype=np.float64)
    best = np.argmax(np.where(np.isfinite(p), p, -np.inf), axis=1).astype(np.int64)
    if hysteresis_db <= 0.0 or p.shape[0] == 0:
        return best
    out = np.empty_like(best)
    current = int(best[0])
    rows = np.arange(p.shape[0])
    best_power = p[rows, best]
    for i in range(p.shape[0]):
        if best[i] != current and best_power[i] > p[i, current] + hysteresis_db:
            current = int(best[i])
        out[i] = current
    return out


@dataclass(frozen=True)
class DwellStats:
    """Switches of the strongest electrode and the dwell times between them.

    Attributes
    ----------
    duration_s : float
        Time covered by valid frames (gaps longer than ``max_gap_s`` are
        excluded).
    n_switches : int
    switches_per_min : float
    dwell_times_s : numpy.ndarray
        Durations of completed dwells (both ends observed, no gap inside).
    still_fraction : float
        Fraction of ``duration_s`` spent in dwells of at least
        ``still_min_s``.
    """

    duration_s: float
    n_switches: int
    switches_per_min: float
    dwell_times_s: FloatArray
    still_fraction: float


def dwell_stats(
    times_s: npt.ArrayLike,
    sequence: npt.ArrayLike,
    *,
    max_gap_s: float = 2.0,
    still_min_s: float = 60.0,
) -> DwellStats:
    """Count switches of a label sequence and measure dwell times.

    Parameters
    ----------
    times_s : array_like, shape (n,)
        Increasing frame times.
    sequence : array_like of int, shape (n,)
        Label per frame (e.g. from :func:`strongest_electrode`).
    max_gap_s : float
        Frames further apart than this break the sequence: no switch is
        counted across the gap and dwells touching it are not completed.
    still_min_s : float
        Minimum dwell for :attr:`DwellStats.still_fraction`.

    Returns
    -------
    DwellStats
    """
    t = np.asarray(times_s, dtype=np.float64)
    s = np.asarray(sequence)
    if t.size < 2:
        return DwellStats(0.0, 0, float("nan"), np.zeros(0), float("nan"))
    dt = np.diff(t)
    joined = dt <= max_gap_s
    duration = float(np.sum(dt[joined]))
    changed = (s[1:] != s[:-1]) & joined
    n_switches = int(np.count_nonzero(changed))
    # Boundaries of runs: label changes or gaps.
    breaks = np.flatnonzero((s[1:] != s[:-1]) | ~joined) + 1
    starts = np.concatenate([[0], breaks])
    stops = np.concatenate([breaks, [t.size]])
    durations = []
    still_time = 0.0
    for a, b in zip(starts, stops, strict=True):
        run_duration = float(t[b - 1] - t[a])
        if run_duration >= still_min_s:
            still_time += run_duration
        # Completed: entered and left by a label change, not by a gap or edge.
        entered = a > 0 and joined[a - 1]
        left = b < t.size and joined[b - 1]
        if entered and left:
            durations.append(float(t[b] - t[a]))
    return DwellStats(
        duration_s=duration,
        n_switches=n_switches,
        switches_per_min=60.0 * n_switches / duration if duration > 0 else float("nan"),
        dwell_times_s=np.asarray(durations),
        still_fraction=still_time / duration if duration > 0 else float("nan"),
    )
