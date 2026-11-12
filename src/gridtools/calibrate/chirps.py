"""Chirp timing statistics from detections (e.g. chirpdetector v1).

Detections are biased toward detectable chirps (high SNR, large excursion,
no overlap with the other fish), and v1 assigned emitters with an
unvalidated model; everything here describes *v1-detected, v1-assigned*
chirps, not all chirps.

A chirp's time is the centre of its detection box. v1 did not deduplicate
detections across overlapping analysis windows (audit item D4), so
:func:`merge_duplicates` first merges detections of the same fish that are
closer than a minimum interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "CrossFishStats",
    "coincidence_fraction",
    "cross_fish_stats",
    "merge_duplicates",
    "next_event_latencies",
]

FloatArray = npt.NDArray[np.float64]


def merge_duplicates(
    times_s: npt.ArrayLike, min_interval_s: float = 0.05
) -> tuple[FloatArray, int]:
    """Merge events closer than ``min_interval_s`` (keep the first of a cluster).

    Parameters
    ----------
    times_s : array_like
        Event times (any order).
    min_interval_s : float
        Events within this interval of the previous kept event are dropped.

    Returns
    -------
    times : numpy.ndarray
        Sorted, merged times.
    n_merged : int
        Number of dropped events.
    """
    t = np.sort(np.asarray(times_s, dtype=np.float64))
    if t.size == 0:
        return t, 0
    keep = np.ones(t.size, dtype=bool)
    last = t[0]
    for i in range(1, t.size):
        if t[i] - last < min_interval_s:
            keep[i] = False
        else:
            last = t[i]
    return t[keep], int(np.count_nonzero(~keep))


def next_event_latencies(a: npt.ArrayLike, b: npt.ArrayLike) -> FloatArray:
    """Time from each event of ``a`` to the next event of ``b`` (>= 0).

    Parameters
    ----------
    a, b : array_like
        Sorted event times.

    Returns
    -------
    numpy.ndarray, shape (len(a),)
        NaN where no event of ``b`` follows.
    """
    ta = np.asarray(a, dtype=np.float64)
    tb = np.asarray(b, dtype=np.float64)
    idx = np.searchsorted(tb, ta, side="left")
    out = np.full(ta.size, np.nan)
    ok = idx < tb.size
    out[ok] = tb[idx[ok]] - ta[ok]
    return out


def coincidence_fraction(a: npt.ArrayLike, b: npt.ArrayLike, window_s: float) -> float:
    """Fraction of events of ``a`` with an event of ``b`` within +/- ``window_s``.

    Parameters
    ----------
    a, b : array_like
        Sorted event times.
    window_s : float

    Returns
    -------
    float
        NaN if ``a`` is empty.
    """
    ta = np.asarray(a, dtype=np.float64)
    tb = np.asarray(b, dtype=np.float64)
    if ta.size == 0:
        return float("nan")
    if tb.size == 0:
        return 0.0
    lo = np.searchsorted(tb, ta - window_s, side="left")
    hi = np.searchsorted(tb, ta + window_s, side="right")
    return float(np.mean(hi > lo))


def _circular_shift(
    times: FloatArray, shift_s: float, t0_s: float, duration_s: float
) -> FloatArray:
    return np.sort(t0_s + np.mod(times - t0_s + shift_s, duration_s))


@dataclass(frozen=True)
class CrossFishStats:
    """Timing of one fish's chirps relative to another's.

    Attributes
    ----------
    latency_edges_s : numpy.ndarray, shape (L + 1,)
        Bin edges of the next-chirp latency histogram.
    latency_counts : numpy.ndarray, shape (L,)
        Observed counts of the latency from a chirp of A to the next chirp
        of B.
    latency_null_counts : numpy.ndarray, shape (L,)
        Mean counts under the null: B's whole chirp train shifted by a random
        offset of ``min_shift_s`` to ``max_shift_s`` (either sign, wrapped at
        the ends). This keeps both trains' own structure and their slow
        co-modulation (both fish chirp more during fights) and removes only
        the alignment on time-scales shorter than the shift.
    coincidence : float
        Observed fraction of A's chirps with a chirp of B within
        +/- ``window_s``.
    coincidence_null : float
        Mean of that fraction under the null.
    coincidence_null_sd : float
    window_s : float
    n_a, n_b : int
    """

    latency_edges_s: FloatArray
    latency_counts: FloatArray
    latency_null_counts: FloatArray
    coincidence: float
    coincidence_null: float
    coincidence_null_sd: float
    window_s: float
    n_a: int
    n_b: int


def cross_fish_stats(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    t0_s: float,
    duration_s: float,
    rng: np.random.Generator,
    *,
    latency_edges_s: npt.ArrayLike | None = None,
    window_s: float = 0.25,
    n_shuffles: int = 200,
    min_shift_s: float = 5.0,
    max_shift_s: float = 30.0,
) -> CrossFishStats:
    """Next-chirp latencies and coincidences of A's chirps with B's.

    Parameters
    ----------
    a, b : array_like
        Sorted chirp times of fish A and B within ``[t0_s, t0_s + duration_s)``.
    t0_s, duration_s : float
        Observation window (for the circular shift).
    rng : numpy.random.Generator
        Draws the shifts.
    latency_edges_s : array_like, optional
        Histogram edges (default 0-10 s in 0.1-s bins).
    window_s : float
        Half-width of the coincidence window (default 0.25 s).
    n_shuffles : int
        Null realisations.
    min_shift_s, max_shift_s : float
        Range of the absolute shift of the null (default 5-30 s, short
        enough to keep the slow co-modulation of the two fish's chirp
        rates, long enough to remove responses).

    Returns
    -------
    CrossFishStats
    """
    ta = np.sort(np.asarray(a, dtype=np.float64))
    tb = np.sort(np.asarray(b, dtype=np.float64))
    edges = (
        np.linspace(0.0, 10.0, 101)
        if latency_edges_s is None
        else np.asarray(latency_edges_s, dtype=np.float64)
    )

    def histogram(tb_: FloatArray) -> FloatArray:
        lat = next_event_latencies(ta, tb_)
        return np.histogram(lat[np.isfinite(lat)], edges)[0].astype(np.float64)

    observed = histogram(tb)
    null = np.zeros(edges.size - 1)
    coincidences = np.empty(n_shuffles)
    for s in range(n_shuffles):
        offset = float(rng.uniform(min_shift_s, max_shift_s)) * rng.choice((-1.0, 1.0))
        shifted = _circular_shift(tb, offset, t0_s, duration_s)
        null += histogram(shifted)
        coincidences[s] = coincidence_fraction(ta, shifted, window_s)
    return CrossFishStats(
        latency_edges_s=edges,
        latency_counts=observed,
        latency_null_counts=null / max(n_shuffles, 1),
        coincidence=coincidence_fraction(ta, tb, window_s),
        coincidence_null=float(np.nanmean(coincidences))
        if n_shuffles
        else float("nan"),
        coincidence_null_sd=float(np.nanstd(coincidences))
        if n_shuffles
        else float("nan"),
        window_s=window_s,
        n_a=int(ta.size),
        n_b=int(tb.size),
    )
