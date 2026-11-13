"""Choosing background segments: chirp screening and stratified selection.

Pure functions on arrays (no I/O), used by
:func:`gridtools.background.build_background_library`.

Rules
-----
1. Candidates start on a grid (``step_s``) and lie at least ``edge_s``
   inside the recording (fish handling at the start, v1 never processed
   the last 16-31 s of a recording).
2. Where chirp detections exist, a candidate is valid only if no detection
   box overlaps the candidate widened by ``margin_s`` on both sides; hand
   labelled spans (the locked test set) are excluded the same way.
3. The usable span of the recording is cut into equal strata (one per
   segment wanted), and each stratum contributes the valid candidate with
   the lowest score: the expected number of chirps v1 missed in it (from
   the local detection rate and v1's recall) plus the fraction of the
   segment where the worst-covered fish has no track detection. Ties are
   broken at random, so recordings without detections get a random
   candidate per stratum. Chosen segments never overlap.
4. Optionally (``fill``), strata that found no free candidate are made up
   for: the remaining valid candidates that overlap no chosen segment are
   added, lowest score first, until ``segments_per_recording`` are chosen.
   With narrow windows (e.g. the last 30 min of a recording) and strata
   about one segment wide, this packs the window instead of leaving gaps.
5. Alternatively (``pack``), strata and scores are ignored: valid
   candidates are taken greedily from the earliest on, each starting at
   least one segment after the previous one (for equal lengths this fits
   the most segments), and if more than ``segments_per_recording`` fit, an
   evenly spread subset is kept. For windows that hold about as many
   segments as wanted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "SelectionConfig",
    "candidate_starts",
    "choose_stratified",
    "count_overlaps",
    "expected_missed_chirps",
    "fill_non_overlapping",
    "frame_coverage",
    "local_rate_per_s",
    "pack_non_overlapping",
    "span_starts",
]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SelectionConfig:
    """Where segments may come from and how they are chosen.

    Attributes
    ----------
    segment_s : float
        Segment length (default 60 s).
    step_s : float
        Grid of candidate starts (default 15 s).
    margin_s : float
        A chirp detection this close to a segment rejects it (default 2 s:
        longer than a chirp's spectrogram footprint with the 205-ms v1
        window, and than a typical burst gap).
    edge_s : float
        Distance from the recording's start and end (default 120 s).
    segments_per_recording : int
        Strata per recording, at most one segment each (default 8).
    rate_window_s : float
        Window centred on a candidate in which the local detection rate is
        measured (default 600 s).
    v1_recall : float
        Recall of chirpdetector v1 on the hand labels (0.764 at threshold
        0.5), used for the expected number of missed chirps.
    fill : bool
        Make up for empty strata from the remaining valid candidates
        (default False; rule 4).
    pack : bool
        Pack the valid candidates greedily instead of choosing per stratum
        (default False; rule 5).
    """

    segment_s: float = 60.0
    step_s: float = 15.0
    margin_s: float = 2.0
    edge_s: float = 120.0
    segments_per_recording: int = 8
    rate_window_s: float = 600.0
    v1_recall: float = 0.764
    fill: bool = False
    pack: bool = False

    def __post_init__(self) -> None:
        for name in ("segment_s", "step_s", "rate_window_s"):
            if not getattr(self, name) > 0:
                msg = f"{name} must be > 0, got {getattr(self, name)!r}"
                raise ValueError(msg)
        for name in ("margin_s", "edge_s"):
            if not getattr(self, name) >= 0:
                msg = f"{name} must be >= 0, got {getattr(self, name)!r}"
                raise ValueError(msg)
        if self.segments_per_recording < 1:
            msg = "segments_per_recording must be >= 1"
            raise ValueError(msg)
        if not 0 < self.v1_recall <= 1:
            msg = f"v1_recall must lie in (0, 1], got {self.v1_recall!r}"
            raise ValueError(msg)


def candidate_starts(
    duration_s: float, segment_s: float, step_s: float, edge_s: float
) -> FloatArray:
    """Start times of candidate segments on a grid of ``step_s``.

    Every candidate ``[start, start + segment_s]`` lies within
    ``[edge_s, duration_s - edge_s]``. Empty if none fits.
    """
    last = duration_s - edge_s - segment_s
    if last < edge_s:
        return np.empty(0)
    n = int(np.floor((last - edge_s) / step_s + 1e-9)) + 1
    return edge_s + step_s * np.arange(n, dtype=np.float64)


def span_starts(
    start_s: float, stop_s: float, segment_s: float, step_s: float
) -> FloatArray:
    """Start times on a grid of ``step_s`` from ``start_s``, segments inside the span.

    Every candidate ``[start, start + segment_s]`` lies within
    ``[start_s, stop_s]``. ``candidate_starts(d, L, step, edge)`` equals
    ``span_starts(edge, d - edge, L, step)``.
    """
    last = stop_s - segment_s
    if last < start_s:
        return np.empty(0)
    n = int(np.floor((last - start_s) / step_s + 1e-9)) + 1
    return start_s + step_s * np.arange(n, dtype=np.float64)


def count_overlaps(
    starts_s: npt.ArrayLike,
    segment_s: float,
    span_starts_s: npt.ArrayLike,
    span_stops_s: npt.ArrayLike,
    margin_s: float = 0.0,
) -> npt.NDArray[np.int64]:
    """Number of spans overlapping each widened candidate.

    A span ``[a, b]`` overlaps the candidate ``[s - margin, s + L + margin]``
    if ``a < s + L + margin`` and ``b > s - margin``. With ``a <= b`` for
    every span, the count is ``#{a < high} - #{b <= low}``.

    Parameters
    ----------
    starts_s : array_like, shape (C,)
        Candidate starts.
    segment_s : float
        Candidate length ``L``.
    span_starts_s, span_stops_s : array_like, shape (D,)
        Spans (e.g. detection boxes), in any order.
    margin_s : float
        Widening of the candidate on both sides.

    Returns
    -------
    numpy.ndarray of int, shape (C,)
    """
    s = np.asarray(starts_s, dtype=np.float64)
    a = np.sort(np.asarray(span_starts_s, dtype=np.float64))
    b = np.sort(np.asarray(span_stops_s, dtype=np.float64))
    if a.size != b.size:
        msg = "span starts and stops differ in length"
        raise ValueError(msg)
    low = s - margin_s
    high = s + segment_s + margin_s
    return (
        np.searchsorted(a, high, side="left") - np.searchsorted(b, low, side="right")
    ).astype(np.int64)


def local_rate_per_s(
    centres_s: npt.ArrayLike, event_times_s: npt.ArrayLike, window_s: float
) -> FloatArray:
    """Events per second in a window of ``window_s`` centred on each time."""
    c = np.asarray(centres_s, dtype=np.float64)
    t = np.sort(np.asarray(event_times_s, dtype=np.float64))
    n = np.searchsorted(t, c + window_s / 2, side="left") - np.searchsorted(
        t, c - window_s / 2, side="left"
    )
    return n / window_s


def expected_missed_chirps(
    local_rate_per_s: npt.ArrayLike, segment_s: float, recall: float
) -> FloatArray:
    """Expected chirps a detector with ``recall`` missed in a segment.

    Detected chirps occur at the local rate; true chirps at rate / recall;
    the missed ones at rate (1 - recall) / recall, independently of whether
    any were detected in the segment (thinning of a Poisson process).
    """
    rate = np.asarray(local_rate_per_s, dtype=np.float64)
    return rate * segment_s * (1.0 - recall) / recall


def frame_coverage(
    starts_s: npt.ArrayLike,
    segment_s: float,
    frame_times_s: npt.ArrayLike,
    detected: npt.ArrayLike,
) -> FloatArray:
    """Fraction of a fish's track frames inside each candidate with a detection.

    Parameters
    ----------
    starts_s : array_like, shape (C,)
    segment_s : float
    frame_times_s : array_like, shape (T,)
        Times of all track frames (sorted).
    detected : array_like of bool, shape (T,)
        Whether the fish has a detection in each frame.

    Returns
    -------
    numpy.ndarray, shape (C,)
        Coverage in [0, 1]; NaN where a candidate contains no frame.
    """
    s = np.asarray(starts_s, dtype=np.float64)
    t = np.asarray(frame_times_s, dtype=np.float64)
    cum = np.concatenate([[0], np.cumsum(np.asarray(detected, dtype=np.int64))])
    lo = np.searchsorted(t, s, side="left")
    hi = np.searchsorted(t, s + segment_s, side="left")
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cum[hi] - cum[lo]) / np.maximum(n, 1), np.nan)


def choose_stratified(
    rng: np.random.Generator,
    starts_s: npt.ArrayLike,
    valid: npt.ArrayLike,
    score: npt.ArrayLike,
    *,
    segment_s: float,
    n_strata: int,
    span_s: tuple[float, float],
) -> npt.NDArray[np.intp]:
    """One valid, lowest-score candidate per time stratum, without overlaps.

    Parameters
    ----------
    rng : numpy.random.Generator
        Breaks ties between equal scores.
    starts_s : array_like, shape (C,)
        Candidate starts (sorted).
    valid : array_like of bool, shape (C,)
    score : array_like, shape (C,)
        Lower is better; NaN counts as invalid.
    segment_s : float
        Candidate length (chosen candidates do not overlap).
    n_strata : int
        Number of equal strata over ``span_s``; a candidate belongs to the
        stratum containing its start.
    span_s : (float, float)
        Span divided into strata.

    Returns
    -------
    numpy.ndarray of int
        Indices of the chosen candidates, in time order.
    """
    s = np.asarray(starts_s, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(
        np.asarray(score, dtype=np.float64)
    )
    sc = np.asarray(score, dtype=np.float64)
    tie = rng.random(s.size)
    edges = np.linspace(span_s[0], span_s[1], n_strata + 1)
    stratum = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_strata - 1)
    chosen: list[int] = []
    for k in range(n_strata):
        idx = np.flatnonzero(ok & (stratum == k))
        if chosen:
            taken = s[np.asarray(chosen)]
            free = np.all(np.abs(s[idx, None] - taken[None, :]) >= segment_s, axis=1)
            idx = idx[free]
        if idx.size == 0:
            continue
        best = idx[np.lexsort((tie[idx], sc[idx]))[0]]
        chosen.append(int(best))
    return np.sort(np.asarray(chosen, dtype=np.intp))


def fill_non_overlapping(
    rng: np.random.Generator,
    starts_s: npt.ArrayLike,
    valid: npt.ArrayLike,
    score: npt.ArrayLike,
    chosen: npt.ArrayLike,
    *,
    segment_s: float,
    n_max: int,
) -> npt.NDArray[np.intp]:
    """Add valid candidates that overlap no chosen one, lowest score first.

    Parameters
    ----------
    rng : numpy.random.Generator
        Breaks ties between equal scores.
    starts_s, valid, score : array_like, shape (C,)
        As for :func:`choose_stratified`.
    chosen : array_like of int
        Indices already chosen (kept).
    segment_s : float
        Candidate length.
    n_max : int
        Stop when this many are chosen in total.

    Returns
    -------
    numpy.ndarray of int
        The chosen indices (old and new), in time order.
    """
    s = np.asarray(starts_s, dtype=np.float64)
    sc = np.asarray(score, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(sc)
    out = [int(i) for i in np.asarray(chosen, dtype=np.intp)]
    tie = rng.random(s.size)
    taken = np.sort(s[np.asarray(out, dtype=np.intp)]) if out else np.empty(0)
    for i in np.flatnonzero(ok)[np.lexsort((tie[ok], sc[ok]))]:
        if len(out) >= n_max:
            break
        k = np.searchsorted(taken, s[i])
        near = taken[max(0, k - 1) : k + 1]
        if near.size and np.min(np.abs(near - s[i])) < segment_s:
            continue
        out.append(int(i))
        taken = np.insert(taken, k, s[i])
    return np.sort(np.asarray(out, dtype=np.intp))


def pack_non_overlapping(
    starts_s: npt.ArrayLike,
    valid: npt.ArrayLike,
    *,
    segment_s: float,
    n_max: int,
) -> npt.NDArray[np.intp]:
    """Most non-overlapping valid candidates, earliest first (rule 5).

    Parameters
    ----------
    starts_s : array_like, shape (C,)
        Candidate starts (sorted).
    valid : array_like of bool, shape (C,)
    segment_s : float
        Candidate length.
    n_max : int
        At most this many; if more fit, an evenly spread subset of them.

    Returns
    -------
    numpy.ndarray of int
        Indices of the chosen candidates, in time order.
    """
    s = np.asarray(starts_s, dtype=np.float64)
    chosen: list[int] = []
    last = -np.inf
    for i in np.flatnonzero(np.asarray(valid, dtype=bool)):
        if s[i] >= last + segment_s - 1e-9:
            chosen.append(int(i))
            last = s[i]
    out = np.asarray(chosen, dtype=np.intp)
    if out.size > n_max:
        out = out[np.unique(np.round(np.linspace(0, out.size - 1, n_max)).astype(int))]
    return out
