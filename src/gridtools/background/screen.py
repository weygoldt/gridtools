"""Chirp screens of background candidates: external detections, gate, windows.

Real fish in a background segment may chirp, and their chirps are not
labelled, so a detector trained on hybrid scenes would be penalised for
finding them. Before a segment is used it is therefore screened with chirp
detectors run on the source recording at a low threshold. The detectors run
outside this package (e.g. the v1 production detector in its frozen
environment, ``scripts/v1_screen.py`` of the workspace); this module reads
their output, decides which candidates pass and which recordings are too
active to be used at all.

Screen directory
----------------
::

    ROOT/screen.json                    detector, weights sha256, store threshold
    ROOT/<dataset>/<recording>.csv      boxes: t_start_s, t_end_s, f_low_hz,
                                        f_high_hz, score (recording time, s/Hz)
    ROOT/<dataset>/<recording>.json     covered_s: [[start, stop], ...] spans
                                        the detector processed, plus run info

Every box with a score at or above ``screen.json``'s ``store_threshold`` is
listed, so a screen's threshold can be changed without re-running the
detector. Spans outside ``covered_s`` were never looked at: a candidate is
screened only if it lies inside a covered span, margin included.

Rules
-----
- A candidate ``[start, start + L]`` fails a screen if any box with
  ``score >= threshold`` overlaps ``[start - margin, start + L + margin]``
  (:func:`count_overlaps`), or if that span is not covered.
- A recording fails the gate if its window holds more than
  ``max_rate_per_h`` boxes per covered hour with ``score >= threshold``
  (box centres; a busy recording carries many chirps the detector missed
  even in segments it passes: at 20/h and recall 0.76, about 0.1 per 60-s
  segment).
- :class:`TimeWindow`: the part of a recording candidates may come from,
  measured from its end (and optionally no earlier than a time after its
  start).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from gridtools.background.select import count_overlaps

__all__ = [
    "BOX_COLUMNS",
    "RecordingGate",
    "Screen",
    "ScreenTable",
    "TimeWindow",
    "covered",
    "gate_rate_per_h",
    "max_score_near",
    "read_screen",
    "read_screen_meta",
]

#: Columns of a screen's box table.
BOX_COLUMNS = ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz", "score")


@dataclass(frozen=True)
class TimeWindow:
    """Where in a recording background candidates may come from.

    The window is ``[max(duration - last_s, min_start_s), duration]``:
    measured from the recording's end, and never earlier than
    ``min_start_s`` after its start. With both set, a recording shorter
    than ``min_start_s + last_s`` gets a shorter window, so the window
    stays inside the period both bounds describe whether the recording was
    started late or stopped early.

    Attributes
    ----------
    last_s : float or None
        Length of the window before the recording's end; None for no bound
        (the whole recording).
    min_start_s : float
        Earliest start of the window after the recording's start (default
        0).
    min_duration_s : float
        Recordings shorter than this are skipped (default 0).
    """

    last_s: float | None = None
    min_start_s: float = 0.0
    min_duration_s: float = 0.0

    def __post_init__(self) -> None:
        if self.last_s is not None and not self.last_s > 0:
            msg = f"last_s must be > 0 or None, got {self.last_s!r}"
            raise ValueError(msg)
        for name in ("min_start_s", "min_duration_s"):
            if not getattr(self, name) >= 0:
                msg = f"{name} must be >= 0, got {getattr(self, name)!r}"
                raise ValueError(msg)

    def span(self, duration_s: float) -> tuple[float, float] | None:
        """The window ``(start, stop)`` of a recording, or None if it is empty.

        None also if the recording is shorter than ``min_duration_s``.
        """
        if duration_s < self.min_duration_s:
            return None
        start = 0.0 if self.last_s is None else duration_s - self.last_s
        start = max(start, self.min_start_s, 0.0)
        if start >= duration_s:
            return None
        return start, duration_s

    def describe(self) -> str:
        """Short text for the manifest."""
        parts = []
        if self.last_s is not None:
            parts.append(f"last {self.last_s:g} s")
        if self.min_start_s > 0:
            parts.append(f"not before {self.min_start_s:g} s")
        if self.min_duration_s > 0:
            parts.append(f"recordings >= {self.min_duration_s:g} s")
        return ", ".join(parts) or "whole recording"


@dataclass(frozen=True)
class Screen:
    """One chirp screen: a detector's boxes and the rule applied to them.

    Attributes
    ----------
    name : str
        Short name; manifest columns are ``screen_<name>_*``.
    root : pathlib.Path
        Screen directory (module docstring).
    threshold : float
        Boxes with at least this score reject a candidate.
    margin_s : float
        A box this close to a candidate rejects it (default 1 s).
    required : bool
        Skip recordings the screen has no table for (default True);
        otherwise their candidates are only screened by the other screens.
    """

    name: str
    root: Path
    threshold: float
    margin_s: float = 1.0
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            msg = f"screen name must be an identifier, got {self.name!r}"
            raise ValueError(msg)
        if not 0 <= self.threshold <= 1:
            msg = f"threshold must lie in [0, 1], got {self.threshold!r}"
            raise ValueError(msg)
        if not self.margin_s >= 0:
            msg = f"margin_s must be >= 0, got {self.margin_s!r}"
            raise ValueError(msg)
        object.__setattr__(self, "root", Path(self.root))

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for ``library.json``."""
        return {
            "name": self.name,
            "root": str(self.root),
            "threshold": self.threshold,
            "margin_s": self.margin_s,
            "required": self.required,
        }


@dataclass(frozen=True)
class RecordingGate:
    """Drop recordings whose window is too chirp-active.

    Attributes
    ----------
    screen : str
        Name of the screen whose boxes are counted.
    threshold : float
        Score of the boxes counted (default 0.5, v1's production threshold).
    max_rate_per_h : float
        Recordings with more boxes per covered hour of their window are
        dropped (default 20).
    """

    screen: str = "v1"
    threshold: float = 0.5
    max_rate_per_h: float = 20.0

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 1:
            msg = f"threshold must lie in [0, 1], got {self.threshold!r}"
            raise ValueError(msg)
        if not self.max_rate_per_h >= 0:
            msg = f"max_rate_per_h must be >= 0, got {self.max_rate_per_h!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ScreenTable:
    """A screen's output for one recording.

    Attributes
    ----------
    boxes : pandas.DataFrame
        :data:`BOX_COLUMNS`, sorted by start time.
    covered_s : numpy.ndarray, shape (K, 2)
        Spans the detector processed (sorted, non-overlapping).
    info : dict
        The recording's run info (``<recording>.json``).
    meta : dict
        The screen's ``screen.json``.
    """

    boxes: pd.DataFrame
    covered_s: npt.NDArray[np.float64]
    info: dict[str, Any]
    meta: dict[str, Any]

    def above(self, threshold: float) -> pd.DataFrame:
        """Boxes with ``score >= threshold``.

        Raises
        ------
        ValueError
            If ``threshold`` is below the screen's store threshold (boxes
            below it were never written).
        """
        store = float(self.meta.get("store_threshold", 0.0))
        if threshold < store - 1e-12:
            msg = f"threshold {threshold} is below the screen's store threshold {store}"
            raise ValueError(msg)
        return pd.DataFrame(
            self.boxes.loc[(self.boxes["score"] >= threshold).to_numpy()]
        )

    def covered_within(self, start_s: float, stop_s: float) -> float:
        """Seconds of ``[start_s, stop_s]`` the detector processed."""
        lo = np.clip(self.covered_s[:, 0], start_s, stop_s)
        hi = np.clip(self.covered_s[:, 1], start_s, stop_s)
        return float(np.sum(hi - lo))


def read_screen_meta(root: str | Path) -> dict[str, Any]:
    """A screen directory's ``screen.json``."""
    path = Path(root) / "screen.json"
    if not path.exists():
        msg = f"{path} does not exist"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text())


def read_screen(root: str | Path, dataset: str, recording: str) -> ScreenTable | None:
    """A recording's screen output, or None if the screen has none.

    Parameters
    ----------
    root : path
        Screen directory.
    dataset, recording : str
        Source dataset and recording name.

    Returns
    -------
    ScreenTable or None
    """
    root = Path(root)
    table_path = root / dataset / f"{recording}.csv"
    info_path = root / dataset / f"{recording}.json"
    if not (table_path.exists() and info_path.exists()):
        return None
    meta = read_screen_meta(root)
    info = json.loads(info_path.read_text())
    raw = pd.read_csv(table_path)
    missing = [c for c in BOX_COLUMNS if c not in raw.columns]
    if missing:
        msg = f"{table_path}: missing columns {missing}"
        raise ValueError(msg)
    boxes = pd.DataFrame({c: raw[c].astype(np.float64) for c in BOX_COLUMNS})
    if len(boxes) and bool((boxes["t_end_s"] < boxes["t_start_s"]).any()):
        msg = f"{table_path}: boxes end before they start"
        raise ValueError(msg)
    spans = np.asarray(info.get("covered_s", []), dtype=np.float64).reshape(-1, 2)
    spans = spans[np.argsort(spans[:, 0], kind="stable")]
    return ScreenTable(
        boxes=boxes.sort_values("t_start_s", kind="stable").reset_index(drop=True),
        covered_s=spans,
        info=info,
        meta=meta,
    )


def covered(
    starts_s: npt.ArrayLike,
    segment_s: float,
    spans_s: npt.ArrayLike,
    margin_s: float = 0.0,
) -> npt.NDArray[np.bool_]:
    """Whether each widened candidate lies inside one covered span.

    Parameters
    ----------
    starts_s : array_like, shape (C,)
    segment_s : float
    spans_s : array_like, shape (K, 2)
        Covered spans (any order; overlapping or abutting spans are merged).
    margin_s : float
        Widening of the candidate on both sides.

    Returns
    -------
    numpy.ndarray of bool, shape (C,)
    """
    s = np.asarray(starts_s, dtype=np.float64)
    spans = np.asarray(spans_s, dtype=np.float64).reshape(-1, 2)
    if spans.size == 0:
        return np.zeros(s.size, dtype=bool)
    spans = spans[np.argsort(spans[:, 0], kind="stable")]
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    m = np.asarray(merged)
    lo = s - margin_s
    hi = s + segment_s + margin_s
    k = np.searchsorted(m[:, 0], lo, side="right") - 1
    ok = k >= 0
    kk = np.clip(k, 0, len(m) - 1)
    return ok & (hi <= m[kk, 1])


def max_score_near(
    starts_s: npt.ArrayLike,
    segment_s: float,
    boxes: pd.DataFrame,
    margin_s: float,
) -> npt.NDArray[np.float64]:
    """Highest box score overlapping each widened candidate (0 if none)."""
    s = np.asarray(starts_s, dtype=np.float64)
    out = np.zeros(s.size)
    if len(boxes) == 0:
        return out
    a = boxes["t_start_s"].to_numpy(dtype=np.float64)
    b = boxes["t_end_s"].to_numpy(dtype=np.float64)
    score = boxes["score"].to_numpy(dtype=np.float64)
    order = np.argsort(a, kind="stable")
    a, b, score = a[order], b[order], score[order]
    longest = float(np.max(b - a)) if a.size else 0.0
    for i, start in enumerate(s):
        low, high = start - margin_s, start + segment_s + margin_s
        lo = np.searchsorted(a, low - longest, side="left")
        hi = np.searchsorted(a, high, side="left")
        sel = b[lo:hi] > low
        if sel.any():
            out[i] = float(score[lo:hi][sel].max())
    return out


def gate_rate_per_h(
    table: ScreenTable, threshold: float, start_s: float, stop_s: float
) -> float:
    """Boxes (centres) per covered hour of ``[start_s, stop_s]``.

    NaN if none of the span is covered.
    """
    hours = table.covered_within(start_s, stop_s) / 3600.0
    if hours <= 0:
        return math.nan
    boxes = table.above(threshold)
    centres = 0.5 * (boxes["t_start_s"] + boxes["t_end_s"]).to_numpy()
    inside = (centres >= start_s) & (centres <= stop_s)
    in_span = np.zeros(centres.size, dtype=bool)
    for a, b in table.covered_s:
        in_span |= (centres >= a) & (centres <= b)
    return float(np.sum(inside & in_span) / hours)


def rejects(
    starts_s: npt.ArrayLike,
    segment_s: float,
    table: ScreenTable,
    threshold: float,
    margin_s: float,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
    """Which candidates a screen rejects, and the boxes near each.

    Returns
    -------
    rejected : numpy.ndarray of bool, shape (C,)
        A box with ``score >= threshold`` overlaps the widened candidate, or
        the widened candidate is not covered.
    n_near : numpy.ndarray of int, shape (C,)
        Number of such boxes.
    """
    boxes = table.above(threshold)
    n_near = count_overlaps(
        starts_s, segment_s, boxes["t_start_s"], boxes["t_end_s"], margin_s
    )
    ok = covered(starts_s, segment_s, table.covered_s, margin_s)
    return (n_near > 0) | ~ok, n_near
