"""Time-frequency labels: audian label files and v1 chirp detections.

audian (https://github.com/bendalab/audian) stores labels as CSV with the
columns in :data:`AUDIAN_COLUMNS`; boxes are spans in time (s) and frequency
(Hz). The hand labels of this project use two categories:
``labeled_region`` (a span in which every chirp was labelled) and ``chirp``.
Additional columns (e.g. an emitter identity) are preserved on reading and
writing.

chirpdetector v1 wrote its detections to ``chirpdetector_bboxes.csv``;
:func:`read_v1_detections` maps them onto the same box columns.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "AUDIAN_COLUMNS",
    "BOX_COLUMNS",
    "read_labels",
    "read_v1_detections",
    "write_labels",
]

AUDIAN_COLUMNS = (
    "category",
    "kind",
    "channel",
    "t_start_s",
    "t_end_s",
    "f_low_hz",
    "f_high_hz",
    "note",
)
"""Columns of an audian label file, in order."""

BOX_COLUMNS = ("t_start_s", "t_end_s", "f_low_hz", "f_high_hz")
"""Columns that define a time-frequency box."""


def _check_boxes(df: pd.DataFrame, source: object) -> None:
    for column in BOX_COLUMNS:
        if column not in df.columns:
            msg = f"{source}: missing column {column!r}"
            raise ValueError(msg)
    boxes = df[list(BOX_COLUMNS)].to_numpy(dtype=np.float64)
    finite = np.isfinite(boxes)
    if not finite[:, :2].all():
        msg = f"{source}: every label needs finite t_start_s and t_end_s"
        raise ValueError(msg)
    if np.any(boxes[:, 1] < boxes[:, 0]):
        msg = f"{source}: t_end_s < t_start_s"
        raise ValueError(msg)
    both = finite[:, 2] & finite[:, 3]
    if np.any(boxes[both, 3] < boxes[both, 2]):
        msg = f"{source}: f_high_hz < f_low_hz"
        raise ValueError(msg)


def read_labels(path: str | Path) -> pd.DataFrame:
    """Read an audian label file.

    Parameters
    ----------
    path
        CSV file with at least the columns in :data:`AUDIAN_COLUMNS`.

    Returns
    -------
    One row per label. Box columns are float (NaN where empty), ``channel``
    is a nullable integer, text columns are strings (empty where missing).
    Extra columns are kept as read.

    Raises
    ------
    ValueError
        If a required column is missing or a box is malformed.
    """
    df = pd.read_csv(path, dtype={"category": str, "kind": str, "note": str})
    missing = [c for c in AUDIAN_COLUMNS if c not in df.columns]
    if missing:
        msg = f"{path}: missing columns {missing}"
        raise ValueError(msg)
    for column in ("category", "kind", "note"):
        df[column] = df[column].fillna("").astype(str)
    df["channel"] = df["channel"].astype("Int64")
    for column in BOX_COLUMNS:
        df[column] = df[column].astype(np.float64)
    _check_boxes(df, path)
    return df


def write_labels(labels: pd.DataFrame, path: str | Path) -> None:
    """Write labels as an audian label file, atomically.

    Columns in :data:`AUDIAN_COLUMNS` come first, in that order; missing
    ones are written empty; extra columns follow. The file is written to a
    temporary file next to ``path`` and then renamed, so a crash never
    leaves a half-written label file.
    """
    path = Path(path)
    df = labels.copy()
    for column in AUDIAN_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    _check_boxes(df, "labels")
    extra = [c for c in df.columns if c not in AUDIAN_COLUMNS]
    df = df[[*AUDIAN_COLUMNS, *extra]]
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="") as f:
            df.to_csv(f, index=False, float_format="%.6f")
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_v1_detections(path: str | Path) -> pd.DataFrame:
    """Read chirpdetector v1 detections (``chirpdetector_bboxes.csv``).

    Parameters
    ----------
    path
        The CSV file, or the recording directory containing it.

    Returns
    -------
    One row per detection, sorted by start time, with columns
    ``t_start_s``, ``t_end_s``, ``f_low_hz``, ``f_high_hz``, ``score``,
    ``emitter_hz`` (v1's emitter EODf) and ``track_id`` (nullable integer;
    v1's assigned track).
    """
    path = Path(path)
    if path.is_dir():
        path = path / "chirpdetector_bboxes.csv"
    raw = pd.read_csv(path)
    df = pd.DataFrame(
        {
            "t_start_s": raw["t1"].astype(np.float64),
            "t_end_s": raw["t2"].astype(np.float64),
            "f_low_hz": raw["f1"].astype(np.float64),
            "f_high_hz": raw["f2"].astype(np.float64),
            "score": raw["score"].astype(np.float64),
            "emitter_hz": raw["emitter_eodf"].astype(np.float64),
            "track_id": raw["track_id"].round().astype("Int64"),
        }
    )
    _check_boxes(df, path)
    return df.sort_values("t_start_s", kind="stable").reset_index(drop=True)
