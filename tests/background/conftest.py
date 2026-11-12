"""Synthetic fishgrid trials for the background library tests.

A trial is a fishgrid directory (``traces-grid1.raw`` + ``fishgrid.cfg``)
with white noise on every channel, two fish (constant-frequency sinusoids
with fixed per-electrode gains) present from ``fish_from_s`` on, tracks in
wavetracker layout covering the fish, v1 detections at given times and,
optionally, hand labels.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridtools.background.sources import SourceRecording

FS = 4000.0
ROWS, COLUMNS = 2, 3
NOISE_SD = 1e-4
FISH_HZ = (712.0, 861.0)
FISH_AMPLITUDE = 2e-3

CONFIG = """*FishGrid
  Grid &1:
      Used1            : true
      Columns1         : {columns}
      Rows1            : {rows}
      ColumnDistance1  : 50.0cm
      RowDistance1     : 50.0cm
  Hardware Settings:
      DAQ board:
          AISampleRate: {rate:.3f}kHz
          AIMaxVolt   : 20.0mV
"""


@dataclass(frozen=True)
class Trial:
    """A synthetic trial on disk plus its ground truth."""

    source: SourceRecording
    samples: np.ndarray
    gains: np.ndarray  # (2, E) per-fish electrode gains
    fish_from_s: float
    detection_times_s: tuple[float, ...]


def fish_gains(n_electrodes: int) -> np.ndarray:
    """Fixed, distinct spatial patterns of the two fish."""
    e = np.arange(n_electrodes)
    return np.vstack([1.0 / (1.0 + e), 1.0 / (1.0 + e[::-1])])


def write_trial(
    directory: Path,
    *,
    duration_s: float = 150.0,
    fish_from_s: float = 30.0,
    detection_times_s: Sequence[float] = (50.0, 51.0, 80.0),
    labels: bool = False,
    seed: int = 0,
    name: str | None = None,
    partial_frame: bool = False,
    noisy_channel: int | None = None,
) -> Trial:
    """Write a synthetic fishgrid trial into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_electrodes = ROWS * COLUMNS
    n = int(duration_s * FS)
    t = np.arange(n) / FS
    x = rng.normal(0.0, NOISE_SD, (n, n_electrodes))
    if noisy_channel is not None:
        x[:, noisy_channel] *= 4.0  # +12 dB floor
    gains = fish_gains(n_electrodes)
    present = t >= fish_from_s
    for f, g in zip(FISH_HZ, gains, strict=True):
        x += (FISH_AMPLITUDE * np.sin(2 * np.pi * f * t) * present)[:, None] * g[
            None, :
        ]
    x = x.astype(np.float32)
    raw = x.tobytes()
    if partial_frame:
        raw += b"\x00" * 8
    (directory / "traces-grid1.raw").write_bytes(raw)
    (directory / "fishgrid.cfg").write_text(
        CONFIG.format(rows=ROWS, columns=COLUMNS, rate=FS / 1000)
    )
    # tracks: frames every 0.5 s, both fish from fish_from_s on, 5 of 6 powers
    times = np.arange(0.25, duration_s, 0.5)
    frames = np.flatnonzero(times >= fish_from_s)
    idx = np.repeat(frames, 2)
    ident = np.tile([3.0, 7.0], frames.size)
    fund = np.tile(FISH_HZ, frames.size)
    sign = np.full((idx.size, n_electrodes - 1), -60.0)
    np.save(directory / "times.npy", times)
    np.save(directory / "idx_v.npy", idx)
    np.save(directory / "ident_v.npy", ident)
    np.save(directory / "fund_v.npy", fund)
    np.save(directory / "sign_v.npy", sign)
    det = pd.DataFrame(
        {
            "t1": [d - 0.05 for d in detection_times_s],
            "t2": [d + 0.05 for d in detection_times_s],
            "f1": 700.0,
            "f2": 900.0,
            "score": 0.9,
            "emitter_eodf": 700.0,
            "track_id": 3.0,
        }
    )
    det.to_csv(directory / "chirpdetector_bboxes.csv", index=False)
    labels_path = None
    if labels:
        labels_path = directory / "traces-grid1-editable-labels.csv"
        pd.DataFrame(
            {
                "category": ["labeled_region"],
                "kind": [""],
                "channel": [pd.NA],
                "t_start_s": [100.0],
                "t_end_s": [110.0],
                "f_low_hz": [np.nan],
                "f_high_hz": [np.nan],
                "note": [""],
            }
        ).to_csv(labels_path, index=False)
    source = SourceRecording(
        dataset="synthetic",
        name=name or directory.name,
        recording_path=directory,
        config_path=directory / "fishgrid.cfg",
        tracks_dir=directory,
        detections_path=directory / "chirpdetector_bboxes.csv",
        labels_path=labels_path,
        dark_phase_s=60.0,
    )
    return Trial(source, x, gains, fish_from_s, tuple(detection_times_s))


@pytest.fixture
def make_trial(tmp_path: Path) -> Callable[..., Trial]:
    """Factory writing synthetic trials below ``tmp_path/sources``."""

    def make(name: str = "trial-a", **kwargs: object) -> Trial:
        return write_trial(tmp_path / "sources" / name, name=name, **kwargs)  # type: ignore[arg-type]

    return make
