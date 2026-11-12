"""Electrode positions from electrodes.csv next to a wav."""

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from gridtools.io import open_recording


def _write(directory: Path, n_channels: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    data = np.zeros((100, n_channels), dtype=np.float32)
    wavfile.write(directory / "recording.wav", 20_000, data)
    (directory / "fishgrid.cfg").write_text(
        "*FishGrid\n  Grid &1:\n      Columns1 : 3\n      Rows1 : 1\n"
        "      ColumnDistance1 : 50.0cm\n      RowDistance1 : 50.0cm\n"
        "  Hardware Settings:\n      DAQ board:\n          AISampleRate: 20.000kHz\n"
    )
    return directory


def test_electrodes_csv_overrides_the_regular_grid(tmp_path: Path) -> None:
    directory = _write(tmp_path / "a", 3)
    (directory / "electrodes.csv").write_text(
        "channel,x_m,y_m\n2,0.3,0.1\n0,0.0,0.0\n1,0.12,0.25\n"
    )
    recording = open_recording(directory)
    assert recording.geometry is not None
    np.testing.assert_allclose(
        recording.geometry.positions_m, [[0.0, 0.0], [0.12, 0.25], [0.3, 0.1]]
    )


def test_fishgrid_geometry_without_electrodes_csv(tmp_path: Path) -> None:
    recording = open_recording(_write(tmp_path / "b", 3))
    assert recording.geometry is not None
    np.testing.assert_allclose(recording.geometry.positions_m[:, 0], [0.0, 0.5, 1.0])


def test_mismatched_electrodes_csv_is_ignored(tmp_path: Path) -> None:
    directory = _write(tmp_path / "c", 3)
    (directory / "electrodes.csv").write_text("channel,x_m,y_m\n0,0.0,0.0\n")
    assert open_recording(directory).geometry is None
