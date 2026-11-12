from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from gridtools.io import open_recording

from .test_geometry import write_cfg


def make_fishgrid(directory: Path, n_samples: int = 1000) -> np.ndarray:
    write_cfg(directory)
    data = np.random.default_rng(0).standard_normal((n_samples, 15)).astype(np.float32)
    data.tofile(directory / "traces-grid1.raw")
    return data


def test_fishgrid_directory_roundtrip(tmp_path: Path) -> None:
    data = make_fishgrid(tmp_path)
    rec = open_recording(tmp_path)
    assert (rec.n_samples, rec.n_channels, rec.samplerate_hz) == (1000, 15, 20_000.0)
    assert rec.duration_s == pytest.approx(0.05)
    assert rec.geometry is not None
    assert rec.geometry.n_electrodes == 15
    np.testing.assert_array_equal(rec.read(10, 20), data[10:20])
    np.testing.assert_array_equal(rec.read(0, 5, channels=[3, 1]), data[:5, [3, 1]])
    np.testing.assert_array_equal(
        open_recording(tmp_path / "traces-grid1.raw").read(0, 3), data[:3]
    )


def test_read_returns_writable_copy(tmp_path: Path) -> None:
    make_fishgrid(tmp_path)
    rec = open_recording(tmp_path)
    block = rec.read(0, 10)
    block[:] = 0.0
    assert np.any(rec.read(0, 10) != 0.0)


def test_read_time_uses_sample_grid(tmp_path: Path) -> None:
    data = make_fishgrid(tmp_path)
    rec = open_recording(tmp_path)
    np.testing.assert_array_equal(rec.read_time(0.001, 0.002), data[20:40])


def test_read_out_of_range(tmp_path: Path) -> None:
    make_fishgrid(tmp_path)
    rec = open_recording(tmp_path)
    with pytest.raises(IndexError):
        rec.read(990, 1001)
    with pytest.raises(IndexError):
        rec.read(-1, 5)


def test_raw_size_must_match_channels(tmp_path: Path) -> None:
    write_cfg(tmp_path)
    np.zeros(16, dtype=np.float32).tofile(tmp_path / "traces-grid1.raw")
    with pytest.raises(ValueError, match="whole number"):
        open_recording(tmp_path)


def test_int16_wav_is_scaled(tmp_path: Path) -> None:
    pcm = np.array([[0, -32768], [16384, 32767]], dtype=np.int16)
    wavfile.write(tmp_path / "x.wav", 8000, pcm)
    rec = open_recording(tmp_path)
    assert rec.geometry is None
    np.testing.assert_allclose(rec.read(0, 2), [[0.0, -1.0], [0.5, 32767 / 32768]])


def test_missing_recording(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_recording(tmp_path)
    with pytest.raises(FileNotFoundError):
        open_recording(tmp_path / "nope.wav")


def test_fixture_excerpt(fixture_dir: Path) -> None:
    rec = open_recording(fixture_dir)
    assert (rec.n_channels, rec.samplerate_hz, rec.n_samples) == (15, 20_000.0, 400_000)
    block = rec.read(0, 20_000)
    assert block.dtype == np.float32
    assert 0.0 < np.abs(block).max() <= 1.0
