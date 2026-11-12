"""Writers: float wav and fishgrid.cfg."""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from gridtools.io import read_fishgrid_config
from gridtools.sim.output import write_fishgrid_config, write_wav


def test_wav_is_extensible_float_and_round_trips(tmp_path: Path) -> None:
    data = np.random.default_rng(0).normal(0, 3.0, (1001, 15)).astype(np.float32)
    path = write_wav(tmp_path / "x.wav", data, 20_000.0)
    header = path.read_bytes()[:40]
    assert header[:4] == b"RIFF"
    assert int.from_bytes(header[20:22], "little") == 0xFFFE  # extensible
    assert int.from_bytes(header[22:24], "little") == 15
    rate, back = wavfile.read(path, mmap=True)
    assert rate == 20_000
    np.testing.assert_array_equal(back, data)  # values > 1 are kept


def test_wav_mono_and_bad_rates(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "mono.wav", np.zeros(10), 44_100)
    rate, back = wavfile.read(path)
    assert rate == 44_100
    assert back.size == 10
    with pytest.raises(ValueError, match="integer"):
        write_wav(tmp_path / "bad.wav", np.zeros((10, 2)), 20_000.5)


@pytest.mark.parametrize(
    ("rate", "spacing"), [(20_000.0, 0.5), (44_100.0, 0.3), (12_345.0, 1.0 / 3.0)]
)
def test_fishgrid_config_round_trips(
    tmp_path: Path, rate: float, spacing: float
) -> None:
    path = write_fishgrid_config(
        tmp_path / "fishgrid.cfg",
        samplerate_hz=rate,
        rows=3,
        columns=5,
        spacing_m=spacing,
    )
    config = read_fishgrid_config(path)
    assert (config.n_rows, config.n_columns) == (3, 5)
    assert config.samplerate_hz == pytest.approx(rate, rel=1e-12)
    assert config.row_spacing_m == pytest.approx(spacing, rel=1e-12)
    geometry = config.geometry()
    assert geometry.n_electrodes == 15
    assert geometry.positions_m[1, 0] == pytest.approx(spacing)
