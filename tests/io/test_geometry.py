from pathlib import Path

import numpy as np
import pytest

from gridtools.io import GridGeometry, parse_quantity, read_fishgrid_config

CFG = """*FishGrid
  Grid &1:
      Used1            : true
      Columns1         : 5
      Rows1            : 3
      ColumnDistance1  : 50.0cm
      RowDistance1     : 40.0cm
  Hardware Settings:
      DAQ board:
          AISampleRate: 20.000kHz
          AIMaxVolt   : 20.0mV
*Recording
  PathFormat: "%04Y-%02m-%02d-%02H_%02M"
"""


def write_cfg(directory: Path, text: str = CFG) -> Path:
    path = directory / "fishgrid.cfg"
    path.write_text(text)
    return path


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("50.0cm", (50.0, "cm")),
        ("20.000kHz", (20.0, "kHz")),
        ("3", (3.0, "")),
        ("-1e-3V", (-1e-3, "V")),
    ],
)
def test_parse_quantity(text: str, expected: tuple[float, str]) -> None:
    assert parse_quantity(text) == expected


def test_parse_quantity_rejects_text() -> None:
    with pytest.raises(ValueError, match="not a quantity"):
        parse_quantity("true")


def test_read_fishgrid_config_converts_units(tmp_path: Path) -> None:
    config = read_fishgrid_config(write_cfg(tmp_path))
    assert config.samplerate_hz == 20_000.0
    assert (config.n_rows, config.n_columns, config.n_electrodes) == (3, 5, 15)
    assert config.column_spacing_m == pytest.approx(0.5)
    assert config.row_spacing_m == pytest.approx(0.4)
    assert config.max_amplitude_v == pytest.approx(0.02)
    assert config.entries["PathFormat"].startswith('"%04Y')


def test_read_fishgrid_config_accepts_directory(tmp_path: Path) -> None:
    write_cfg(tmp_path)
    assert read_fishgrid_config(tmp_path).n_electrodes == 15


def test_read_fishgrid_config_missing_entry(tmp_path: Path) -> None:
    write_cfg(tmp_path, CFG.replace("Rows1", "Rowz1"))
    with pytest.raises(ValueError, match="Rows1"):
        read_fishgrid_config(tmp_path)


def test_regular_geometry_is_row_major() -> None:
    geometry = GridGeometry.regular(
        n_columns=3, n_rows=2, column_spacing_m=0.5, row_spacing_m=0.4
    )
    np.testing.assert_allclose(
        geometry.positions_m,
        [[0, 0], [0.5, 0], [1.0, 0], [0, 0.4], [0.5, 0.4], [1.0, 0.4]],
    )
    assert geometry.n_electrodes == 6


def test_geometry_is_immutable_and_validated() -> None:
    geometry = GridGeometry(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="read-only"):
        geometry.positions_m[0, 0] = 1.0
    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        GridGeometry(np.zeros((3, 3)))


def test_distances() -> None:
    geometry = GridGeometry.regular(2, 1, 1.0, 1.0)
    np.testing.assert_allclose(geometry.distances_m([0.0, 0.0]), [0.0, 1.0])
    np.testing.assert_allclose(geometry.distances_m([[0.0, 1.0]]), [[1.0, np.sqrt(2)]])
