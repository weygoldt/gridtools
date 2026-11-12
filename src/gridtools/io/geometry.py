"""Electrode geometry and fishgrid configuration files.

A recording made with fishgrid (https://github.com/bendalab/fishgrid) stores
its settings in ``fishgrid.cfg``: an indented ``key : value`` file whose
values carry units (``50.0cm``, ``20.000kHz``, ``20.0mV``). Only the first
grid is supported; recordings in this project never use more than one.

Channel order
-------------
The mapping from channel index to electrode position is not documented in
the configuration files. :meth:`GridGeometry.regular` assumes row-major order
(channel ``r * n_columns + c``). An empirical check on the 2019/2020 tube
competition recordings (which electrode takes over the strongest signal as a
fish moves) could not confirm or reject this order, so treat absolute
positions as provisional. Downstream code should prefer features that do not
depend on the channel order, such as comparing per-electrode power profiles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = [
    "FishgridConfig",
    "GridGeometry",
    "parse_quantity",
    "read_fishgrid_config",
]

_UNIT_SCALE = {
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "hz": 1.0,
    "khz": 1e3,
    "v": 1.0,
    "mv": 1e-3,
    "uv": 1e-6,
    "µv": 1e-6,
}

_QUANTITY = re.compile(r"^\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*([A-Za-zµ]*)\s*$")


def parse_quantity(text: str) -> tuple[float, str]:
    """Split a fishgrid value such as ``"50.0cm"`` into number and unit.

    Parameters
    ----------
    text
        The value as written in ``fishgrid.cfg``.

    Returns
    -------
    value, unit
        The number as written and the unit string (empty if none).

    Raises
    ------
    ValueError
        If ``text`` is not a number with an optional unit.
    """
    match = _QUANTITY.match(text)
    if match is None:
        msg = f"not a quantity: {text!r}"
        raise ValueError(msg)
    return float(match.group(1)), match.group(2)


def _to_si(text: str, expected: tuple[str, ...]) -> float:
    value, unit = parse_quantity(text)
    unit = unit.lower()
    if unit not in expected:
        msg = f"unexpected unit {unit!r} in {text!r}; expected one of {expected}"
        raise ValueError(msg)
    return value * _UNIT_SCALE[unit]


@dataclass(frozen=True)
class GridGeometry:
    """Positions of the electrodes of one grid, indexed by channel.

    Parameters
    ----------
    positions_m
        Array of shape ``(n_electrodes, 2)`` with x and y in metres; row
        ``i`` is the electrode recorded on channel ``i``.
    """

    positions_m: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
            msg = f"positions_m must have shape (n, 2), got {positions.shape}"
            raise ValueError(msg)
        if not np.all(np.isfinite(positions)):
            msg = "positions_m must be finite"
            raise ValueError(msg)
        positions.setflags(write=False)
        object.__setattr__(self, "positions_m", positions)

    @classmethod
    def regular(
        cls,
        n_columns: int,
        n_rows: int,
        column_spacing_m: float,
        row_spacing_m: float,
    ) -> GridGeometry:
        """Rectangular grid with the first electrode at the origin.

        Channels are assumed to be in row-major order: channel
        ``r * n_columns + c`` sits at ``(c * column_spacing_m,
        r * row_spacing_m)``. See the module docstring for why this order is
        provisional.
        """
        if n_columns < 1 or n_rows < 1:
            msg = f"grid needs at least one row and column, got {n_rows}x{n_columns}"
            raise ValueError(msg)
        rows, columns = np.divmod(np.arange(n_rows * n_columns), n_columns)
        positions = np.column_stack(
            [columns * column_spacing_m, rows * row_spacing_m]
        ).astype(np.float64)
        return cls(positions)

    @property
    def n_electrodes(self) -> int:
        """Number of electrodes."""
        return len(self.positions_m)

    def distances_m(self, xy_m: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Distances from points to every electrode.

        Parameters
        ----------
        xy_m
            Points of shape ``(n, 2)`` (or ``(2,)`` for one point) in metres.

        Returns
        -------
        Array of shape ``(n, n_electrodes)`` (or ``(n_electrodes,)``).
        """
        xy = np.asarray(xy_m, dtype=np.float64)
        return np.linalg.norm(xy[..., None, :] - self.positions_m, axis=-1)


@dataclass(frozen=True)
class FishgridConfig:
    """The parts of ``fishgrid.cfg`` this package uses.

    Parameters
    ----------
    samplerate_hz
        Sampling rate of the analog input.
    n_rows, n_columns
        Size of the first grid.
    row_spacing_m, column_spacing_m
        Electrode spacing of the first grid.
    max_amplitude_v
        Input range of the DAQ board in volts (``AIMaxVolt``); the raw
        samples are in the same unit as this value.
    entries
        Every ``key: value`` pair of the file, unparsed.
    """

    samplerate_hz: float
    n_rows: int
    n_columns: int
    row_spacing_m: float
    column_spacing_m: float
    max_amplitude_v: float | None = None
    entries: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def n_electrodes(self) -> int:
        """Number of electrodes in the first grid."""
        return self.n_rows * self.n_columns

    def geometry(self) -> GridGeometry:
        """Electrode positions of the first grid (provisional channel order)."""
        return GridGeometry.regular(
            self.n_columns, self.n_rows, self.column_spacing_m, self.row_spacing_m
        )


def _read_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, value = line.partition(":")
        key = key.strip()
        if not sep or not key or key.startswith("*"):
            continue
        entries.setdefault(key, value.strip())
    return entries


def read_fishgrid_config(path: str | Path) -> FishgridConfig:
    """Read ``fishgrid.cfg``.

    Parameters
    ----------
    path
        The configuration file, or the recording directory containing it.

    Returns
    -------
    The sampling rate and geometry of the first grid.

    Raises
    ------
    FileNotFoundError
        If there is no ``fishgrid.cfg``.
    ValueError
        If a required entry is missing or malformed.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "fishgrid.cfg"
    entries = _read_entries(path)

    def get(key: str) -> str:
        try:
            return entries[key]
        except KeyError:
            msg = f"{path}: missing entry {key!r}"
            raise ValueError(msg) from None

    max_volt = entries.get("AIMaxVolt")
    return FishgridConfig(
        samplerate_hz=_to_si(get("AISampleRate"), ("hz", "khz")),
        n_rows=int(parse_quantity(get("Rows1"))[0]),
        n_columns=int(parse_quantity(get("Columns1"))[0]),
        row_spacing_m=_to_si(get("RowDistance1"), ("m", "cm", "mm")),
        column_spacing_m=_to_si(get("ColumnDistance1"), ("m", "cm", "mm")),
        max_amplitude_v=_to_si(max_volt, ("v", "mv", "uv", "µv")) if max_volt else None,
        entries=entries,
    )
