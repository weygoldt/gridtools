"""Read-only access to multi-channel grid recordings.

Two layouts are supported:

- fishgrid directories: ``traces-grid1.raw`` (interleaved float32, one
  column per electrode) next to ``fishgrid.cfg``. Samples are in volts:
  on the tube competition recordings the quantisation step is 0.625 uV
  and the range +-20.48 mV (verified 2026-11-12), even though the
  configuration and thunderlab label the data mV. Some files end in a
  partial frame (an interrupted write); it is dropped with a warning;
- wav files (PCM or float), e.g. excerpts or simulations. Integer PCM is
  scaled to [-1, 1). Electrode positions come from an ``electrodes.csv``
  (columns ``channel, x_m, y_m``) next to the wav if there is one (layouts
  that are not a full rectangle), else from ``fishgrid.cfg``.

Both are memory-mapped, so opening a 26 GB recording is instant and reading a
few seconds anywhere in it costs only those seconds.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.io import wavfile

from gridtools.io.geometry import GridGeometry, read_fishgrid_config

__all__ = ["Recording", "open_recording"]

log = logging.getLogger(__name__)

_RAW_NAME = "traces-grid1.raw"


@dataclass(frozen=True)
class Recording:
    """A multi-channel recording, read lazily.

    Use :func:`open_recording` to create one.

    Parameters
    ----------
    path
        The file the samples are read from.
    samplerate_hz
        Sampling rate.
    data
        Memory-mapped samples of shape ``(n_samples, n_channels)``.
    scale
        Factor applied to samples on reading (1 for float data).
    geometry
        Electrode positions, if known.
    """

    path: Path
    samplerate_hz: float
    data: npt.NDArray[np.generic]
    scale: float = 1.0
    geometry: GridGeometry | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            msg = f"data must be 2-D (samples, channels), got {self.data.shape}"
            raise ValueError(msg)
        if self.samplerate_hz <= 0:
            msg = f"samplerate_hz must be positive, got {self.samplerate_hz}"
            raise ValueError(msg)
        if self.geometry is not None and self.geometry.n_electrodes != self.n_channels:
            msg = (
                f"geometry has {self.geometry.n_electrodes} electrodes but the "
                f"recording has {self.n_channels} channels"
            )
            raise ValueError(msg)

    @property
    def n_samples(self) -> int:
        """Number of samples per channel."""
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of channels (electrodes)."""
        return int(self.data.shape[1])

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.n_samples / self.samplerate_hz

    def read(
        self,
        start: int,
        stop: int,
        channels: Sequence[int] | None = None,
    ) -> npt.NDArray[np.float32]:
        """Read samples ``[start, stop)``.

        Parameters
        ----------
        start, stop
            Sample indices with ``0 <= start <= stop <= n_samples``.
        channels
            Channels to read, in this order; all if None.

        Returns
        -------
        Float32 array of shape ``(stop - start, n_read_channels)``; a copy,
        safe to modify.

        Raises
        ------
        IndexError
            If the span lies outside the recording.
        """
        if not 0 <= start <= stop <= self.n_samples:
            msg = (
                f"span [{start}, {stop}) outside recording of {self.n_samples} samples"
            )
            raise IndexError(msg)
        block = self.data[start:stop]
        if channels is not None:
            block = block[:, list(channels)]
        out = np.asarray(block, dtype=np.float32)
        if self.scale != 1.0:
            out = out * np.float32(self.scale)
        elif np.shares_memory(out, self.data):
            out = out.copy()
        return out

    def time_to_sample(self, t_s: float) -> int:
        """Index of the sample at or just after time ``t_s``."""
        return int(np.ceil(t_s * self.samplerate_hz - 1e-9))

    def read_time(
        self,
        start_s: float,
        stop_s: float,
        channels: Sequence[int] | None = None,
    ) -> npt.NDArray[np.float32]:
        """Read the samples in ``[start_s, stop_s)`` seconds; see :meth:`read`."""
        return self.read(
            self.time_to_sample(start_s), self.time_to_sample(stop_s), channels
        )


def _open_raw(path: Path) -> Recording:
    config = read_fishgrid_config(path.parent)
    n_channels = config.n_electrodes
    itemsize = np.dtype(np.float32).itemsize
    frame_bytes = itemsize * n_channels
    n_frames, partial = divmod(path.stat().st_size, frame_bytes)
    if n_frames == 0:
        msg = f"{path}: no complete {n_channels}-channel float32 frame"
        raise ValueError(msg)
    if partial:
        log.warning(
            "%s: dropping a trailing partial frame (%d of %d bytes)",
            path,
            partial,
            frame_bytes,
        )
    data = np.memmap(path, dtype=np.float32, mode="r", shape=(n_frames, n_channels))
    return Recording(path, config.samplerate_hz, data, 1.0, config.geometry())


def _open_wav(path: Path) -> Recording:
    rate, samples = wavfile.read(path, mmap=True)
    data: npt.NDArray[np.generic] = samples[:, None] if samples.ndim == 1 else samples
    if data.dtype.kind == "u":  # 8-bit wav is unsigned
        msg = f"{path}: unsigned 8-bit wav is not supported"
        raise ValueError(msg)
    scale = 1.0 / 2 ** (8 * data.dtype.itemsize - 1) if data.dtype.kind == "i" else 1.0
    geometry = None
    config_path = path.parent / "fishgrid.cfg"
    electrodes_path = path.parent / "electrodes.csv"
    if electrodes_path.exists():
        geometry = _read_electrodes(electrodes_path, data.shape[1])
    elif config_path.exists():
        config = read_fishgrid_config(config_path)
        if config.n_electrodes == data.shape[1]:
            geometry = config.geometry()
    return Recording(path, float(rate), data, scale, geometry)


def _read_electrodes(path: Path, n_channels: int) -> GridGeometry | None:
    """Electrode positions from ``electrodes.csv`` (columns channel, x_m, y_m)."""
    table = np.genfromtxt(path, delimiter=",", names=True)
    names = table.dtype.names or ()
    if not {"channel", "x_m", "y_m"} <= set(names):
        log.warning("%s: expected columns channel, x_m, y_m; ignored", path)
        return None
    table = np.atleast_1d(table)
    order = np.argsort(table["channel"])
    if table.size != n_channels:
        log.warning(
            "%s: %d rows for %d channels; ignored", path, table.size, n_channels
        )
        return None
    return GridGeometry(np.column_stack([table["x_m"][order], table["y_m"][order]]))


def open_recording(path: str | Path) -> Recording:
    """Open a grid recording for reading.

    Parameters
    ----------
    path
        A fishgrid directory (with ``traces-grid1.raw`` and ``fishgrid.cfg``),
        a ``.raw`` file inside one, a ``.wav`` file, or a directory holding
        exactly one ``.wav`` file.

    Returns
    -------
    The recording; samples are memory-mapped, not loaded.

    Raises
    ------
    FileNotFoundError
        If no supported recording is found.
    ValueError
        If the files are inconsistent (e.g. raw size vs channel count).
    """
    path = Path(path)
    if path.is_dir():
        if (path / _RAW_NAME).exists():
            return _open_raw(path / _RAW_NAME)
        wavs = sorted(path.glob("*.wav"))
        if len(wavs) == 1:
            return _open_wav(wavs[0])
        msg = (
            f"{path}: no {_RAW_NAME} and not exactly one .wav file ({len(wavs)} found)"
        )
        raise FileNotFoundError(msg)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".raw":
        return _open_raw(path)
    if path.suffix == ".wav":
        return _open_wav(path)
    msg = f"{path}: unsupported recording format {path.suffix!r}"
    raise ValueError(msg)
