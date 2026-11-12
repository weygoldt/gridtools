"""Read the raw chirp extractions (``extracted_chirps.h5``).

The file was written by the MSc thesis extraction script
(``extract_chirpchamber_and_livios_chirps.py``). Each row is a 0.4 s snippet
at 20 kHz around one detected chirp, with four traces:

``orig_freq``
    Instantaneous EOD frequency from interpolated zero crossings, cubic
    spline through one value per EOD cycle, minus its value at the first
    sample of the extraction window (so it is relative, in Hz).
``smooth_freq``
    The same trace smoothed with a Gaussian (sigma = 20 samples = 1 ms)
    before cutting, minus its own first-sample value.
``orig_amp``, ``smooth_amp``
    Per-cycle peak amplitude (and its smoothed version), divided by the
    maximum within the window.

After cutting, the script centred each chirp with ``np.roll``, a circular
shift. The first sample of the original window (exactly ``0.0`` in both
frequency traces) therefore sits somewhere inside the stored row, and the
row is not contiguous in time across that point. :func:`unroll_rows` undoes
the shift.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)

TRACE_NAMES: tuple[str, ...] = ("orig_freq", "smooth_freq", "orig_amp", "smooth_amp")


@dataclass(frozen=True, slots=True)
class SourceChunk:
    """A block of consecutive rows from the source file.

    Attributes
    ----------
    start_row
        Index of the first row in the source file.
    traces
        Trace name -> float64 array of shape (n_rows, n_samples). These are
        fresh copies; the source file is never written.
    raw_hashes
        One 16-byte digest per row over the raw float32 bytes of all four
        traces, for exact-duplicate detection.
    """

    start_row: int
    traces: dict[str, npt.NDArray[np.float64]]
    raw_hashes: list[bytes]

    @property
    def n_rows(self) -> int:
        """Number of rows in the chunk."""
        return next(iter(self.traces.values())).shape[0]


def read_source_shape(h5_path: Path) -> tuple[int, int, float]:
    """Return ``(n_rows, n_samples, samplerate_hz)`` of the source file."""
    with h5py.File(h5_path, "r") as f:
        shapes = {name: _dataset(f, name).shape for name in TRACE_NAMES}
        if len(set(shapes.values())) != 1:
            msg = f"trace datasets differ in shape: {shapes}"
            raise ValueError(msg)
        n_rows, n_samples = shapes[TRACE_NAMES[0]]
        samplerate_hz = float(_dataset(f, "samplerate")[()])
    return int(n_rows), int(n_samples), samplerate_hz


def iter_source_chunks(h5_path: Path, chunk_rows: int) -> Iterator[SourceChunk]:
    """Yield the source file in blocks of ``chunk_rows`` rows.

    Parameters
    ----------
    h5_path
        Path to ``extracted_chirps.h5`` (opened read-only).
    chunk_rows
        Rows per block; each block needs about
        ``chunk_rows * n_samples * 4 traces * 8 bytes`` of memory.

    Yields
    ------
    SourceChunk
        Float64 copies of the four traces plus per-row raw hashes.
    """
    n_rows, _, _ = read_source_shape(h5_path)
    with h5py.File(h5_path, "r") as f:
        for start in range(0, n_rows, chunk_rows):
            stop = min(start + chunk_rows, n_rows)
            raw = {name: _dataset(f, name)[start:stop] for name in TRACE_NAMES}
            hashes = [
                _row_hash([raw[name][i] for name in TRACE_NAMES])
                for i in range(stop - start)
            ]
            traces = {name: raw[name].astype(np.float64) for name in TRACE_NAMES}
            yield SourceChunk(start_row=start, traces=traces, raw_hashes=hashes)


def find_wrap_index(freq_row: npt.NDArray[np.floating]) -> int:
    """Locate the original window start inside a rolled frequency row.

    Parameters
    ----------
    freq_row
        One row of ``smooth_freq`` or ``orig_freq`` as stored.

    Returns
    -------
    int
        Index of the single sample that is exactly ``0.0``, or ``-1`` if
        there is no such sample or more than one.
    """
    zeros = np.flatnonzero(freq_row == 0.0)
    if zeros.size != 1:
        return -1
    return int(zeros[0])


def unroll_rows(
    rows: npt.NDArray[np.float64], wrap_index: int
) -> npt.NDArray[np.float64]:
    """Undo the circular shift so the row is contiguous in time.

    Parameters
    ----------
    rows
        Array of shape (..., n_samples).
    wrap_index
        Output of :func:`find_wrap_index`; ``-1`` returns a copy unchanged.

    Returns
    -------
    numpy.ndarray
        New array (never a view) whose first sample is the first sample of
        the original extraction window.
    """
    if wrap_index < 0:
        return np.array(rows, dtype=np.float64, copy=True)
    return np.roll(rows, -wrap_index, axis=-1)


def describe_source_file(path: Path, *, sha256: bool = True) -> dict[str, object]:
    """Return provenance information for a source file.

    Parameters
    ----------
    path
        File to describe.
    sha256
        Also hash the file content (reads the whole file once).

    Returns
    -------
    dict
        ``path``, ``size_bytes``, ``mtime_ns``, ``mtime_iso`` and optionally
        ``sha256``.
    """
    stat = os.stat(path)
    info: dict[str, object] = {
        "path": str(Path(path).resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mtime_iso": _iso_utc(stat.st_mtime_ns),
    }
    if sha256:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            while block := f.read(1 << 24):
                digest.update(block)
        info["sha256"] = digest.hexdigest()
    return info


def _iso_utc(mtime_ns: int) -> str:
    import datetime

    return datetime.datetime.fromtimestamp(mtime_ns / 1e9, tz=datetime.UTC).isoformat()


def _dataset(f: h5py.File, name: str) -> h5py.Dataset:
    obj = f[name]
    if not isinstance(obj, h5py.Dataset):
        msg = f"{name!r} is not a dataset"
        raise TypeError(msg)
    return obj


def _row_hash(parts: list[npt.NDArray[np.float32]]) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        digest.update(np.ascontiguousarray(part).tobytes())
    return digest.digest()
