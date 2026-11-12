"""Writers for scene files: float wav and a minimal ``fishgrid.cfg``."""

import struct
from pathlib import Path

import numpy as np
import numpy.typing as npt

from gridtools.io.geometry import read_fishgrid_config

__all__ = ["write_fishgrid_config", "write_wav"]

# KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
_GUID_IEEE_FLOAT = bytes.fromhex("0300000000001000800000aa00389b71")
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE
_MAX_RIFF_BYTES = 2**32 - 1


def write_wav(path: str | Path, data: npt.ArrayLike, samplerate_hz: float) -> Path:
    """Write samples as a 32-bit float WAVE_FORMAT_EXTENSIBLE file.

    The WAV specification requires WAVE_FORMAT_EXTENSIBLE for more than two
    channels or more than 16 bits per sample; ``scipy.io.wavfile.write``
    writes plain WAVE_FORMAT_IEEE_FLOAT instead. The file written here is
    read by scipy (also memory-mapped, as :func:`gridtools.io.open_recording`
    does), libsndfile/soundfile, audioio (with soundfile installed, as in
    audian) and ffmpeg. Values are written as they are (no scaling or
    clipping); float wav may exceed [-1, 1].

    Parameters
    ----------
    path : path
        Output file.
    data : array_like, shape (n_samples, n_channels) or (n_samples,)
        Samples; converted to little-endian float32.
    samplerate_hz : float
        Sampling rate; must be a positive integer number of Hz.

    Returns
    -------
    pathlib.Path
    """
    path = Path(path)
    samples = np.asarray(data, dtype="<f4")
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2 or samples.shape[1] < 1:
        msg = f"data must have shape (n_samples, n_channels), got {samples.shape}"
        raise ValueError(msg)
    rate = round(samplerate_hz)
    if rate <= 0 or rate != samplerate_hz:
        msg = f"samplerate_hz must be a positive integer, got {samplerate_hz!r}"
        raise ValueError(msg)
    n_frames, n_channels = samples.shape
    block_align = 4 * n_channels
    data_bytes = n_frames * block_align
    fmt = (
        struct.pack(
            "<HHIIHHHHI",
            _WAVE_FORMAT_EXTENSIBLE,
            n_channels,
            rate,
            rate * block_align,
            block_align,
            32,  # bits per sample
            22,  # size of the extension
            32,  # valid bits per sample
            0,  # channel mask: no speaker positions
        )
        + _GUID_IEEE_FLOAT
    )
    fact = struct.pack("<I", n_frames)
    riff_bytes = 4 + (8 + len(fmt)) + (8 + len(fact)) + (8 + data_bytes)
    if riff_bytes > _MAX_RIFF_BYTES:
        msg = f"{path}: {data_bytes} bytes of samples exceed the 4 GiB WAV limit"
        raise ValueError(msg)
    with path.open("wb") as f:
        f.write(b"RIFF" + struct.pack("<I", riff_bytes) + b"WAVE")
        f.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
        f.write(b"fact" + struct.pack("<I", len(fact)) + fact)
        f.write(b"data" + struct.pack("<I", data_bytes))
        np.ascontiguousarray(samples).tofile(f)
    return path


def _quantity(value: float, units: tuple[tuple[str, float], ...]) -> str:
    """Format a value with the first unit that represents it exactly."""
    for unit, scale in units:
        text = f"{value / scale:.3f}{unit}"
        if float(text[: -len(unit)]) * scale == value:
            return text
    return f"{value!r}{units[-1][0]}"


def write_fishgrid_config(
    path: str | Path,
    *,
    samplerate_hz: float,
    rows: int,
    columns: int,
    spacing_m: float,
) -> Path:
    """Write a minimal ``fishgrid.cfg`` for a simulated recording.

    Only the entries :func:`gridtools.io.read_fishgrid_config` needs
    (``AISampleRate``, ``Rows1``, ``Columns1``, ``RowDistance1``,
    ``ColumnDistance1``), in fishgrid's layout. The file is read back to
    check that it round-trips.

    Returns
    -------
    pathlib.Path
    """
    path = Path(path)
    rate = _quantity(samplerate_hz, (("kHz", 1e3), ("Hz", 1.0)))
    distance = _quantity(spacing_m, (("cm", 1e-2), ("m", 1.0)))
    text = (
        "*FishGrid\n"
        "  Grid &1:\n"
        "      Used1            : true\n"
        f"      Columns1         : {columns}\n"
        f"      Rows1            : {rows}\n"
        f"      ColumnDistance1  : {distance}\n"
        f"      RowDistance1     : {distance}\n"
        "  Hardware Settings:\n"
        "      DAQ board:\n"
        f"          AISampleRate: {rate}\n"
        "  Recording:\n"
        "      General:\n"
        "          Experiment.Name  : gridtools simulation\n"
    )
    path.write_text(text, encoding="utf-8")
    config = read_fishgrid_config(path)
    checks = (
        (config.samplerate_hz, samplerate_hz),
        (config.row_spacing_m, spacing_m),
        (config.column_spacing_m, spacing_m),
    )
    if (config.n_rows, config.n_columns) != (rows, columns) or any(
        not np.isclose(a, b, rtol=1e-12, atol=0) for a, b in checks
    ):
        msg = f"{path}: fishgrid.cfg does not round-trip"
        raise ValueError(msg)
    return path
