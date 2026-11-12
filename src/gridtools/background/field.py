"""Fish-free background segments from field logger recordings.

The field loggers (Teensy-based, 16 channels, int16, 48 or 96 kHz) write
consecutive WavPack chunks (``*.wv``, 60 s in 2023, 300 s since 2025) into
one folder per deployment. They are decoded with ``ffmpeg`` (read-only, only
the window needed) to float32 in units of the int16 full scale; the
amplifier gain is not calibrated, so levels are relative (the SNR scaling
absorbs the unit).

Fish-free test
--------------
A window ``[start - margin, stop + margin]`` is fish-free if no electrode
shows a spectral line more than ``threshold_db`` above the running median
(100 Hz) of its own Welch PSD (2.9 Hz bins at 48 kHz) anywhere in
``band_hz`` (default 300-2000 Hz, the wave-fish band; harmonics of
lower-frequency fish such as *Sternopygus* fall into it too). Testing each
electrode separately keeps a weak fish near one electrode from being diluted
by the other fifteen. With a 70-s Welch average the PSD scatters by about
0.5 dB, so an 8-dB line is a real line. Pulse fish and electric eels are
broadband and do not fail the test unless they fire very regularly.

Data sets
---------
``field_iriri2026``
    Rio Iriri 2026, ``<site>/gridNN/<deployment>``: nine 4 x 4 grids per
    site at 60 cm, 300-s chunks at 48 kHz; electrode positions and channel
    map in each grid's ``electrode_layout.json`` (metadata/README.md).
``field_flona2025``
    FLONA (Rio Falsino) 2025, ``site_N/<deployment>`` (calibration
    deployments excluded): 4 x 4 grids at 60 cm with
    ``electrode_layout.json`` per site.
``field_flona2023``
    FLONA 2023 single-logger deployments (16 channels, 60-s chunks, 48 or
    96 kHz). No electrode layout was recorded: the segments get a regular
    4 x 4 layout at 0.5 m and ``geometry_known=False``.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import shutil
import subprocess
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.io import wavfile

from gridtools.background.levels import tonal_peaks, welch_psd

__all__ = [
    "FLONA2023_ROOT",
    "FLONA2025_ROOT",
    "IRIRI2026_ROOT",
    "FieldFolder",
    "FieldScanConfig",
    "decode_wv",
    "fish_free_test",
    "flona2023_folders",
    "flona2025_folders",
    "iriri2026_folders",
    "probe_wv",
]

FLONA2023_ROOT = Path("/mnt/data1/efish/fielddata/flona2023")
FLONA2025_ROOT = Path("/mnt/data1/efish/fielddata/flona2025_workspace/flona2025")
IRIRI2026_ROOT = Path(
    "/run/media/weygoldt/data1/efish/fielddata/iriri2026/iriri2026_recordings"
)

FIELD_UNITS = "int16 full scale (field logger, gain not calibrated)"

_TIME = re.compile(r"(\d{8}T\d{4,6})")


@dataclass(frozen=True)
class FieldFolder:
    """One logger deployment: a folder of consecutive WavPack chunks.

    Attributes
    ----------
    dataset : str
        ``field_iriri2026``, ``field_flona2025`` or ``field_flona2023``.
    site : str
        Recording site.
    name : str
        Deployment name, unique within the dataset (``site/.../folder``).
    path : pathlib.Path
        The folder.
    layout_path : pathlib.Path or None
        ``electrode_layout.json`` (coordinates in cm, one per channel).
    group : str
        Split group: deployments of one grid share their noise, so splits
        should keep them together.
    """

    dataset: str
    site: str
    name: str
    path: Path
    layout_path: Path | None = None
    group: str = ""

    @property
    def key(self) -> str:
        """``dataset/name``."""
        return f"{self.dataset}/{self.name}"

    def chunks(self) -> list[Path]:
        """WavPack chunks in time order (message files excluded)."""
        return sorted(p for p in self.path.glob("*.wv") if "-message" not in p.name)


def _folders(pattern_dirs: list[Path]) -> list[Path]:
    return sorted(d for d in pattern_dirs if d.is_dir() and any(d.glob("*.wv")))


def iriri2026_folders(root: str | Path = IRIRI2026_ROOT) -> list[FieldFolder]:
    """Deployments of the Rio Iriri 2026 grids (``<site>/gridNN/<deployment>``)."""
    root = Path(root)
    out = []
    for site in sorted(p for p in root.glob("iriri*") if p.is_dir()):
        for grid in sorted(site.glob("grid*")):
            layout = grid / "electrode_layout.json"
            for d in _folders(list(grid.glob("iriri*"))):
                out.append(
                    FieldFolder(
                        dataset="field_iriri2026",
                        site=site.name,
                        name=f"{site.name}/{grid.name}/{d.name}",
                        path=d,
                        layout_path=layout if layout.exists() else None,
                        group=f"field_iriri2026/{site.name}/{grid.name}",
                    )
                )
    return out


def flona2025_folders(root: str | Path = FLONA2025_ROOT) -> list[FieldFolder]:
    """Grid deployments of FLONA 2025 (``site_N/<deployment>``, no calibrations)."""
    root = Path(root)
    out = []
    for site in sorted(p for p in root.glob("site_*") if p.is_dir()):
        layout = site / "electrode_layout.json"
        for d in _folders([p for p in site.iterdir() if "calib" not in p.name]):
            out.append(
                FieldFolder(
                    dataset="field_flona2025",
                    site=site.name,
                    name=f"{site.name}/{d.name}",
                    path=d,
                    layout_path=layout if layout.exists() else None,
                    group=f"field_flona2025/{site.name}",
                )
            )
    return out


def flona2023_folders(root: str | Path = FLONA2023_ROOT) -> list[FieldFolder]:
    """Single-logger deployments of FLONA 2023 (``<site>/<date>-*logger``)."""
    root = Path(root)
    out = []
    for site in sorted(p for p in root.glob("rio-*") if p.is_dir()):
        dirs = [d for d in site.iterdir() if d.is_dir() and "logger" in d.name]
        dirs += [
            d for d in site.iterdir() if d.is_dir() and d.name.endswith("-afternoon")
        ]
        for d in _folders(dirs):
            out.append(
                FieldFolder(
                    dataset="field_flona2023",
                    site=site.name,
                    name=f"{site.name}/{d.name}",
                    path=d,
                    layout_path=None,
                    group=f"field_flona2023/{site.name}/{d.name}",
                )
            )
    return out


def probe_wv(path: str | Path) -> tuple[float, int, float]:
    """Sampling rate, channel count and duration of a WavPack file (ffprobe)."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    return (
        float(stream["sample_rate"]),
        int(stream["channels"]),
        float(stream["duration"]),
    )


def decode_wv(
    path: str | Path, start_s: float, duration_s: float, n_channels: int
) -> npt.NDArray[np.float32]:
    """Decode ``duration_s`` from ``start_s`` of a WavPack file (int16 -> [-1, 1)).

    Returns
    -------
    numpy.ndarray of float32, shape (n, n_channels)
    """
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-ss",
            f"{start_s:.6f}",
            "-t",
            f"{duration_s:.6f}",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype="<f4").reshape(-1, n_channels).copy()


def fish_free_test(
    x: npt.ArrayLike,
    samplerate_hz: float,
    *,
    band_hz: tuple[float, float] = (300.0, 2000.0),
    threshold_db: float = 8.0,
) -> tuple[bool, list[float], float]:
    """Per-electrode line test (see module docstring).

    Returns
    -------
    fish_free : bool
    peaks_hz : list of float
        Lines found on any electrode.
    max_excess_db : float
        Height of the strongest line (0 if none).
    """
    freqs, psd = welch_psd(x, samplerate_hz)
    found: list[float] = []
    best = 0.0
    for e in range(psd.shape[1]):
        peaks, excess = tonal_peaks(
            freqs, psd[:, e], band_hz=band_hz, threshold_db=threshold_db, mains_hz=None
        )
        found += [float(f) for f in peaks]
        if excess.size:
            best = max(best, float(excess.max()))
    return not found, sorted(set(np.round(found, 1).tolist())), best


@dataclass(frozen=True)
class FieldScanConfig:
    """How field deployments are scanned for fish-free segments.

    Attributes
    ----------
    probes_per_folder : int
        Windows tested per deployment, spread evenly over its chunks
        (default 8).
    segments_per_folder : int
        At most this many fish-free windows kept per deployment, spread over
        the passing ones (default 2).
    margin_s : float
        The window tested extends this far beyond the segment on both sides
        (default 2 s, as the chirp margin of the grid recordings; 60-s
        chunks then give 56-s segments).
    band_hz : (float, float)
        Band of the line test (default 300-2000 Hz).
    threshold_db : float
        Line threshold (default 8 dB).
    default_spacing_m : float
        Electrode spacing of the regular 4 x 4 layout used when a deployment
        has no layout file (default 0.5 m).
    skip_edges_s : float
        Chunks this close to the start or end of a deployment are not
        tested (default 30 min: deployment and retrieval disturb the
        water and the cables; an Iriri grid 6 min after the logger started
        was full of broadband transients). Ignored for deployments too
        short to leave anything.
    """

    probes_per_folder: int = 8
    segments_per_folder: int = 2
    margin_s: float = 2.0
    band_hz: tuple[float, float] = (300.0, 2000.0)
    threshold_db: float = 8.0
    default_spacing_m: float = 0.5
    skip_edges_s: float = 1800.0


def _layout(
    folder: FieldFolder, n_channels: int, spacing_m: float
) -> tuple[np.ndarray, float, bool]:
    """Electrode positions (m, x/y), spacing and whether they are known."""
    if folder.layout_path is not None:
        data = json.loads(folder.layout_path.read_text())
        coords = np.asarray(data["coordinates"], dtype=np.float64)[:, :2] / 100.0
        if len(coords) == n_channels:
            d = np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(-1))
            pitch = float(np.min(d[d > 0])) if n_channels > 1 else spacing_m
            return coords, pitch, True
    side = math.isqrt(n_channels)
    rows, cols = (side, side) if side * side == n_channels else (1, n_channels)
    r, c = np.divmod(np.arange(rows * cols), cols)
    return (
        np.column_stack([c * spacing_m, r * spacing_m]).astype(float),
        spacing_m,
        False,
    )


def _write_config(
    path: Path, rows: int, cols: int, spacing_m: float, rate: float
) -> None:
    path.write_text(
        "*FishGrid\n  Grid &1:\n"
        f"      Columns1         : {cols}\n"
        f"      Rows1            : {rows}\n"
        f"      ColumnDistance1  : {spacing_m * 100:.1f}cm\n"
        f"      RowDistance1     : {spacing_m * 100:.1f}cm\n"
        "  Hardware Settings:\n      DAQ board:\n"
        f"          AISampleRate: {rate / 1000:.3f}kHz\n"
        "*Note: written by gridtools.background for a field logger segment; "
        "electrode positions are in the library manifest\n"
    )


def chunk_time(path: Path) -> datetime | None:
    """Start time of a chunk from its file name (``...YYYYmmddTHHMM[SS]...``)."""
    match = _TIME.search(path.name)
    if match is None:
        return None
    text = match.group(1)
    fmt = "%Y%m%dT%H%M%S" if len(text) == 15 else "%Y%m%dT%H%M"
    try:
        return datetime.strptime(text, fmt)
    except ValueError:
        return None


def _segment_id(
    folder: FieldFolder,
    chunk: int,
    chunk_s: float,
    offset_s: float,
    scan: FieldScanConfig,
) -> str:
    start_s = chunk * chunk_s + offset_s + scan.margin_s
    return f"{folder.dataset}_{folder.name.replace('/', '_')}_{round(start_s):07d}"


def scan_folder(
    folder: FieldFolder,
    out_dir: Path,
    segment_s: float,
    scan: FieldScanConfig,
    seed: int,
    measure: Any,
    *,
    vetoed: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Test windows of one deployment and write the fish-free ones.

    ``measure(x, rate) -> dict`` computes the manifest's level columns
    (passed in by the build to keep one definition). Windows whose segment
    id is in ``vetoed`` (rejected by the author) are not kept.

    Returns
    -------
    rows : list of dict
        Manifest rows of the kept segments.
    info : dict
        Scan statistics (probes, audio scanned, fish-free windows, peaks).
    """
    chunks = folder.chunks()
    info: dict[str, Any] = {
        "dataset": folder.dataset,
        "recording": folder.name,
        "site": folder.site,
        "n_chunks": len(chunks),
    }
    if not chunks:
        info["skipped"] = "no chunks"
        return [], info
    rate, n_channels, chunk_s = probe_wv(chunks[len(chunks) // 2])
    info.update(samplerate_hz=rate, n_electrodes=n_channels, chunk_s=chunk_s)
    length = min(segment_s, chunk_s - 2 * scan.margin_s)
    if length < 10:
        info["skipped"] = f"chunks of {chunk_s} s too short"
        return [], info
    key = zlib.crc32(folder.key.encode("utf-8"))
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(key,)))
    # skip deployment and retrieval (people in the water, cables handled)
    skip = math.ceil(scan.skip_edges_s / chunk_s)
    first, last = skip, len(chunks) - skip
    if last - first < 1:
        first, last = 0, len(chunks)
    n_probes = min(scan.probes_per_folder, last - first)
    edges = np.linspace(first, last, n_probes + 1)
    picks = sorted(
        {
            int(min(len(chunks) - 1, lo + rng.integers(0, max(1, int(hi - lo)))))
            for lo, hi in itertools.pairwise(edges)
        }
    )
    window = length + 2 * scan.margin_s
    results = []
    scanned_s = 0.0
    for i in picks:
        path = chunks[i]
        slack = chunk_s - window
        offset = float(rng.uniform(0, slack)) if slack > 0 else 0.0
        try:
            x = decode_wv(path, offset, window, n_channels)
        except (subprocess.CalledProcessError, ValueError):
            continue
        if x.shape[0] < int(0.99 * window * rate):
            continue
        scanned_s += x.shape[0] / rate
        ok, peaks, excess = fish_free_test(
            x, rate, band_hz=scan.band_hz, threshold_db=scan.threshold_db
        )
        clipped = bool(np.max(np.abs(x)) >= 0.999)
        results.append((i, path, offset, ok and not clipped, peaks, excess))
    passing = [
        r
        for r in results
        if r[3] and _segment_id(folder, r[0], chunk_s, r[2], scan) not in vetoed
    ]
    info.update(
        n_probes=len(results),
        scanned_h=scanned_s / 3600,
        span_h=len(chunks) * chunk_s / 3600,
        n_fish_free=len(passing),
        fish_free_fraction=len(passing) / len(results) if results else math.nan,
        lines_hz_examples=sorted({p for r in results for p in r[4]})[:20],
    )
    if not passing:
        return [], info
    keep = np.unique(
        np.round(
            np.linspace(
                0, len(passing) - 1, min(scan.segments_per_folder, len(passing))
            )
        )
    ).astype(int)
    positions, pitch, known = _layout(folder, n_channels, scan.default_spacing_m)
    side = math.isqrt(n_channels)
    rows_, cols_ = (side, side) if side * side == n_channels else (1, n_channels)
    out_rows = []
    for k in keep:
        i, path, offset, _, _, excess = passing[k]
        start_in_chunk = offset + scan.margin_s
        x = decode_wv(path, start_in_chunk, length, n_channels)
        n = round(length * rate)
        x = x[:n]
        start_s = i * chunk_s + start_in_chunk  # nominal time in the deployment
        sid = _segment_id(folder, i, chunk_s, offset, scan)
        seg_dir = out_dir / "segments" / sid
        seg_dir.mkdir(parents=True, exist_ok=True)
        wavfile.write(
            seg_dir / "background.wav",
            round(rate),
            np.ascontiguousarray(x, dtype="<f4"),
        )
        _write_config(seg_dir / "fishgrid.cfg", rows_, cols_, pitch, rate)
        if folder.layout_path is not None:
            shutil.copyfile(folder.layout_path, seg_dir / "electrode_layout.json")
        m = measure(x, rate)
        when = chunk_time(path)
        hour = (
            None
            if when is None
            else (when.hour + (when.minute + start_in_chunk / 60) / 60) % 24
        )
        details = {
            "segment_id": sid,
            "source": {
                "dataset": folder.dataset,
                "deployment": str(folder.path),
                "file": str(path),
                "offset_in_file_s": start_in_chunk,
                "n_samples": n,
                "layout": None
                if folder.layout_path is None
                else str(folder.layout_path),
                "chunk_start": None if when is None else when.isoformat(),
            },
            "fish_free_test": {
                "band_hz": list(scan.band_hz),
                "threshold_db": scan.threshold_db,
                "margin_s": scan.margin_s,
                "max_excess_db": excess,
            },
            **m["details"],
        }
        (seg_dir / "segment.json").write_text(json.dumps(details, indent=1) + "\n")
        row = {
            "segment_id": sid,
            "dataset": folder.dataset,
            "recording": folder.name,
            "split_group": folder.group or folder.key,
            "path": str(Path("segments") / sid),
            "segment_class": "fish_free",
            "screened": True,
            "weight": 1.0,
            "start_s": float(start_s),
            "stop_s": float(start_s + n / rate),
            "duration_s": float(n / rate),
            "dark_phase": None if hour is None else bool(hour >= 18 or hour < 6),
            "samplerate_hz": float(rate),
            "n_samples": n,
            "n_electrodes": n_channels,
            "n_rows": rows_,
            "n_columns": cols_,
            "row_spacing_m": pitch,
            "column_spacing_m": pitch,
            "geometry_known": known,
            "spacing_known": known,
            "electrode_positions_m": positions.tolist(),
            "units": FIELD_UNITS,
            "tracks_available": False,
            "track_quality": "none",
            "n_fish": 0,
            "fish_ids": [],
            "fish_eodf_hz": [],
            "fish_coverage": [],
            "fish_power": [],
            "fish_snr_db": [],
            "untracked_peaks_hz": [],
            "tracks_complete": True,
            **m["columns"],
        }
        out_rows.append(row)
    info["n_segments"] = len(out_rows)
    return out_rows, info
