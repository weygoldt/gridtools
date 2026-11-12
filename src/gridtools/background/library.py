"""Load a background library and draw segments for hybrid scenes.

A library directory (made by :func:`gridtools.background.build_background_library`)
holds ``manifest.parquet`` (and the same table as ``manifest.csv``),
``library.json`` and one directory per segment:

``segments/<segment_id>/background.wav``
    Float32 samples at the source rate, in the source's unit (raw ADC volts
    for fishgrid recordings), one channel per electrode in the source's
    channel order.
``segments/<segment_id>/fishgrid.cfg``
    The source's configuration, so :func:`gridtools.io.open_recording`
    returns the segment with its geometry.
``segments/<segment_id>/{times,fund_v,ident_v,idx_v,sign_v}.npy``
    The real fish's tracks clipped to the segment (times relative to the
    segment start; only assigned detections; ``sign_v`` padded with NaN
    to one column per channel). Absent for fish-free segments.
``segments/<segment_id>/segment.json``
    Per-electrode floors, per-fish measurements, tonal peaks, hum and
    clicks.

Hybrid scenes
-------------
A hybrid scene takes its electrodes from the background: draw a segment
first (:meth:`BackgroundLibrary.sample` with the scene's electrode count),
then place the simulated fish on ``segment.geometry`` (the positions of the
chosen channels: the whole grid, or a contiguous run or random subset of a
larger one) with ``segment.samplerate_hz`` and ``segment.n_samples``. The
scene never maps channels to positions of its own, so real and simulated
fish share one geometry. Dead and noisy channels are avoided by default.
Add the background after the scene's channel gains (it already went
through the real recording chain):

.. code-block:: python

    lib = BackgroundLibrary.load()
    segment = lib.select(screened=True).sample(rng, 60.0, n_electrodes=9)
    ...  # simulate fish on segment.geometry, get their fundamentals
    signal = fish_signal_level(fundamentals)          # unit**2 of the scene
    scale = scale_for_snr(segment.level, signal, snr_db)
    recording = add_background(recording, segment, scale)
    tracks, real_ids = merge_tracks(scene_tracks, segment.tracks,
                                    power_offset_db=20 * log10(scale))

The real fish join the scene's tracks under new identities
(:func:`merge_tracks`, default from 1000 on) and should be listed in the
scene's metadata with ``real_fish=True`` (:meth:`BackgroundSegment.real_fish`):
they are tracked, non-chirping candidates for the assignment model.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from gridtools.io import GridGeometry, Tracks, load_tracks, open_recording

__all__ = [
    "CHANNEL_MODES",
    "DEFAULT_LIBRARY",
    "JSON_COLUMNS",
    "MANIFEST_COLUMNS",
    "REAL_FISH_IDENT_OFFSET",
    "BackgroundLibrary",
    "BackgroundSegment",
    "add_background",
    "choose_channels",
    "clip_tracks",
    "merge_tracks",
    "read_manifest",
    "write_manifest",
]

DEFAULT_LIBRARY = Path("/mnt/data1/chirp_v2/background/v1")

#: Real fish get identities from this value on when merged into a scene.
REAL_FISH_IDENT_OFFSET = 1000

#: How :meth:`BackgroundLibrary.sample` picks a channel subset.
CHANNEL_MODES = ("contiguous", "random")

MANIFEST_COLUMNS: tuple[str, ...] = (
    "segment_id",
    "dataset",
    "recording",
    "split_group",
    "path",
    "segment_class",
    "screened",
    "weight",
    "start_s",
    "stop_s",
    "duration_s",
    "dark_phase",
    "samplerate_hz",
    "n_samples",
    "n_electrodes",
    "n_rows",
    "n_columns",
    "row_spacing_m",
    "column_spacing_m",
    "geometry_known",
    "spacing_known",
    "electrode_positions_m",
    "units",
    "n_v1_detections_margin",
    "nearest_v1_detection_s",
    "v1_rate_per_h",
    "expected_missed_chirps",
    "selection_score",
    "tracks_available",
    "track_quality",
    "track_power_nfft",
    "n_fish",
    "fish_ids",
    "fish_eodf_hz",
    "fish_coverage",
    "fish_power",
    "fish_snr_db",
    "real_snr_max_db",
    "real_level_diff_db",
    "untracked_peaks_hz",
    "tracks_complete",
    "dead_channels",
    "noisy_channels",
    "noise_psd",
    "noise_psd_db",
    "noise_psd_spread_db",
    "level",
    "level_db",
    "reference_bandwidth_hz",
    "level_band_low_hz",
    "level_band_high_hz",
    "tonal_fraction",
    "hum_amplitude",
    "hum_harmonics_amplitude",
    "n_clicks",
    "click_rate_per_s",
    "max_abs",
    "accept",
)
"""Columns of ``manifest.parquet``, in order (documented in the report)."""

#: Manifest columns holding lists, stored as JSON text in parquet and CSV.
JSON_COLUMNS: tuple[str, ...] = (
    "electrode_positions_m",
    "fish_ids",
    "fish_eodf_hz",
    "fish_coverage",
    "fish_power",
    "fish_snr_db",
    "untracked_peaks_hz",
    "dead_channels",
    "noisy_channels",
)


def write_manifest(manifest: pd.DataFrame, directory: str | Path) -> None:
    """Write ``manifest.parquet`` and ``manifest.csv`` (list columns as JSON)."""
    directory = Path(directory)
    df = manifest.copy()
    for column in JSON_COLUMNS:
        if column in df.columns:
            df[column] = [json.dumps(_plain(v)) for v in df[column]]
    df.to_parquet(directory / "manifest.parquet", index=False)
    df.to_csv(directory / "manifest.csv", index=False, float_format="%.10g")


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a manifest (parquet or CSV) and decode the list columns.

    Parameters
    ----------
    path : path
        ``manifest.parquet``, ``manifest.csv`` or the library directory.

    Returns
    -------
    pandas.DataFrame
        One row per segment; list columns hold Python lists.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.parquet"
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    for column in JSON_COLUMNS:
        if column in df.columns:
            df[column] = [
                json.loads(v) if isinstance(v, str) else [] for v in df[column]
            ]
    return df


def clip_tracks(
    tracks: Tracks,
    start_s: float,
    stop_s: float,
    *,
    n_electrodes: int | None = None,
) -> Tracks | None:
    """Assigned detections within ``[start_s, stop_s)``, shifted to start at 0.

    Parameters
    ----------
    tracks : Tracks
        Wavetracker tracks of a recording.
    start_s, stop_s : float
        Span to keep (recording time).
    n_electrodes : int, optional
        Pad ``powers_db`` with NaN columns to this many electrodes (e.g. the
        2019 tracks have 8 columns for 9 channels; the missing channel
        carries no fish signal and was not tracked).

    Returns
    -------
    Tracks or None
        None if no assigned detection lies in the span. ``times_s`` holds
        only the frames in the span, minus ``start_s``.
    """
    frames = np.flatnonzero((tracks.times_s >= start_s) & (tracks.times_s < stop_s))
    if frames.size == 0:
        return None
    first, last = int(frames[0]), int(frames[-1])
    rows = np.flatnonzero(
        np.isfinite(tracks.idents)
        & (tracks.time_indices >= first)
        & (tracks.time_indices <= last)
    )
    if rows.size == 0:
        return None
    powers = None
    if tracks.powers_db is not None:
        powers = np.asarray(tracks.powers_db[rows], dtype=np.float64)
        if n_electrodes is not None and powers.shape[1] < n_electrodes:
            pad = np.full((len(rows), n_electrodes - powers.shape[1]), np.nan)
            powers = np.hstack([powers, pad])
    return Tracks(
        times_s=tracks.times_s[first : last + 1] - start_s,
        freqs_hz=tracks.freqs_hz[rows],
        idents=tracks.idents[rows],
        time_indices=tracks.time_indices[rows] - first,
        powers_db=powers,
    )


def merge_tracks(
    scene_tracks: Tracks | None,
    real_tracks: Tracks | None,
    *,
    ident_offset: int | None = None,
    power_offset_db: float = 0.0,
) -> tuple[Tracks | None, dict[int, int]]:
    """Merge a background's real-fish tracks into a scene's tracks.

    The frame grids are united (sorted unique times), so both sets of
    detections keep their exact times. Real fish get new identities
    ``ident_offset + k`` (``k`` in order of their original identity), so
    they never collide with simulated fish.

    Parameters
    ----------
    scene_tracks : Tracks or None
        Ground-truth tracks of the simulated fish.
    real_tracks : Tracks or None
        The background segment's tracks (times relative to the segment start,
        which must coincide with the scene start).
    ident_offset : int, optional
        First identity of the real fish. Default: the smallest multiple of
        :data:`REAL_FISH_IDENT_OFFSET` above every simulated identity.
    power_offset_db : float
        Added to the real fish's ``powers_db``: ``20 log10(scale)`` for a
        background scaled by ``scale``, plus ``10 log10(nfft_scene /
        nfft_track)`` to convert the tracker's PSD reference (manifest
        ``track_power_nfft``) to the scene's.

    Returns
    -------
    tracks : Tracks or None
        The merged tracks; None if both inputs are None. Powers are NaN
        where one side has none or fewer electrode columns.
    real_ids : dict of int to int
        New identity -> original identity of every real fish.
    """
    if real_tracks is None or real_tracks.ids.size == 0:
        return scene_tracks, {}
    scene_max = (
        -1
        if scene_tracks is None or scene_tracks.ids.size == 0
        else int(scene_tracks.ids.max())
    )
    if ident_offset is None:
        ident_offset = REAL_FISH_IDENT_OFFSET * (
            max(scene_max, 0) // REAL_FISH_IDENT_OFFSET + 1
        )
    if ident_offset <= scene_max:
        msg = (
            f"ident_offset {ident_offset} collides with simulated identity {scene_max}"
        )
        raise ValueError(msg)
    original = real_tracks.ids
    mapping = {int(o): ident_offset + k for k, o in enumerate(original)}
    real_ident = real_tracks.idents.copy()
    assigned = np.isfinite(real_ident)
    real_ident[assigned] = [mapping[int(i)] for i in real_ident[assigned]]
    parts = [] if scene_tracks is None else [scene_tracks]
    parts.append(
        Tracks(
            times_s=real_tracks.times_s,
            freqs_hz=real_tracks.freqs_hz,
            idents=real_ident,
            time_indices=real_tracks.time_indices,
            powers_db=None
            if real_tracks.powers_db is None
            else real_tracks.powers_db + power_offset_db,
        )
    )
    times = np.unique(np.concatenate([p.times_s for p in parts]))
    widths = [p.n_electrodes for p in parts if p.n_electrodes is not None]
    width = max(widths) if widths else None
    freqs, idents, indices, powers = [], [], [], []
    for p in parts:
        freqs.append(p.freqs_hz)
        idents.append(p.idents)
        indices.append(np.searchsorted(times, p.times_s)[p.time_indices])
        if width is not None:
            block = np.full((len(p.freqs_hz), width), np.nan)
            if p.powers_db is not None:
                block[:, : p.powers_db.shape[1]] = p.powers_db
            powers.append(block)
    index = np.concatenate(indices)
    ident = np.concatenate(idents)
    order = np.lexsort((ident, index))
    merged = Tracks(
        times_s=times,
        freqs_hz=np.concatenate(freqs)[order],
        idents=ident[order],
        time_indices=index[order],
        powers_db=None if width is None else np.concatenate(powers)[order],
    )
    return merged, {new: old for old, new in mapping.items()}


@dataclass(frozen=True, eq=False)
class BackgroundSegment:
    """A background excerpt ready to be added to a scene.

    Attributes
    ----------
    segment_id : str
        Library segment the excerpt comes from.
    data : numpy.ndarray, shape (n_samples, n_electrodes)
        Read-only, memory-mapped samples (source unit); use :meth:`read`
        for a float32 copy.
    samplerate_hz : float
    geometry : GridGeometry
        Electrode positions of the source grid (channel order of ``data``).
    tracks : Tracks or None
        Real fish tracks of the excerpt (times from its first sample).
    info : mapping
        The segment's manifest row; for a channel subset, ``n_electrodes``,
        ``electrode_positions_m``, ``level``, ``dead_channels``,
        ``noisy_channels`` and ``fish_snr_db`` refer to the subset.
    offset_s : float
        Start of the excerpt within the stored segment.
    channels : tuple of int
        Channels of the stored segment in ``data`` (all, or a subset).
    """

    segment_id: str
    data: npt.NDArray[np.floating]
    samplerate_hz: float
    geometry: GridGeometry
    tracks: Tracks | None
    info: Mapping[str, Any] = field(repr=False)
    offset_s: float = 0.0
    channels: tuple[int, ...] = ()

    @property
    def n_samples(self) -> int:
        """Samples per channel."""
        return int(self.data.shape[0])

    @property
    def n_electrodes(self) -> int:
        """Number of channels (electrodes)."""
        return int(self.data.shape[1])

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.n_samples / self.samplerate_hz

    @property
    def level(self) -> float:
        """Background level of the stored segment (``unit**2``, reference bandwidth)."""
        return float(self.info["level"])

    @property
    def dead_channels(self) -> list[int]:
        """Columns of ``data`` without fish signal in the source (0-based)."""
        return [int(c) for c in self.info.get("dead_channels", [])]

    @property
    def noisy_channels(self) -> list[int]:
        """Columns of ``data`` whose floor is > 6 dB above the source's median."""
        return [int(c) for c in self.info.get("noisy_channels", [])]

    @property
    def geometry_known(self) -> bool:
        """Whether the grid layout (rows, columns, channel order) is known."""
        return bool(self.info.get("geometry_known", True))

    @property
    def positions_nominal(self) -> bool:
        """True if :attr:`geometry` uses a nominal spacing, not measured positions.

        The competition grids' ``fishgrid.cfg`` spacing (0.5 m) conflicts with
        the tank size in the paper; their true spacing is unknown.
        """
        return not bool(self.info.get("spacing_known", False))

    def read(self, start: int = 0, stop: int | None = None) -> npt.NDArray[np.float32]:
        """Float32 copy of samples ``[start, stop)``."""
        stop = self.n_samples if stop is None else stop
        return np.array(self.data[start:stop], dtype=np.float32)

    def real_fish(self, real_ids: Mapping[int, int] | None = None) -> pd.DataFrame:
        """Table of the real fish, for a scene's metadata.

        Parameters
        ----------
        real_ids : mapping, optional
            New identity -> original identity (from :func:`merge_tracks`).

        Returns
        -------
        pandas.DataFrame
            Columns ``ident`` (scene identity, or the original if no mapping),
            ``source_ident``, ``real_fish`` (True), ``eodf_hz``, ``snr_db``,
            ``source_dataset``, ``source_recording``, ``segment_id``.
        """
        ids = [int(i) for i in self.info.get("fish_ids", [])]
        eodf = list(self.info.get("fish_eodf_hz", [])) or [np.nan] * len(ids)
        snr = list(self.info.get("fish_snr_db", [])) or [np.nan] * len(ids)
        inverse = {old: new for new, old in (real_ids or {}).items()}
        return pd.DataFrame(
            {
                "ident": [inverse.get(i, i) for i in ids],
                "source_ident": ids,
                "real_fish": True,
                "eodf_hz": eodf,
                "snr_db": snr,
                "source_dataset": self.info.get("dataset", ""),
                "source_recording": self.info.get("recording", ""),
                "segment_id": self.segment_id,
            }
        )


def add_background(
    recording: npt.ArrayLike,
    segment: BackgroundSegment,
    scale: float,
    *,
    geometry: GridGeometry | None = None,
    samplerate_hz: float | None = None,
    out: npt.NDArray[np.float32] | None = None,
    chunk_samples: int = 1 << 18,
) -> npt.NDArray[np.float32]:
    """Add a scaled background to a scene recording.

    Parameters
    ----------
    recording : array_like, shape (n_samples, n_electrodes)
        The simulated fish (and any other scene components), after channel
        gains.
    segment : BackgroundSegment
        Background with exactly the same shape.
    scale : float
        Factor on the background (:func:`scale_for_snr`).
    geometry : GridGeometry, optional
        The scene's electrode positions; must equal the segment's.
    samplerate_hz : float, optional
        The scene's sampling rate; must equal the segment's.
    out : numpy.ndarray of float32, optional
        Output array; may be ``recording`` itself (in place).
    chunk_samples : int
        Samples per chunk (bounds the memory of the float64 temporary).

    Returns
    -------
    numpy.ndarray of float32
        ``recording + scale * background``.

    Raises
    ------
    ValueError
        On a shape, geometry or sampling-rate mismatch, or a bad scale.
    """
    x = np.asarray(recording)
    if x.shape != segment.data.shape:
        msg = (
            f"recording has shape {x.shape} but the background segment "
            f"{segment.segment_id} has {segment.data.shape}; a hybrid scene must "
            "use the segment's grid, rate and duration"
        )
        raise ValueError(msg)
    if geometry is not None and not (
        geometry.positions_m.shape == segment.geometry.positions_m.shape
        and np.allclose(geometry.positions_m, segment.geometry.positions_m)
    ):
        msg = "scene geometry differs from the background segment's geometry"
        raise ValueError(msg)
    if samplerate_hz is not None and not math.isclose(
        samplerate_hz, segment.samplerate_hz
    ):
        msg = (
            f"scene rate {samplerate_hz} Hz differs from the background's "
            f"{segment.samplerate_hz} Hz"
        )
        raise ValueError(msg)
    if not (math.isfinite(scale) and scale >= 0):
        msg = f"scale must be finite and >= 0, got {scale!r}"
        raise ValueError(msg)
    if out is None:
        out = np.empty(x.shape, dtype=np.float32)
    elif out.shape != x.shape or out.dtype != np.float32:
        msg = f"out must be float32 with shape {x.shape}"
        raise ValueError(msg)
    for start in range(0, x.shape[0], chunk_samples):
        stop = min(x.shape[0], start + chunk_samples)
        acc = np.asarray(x[start:stop], dtype=np.float64)
        acc = acc + scale * np.asarray(segment.data[start:stop], dtype=np.float64)
        out[start:stop] = acc
    return out


def _resample(
    data: npt.NDArray[np.floating], from_hz: float, to_hz: float
) -> npt.NDArray[np.float32]:
    """Polyphase resampling along the first axis (rational approximation)."""
    from fractions import Fraction

    from scipy.signal import resample_poly

    ratio = Fraction(to_hz / from_hz).limit_denominator(1000)
    out = resample_poly(
        np.asarray(data, dtype=np.float64), ratio.numerator, ratio.denominator, axis=0
    )
    return np.asarray(out, dtype=np.float32)


def choose_channels(
    rng: np.random.Generator,
    n_channels: int,
    k: int,
    bad: Sequence[int] = (),
    mode: str = "contiguous",
) -> tuple[int, ...] | None:
    """Pick ``k`` of ``n_channels`` channels, avoiding ``bad`` ones.

    Parameters
    ----------
    rng : numpy.random.Generator
    n_channels : int
        Channels of the segment.
    k : int
        Channels wanted (1 <= k).
    bad : sequence of int
        Channels that must not be chosen.
    mode : {"contiguous", "random"}
        ``"contiguous"``: a run of ``k`` consecutive channels in channel order
        (uniform over the runs without bad channels; on a row-major grid a
        run may wrap to the next row). ``"random"``: ``k`` distinct allowed
        channels, uniformly, in ascending order.

    Returns
    -------
    tuple of int or None
        The channels, or None if no choice avoids the bad channels.
    """
    if mode not in CHANNEL_MODES:
        msg = f"mode must be one of {CHANNEL_MODES}, got {mode!r}"
        raise ValueError(msg)
    if k < 1:
        msg = f"k must be >= 1, got {k}"
        raise ValueError(msg)
    allowed = np.ones(n_channels, dtype=bool)
    allowed[[c for c in bad if 0 <= c < n_channels]] = False
    if k > n_channels:
        return None
    if mode == "contiguous":
        runs = [s for s in range(n_channels - k + 1) if bool(allowed[s : s + k].all())]
        if not runs:
            return None
        start = runs[int(rng.integers(len(runs)))]
        return tuple(range(start, start + k))
    pool = np.flatnonzero(allowed)
    if pool.size < k:
        return None
    return tuple(int(c) for c in np.sort(rng.choice(pool, size=k, replace=False)))


def _bad_channels(row: Mapping[str, Any]) -> list[int]:
    return sorted(
        {int(c) for c in row.get("dead_channels", []) or []}
        | {int(c) for c in row.get("noisy_channels", []) or []}
    )


def _subset_info(
    info: dict[str, Any], directory: Path, chosen: tuple[int, ...]
) -> dict[str, Any]:
    """Manifest row restricted to a channel subset (see :meth:`segment`)."""
    out = dict(info)
    index = {c: i for i, c in enumerate(chosen)}
    out["source_channels"] = list(chosen)
    out["n_electrodes"] = len(chosen)
    positions = info.get("electrode_positions_m")
    if isinstance(positions, list) and positions:
        out["electrode_positions_m"] = [positions[c] for c in chosen]
    for key in ("dead_channels", "noisy_channels"):
        out[key] = [index[c] for c in info.get(key, []) or [] if c in index]
    details_path = directory / "segment.json"
    if details_path.exists():
        details = json.loads(details_path.read_text())
        floors = np.asarray(details.get("noise_psd_per_electrode", []), dtype=float)
        if floors.size > max(chosen):
            noise = float(np.median(floors[list(chosen)]))
            out["noise_psd"] = noise
            out["level"] = noise * float(info["reference_bandwidth_hz"])
            out["level_db"] = 10 * math.log10(out["level"])
            out["noise_psd_db"] = 10 * math.log10(noise)
            snrs = []
            for fish in details.get("fish", []):
                power = np.asarray(fish.get("power_per_electrode", []), dtype=float)
                best = (
                    float(np.nanmax(power[list(chosen)]))
                    if power.size > max(chosen)
                    and np.isfinite(power[list(chosen)]).any()
                    else math.nan
                )
                snrs.append(
                    10 * math.log10(best / out["level"]) if best > 0 else math.nan
                )
            if snrs:
                out["fish_snr_db"] = snrs
    return out


class BackgroundLibrary:
    """A background library: manifest plus segment files.

    Parameters
    ----------
    root : path
        Library directory.
    manifest : pandas.DataFrame
        Segment table (a subset of the library's after :meth:`select`).
    """

    def __init__(self, root: str | Path, manifest: pd.DataFrame) -> None:
        self.root = Path(root)
        self.manifest = manifest.reset_index(drop=True)

    @classmethod
    def load(
        cls, root: str | Path = DEFAULT_LIBRARY, *, include_rejected: bool = False
    ) -> BackgroundLibrary:
        """Load the manifest of a built library.

        ``accept`` and ``note`` come from the author's ``review.csv`` if
        there is one (it may be newer than the manifest); rejected segments
        are dropped unless ``include_rejected``.
        """
        root = Path(root)
        manifest = read_manifest(root / "manifest.parquet")
        if "accept" not in manifest.columns:
            manifest["accept"] = True
        manifest["note"] = ""
        if (root / "review.csv").exists():
            from gridtools.background.review import read_review

            review = read_review(root)
            if review is not None:
                accept = dict(zip(review["segment_id"], review["accept"], strict=True))
                note = dict(zip(review["segment_id"], review["note"], strict=True))
                ids = manifest["segment_id"].astype(str).tolist()
                manifest["accept"] = [
                    bool(accept.get(i, a))
                    for i, a in zip(ids, manifest["accept"], strict=True)
                ]
                manifest["note"] = [str(note.get(i, "")) for i in ids]
        if not include_rejected:
            manifest = pd.DataFrame(
                manifest.loc[manifest["accept"].astype(bool).to_numpy()]
            )
        return cls(root, manifest)

    def __len__(self) -> int:
        return len(self.manifest)

    def __repr__(self) -> str:
        return f"BackgroundLibrary({str(self.root)!r}, {len(self)} segments)"

    def select(
        self,
        *,
        datasets: Sequence[str] | None = None,
        segment_class: str | Sequence[str] | None = None,
        screened: bool | None = None,
        geometry_known: bool | None = None,
        tracks_complete: bool | None = None,
        n_electrodes: int | None = None,
        samplerate_hz: float | None = None,
        min_duration_s: float | None = None,
        exclude_groups: Sequence[str] | None = None,
        include_groups: Sequence[str] | None = None,
    ) -> BackgroundLibrary:
        """Subset of the library; every given criterion must hold.

        Parameters
        ----------
        datasets : sequence of str, optional
            Keep these datasets.
        segment_class : str or sequence of str, optional
            ``"fish"`` and/or ``"fish_free"``.
        screened, geometry_known, tracks_complete : bool, optional
            Required value of the manifest flag.
        n_electrodes : int, optional
        samplerate_hz : float, optional
        min_duration_s : float, optional
        exclude_groups, include_groups : sequence of str, optional
            Split groups (``dataset/recording``) to drop or keep, for
            train/validation/test splits by recording.

        Returns
        -------
        BackgroundLibrary
        """
        m = self.manifest
        keep = np.ones(len(m), dtype=bool)
        if datasets is not None:
            keep &= m["dataset"].isin(list(datasets)).to_numpy()
        if segment_class is not None:
            classes = (
                [segment_class]
                if isinstance(segment_class, str)
                else list(segment_class)
            )
            keep &= m["segment_class"].isin(classes).to_numpy()
        for name, value in (
            ("screened", screened),
            ("geometry_known", geometry_known),
            ("tracks_complete", tracks_complete),
        ):
            if value is not None:
                keep &= (m[name].astype(bool) == value).to_numpy()
        if n_electrodes is not None:
            keep &= (m["n_electrodes"] == n_electrodes).to_numpy()
        if samplerate_hz is not None:
            keep &= np.isclose(m["samplerate_hz"].to_numpy(dtype=float), samplerate_hz)
        if min_duration_s is not None:
            keep &= m["duration_s"].to_numpy(dtype=float) >= min_duration_s - 1e-9
        if exclude_groups is not None:
            keep &= ~m["split_group"].isin(list(exclude_groups)).to_numpy()
        if include_groups is not None:
            keep &= m["split_group"].isin(list(include_groups)).to_numpy()
        return BackgroundLibrary(self.root, pd.DataFrame(m.loc[keep]))

    def segment_dir(self, segment_id: str) -> Path:
        """Directory of a segment."""
        row = self._row(segment_id)
        return self.root / str(row["path"])

    def _row(self, segment_id: str) -> pd.Series:
        rows = self.manifest[self.manifest["segment_id"] == segment_id]
        if rows.empty:
            raise KeyError(segment_id)
        return rows.iloc[0]

    def segment(
        self,
        segment_id: str,
        *,
        offset_s: float = 0.0,
        duration_s: float | None = None,
        samplerate_hz: float | None = None,
        channels: Sequence[int] | None = None,
    ) -> BackgroundSegment:
        """Load a segment (or an excerpt of it).

        Parameters
        ----------
        segment_id : str
        offset_s : float
            Start of the excerpt within the segment.
        duration_s : float, optional
            Length of the excerpt (default: to the end).
        samplerate_hz : float, optional
            Resample to this rate if the segment has another one (polyphase,
            :func:`scipy.signal.resample_poly`; the excerpt is then held in
            memory instead of memory-mapped). The level stays valid: it is a
            density in 500-1500 Hz, far below the anti-aliasing cut-off.
        channels : sequence of int, optional
            Keep only these channels (in this order). Positions, the tracks'
            ``powers_db`` columns, dead and noisy channels, the level (median
            floor of the chosen electrodes) and the real fish's SNRs (their
            strongest chosen electrode, from the per-electrode medians) are
            restricted to them.

        Returns
        -------
        BackgroundSegment
            Geometry from the manifest's ``electrode_positions_m`` (exact
            positions of field grids, nominal ones of the competition grids),
            else from ``fishgrid.cfg``.
        """
        row = self._row(segment_id)
        directory = self.root / str(row["path"])
        rec = open_recording(directory / "background.wav")
        positions = row.get("electrode_positions_m")
        if isinstance(positions, list) and len(positions) == rec.n_channels:
            geometry = GridGeometry(np.asarray(positions, dtype=np.float64))
        elif rec.geometry is not None:
            geometry = rec.geometry
        else:
            msg = f"{directory}: no electrode positions"
            raise ValueError(msg)
        fs = rec.samplerate_hz
        start = round(offset_s * fs)
        n = rec.n_samples - start if duration_s is None else round(duration_s * fs)
        if start < 0 or n < 1 or start + n > rec.n_samples:
            msg = (
                f"excerpt [{offset_s}, {offset_s + n / fs}] s outside segment "
                f"{segment_id} of {rec.duration_s} s"
            )
            raise ValueError(msg)
        tracks = None
        if (directory / "ident_v.npy").exists():
            full = load_tracks(directory)
            t0 = start / fs
            tracks = clip_tracks(full, t0, t0 + n / fs)
        data = np.asarray(rec.data[start : start + n], dtype=np.float32)
        if rec.scale != 1.0:
            data = data * np.float32(rec.scale)
        info = row.to_dict()
        chosen = tuple(range(rec.n_channels)) if channels is None else tuple(channels)
        if channels is not None:
            if not chosen or any(not 0 <= c < rec.n_channels for c in chosen):
                msg = f"channels {chosen} not in 0..{rec.n_channels - 1}"
                raise ValueError(msg)
            data = data[:, list(chosen)]
            geometry = GridGeometry(geometry.positions_m[list(chosen)])
            if tracks is not None and tracks.powers_db is not None:
                tracks = Tracks(
                    times_s=tracks.times_s,
                    freqs_hz=tracks.freqs_hz,
                    idents=tracks.idents,
                    time_indices=tracks.time_indices,
                    powers_db=tracks.powers_db[:, list(chosen)],
                )
            info = _subset_info(info, directory, chosen)
        rate = fs
        if samplerate_hz is not None and not math.isclose(samplerate_hz, fs):
            data, rate = _resample(data, fs, samplerate_hz), float(samplerate_hz)
            if duration_s is not None:
                data = data[: round(duration_s * rate)]
        return BackgroundSegment(
            segment_id=segment_id,
            data=data,
            samplerate_hz=rate,
            geometry=geometry,
            tracks=tracks,
            info=info,
            offset_s=start / fs,
            channels=chosen,
        )

    def weights(
        self, weights: str | npt.ArrayLike | None = "weight"
    ) -> npt.NDArray[np.float64]:
        """Normalised sampling weights of the segments.

        Parameters
        ----------
        weights : str, array_like or None
            A manifest column (default ``"weight"``: 1 for screened and
            fish-free segments, lower for unscreened ones), explicit
            non-negative weights, or None for uniform.
        """
        if weights is None:
            w = np.ones(len(self))
        elif isinstance(weights, str):
            w = self.manifest[weights].to_numpy(dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(self),) or np.any(w < 0) or not np.all(np.isfinite(w)):
            msg = f"weights must be {len(self)} finite non-negative values"
            raise ValueError(msg)
        if w.sum() <= 0:
            msg = "weights sum to zero"
            raise ValueError(msg)
        return w / w.sum()

    def sample(
        self,
        rng: np.random.Generator,
        duration_s: float,
        *,
        n_electrodes: int | None = None,
        channels: str = "contiguous",
        avoid_bad_channels: bool = True,
        geometry: GridGeometry | None = None,
        samplerate_hz: float | None = None,
        resample: bool = False,
        weights: str | npt.ArrayLike | None = "weight",
    ) -> BackgroundSegment:
        """Draw a segment, a random excerpt of ``duration_s`` and its channels.

        Parameters
        ----------
        rng : numpy.random.Generator
            Source of the segment choice and the excerpt offset.
        duration_s : float
            Excerpt length; segments shorter than this are not eligible.
        n_electrodes : int, optional
            Electrodes of the scene. Segments with at least this many
            channels are eligible; the excerpt keeps ``n_electrodes`` of them
            (:func:`choose_channels`), so a single-electrode scene can use any
            segment. Default: all channels of the drawn segment.
        channels : {"contiguous", "random"}
            How the subset is chosen (default a contiguous run in channel
            order).
        avoid_bad_channels : bool
            Never choose dead or noisy channels (default True). With
            ``n_electrodes`` equal to a segment's channel count this excludes
            segments with bad channels; without ``n_electrodes`` it has no
            effect (the whole grid, bad channels listed in the segment).
        geometry : GridGeometry, optional
            Only segments with exactly these electrode positions (whole grid,
            no subset).
        samplerate_hz : float, optional
            Only segments at this rate, or, with ``resample``, every segment
            resampled to it (field loggers record at 48 or 96 kHz).
        resample : bool
            See ``samplerate_hz``.
        weights : str, array_like or None
            See :meth:`weights` (array_like: one value per segment of this
            library, before the eligibility filter).

        Returns
        -------
        BackgroundSegment

        Raises
        ------
        ValueError
            If no segment is eligible.
        """
        if channels not in CHANNEL_MODES:
            msg = f"channels must be one of {CHANNEL_MODES}, got {channels!r}"
            raise ValueError(msg)
        w = self.weights(weights)
        m = self.manifest
        eligible = m["duration_s"].to_numpy(dtype=float) >= duration_s - 1e-9
        if n_electrodes is not None:
            probe = np.random.default_rng(0)  # feasibility only, not the draw
            eligible &= np.array(
                [
                    choose_channels(
                        probe,
                        int(r["n_electrodes"]),
                        n_electrodes,
                        _bad_channels(r) if avoid_bad_channels else (),
                        channels,
                    )
                    is not None
                    for r in m.to_dict("records")
                ],
                dtype=bool,
            )
        if samplerate_hz is not None and not resample:
            eligible &= np.isclose(
                m["samplerate_hz"].to_numpy(dtype=float), samplerate_hz
            )
        if geometry is not None:
            target = np.asarray(geometry.positions_m)
            eligible &= np.array(
                [
                    np.shape(p) == target.shape and np.allclose(p, target)
                    for p in m["electrode_positions_m"]
                ],
                dtype=bool,
            )
        w = np.where(eligible, w, 0.0)
        if w.sum() <= 0:
            msg = (
                f"no background segment of >= {duration_s} s matches "
                f"(n_electrodes={n_electrodes}, geometry given={geometry is not None}, "
                f"samplerate_hz={samplerate_hz})"
            )
            raise ValueError(msg)
        index = int(rng.choice(len(m), p=w / w.sum()))
        row = m.iloc[index]
        fs = float(row["samplerate_hz"])
        n = round(duration_s * fs)
        slack = int(row["n_samples"]) - n
        start = int(rng.integers(0, slack + 1)) if slack > 0 else 0
        subset = None
        if n_electrodes is not None:
            subset = choose_channels(
                rng,
                int(row["n_electrodes"]),
                n_electrodes,
                _bad_channels(row.to_dict()) if avoid_bad_channels else (),
                channels,
            )
        return self.segment(
            str(row["segment_id"]),
            offset_s=start / fs,
            duration_s=n / fs,
            samplerate_hz=samplerate_hz if resample else None,
            channels=subset,
        )
