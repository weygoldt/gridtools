"""Real backgrounds in simulated scenes (hybrid scenes).

A hybrid scene replaces the synthetic electrode noise, hum and clicks by an
excerpt of a real grid recording from the background library
(:mod:`gridtools.sim.background`, built by :mod:`gridtools.background`).
The excerpt fixes the grid: the scene uses its channels with their
positions and order, and the simulated fish are rendered on them. The
recording is

    recording = sum_i render_fish(i) + scale * background

(float32 rounding of the sum, then the scene's clipping if enabled), where
``scale`` sets the simulated fish's SNR against the background and converts
the background's unit (ADC volts, or int16 full scale for field loggers)
into the scene's. Real fish in the excerpt keep their tracks and join the
scene's tracks with identities from :data:`REAL_FISH_IDENT_OFFSET` on; they
never chirp in the ground truth.

This module draws the segment and its channels (scene stream
``"background"``), derives the scene's layout from them, and reproduces a
written scene's background from its ``scene.json`` record. Scene assembly
is in :mod:`gridtools.sim.scene`; the config is
:class:`~gridtools.sim.config.HybridConfig` (TOML section ``[background]``).
"""

import dataclasses
import functools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gridtools.background.levels import SnrDistribution, scale_for_snr
from gridtools.background.library import (
    REAL_FISH_IDENT_OFFSET,
    BackgroundLibrary,
    BackgroundSegment,
    choose_channels,
)
from gridtools.sim._common import Float32Array
from gridtools.sim.config import HybridConfig

__all__ = [
    "REAL_FISH_IDENT_OFFSET",
    "HybridBackground",
    "background_library",
    "candidates",
    "check_duration",
    "draw_segment",
    "load_segment",
    "natural_scale",
    "segment_spacing",
    "target_snr",
]

_EPS_S = 1e-9


@functools.lru_cache(maxsize=4)
def background_library(
    root: str, *, include_rejected: bool = False
) -> BackgroundLibrary:
    """The background library at ``root`` (manifest cached per process).

    Scenes draw from the accepted segments; a written scene's background is
    reproduced with ``include_rejected=True``, so a later review that
    rejects its segment does not break it.
    """
    return BackgroundLibrary.load(root, include_rejected=include_rejected)


def _library(cfg: HybridConfig) -> str:
    """The configured library directory; hybrid scenes need one."""
    if cfg.library is None:
        msg = 'hybrid scenes need [background] library = "PATH"'
        raise ValueError(msg)
    return cfg.library


@functools.lru_cache(maxsize=8)
def candidates(cfg: HybridConfig) -> tuple[pd.DataFrame, np.ndarray]:
    """Eligible segments (manifest rows) and their unnormalised weights.

    Weight 1 for screened and fish-free segments, ``unscreened_weight`` for
    the others, times ``dataset_weights``.

    Raises
    ------
    ValueError
        If a dataset in ``datasets`` or ``dataset_weights`` is not in the
        library, or nothing is left.
    """
    lib = background_library(_library(cfg))
    known = set(lib.manifest["dataset"])
    named = set(cfg.datasets) | {name for name, _ in cfg.dataset_weights}
    unknown = sorted(named - known)
    if unknown:
        msg = (
            f"background datasets {unknown} are not in {cfg.library} "
            f"(it has {sorted(known)})"
        )
        raise ValueError(msg)
    selected = lib.select(
        datasets=cfg.datasets or None,
        include_groups=cfg.include_groups or None,
        exclude_groups=cfg.exclude_groups or None,
    )
    m = selected.manifest.reset_index(drop=True)
    if m.empty:
        msg = f"no background segment of {cfg.library} matches the [background] filters"
        raise ValueError(msg)
    screened = m["screened"].to_numpy(dtype=bool)
    weights = np.where(screened, 1.0, cfg.unscreened_weight)
    factors = dict(cfg.dataset_weights)
    weights = weights * np.array(
        [factors.get(str(d), 1.0) for d in m["dataset"]], dtype=np.float64
    )
    return m, weights


def check_duration(cfg: HybridConfig, duration_s: float) -> None:
    """Refuse hybrid scenes longer than the longest eligible segment.

    The library's segments are 60 s; scenes longer than that would need
    several segments (a cut in the background), so they are refused rather
    than tiled. Set ``background.prob = 0`` for longer scenes.
    """
    m, weights = candidates(cfg)
    longest = (
        float(m.loc[weights > 0, "duration_s"].max()) if np.any(weights > 0) else 0.0
    )
    if duration_s > longest + _EPS_S:
        msg = (
            f"hybrid scenes are limited to the background segments' {longest:g} s, "
            f"got duration_s = {duration_s:g}; set background.prob = 0 for longer "
            "scenes"
        )
        raise ValueError(msg)


def _bad_channels(row: Mapping[str, Any]) -> list[int]:
    return sorted(
        {int(c) for c in row.get("dead_channels", []) or []}
        | {int(c) for c in row.get("noisy_channels", []) or []}
    )


@functools.lru_cache(maxsize=64)
def _floors(root: str, path: str) -> np.ndarray | None:
    details = Path(root) / path / "segment.json"
    if not details.exists():
        return None
    floors = json.loads(details.read_text()).get("noise_psd_per_electrode")
    return None if not floors else np.asarray(floors, dtype=np.float64)


def silent_channels(lib: BackgroundLibrary, segment_id: str) -> list[int]:
    """Channels of a stored segment without any signal (zero noise floor).

    Some field segments (flona2025) have half of the logger's channels at
    exactly zero without listing them as dead; they count as dead here, are
    never chosen for a channel subset, and do not enter the level.
    """
    row = lib.manifest.loc[lib.manifest["segment_id"] == segment_id]
    if row.empty:
        raise KeyError(segment_id)
    floors = _floors(str(lib.root), str(row["path"].iloc[0]))
    if floors is None:
        return []
    return [int(c) for c in np.flatnonzero(~(floors > 0.0))]


def _choose(
    rng: np.random.Generator,
    row: Mapping[str, Any],
    k: int,
    cfg: HybridConfig,
    silent: Sequence[int] = (),
) -> tuple[int, ...] | None:
    """``k`` channels of a segment, or None for the whole grid.

    The whole grid keeps its bad channels (they are part of the recording).
    A subset never contains silent channels (``k`` is capped by the others)
    and avoids dead and noisy ones if ``avoid_bad_channels``; if no run of
    ``k`` allowed channels exists in ``contiguous`` mode, a random subset of
    allowed channels is drawn, and only if that fails too may dead or noisy
    channels be chosen.
    """
    n = int(row["n_electrodes"])
    if k >= n:
        return None
    k = min(k, n - len(silent))
    bad = sorted(
        set(_bad_channels(row) if cfg.avoid_bad_channels else []) | set(silent)
    )
    for mode, avoid in (
        (cfg.channels, bad),
        ("random", bad),
        (cfg.channels, list(silent)),
        ("random", list(silent)),
    ):
        chosen = choose_channels(rng, n, k, avoid, mode)
        if chosen is not None:
            return chosen
    msg = f"cannot choose {k} of {n} channels"  # unreachable for 1 <= k < n
    raise RuntimeError(msg)


def load_segment(
    lib: BackgroundLibrary,
    segment_id: str,
    *,
    offset_s: float,
    source_n_samples: int,
    channels: Sequence[int] | None,
    samplerate_hz: float,
    n_samples: int,
) -> BackgroundSegment:
    """Load an excerpt at the scene rate with exactly ``n_samples`` samples.

    ``source_n_samples`` samples from ``offset_s`` at the segment's own
    rate, resampled to ``samplerate_hz`` if that differs (polyphase), then
    cut to ``n_samples``. Deterministic: the same arguments give the same
    samples.
    """
    row = lib.manifest.loc[lib.manifest["segment_id"] == segment_id]
    if row.empty:
        raise KeyError(segment_id)
    source_hz = float(row["samplerate_hz"].iloc[0])
    resample = not math.isclose(source_hz, samplerate_hz)
    segment = lib.segment(
        segment_id,
        offset_s=offset_s,
        duration_s=source_n_samples / source_hz,
        samplerate_hz=samplerate_hz if resample else None,
        channels=None if channels is None else list(channels),
    )
    if segment.n_samples < n_samples:
        msg = (
            f"background {segment_id}: {segment.n_samples} samples at "
            f"{samplerate_hz} Hz, the scene needs {n_samples}"
        )
        raise ValueError(msg)
    if segment.n_samples > n_samples:
        segment = dataclasses.replace(segment, data=segment.data[:n_samples])
    silent = silent_channels(lib, segment_id)
    columns = [i for i, c in enumerate(segment.channels) if c in set(silent)]
    if columns:  # silent channels: dead, and not part of the level
        floors = _floors(str(lib.root), str(row["path"].iloc[0]))
        assert floors is not None
        live = [c for c in segment.channels if c not in set(silent)]
        if not live:
            msg = f"background {segment_id}: only silent channels chosen"
            raise ValueError(msg)
        info = dict(segment.info)
        info["silent_channels"] = columns
        info["dead_channels"] = sorted(set(segment.dead_channels) | set(columns))
        info["level"] = float(np.median(floors[live])) * float(
            info["reference_bandwidth_hz"]
        )
        info["level_db"] = 10.0 * math.log10(info["level"])
        segment = dataclasses.replace(segment, info=info)
    return segment


def draw_segment(
    rng: np.random.Generator,
    cfg: HybridConfig,
    k: int | None,
    samplerate_hz: float,
    n_samples: int,
) -> tuple[BackgroundSegment, dict[str, Any]]:
    """Draw a segment, its channels and an excerpt for a scene.

    Parameters
    ----------
    rng : numpy.random.Generator
        The scene's ``"background"`` stream.
    cfg : HybridConfig
    k : int or None
        Electrodes wanted (capped by the segment's channel count); None for
        the segment's whole grid.
    samplerate_hz : float
        Scene rate; the excerpt is resampled to it if needed.
    n_samples : int
        Scene length in samples.

    Returns
    -------
    segment : BackgroundSegment
    choice : dict
        What reproduces the excerpt with :func:`load_segment` (segment id,
        offset, source samples, channels or None).
    """
    m, weights = candidates(cfg)
    duration_s = n_samples / samplerate_hz
    eligible = m["duration_s"].to_numpy(dtype=float) >= duration_s - _EPS_S
    w = np.where(eligible, weights, 0.0)
    if w.sum() <= 0:
        msg = f"no background segment of >= {duration_s:g} s with nonzero weight"
        raise ValueError(msg)
    index = int(rng.choice(len(m), p=w / w.sum()))
    row = m.iloc[index].to_dict()
    n_channels = int(row["n_electrodes"])
    lib = background_library(_library(cfg))
    silent = silent_channels(lib, str(row["segment_id"]))
    channels = _choose(
        rng, row, n_channels if k is None else min(k, n_channels), cfg, silent
    )
    source_hz = float(row["samplerate_hz"])
    source_n = math.ceil(duration_s * source_hz - 1e-6)
    slack = int(row["n_samples"]) - source_n
    start = int(rng.integers(0, slack + 1)) if slack > 0 else 0
    choice = {
        "segment_id": str(row["segment_id"]),
        "offset_s": start / source_hz,
        "source_n_samples": source_n,
        "channels": channels,
    }
    segment = load_segment(
        lib,
        choice["segment_id"],
        offset_s=choice["offset_s"],
        source_n_samples=source_n,
        channels=channels,
        samplerate_hz=samplerate_hz,
        n_samples=n_samples,
    )
    return segment, choice


#: Unit of the scenes and of the competition recordings.
VOLT_UNITS_PREFIX = "V at the ADC"


@functools.lru_cache(maxsize=16)
def natural_scale(root: str, dataset: str) -> float:
    """Scale that puts a dataset's backgrounds at their own absolute level.

    1 for recordings in ADC volts (the scenes' unit). Field loggers store
    int16 full scale with an uncalibrated gain; their natural scale maps the
    dataset's median background level onto the median level of the
    library's volt recordings, so the background lands in the range of real
    grid recordings.
    """
    m = background_library(root).manifest
    units = m["units"].astype(str)
    volts = units.str.startswith(VOLT_UNITS_PREFIX).to_numpy()
    own = (m["dataset"] == dataset).to_numpy()
    if not own.any() or bool(volts[own].all()) or not volts.any():
        return 1.0
    reference = float(np.median(m.loc[volts, "level"].to_numpy(dtype=float)))
    level = float(np.median(m.loc[own, "level"].to_numpy(dtype=float)))
    return math.sqrt(reference / level)


def segment_spacing(segment: BackgroundSegment) -> float:
    """Electrode spacing of an excerpt: median nearest-neighbour distance.

    One electrode: the segment's nominal grid spacing.
    """
    positions = np.asarray(segment.geometry.positions_m, dtype=np.float64)
    if positions.shape[0] >= 2:
        d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return float(np.median(d.min(axis=1)))
    for key in ("column_spacing_m", "row_spacing_m"):
        value = segment.info.get(key)
        if value is not None and np.isfinite(value) and value > 0:
            return float(value)
    return 0.5


def real_snr_max_db(segment: BackgroundSegment) -> float:
    """SNR of the excerpt's loudest real fish on its chosen electrodes (NaN if none)."""
    snrs = np.asarray(segment.info.get("fish_snr_db", []) or [], dtype=np.float64)
    if snrs.size and np.isfinite(snrs).any():
        return float(np.nanmax(snrs))
    return float(segment.info.get("real_snr_max_db", math.nan))


def target_snr(
    rng: np.random.Generator, cfg: HybridConfig, segment: BackgroundSegment
) -> tuple[float, float | None]:
    """Draw the simulated fish's target SNR (dB) and the cap that applied.

    The cap keeps the excerpt's real fish at most ``max_real_gap_db``
    louder than the simulated ones: the SNR is drawn at or above
    ``real_snr_max_db - max_real_gap_db`` (on the chosen electrodes).
    """
    real = real_snr_max_db(segment)
    min_db = (
        None
        if cfg.max_real_gap_db is None or not math.isfinite(real)
        else real - cfg.max_real_gap_db
    )
    dist = SnrDistribution(
        hard_db=cfg.snr_hard_db,
        typical_db=cfg.snr_typical_db,
        hard_fraction=cfg.snr_hard_fraction,
    )
    return float(dist.sample(rng, min_db=min_db)), min_db


@dataclass(frozen=True, eq=False)
class HybridBackground:
    """The real background of a hybrid scene.

    Attributes
    ----------
    config : HybridConfig
    segment : BackgroundSegment
        The excerpt at the scene rate, ``n_samples`` long, chosen channels.
    choice : dict
        Arguments of :func:`load_segment` that reproduce ``segment``.
    source_samplerate_hz : float
        Rate of the stored segment.
    signal_level : float
        Level of the simulated fish (:func:`fish_signal_level`, scene unit),
        after ``fish_gain``.
    snr_db : float
        Target SNR of the simulated fish against the background.
    min_snr_db : float or None
        Lower bound of the SNR draw from the real-fish gap cap.
    scale : float
        Factor applied to the background.
    fish_gain : float
        Factor applied to the simulated fish's amplitudes (<= 1).
    natural_scale : float
        Scale at which the background has its own absolute level in the
        scene's unit (see :func:`natural_scale`).
    """

    config: HybridConfig
    segment: BackgroundSegment
    choice: dict[str, Any]
    source_samplerate_hz: float
    signal_level: float
    snr_db: float
    min_snr_db: float | None
    scale: float
    fish_gain: float = 1.0
    natural_scale: float = 1.0

    @classmethod
    def build(
        cls,
        cfg: HybridConfig,
        segment: BackgroundSegment,
        choice: dict[str, Any],
        signal_level: float,
        rng: np.random.Generator,
    ) -> "HybridBackground":
        """Draw the SNR and split it into a background scale and a fish gain.

        :func:`scale_for_snr` gives the ratio ``s`` of background scale to
        fish gain. Unless ``cfg.amplify_background``, nothing is amplified
        beyond its own level: up to the background's natural scale the
        background is attenuated (fish gain 1, amplitudes as drawn); a
        harder SNR attenuates the simulated fish instead (background at its
        natural scale, fish gain ``natural / s``). Amplifying the background
        by up to 40 dB would amplify its real fish and clicks with it and
        clip about a third of the scenes at ``noise.clip_limit``.
        """
        snr, min_db = target_snr(rng, cfg, segment)
        ratio = scale_for_snr(segment.level, signal_level, snr)
        natural = natural_scale(_library(cfg), str(segment.info.get("dataset", "")))
        scale, fish_gain = ratio, 1.0
        if not cfg.amplify_background and ratio > natural:
            scale, fish_gain = natural, natural / ratio
        return cls(
            config=cfg,
            segment=segment,
            choice=choice,
            source_samplerate_hz=float(segment.info["samplerate_hz"]),
            signal_level=signal_level * fish_gain**2,
            snr_db=snr,
            min_snr_db=min_db,
            scale=scale,
            fish_gain=fish_gain,
            natural_scale=natural,
        )

    def scaled(self, start: int = 0, stop: int | None = None) -> Float32Array:
        """``scale * background`` over samples ``[start, stop)`` as float32."""
        data = np.asarray(self.segment.data[start:stop], dtype=np.float64)
        return (self.scale * data).astype(np.float32)

    def record(self) -> dict[str, Any]:
        """The ``background`` entry of ``scene.json`` (without the real fish)."""
        info = self.segment.info
        channels = self.choice["channels"]
        return {
            "used": True,
            "library": self.config.library,
            "segment_id": self.segment.segment_id,
            "dataset": str(info.get("dataset", "")),
            "recording": str(info.get("recording", "")),
            "split_group": str(info.get("split_group", "")),
            "segment_class": str(info.get("segment_class", "")),
            "screened": bool(info.get("screened", False)),
            "offset_s": float(self.choice["offset_s"]),
            "source_n_samples": int(self.choice["source_n_samples"]),
            "source_samplerate_hz": self.source_samplerate_hz,
            "resampled": not math.isclose(
                self.source_samplerate_hz, self.segment.samplerate_hz
            ),
            "whole_grid": channels is None,
            "channels": list(self.segment.channels),
            "geometry_known": self.segment.geometry_known,
            "positions_nominal": self.segment.positions_nominal,
            "source_units": str(info.get("units", "")),
            "dead_channels": self.segment.dead_channels,
            "noisy_channels": self.segment.noisy_channels,
            "silent_channels": list(info.get("silent_channels", [])),
            "level": self.segment.level,
            "signal_level": self.signal_level,
            "snr_db": self.snr_db,
            "min_snr_db": self.min_snr_db,
            "real_snr_max_db": real_snr_max_db(self.segment),
            "scale": self.scale,
            "fish_gain": self.fish_gain,
            "natural_scale": self.natural_scale,
            "tracks_complete": bool(info.get("tracks_complete", True)),
            "definition": (
                "recording = sum of the simulated fish + scale * background "
                "(source samples from offset_s, channels in this order, resampled "
                "to the scene rate if resampled); scale / fish_gain = "
                "sqrt(signal_level / 10**(snr_db/10) / level) with signal_level "
                "before fish_gain; fish_gain < 1 (already in the fish amplitudes) "
                "only where scale would exceed natural_scale; levels in unit**2 in "
                "the reference bandwidth (gridtools.background.levels)"
            ),
        }


def reproduce(
    record: Mapping[str, Any], samplerate_hz: float, n_samples: int
) -> HybridBackground:
    """Rebuild the background of a written scene from its ``scene.json`` record."""
    lib = background_library(str(record["library"]), include_rejected=True)
    channels = None if record.get("whole_grid") else tuple(record["channels"])
    choice = {
        "segment_id": str(record["segment_id"]),
        "offset_s": float(record["offset_s"]),
        "source_n_samples": int(record["source_n_samples"]),
        "channels": channels,
    }
    segment = load_segment(
        lib,
        choice["segment_id"],
        offset_s=choice["offset_s"],
        source_n_samples=choice["source_n_samples"],
        channels=channels,
        samplerate_hz=samplerate_hz,
        n_samples=n_samples,
    )
    return HybridBackground(
        config=HybridConfig(library=str(record["library"])),
        segment=segment,
        choice=choice,
        source_samplerate_hz=float(record["source_samplerate_hz"]),
        signal_level=float(record["signal_level"]),
        snr_db=float(record["snr_db"]),
        min_snr_db=record.get("min_snr_db"),
        scale=float(record["scale"]),
        fish_gain=float(record.get("fish_gain", 1.0)),
        natural_scale=float(record.get("natural_scale", 1.0)),
    )
