"""Scene assembly: complete synthetic grid recordings with exact ground truth.

``simulate_scene(config, seed)`` draws a scene from a
:class:`~gridtools.sim.config.SceneConfig` and renders it;
``write_scene(scene, out_dir)`` writes it to a directory:

``recording.wav``
    Multichannel float32 wav (WAVE_FORMAT_EXTENSIBLE), one channel per
    electrode in row-major grid order. Unit: the EOD potential on the body
    axis at the field's reference distance of a fish with amplitude 1
    (nominally mV; see :class:`~gridtools.sim.config.FieldConfig`). Not
    normalised, not clipped unless ``noise.clip_limit`` is set.
``fishgrid.cfg``
    Sampling rate and grid geometry, so :func:`gridtools.io.open_recording`
    finds the electrode positions.
``times.npy``, ``fund_v.npy``, ``ident_v.npy``, ``idx_v.npy``, ``sign_v.npy``
    Ground-truth tracks in wavetracker layout: baseline EODf without chirp
    excursions (but with drift and ramps) on frames every
    ``tracks.step_s`` from 0 s up to the first frame at or after the last
    sample; ``ident_v`` = fish id; ``sign_v`` = power of each fish's
    fundamental on each electrode in dB, see :func:`fish_power_db`.
``chirps.csv``, ``chirps.parquet``
    One row per chirp (columns: :data:`CHIRP_COLUMNS`). A label box spans
    ``t_onset_s``-``t_offset_s`` (10 % crossings of the excursion) x
    ``f_low_hz``-``f_high_hz`` (baseline + min/max of the excursion), centred
    at ``t_center_s``; ``emitter_id`` is the track ident. The column name
    ``visible`` is reserved for footprint-based visibility labels.
``clicks.csv``
    One row per click artefact.
``paths.npz``
    Positions, headings, velocities and rest states of every fish at the
    movement rate.
``chirp_traces.npz``
    The placed chirp traces exactly as rendered (with ``paths.npz`` and
    ``scene.json`` enough to re-render any fish without the template
    library).
``scene.json``
    ``duration_s`` and ``samplerate_hz`` at the top level, resolved config,
    seed and random streams, versions, template library manifest hash,
    electrode positions, per-fish parameters and the render model.
``spectrogram_overview.png``, ``spectrogram_zooms.png``
    Annotated spectrograms (:mod:`gridtools.sim.plot`).

Hybrid scenes. With probability ``background.prob`` a scene uses a real
background segment (:mod:`gridtools.sim.hybrid`): the segment's electrodes
(all, or a subset drawn from the usual electrode-count distribution, with
their positions and channel order) replace the drawn layout, the segment is
resampled to the scene rate, synthetic noise, hum and clicks are off, and
the real fish join the tracks with identities from 1000 on (``scene.json``
``real_fish``; they never chirp, ``chirps.csv`` lists simulated chirps only).
``scene.json`` ``background`` records the segment, offset, channels, scale
and SNR, so :func:`render_background` reloads it from the library.

Randomness. One seed sequence per scene; every component draws from its own
child sequence with a fixed spawn key (:mod:`gridtools.sim.seeding`), so
changing one component's config does not reshuffle the others (e.g. the
noise config leaves the fish untouched).
"""

import dataclasses
import functools
import importlib.metadata
import json
import logging
import math
import platform
import subprocess
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from gridtools.background.levels import fish_signal_level
from gridtools.background.library import add_background, merge_tracks
from gridtools.io import GridGeometry, Tracks, save_tracks
from gridtools.sim._common import Float32Array, FloatArray
from gridtools.sim.chirps import (
    ChirpPlacement,
    ChirpSchedule,
    TemplatePool,
    amplitude_from_frequency,
    place_chirp,
    random_dips,
    schedule_chirps,
)
from gridtools.sim.clicks import Click, add_clicks, click_table, draw_clicks
from gridtools.sim.config import SceneConfig
from gridtools.sim.eod import (
    APTERONOTUS_LEPTORHYNCHUS,
    Harmonics,
    _smooth_noise,
    jitter_harmonics,
)
from gridtools.sim.field import DipoleParams, dipole_gain
from gridtools.sim.hybrid import (
    REAL_FISH_IDENT_OFFSET,
    HybridBackground,
    check_duration,
    draw_segment,
    reproduce,
    segment_spacing,
)
from gridtools.sim.mix import mix
from gridtools.sim.movement import Arena, FishPath, simulate_path
from gridtools.sim.noise import (
    ColoredNoiseParams,
    ElectrodeFaultParams,
    ElectrodeFaults,
    GainMismatchParams,
    Hum,
    HumParams,
    colored_noise,
    electrode_faults,
    gain_mismatch,
    powerline_hum,
)
from gridtools.sim.output import write_fishgrid_config, write_wav
from gridtools.sim.render import (
    ChirpSelection,
    DriftKnots,
    FishModel,
    PhaseMode,
    Ramp,
    SceneModel,
)
from gridtools.sim.seeding import (
    FISH_STREAMS,
    SEED_SCHEME,
    STREAMS,
    fish_stream,
    stream,
)

if TYPE_CHECKING:
    from gridtools.sim.footprint import HannSpectrogram

__all__ = [
    "CHIRP_COLUMNS",
    "DEFAULT_ROOT",
    "FISH_STREAMS",
    "FOOTPRINT_COLUMNS",
    "STREAMS",
    "Scene",
    "ScenePlan",
    "fish_power_db",
    "load_scene_model",
    "plan_scene",
    "render_background",
    "render_fish",
    "scene_dir",
    "simulate_scene",
    "write_scene",
]

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
DEFAULT_ROOT = Path("/mnt/data1/chirp_v2/scenes")
OVERLAP_WINDOW_S = 0.25

#: Columns of ``chirps.csv``.
CHIRP_COLUMNS: tuple[str, ...] = (
    "chirp_id",
    "emitter_id",
    "t_onset_s",
    "t_center_s",
    "t_peak_s",
    "t_offset_s",
    "t_support_start_s",
    "t_support_stop_s",
    "duration_s",
    "baseline_hz",
    "excursion_hz",
    "undershoot_hz",
    "f_low_hz",
    "f_high_hz",
    "dip",
    "amplitude_mode",
    "template_id",
    "family",
    "is_response",
    "response_to",
    "overlaps_other_chirp",
    "dt_nearest_other_chirp_s",
    "min_df_hz",
    "nearest_fish_id",
    "truncated",
    "emitter_x_m",
    "emitter_y_m",
    "emitter_power_db_max",
)

#: Footprint columns of ``chirps.csv`` (reference front end, see
#: :mod:`gridtools.sim.footprint`); ``visible`` is the label/ignore flag.
FOOTPRINT_COLUMNS: tuple[str, ...] = (
    "box_t0_s",
    "box_t1_s",
    "box_f0_hz",
    "box_f1_hz",
    "snr_db",
    "snr_baseline_db",
    "snr_excursion_db",
    "visible_fraction",
    "visible",
    "harmonic_box_t0_s",
    "harmonic_box_t1_s",
    "harmonic_box_f0_hz",
    "harmonic_box_f1_hz",
)


def fish_power_db(
    fundamental_amplitude: np.ndarray | float,
    samplerate_hz: float,
    nfft: int,
) -> np.ndarray:
    """Noise-free power density of an EOD fundamental, in dB re 1 V^2/Hz.

    For a sinusoid of amplitude ``a`` centred on a bin of a Hann-window
    spectrum with ``specgram`` scaling the peak density is
    ``(a**2 / 2) / ENBW`` with ``ENBW = 1.5 * samplerate_hz / nfft``. Used for
    ``emitter_power_db_max`` in ``chirps.csv`` (the fish's own power without
    noise, other fish or scalloping); ``sign_v`` itself is measured on the
    recording (:func:`sign_v_powers`). Floored at -300 dB.

    Parameters
    ----------
    fundamental_amplitude : array_like
        Amplitude of the fundamental (any sign).
    samplerate_hz : float
        Sampling rate.
    nfft : int
        Window length of the reference spectrum.

    Returns
    -------
    numpy.ndarray
        Power in dB.
    """
    enbw_hz = 1.5 * samplerate_hz / nfft
    power = 0.5 * np.square(np.asarray(fundamental_amplitude, dtype=np.float64))
    return 10.0 * np.log10(np.maximum(power / enbw_hz, 1e-30))


# -- planning -------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ScenePlan:
    """All random draws of a scene except the full-rate signals.

    Attributes
    ----------
    config : SceneConfig
    seed : int
    rows, columns : int
        Grid size for ``fishgrid.cfg`` (``1 x E`` for irregular layouts).
    spacing_m : float
        Electrode spacing.
    layout : Layout
        Electrode layout.
    geometry : GridGeometry
        Electrode positions.
    arena : Arena
        Tank around the grid.
    model : SceneModel
        Render model of the fish (frequency, chirps, paths, channel gains).
    schedule : ChirpSchedule
        Chirp timing statistics.
    fish_info : list of dict
        Per-fish parameters for ``scene.json``.
    levels : dict
        Reference amplitude, SNR, noise SD and exponent.
    channels : dict
        Gain mismatch and fault flags.
    pool : TemplatePool
        Template source.
    background : HybridBackground or None
        The real background of a hybrid scene (None: synthetic background).
    """

    config: SceneConfig
    seed: int
    rows: int
    columns: int
    spacing_m: float
    layout: "Layout"
    geometry: GridGeometry
    arena: Arena
    model: SceneModel
    schedule: ChirpSchedule
    fish_info: list[dict[str, Any]]
    levels: dict[str, Any]
    channels: dict[str, Any]
    pool: TemplatePool
    background: HybridBackground | None = None

    @property
    def hybrid(self) -> bool:
        """Whether the scene has a real background."""
        return self.background is not None


def _drift_knots(
    rng: np.random.Generator,
    duration_s: float,
    *,
    sd: float,
    timescale_s: float,
    control_rate_hz: float,
) -> DriftKnots:
    """Knots of a smooth drift, as :func:`gridtools.sim.eod.baseline_frequency`."""
    pad = 2
    n_control = math.ceil(duration_s * control_rate_hz) + 1 + 2 * pad
    times = (np.arange(n_control, dtype=np.float64) - pad) / control_rate_hz
    values = _smooth_noise(rng, n_control, sd, timescale_s * control_rate_hz)
    return DriftKnots(times, values)


def _ramp(duration_s: float, scene_s: float, start_u: float, delta_hz: float) -> Ramp:
    duration = min(duration_s, 0.8 * scene_s)
    return Ramp(
        start_s=start_u * (scene_s - duration), duration_s=duration, delta_hz=delta_hz
    )


def _population(
    config: SceneConfig, seed: int
) -> tuple[list[float], list[list[Ramp]], list[dict[str, Any]]]:
    """Baselines, ramps and pair info per fish (population stream)."""
    pop = config.population
    rng = stream(seed, "population")
    scene_s = config.duration_s
    n_fish = int(rng.integers(pop.n_fish[0], pop.n_fish[1] + 1))
    lo, hi = pop.baseline_uniform_hz
    baselines: list[float] = []
    ramps: list[list[Ramp]] = []
    info: list[dict[str, Any]] = []
    for i in range(n_fish):
        # fixed number of draws per fish, whatever the branches
        u = rng.random(15)
        z = float(rng.standard_normal())
        close = i > 0 and u[1] < pop.close_pair_prob
        partner = min(int(u[2] * i), i - 1) if i > 0 else -1
        df = pop.close_pair_df_hz[0] + u[3] * (
            pop.close_pair_df_hz[1] - pop.close_pair_df_hz[0]
        )
        sign = 1.0 if u[4] < 0.5 else -1.0
        cross = close and u[5] < pop.crossing_prob
        overshoot = pop.ramp_overshoot_hz[0] + u[6] * (
            pop.ramp_overshoot_hz[1] - pop.ramp_overshoot_hz[0]
        )
        dur_lo, dur_hi = pop.ramp_duration_s
        cross_dur = dur_lo + u[7] * (dur_hi - dur_lo)
        random_ramp = u[8] < pop.ramp_prob
        random_df = pop.ramp_df_hz[0] + u[9] * (pop.ramp_df_hz[1] - pop.ramp_df_hz[0])
        random_sign = 1.0 if u[10] < 0.5 else -1.0
        random_dur = dur_lo + u[11] * (dur_hi - dur_lo)
        start_u = u[12]
        if u[13] < pop.baseline_uniform_prob:
            group = "uniform"
            drawn = lo + u[0] * (hi - lo)
        else:
            group = "male" if u[14] < pop.male_prob else "female"
            mean, sd = pop.male_hz if group == "male" else pop.female_hz
            drawn = min(max(mean + sd * z, lo), hi)

        fish_ramps: list[Ramp] = []
        entry: dict[str, Any] = {
            "baseline_group": group,
            "close_to": -1,
            "close_df_hz": None,
            "crossing": False,
        }
        if close:
            f = baselines[partner] + sign * df
            if not lo <= f <= hi:
                sign = -sign
                f = baselines[partner] + sign * df
            entry.update(
                baseline_group="close pair",
                close_to=partner,
                close_df_hz=float(f - baselines[partner]),
            )
            if cross:
                # move towards and past the partner
                fish_ramps.append(
                    _ramp(cross_dur, scene_s, start_u, -sign * (df + overshoot))
                )
                entry["crossing"] = True
        else:
            f = drawn
        if random_ramp:
            fish_ramps.append(
                _ramp(random_dur, scene_s, 1.0 - start_u, random_sign * random_df)
            )
        baselines.append(float(f))
        ramps.append(fish_ramps)
        info.append(entry)
    return baselines, ramps, info


def _shared_knots(
    config: SceneConfig, seed: int, duration_s: float
) -> tuple[DriftKnots, float]:
    """Shared relative drift: linear trend (centred) plus a stationary part."""
    pop = config.population
    rng = stream(seed, "shared_drift")
    mean, sd = pop.shared_trend_per_h
    trend_per_h = float(mean + sd * rng.standard_normal())
    shared = pop.shared_drift
    knots = _drift_knots(
        rng,
        duration_s,
        sd=shared.sd_rel,
        timescale_s=shared.timescale_s,
        control_rate_hz=shared.control_rate_hz,
    )
    values = knots.values + trend_per_h * (knots.times_s - duration_s / 2) / 3600.0
    return DriftKnots(knots.times_s, values), trend_per_h


@dataclass(frozen=True, eq=False)
class Layout:
    """Electrode positions of a scene.

    Attributes
    ----------
    positions_m : numpy.ndarray, shape (E, 2)
        Electrode positions in metres, by channel.
    rows, columns : int
        Grid size written to ``fishgrid.cfg``; ``1 x E`` for layouts that are
        not a full rectangle.
    kind : str
        ``"real"``, ``"single"``, ``"pair"``, ``"line"``, ``"grid"`` or
        ``"irregular"`` (a near-square grid with positions missing), or
        ``"background"`` (the channels of a real background segment).
    regular : bool
        Whether the positions are exactly ``GridGeometry.regular(columns,
        rows, spacing)`` (no jitter, rotation or gaps). Otherwise the
        positions are written to ``electrodes.csv``.
    spacing_m : float
        Electrode spacing.
    """

    positions_m: FloatArray
    rows: int
    columns: int
    kind: str
    regular: bool
    spacing_m: float

    @property
    def n_electrodes(self) -> int:
        """Number of electrodes."""
        return int(self.positions_m.shape[0])


def _rectangle(n: int, max_aspect: float) -> tuple[int, int] | None:
    """Most square r x c (r <= c, r >= 2) with r * c = n and c / r <= max_aspect."""
    best = None
    for r in range(2, math.isqrt(n) + 1):
        if n % r == 0 and (n // r) / r <= max_aspect:
            best = (r, n // r)
    return best


def draw_layout(config: SceneConfig, seed: int) -> Layout:
    """Electrode layout of a scene (grid stream); see :class:`GridConfig`."""
    grid = config.grid
    rng = stream(seed, "grid")
    u = rng.random(8)
    spacing = float(rng.uniform(*grid.spacing_m))
    if u[0] < grid.real_layout_prob:
        rows, columns = grid.layouts[
            min(int(u[1] * len(grid.layouts)), len(grid.layouts) - 1)
        ]
        positions = GridGeometry.regular(columns, rows, spacing, spacing).positions_m
        return Layout(np.array(positions), rows, columns, "real", True, spacing)

    lo, hi = grid.n_electrodes
    small = (lo, min(grid.small_max, hi))
    large = (max(grid.small_max + 1, lo), hi)
    use_small = (u[2] < grid.small_prob and small[0] <= small[1]) or large[0] > large[1]
    first, last = small if use_small else large
    n = first + min(int(u[3] * (last - first + 1)), last - first)

    if n == 1:
        rows, columns, kind = 1, 1, "single"
    elif n == 2:
        rows, columns, kind = 1, 2, "pair"
    elif n <= 4:
        if u[4] < 0.5:
            rows, columns, kind = 1, n, "line"
        else:
            rows, columns, kind = 2, 2, "grid" if n == 4 else "irregular"
    else:
        shape = _rectangle(n, grid.max_aspect)
        if shape is not None:
            rows, columns, kind = shape[0], shape[1], "grid"
        else:
            columns = math.ceil(math.sqrt(n))
            rows = math.ceil(n / columns)
            kind = "irregular"
    if u[5] < 0.5:
        rows, columns = columns, rows
    positions = np.array(
        GridGeometry.regular(columns, rows, spacing, spacing).positions_m
    )
    if rows * columns > n:  # near-square grid with positions missing
        keep = np.sort(rng.permutation(rows * columns)[:n])
        positions = positions[keep]
    jitter = rng.standard_normal(positions.shape) * grid.jitter_rel * spacing
    positions = positions + jitter
    rotated = u[6] < grid.rotation_prob
    if rotated and n > 1:
        angle = 2.0 * math.pi * u[7]
        centre = positions.mean(axis=0)
        c, s_ = math.cos(angle), math.sin(angle)
        positions = (positions - centre) @ np.array([[c, s_], [-s_, c]]) + centre
    positions = positions - positions.min(axis=0)
    regular = kind != "irregular" and grid.jitter_rel == 0.0 and not rotated
    if kind == "irregular":
        rows, columns = 1, n
    return Layout(positions, rows, columns, kind, regular, spacing)


def layout_arena(layout: Layout, margin_rel: float) -> Arena:
    """Bounding box of the electrodes plus a margin, at least a spacing wide."""
    spacing = layout.spacing_m
    lo = layout.positions_m.min(axis=0) - margin_rel * spacing
    hi = layout.positions_m.max(axis=0) + margin_rel * spacing
    for axis in range(2):
        width = hi[axis] - lo[axis]
        if width < spacing:
            pad = (spacing - width) / 2.0
            lo[axis] -= pad
            hi[axis] += pad
    return Arena(float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))


def background_layout(segment: Any) -> Layout:
    """Layout of a hybrid scene: the background excerpt's electrodes.

    Positions and channel order are the segment's (nominal for the
    competition grids). The whole grid of a segment keeps its rows and
    columns if its positions are the regular row-major grid; anything else
    (a channel subset, a single line of unknown geometry) is written as
    ``1 x E`` with ``electrodes.csv``.
    """
    positions = np.array(segment.geometry.positions_m, dtype=np.float64)
    n = positions.shape[0]
    spacing = segment_spacing(segment)
    info = segment.info
    rows, columns = int(info.get("n_rows", 1) or 1), int(info.get("n_columns", n) or n)
    whole = tuple(segment.channels) == tuple(range(n)) and rows * columns == n
    regular = whole and bool(
        np.allclose(
            positions,
            GridGeometry.regular(columns, rows, spacing, spacing).positions_m,
        )
    )
    if not regular:
        rows, columns = 1, n
    return Layout(positions, rows, columns, "background", regular, spacing)


def _track_times(model: SceneModel, step_s: float) -> FloatArray:
    """Frame times of the ground-truth tracks.

    Up to the first frame at or after the last sample, so every chirp lies
    within the tracks (wavetracker's frames would stop earlier).
    """
    last_s = (model.n_samples - 1) / model.samplerate_hz
    n_frames = math.ceil(last_s / step_s - 1e-9) + 1
    return np.arange(n_frames, dtype=np.float64) * step_s


def fish_fundamentals(model: SceneModel, times_s: np.ndarray) -> list[np.ndarray]:
    """Noise-free fundamental amplitude of every fish on every electrode.

    Amplitude x relative amplitude of the fundamental x field gain x channel
    gain, shape (len(times_s), E) per fish: the input of
    :func:`gridtools.background.fish_signal_level`.
    """
    return [
        model.gain_track(i).evaluate(times_s)
        * model.channel_gain
        * fish.amplitude
        * fish.harmonics.amplitudes[0]
        for i, fish in enumerate(model.fish)
    ]


def median_strongest_gain(
    electrode_xy_m: np.ndarray,
    arena: Arena,
    exponent: float,
    regulariser_m: float,
    reference_distance_m: float,
    *,
    n_positions: int = 41,
    n_headings: int = 24,
) -> float:
    """Median over the arena of the largest |gain| across electrodes.

    Positions on a regular ``n_positions`` x ``n_positions`` grid over the
    arena, headings equally spaced; deterministic, so it consumes no random
    numbers. Used to keep the calibrated fundamental at the strongest
    electrode fixed when the field, the spacing or the layout changes.
    """
    xs = np.linspace(arena.x_min_m, arena.x_max_m, n_positions)
    ys = np.linspace(arena.y_min_m, arena.y_max_m, n_positions)
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel()])
    headings = (np.arange(n_headings) + 0.5) * 2.0 * np.pi / n_headings
    xy = np.repeat(points, n_headings, axis=0)
    heading = np.tile(headings, points.shape[0])
    gains = dipole_gain(
        xy,
        heading,
        electrode_xy_m,
        fish_depth_m=0.0,
        electrode_depth_m=0.0,
        min_distance_m=regulariser_m,
        distance_exponent=exponent,
        reference_distance_m=reference_distance_m,
    )
    return float(np.median(np.max(np.abs(gains), axis=1)))


def calibrated_strongest_gain(field_cfg: Any) -> float:
    """Median strongest-electrode gain of the calibrated field (3 x 5 grid)."""
    spacing = field_cfg.calibration_spacing_m
    electrodes = GridGeometry.regular(5, 3, spacing, spacing).positions_m
    arena = Arena(0.0, 4 * spacing, 0.0, 2 * spacing)
    return median_strongest_gain(
        np.asarray(electrodes),
        arena,
        field_cfg.calibration_exponent,
        field_cfg.calibration_regulariser_rel * spacing,
        field_cfg.reference_distance_m,
    )


def plan_scene(config: SceneConfig, seed: int) -> ScenePlan:
    """Draw everything about a scene except its full-rate signals.

    Cheap (no audio), so population and timing statistics can be checked
    over many seeds.

    Parameters
    ----------
    config : SceneConfig
    seed : int
        Non-negative scene seed.

    Returns
    -------
    ScenePlan
    """
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer) or seed < 0:
        msg = f"seed must be a non-negative integer, got {seed!r}"
        raise ValueError(msg)
    seed = int(seed)
    if config.samplerate_choices_hz:  # augmentation: the rate is drawn per scene
        choices = config.samplerate_choices_hz
        rate = choices[int(stream(seed, "samplerate").integers(len(choices)))]
        config = dataclasses.replace(
            config, samplerate_hz=float(rate), samplerate_choices_hz=()
        )
    fs = config.samplerate_hz
    n_samples = config.n_samples
    duration_s = n_samples / fs
    pool = TemplatePool.open(config.chirps)

    # electrodes and arena; a real background fixes them (hybrid scene)
    layout = draw_layout(config, seed)
    hybrid_cfg = config.background
    background_rng = stream(seed, "background")
    segment = None
    choice: dict[str, Any] = {}
    if hybrid_cfg.prob > 0.0:
        check_duration(hybrid_cfg, duration_s)
        if background_rng.random() < hybrid_cfg.prob:
            # a "real" layout means the segment's whole grid, any other its
            # electrode count (capped by the segment's channels)
            k = None if layout.kind == "real" else layout.n_electrodes
            segment, choice = draw_segment(background_rng, hybrid_cfg, k, fs, n_samples)
            layout = background_layout(segment)
    spacing = layout.spacing_m
    rows, columns = layout.rows, layout.columns
    geometry = GridGeometry(layout.positions_m)
    arena = layout_arena(layout, config.grid.arena_margin_rel)

    # field along the fitted ridge, amplitude normalised to the calibration
    field_cfg = config.field
    exponent = float(stream(seed, "field").uniform(*field_cfg.exponent))
    regulariser_rel = field_cfg.regulariser_rel(exponent)
    dipole = DipoleParams(
        fish_depth_m=0.0,
        electrode_depth_m=0.0,
        min_distance_m=regulariser_rel * spacing,
        distance_exponent=exponent,
        reference_distance_m=field_cfg.reference_distance_m,
    )
    gain_scene = median_strongest_gain(
        layout.positions_m,
        arena,
        exponent,
        regulariser_rel * spacing,
        field_cfg.reference_distance_m,
    )
    gain_calibrated = calibrated_strongest_gain(field_cfg)
    amplitude_scale = gain_calibrated / gain_scene

    # movement regime: once per scene or per fish
    movement = config.movement
    scene_light = bool(stream(seed, "regime").random() < movement.light_prob)

    baselines, ramps, pair_info = _population(config, seed)
    n_fish = len(baselines)
    shared_knots, trend_per_h = _shared_knots(config, seed, duration_s)

    # per-fish body, waveform, drift, path, chirp rate
    pop = config.population
    bodies = []
    for i in range(n_fish):
        body = fish_stream(seed, i, "body")
        z = float(
            np.clip(
                body.standard_normal(),
                -field_cfg.amplitude_clip_sd,
                field_cfg.amplitude_clip_sd,
            )
        )
        amplitude = field_cfg.amplitude_median_v * math.exp(
            field_cfg.amplitude_log_sd * z
        )
        amplitude *= amplitude_scale
        phase0 = float(body.uniform(0.0, 2.0 * np.pi))
        harmonics = jitter_harmonics(
            fish_stream(seed, i, "harmonics"),
            APTERONOTUS_LEPTORHYNCHUS,
            pop.harmonics,
        )
        drift_rng = fish_stream(seed, i, "drift")
        factors = np.exp(
            drift_rng.standard_normal(2)
            * [pop.drift_sd_log_sd, pop.drift_timescale_log_sd]
        )
        drift_sd = pop.drift.sd_hz * float(factors[0])
        drift_tau = max(
            pop.drift.timescale_s * float(factors[1]), 2.0 / pop.drift.control_rate_hz
        )
        drift_knots = _drift_knots(
            drift_rng,
            duration_s,
            sd=drift_sd,
            timescale_s=drift_tau,
            control_rate_hz=pop.drift.control_rate_hz,
        )
        if movement.per == "scene":
            light = scene_light
        else:
            light = bool(fish_stream(seed, i, "regime").random() < movement.light_prob)
        path = simulate_path(
            fish_stream(seed, i, "movement"),
            duration_s,
            arena,
            movement.light if light else movement.dark,
        )
        bodies.append(
            (
                amplitude,
                phase0,
                harmonics,
                drift_knots,
                path,
                drift_sd,
                drift_tau,
                light,
            )
        )

    # chirp timing and templates (one template cycler per fish)
    timing_rngs = [fish_stream(seed, i, "timing") for i in range(n_fish)]
    rates = [float(rng.uniform(*config.chirps.rate_hz)) for rng in timing_rngs]
    cyclers = [pool.cycler(fish_stream(seed, i, "templates")) for i in range(n_fish)]

    def draw(fish_id: int) -> tuple[int, int, int]:
        template_id = int(cyclers[fish_id].draw(1)[0])
        before, after = pool.support(template_id, fs)
        return template_id, before, after

    schedule = schedule_chirps(
        config.chirps,
        rates,
        timing_rngs,
        stream(seed, "responses"),
        draw,
        n_samples=n_samples,
        samplerate_hz=fs,
    )

    # templates and placement
    placements = _place_chirps(config, seed, schedule, pool, n_fish, n_samples)

    fish_models = []
    for i, (amplitude, phase0, harmonics, drift_knots, path, *_) in enumerate(bodies):
        own = sorted((p for p in placements if p.fish_id == i), key=lambda p: p.start)
        fish_models.append(
            FishModel(
                fish_id=i,
                baseline_hz=baselines[i],
                amplitude=amplitude,
                phase0_rad=phase0,
                harmonics=harmonics,
                dipole=dipole,
                drift=drift_knots,
                ramps=tuple(ramps[i]),
                path=path,
                chirps=tuple(own),
            )
        )

    # channels
    n_electrodes = geometry.n_electrodes
    noise_cfg = config.noise
    mismatch = gain_mismatch(
        stream(seed, "channels"),
        n_electrodes,
        GainMismatchParams(log_sd=noise_cfg.gain_mismatch_log_sd),
    )
    if segment is None:
        faults = electrode_faults(
            stream(seed, "faults"),
            n_electrodes,
            ElectrodeFaultParams(
                dead_prob=noise_cfg.dead_prob,
                noisy_prob=noise_cfg.noisy_prob,
                dead_gain=noise_cfg.dead_gain,
                noisy_factor=noise_cfg.noisy_factor,
            ),
        )
    else:  # the segment's own dead (no fish signal) and noisy channels
        dead = np.zeros(n_electrodes, dtype=bool)
        dead[segment.dead_channels] = True
        noisy = np.zeros(n_electrodes, dtype=bool)
        noisy[segment.noisy_channels] = True
        faults = ElectrodeFaults(
            dead=dead,
            noisy=noisy,
            signal_gain=np.where(dead, noise_cfg.dead_gain, 1.0).astype(np.float32),
            noise_scale=np.ones(n_electrodes, dtype=np.float32),
        )
    if faults.dead.all():  # keep at least one live electrode
        dead = faults.dead.copy()
        dead[0] = False
        faults = ElectrodeFaults(
            dead=dead,
            noisy=faults.noisy,
            signal_gain=np.where(dead, noise_cfg.dead_gain, 1.0).astype(np.float32),
            noise_scale=faults.noise_scale,
        )
    channel_gain = mismatch.astype(np.float64) * faults.signal_gain.astype(np.float64)
    floor_spread = 10.0 ** (
        stream(seed, "floors").standard_normal(n_electrodes)
        * noise_cfg.electrode_floor_sd_db
        / 20.0
    )
    noise_scale = faults.noise_scale.astype(np.float64) * floor_spread

    # reference electrode: in the tank (augmentation) or at infinity
    ref_rng = stream(seed, "reference")
    in_tank = bool(ref_rng.random() < config.grid.reference_prob)
    ref_margin = config.grid.reference_margin_rel * spacing
    ref_u = ref_rng.random(2)
    reference_xy = None
    if in_tank:
        reference_xy = np.array(
            [
                arena.x_min_m
                - ref_margin
                + ref_u[0] * (arena.width_m + 2 * ref_margin),
                arena.y_min_m
                - ref_margin
                + ref_u[1] * (arena.height_m + 2 * ref_margin),
            ]
        )

    model = SceneModel(
        samplerate_hz=fs,
        n_samples=n_samples,
        electrode_xy_m=geometry.positions_m,
        channel_gain=channel_gain,
        shared_drift=shared_knots,
        fish=tuple(fish_models),
        reference_xy_m=reference_xy,
    )

    # noise levels in volts
    levels_rng = stream(seed, "levels")
    modes = noise_cfg.floor_modes
    weights = np.array([w for _, _, w in modes], dtype=np.float64)
    pick = int(
        np.searchsorted(
            np.cumsum(weights / weights.sum()), levels_rng.random(), side="right"
        )
    )
    pick = min(pick, len(modes) - 1)
    floor_db = float(modes[pick][0] + modes[pick][1] * levels_rng.standard_normal())
    harder = bool(levels_rng.random() < noise_cfg.harder_prob)
    harder_db = float(levels_rng.uniform(*noise_cfg.harder_db))
    extra_db = harder_db if harder else 0.0
    floor_db += extra_db
    noise_exponent = float(levels_rng.uniform(*noise_cfg.exponent))
    hum_lo, hum_hi = noise_cfg.hum_n_harmonics
    hum_n = int(levels_rng.integers(hum_lo, hum_hi + 1))
    amp_lo, amp_hi = noise_cfg.hum_amplitude_v
    hum_amplitude = math.exp(levels_rng.uniform(math.log(amp_lo), math.log(amp_hi)))
    levels: dict[str, Any] = {
        "noise_floor_db_v2_hz": floor_db,
        "noise_floor_mode": pick,
        "harder": harder,
        "harder_db": extra_db,
        "noise_sd": math.sqrt(10.0 ** (floor_db / 10.0) * fs / 2.0),
        "noise_exponent": noise_exponent,
        "hum_n_harmonics": hum_n,
        "hum_amplitude_v": hum_amplitude,
        "spacing_m": spacing,
        "field_exponent": exponent,
        "field_regulariser_rel": regulariser_rel,
        "field_regulariser_m": regulariser_rel * spacing,
        "median_strongest_gain": gain_scene,
        "median_strongest_gain_calibrated": gain_calibrated,
        "amplitude_scale": amplitude_scale,
        "shared_trend_per_h": trend_per_h,
        "movement_light": scene_light if movement.per == "scene" else None,
        "reference": "in tank" if in_tank else "distant",
        "reference_xy_m": None if reference_xy is None else reference_xy.tolist(),
        "background": "synthetic" if segment is None else "real",
    }
    background = None
    if segment is not None:
        # the real background replaces the synthetic noise, hum and clicks
        for key in (
            "noise_floor_db_v2_hz",
            "noise_floor_mode",
            "harder",
            "harder_db",
            "noise_sd",
            "noise_exponent",
            "hum_n_harmonics",
            "hum_amplitude_v",
        ):
            levels[key] = None
        signal = fish_signal_level(
            fish_fundamentals(model, _track_times(model, config.tracks.step_s))
        )
        background = HybridBackground.build(
            hybrid_cfg, segment, choice, signal, background_rng
        )
        if background.fish_gain != 1.0:  # hard SNR: quieter simulated fish
            fish_models = [
                dataclasses.replace(f, amplitude=f.amplitude * background.fish_gain)
                for f in fish_models
            ]
            model = dataclasses.replace(model, fish=tuple(fish_models))
        levels["background_segment"] = segment.segment_id
        levels["background_dataset"] = str(segment.info.get("dataset", ""))
        levels["background_snr_db"] = background.snr_db
        levels["background_scale"] = background.scale
        levels["fish_gain"] = background.fish_gain
    channels = {
        "gain_mismatch": mismatch.astype(np.float64).tolist(),
        "dead": faults.dead.tolist(),
        "noisy": faults.noisy.tolist(),
        "noise_scale": noise_scale.tolist(),
        "channel_gain": channel_gain.tolist(),
    }

    fish_info = []
    for i, fish in enumerate(fish_models):
        n_resp = sum(1 for c in fish.chirps if c.is_response)
        _, _, _, _, _, drift_sd, drift_tau, light = bodies[i]
        fish_info.append(
            {
                "fish_id": i,
                "baseline_hz": fish.baseline_hz,
                "amplitude_v": fish.amplitude,
                "phase0_rad": fish.phase0_rad,
                "drift_sd_hz": drift_sd,
                "drift_timescale_s": drift_tau,
                "movement": "light" if light else "dark",
                "chirp_rate_hz": rates[i],
                "n_chirps": len(fish.chirps),
                "n_responses": n_resp,
                "ramps": [vars(r) for r in fish.ramps],
                **pair_info[i],
                "real_fish": False,
            }
        )

    return ScenePlan(
        config=config,
        seed=seed,
        rows=rows,
        columns=columns,
        spacing_m=spacing,
        layout=layout,
        geometry=geometry,
        arena=arena,
        model=model,
        schedule=schedule,
        fish_info=fish_info,
        levels=levels,
        channels=channels,
        pool=pool,
        background=background,
    )


def _place_chirps(
    config: SceneConfig,
    seed: int,
    schedule: ChirpSchedule,
    pool: TemplatePool,
    n_fish: int,
    n_samples: int,
) -> list[ChirpPlacement]:
    """Draw amplitude dips per fish (own stream) and place the scheduled chirps."""
    chirp_cfg = config.chirps
    fs = config.samplerate_hz
    by_fish: dict[int, list[int]] = {i: [] for i in range(n_fish)}
    for chirp_id, event in enumerate(schedule.events):
        by_fish[event.fish_id].append(chirp_id)
    placements = []
    for fish_id, ids in by_fish.items():
        dips = random_dips(fish_stream(seed, fish_id, "dips"), len(ids), chirp_cfg.dip)
        for chirp_id, dip in zip(ids, dips, strict=True):
            event = schedule.events[chirp_id]
            delta, measured, centre = pool.scene_traces(event.template_id, fs)
            if chirp_cfg.amplitude_mode == "frequency":
                amp = amplitude_from_frequency(delta, float(dip)).astype(np.float32)
                dip_value = float(dip)
            else:
                amp = measured
                dip_value = float(1.0 - np.min(amp))
            placement = place_chirp(
                chirp_id=chirp_id,
                fish_id=fish_id,
                template_id=event.template_id,
                delta_f_hz=delta,
                amp_factor=amp,
                centre_index_template=centre,
                center_index=event.center_index,
                dip=dip_value,
                n_samples=n_samples,
                is_response=event.is_response,
                response_to=event.trigger,
            )
            if placement is None:
                msg = f"template {event.template_id} has no support inside the scene"
                raise ValueError(msg)
            placements.append(placement)
    return placements


# -- rendering --------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Scene:
    """A simulated scene in memory.

    Attributes
    ----------
    plan : ScenePlan
        Random draws and the render model.
    recording : numpy.ndarray of float32, shape (n_samples, n_electrodes)
    chirps : pandas.DataFrame
        Ground truth, columns :data:`CHIRP_COLUMNS`.
    clicks : pandas.DataFrame
        Click artefacts.
    tracks : Tracks
        Ground-truth tracks (wavetracker layout).
    click_rate_hz : float
        The scene's click rate.
    timings_s : dict of float
        Wall-clock time of the simulation stages.
    real_fish : pandas.DataFrame
        Real fish of a hybrid scene's background (columns of
        :meth:`gridtools.background.BackgroundSegment.real_fish`; ``ident``
        is the track identity, from 1000 on); empty otherwise.
    """

    plan: ScenePlan
    recording: Float32Array
    chirps: pd.DataFrame
    clicks: pd.DataFrame
    tracks: Tracks
    click_rate_hz: float
    timings_s: dict[str, float] = field(default_factory=dict)
    real_fish: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def config(self) -> SceneConfig:
        """The scene's config."""
        return self.plan.config

    @property
    def seed(self) -> int:
        """The scene's seed."""
        return self.plan.seed

    @property
    def model(self) -> SceneModel:
        """The render model."""
        return self.plan.model

    @property
    def samplerate_hz(self) -> float:
        """Sampling rate in Hz."""
        return self.plan.config.samplerate_hz


def _background_parts(
    plan: ScenePlan, *, workers: int = 1
) -> tuple[list[Any], float, list[Click]]:
    """Additive components (noise array, hum), click rate and clicks (deterministic).

    The coloured noise comes first when enabled, so it can serve as the
    output buffer of the mix.
    """
    config = plan.config
    seed = plan.seed
    model = plan.model
    noise_cfg = config.noise
    components: list[Any] = []
    if noise_cfg.colored:
        components.append(
            colored_noise(
                stream(seed, "noise"),
                model.n_samples,
                model.n_electrodes,
                model.samplerate_hz,
                ColoredNoiseParams(
                    sd=plan.levels["noise_sd"],
                    exponent=plan.levels["noise_exponent"],
                    f_min_hz=noise_cfg.f_min_hz,
                ),
                sd_scale=np.asarray(plan.channels["noise_scale"], dtype=np.float32),
                workers=workers,
            )
        )
    if noise_cfg.hum:
        hum = powerline_hum(
            stream(seed, "hum"),
            model.n_samples,
            model.n_electrodes,
            model.samplerate_hz,
            HumParams(
                line_freq_hz=noise_cfg.hum_line_freq_hz,
                n_harmonics=int(plan.levels["hum_n_harmonics"]),
                amplitude_range=(
                    plan.levels["hum_amplitude_v"],
                    plan.levels["hum_amplitude_v"],
                ),
                harmonic_decay=noise_cfg.hum_harmonic_decay,
                coupling_log_sd=noise_cfg.hum_coupling_log_sd,
            ),
        )
        # the drawn amplitude is the line on the electrode with most hum
        coupling = hum.coupling / np.max(hum.coupling)
        components.append(
            Hum(
                source=hum.source,
                coupling=coupling.astype(np.float32),
                frequencies_hz=hum.frequencies_hz,
                amplitudes=hum.amplitudes,
            )
        )
    rate, clicks = draw_clicks(
        stream(seed, "clicks"),
        config.clicks,
        n_samples=model.n_samples,
        n_electrodes=model.n_electrodes,
        samplerate_hz=model.samplerate_hz,
        noise_sd=plan.levels["noise_sd"],
    )
    return components, rate, clicks


def simulate_scene(config: SceneConfig, seed: int, *, workers: int = 1) -> Scene:
    """Draw and render one scene.

    Parameters
    ----------
    config : SceneConfig
    seed : int
        Non-negative scene seed; the result is a deterministic function of
        ``(config, seed)`` and the template library.
    workers : int
        Threads for the noise FFT and mixing (default 1); the result does not
        depend on it.

    Returns
    -------
    Scene
    """
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    plan = plan_scene(config, seed)
    config = plan.config  # resolved (e.g. a drawn sampling rate)
    model = plan.model
    timings["plan"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    eods = [model.waveform(i, 0, model.n_samples) for i in range(len(model.fish))]
    gains = [model.gain_track(i) for i in range(len(model.fish))]
    timings["fish"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    background = plan.background
    if background is None:
        components, click_rate, clicks = _background_parts(plan, workers=workers)
    else:  # the real background has its own noise, hum and clicks
        components, click_rate, clicks = [], 0.0, []
    timings["noise"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    buffer = (
        components[0] if components and isinstance(components[0], np.ndarray) else None
    )
    if buffer is None and not eods:
        buffer = np.zeros((model.n_samples, model.n_electrodes), dtype=np.float32)
        components = [buffer, *components]
    recording = mix(
        eods,
        gains,
        samplerate_hz=model.samplerate_hz,
        additive=components,
        channel_gain=model.channel_gain,
        clip=None,
        gain_update_hz=model.gain_update_hz,
        workers=workers,
        out=buffer,
    )
    if background is not None:
        add_background(
            recording,
            background.segment,
            background.scale,
            geometry=plan.geometry,
            samplerate_hz=model.samplerate_hz,
            out=recording,
        )
    add_clicks(recording, clicks)
    if config.noise.clip_limit is not None:
        limit = np.float32(config.noise.clip_limit)
        np.clip(recording, -limit, limit, out=recording)
    timings["mix"] = time.perf_counter() - t3

    t4 = time.perf_counter()
    chirps = chirp_table(plan)
    tracks = ground_truth_tracks(plan, recording, workers=workers)
    real_fish = pd.DataFrame()
    if background is not None:
        tracks, real_fish = _merge_real_fish(plan, tracks)
    timings["tables"] = time.perf_counter() - t4
    if config.footprint.enabled:
        t5 = time.perf_counter()
        chirps = add_footprints(chirps, plan, recording, workers=workers)
        timings["footprints"] = time.perf_counter() - t5
    timings["simulate_total"] = time.perf_counter() - t0
    logger.info(
        "scene %s/%06d: %d fish, %d chirps, %d clicks, %s background, %.2f s",
        config.name,
        seed,
        len(model.fish),
        len(chirps),
        len(clicks),
        "synthetic"
        if background is None
        else f"real ({background.segment.segment_id}, {len(real_fish)} real fish, "
        f"SNR {background.snr_db:.1f} dB)",
        timings["simulate_total"],
    )
    return Scene(
        plan=plan,
        recording=recording,
        chirps=chirps,
        clicks=click_table(clicks, model.samplerate_hz),
        tracks=tracks,
        click_rate_hz=click_rate,
        timings_s=timings,
        real_fish=real_fish,
    )


def _merge_real_fish(plan: ScenePlan, tracks: Tracks) -> tuple[Tracks, pd.DataFrame]:
    """Add the background's real fish to the tracks (identities from 1000 on).

    Their ``sign_v`` is the tracker's, shifted by ``20 log10(scale)`` (the
    background's scale) and by the ratio of the PSD frames' durations
    (scene ``nfft / fs`` over the tracker's), which turns the power of a
    tone in the tracker's PSD into the scene's definition.
    """
    background = plan.background
    assert background is not None
    segment = background.segment
    real = segment.tracks
    if real is None or real.ids.size == 0:
        return tracks, pd.DataFrame()
    fs = plan.model.samplerate_hz
    scene_window_s = plan.config.tracks.power_nfft(fs) / fs
    track_window_s = float(segment.info["track_power_nfft"]) / float(
        segment.info["samplerate_hz"]
    )
    offset_db = 20.0 * math.log10(background.scale) + 10.0 * math.log10(
        scene_window_s / track_window_s
    )
    merged, real_ids = merge_tracks(
        tracks,
        real,
        ident_offset=REAL_FISH_IDENT_OFFSET,
        power_offset_db=offset_db,
    )
    assert merged is not None
    table = segment.real_fish(real_ids)
    table = table.loc[table["source_ident"].isin(list(real_ids.values()))]
    missing = sorted(set(real_ids.values()) - set(table["source_ident"]))
    if missing:  # tracked in the excerpt but not listed in the manifest
        inverse = {old: new for new, old in real_ids.items()}
        extra = pd.DataFrame(
            {
                "ident": [inverse[i] for i in missing],
                "source_ident": missing,
                "real_fish": True,
                "eodf_hz": np.nan,
                "snr_db": np.nan,
                "source_dataset": str(segment.info.get("dataset", "")),
                "source_recording": str(segment.info.get("recording", "")),
                "segment_id": segment.segment_id,
            }
        )
        table = pd.concat([table, extra], ignore_index=True)
    table = table.sort_values(by="ident").reset_index(drop=True)
    table["power_offset_db"] = offset_db
    return merged, table


def chirp_table(plan: ScenePlan) -> pd.DataFrame:
    """Ground-truth table of all chirps (columns :data:`CHIRP_COLUMNS`)."""
    model = plan.model
    fs = model.samplerate_hz
    chirps = sorted(model.chirps.values(), key=lambda c: c.chirp_id)
    n_fish = len(model.fish)
    rows = []
    centres = np.array([c.center_index / fs for c in chirps])
    emitters = np.array([c.fish_id for c in chirps])
    power_nfft = plan.config.tracks.power_nfft(fs)
    for chirp in chirps:
        t_c = chirp.center_index / fs
        fish = model.fish[chirp.fish_id]
        baseline = float(model.base_frequency_at(chirp.fish_id, [t_c])[0])
        others = emitters != chirp.fish_id
        if np.any(others):
            dt = np.abs(centres[others] - t_c)
            dt_min = float(dt.min())
        else:
            dt_min = math.nan
        if n_fish > 1:
            f_all = np.array(
                [model.base_frequency_at(j, [t_c])[0] for j in range(n_fish)]
            )
            df = np.abs(f_all - baseline)
            df[chirp.fish_id] = np.inf
            nearest = int(np.argmin(df))
            min_df = float(df[nearest])
        else:
            nearest, min_df = -1, math.nan
        x = float(np.interp(t_c, fish.path.times_s, fish.path.xy_m[:, 0]))
        y = float(np.interp(t_c, fish.path.times_s, fish.path.xy_m[:, 1]))
        gains = model.gain_track(chirp.fish_id).evaluate([t_c])[0]
        fundamental = (
            model.channel_gain * gains * fish.amplitude * fish.harmonics.amplitudes[0]
        )
        power = fish_power_db(fundamental, fs, power_nfft)
        delta_min = -chirp.undershoot_hz
        rows.append(
            {
                "chirp_id": chirp.chirp_id,
                "emitter_id": chirp.fish_id,
                "t_onset_s": chirp.onset_index / fs,
                "t_center_s": t_c,
                "t_peak_s": chirp.peak_index / fs,
                "t_offset_s": chirp.offset_index / fs,
                "t_support_start_s": chirp.support_start / fs,
                "t_support_stop_s": chirp.support_stop / fs,
                "duration_s": (chirp.offset_index - chirp.onset_index) / fs,
                "baseline_hz": baseline,
                "excursion_hz": chirp.excursion_hz,
                "undershoot_hz": chirp.undershoot_hz,
                "f_low_hz": baseline + min(delta_min, 0.0),
                "f_high_hz": baseline + chirp.excursion_hz,
                "dip": chirp.dip,
                "amplitude_mode": plan.config.chirps.amplitude_mode,
                "template_id": chirp.template_id,
                "family": plan.pool.family(chirp.template_id),
                "is_response": chirp.is_response,
                "response_to": chirp.response_to,
                "overlaps_other_chirp": bool(dt_min <= OVERLAP_WINDOW_S),
                "dt_nearest_other_chirp_s": dt_min,
                "min_df_hz": min_df,
                "nearest_fish_id": nearest,
                "truncated": bool(
                    chirp.support_start < 0 or chirp.support_stop > model.n_samples
                ),
                "emitter_x_m": x,
                "emitter_y_m": y,
                "emitter_power_db_max": float(power.max()),
            }
        )
    table = pd.DataFrame(rows, columns=list(CHIRP_COLUMNS))
    return table.astype(
        {
            "chirp_id": np.int64,
            "emitter_id": np.int64,
            "template_id": np.int64,
            "response_to": np.int64,
            "nearest_fish_id": np.int64,
            "is_response": bool,
            "overlaps_other_chirp": bool,
            "truncated": bool,
        }
    )


def reference_spectrogram(
    config: SceneConfig, *, workers: int = 1
) -> "HannSpectrogram":
    """The front end of the footprint columns (``config.footprint``)."""
    from gridtools.sim.footprint import HannSpectrogram

    fp = config.footprint
    return HannSpectrogram(
        window_s=fp.window_s,
        hop_s=fp.hop_s,
        f_min_hz=fp.f_min_hz,
        f_max_hz=fp.f_max_hz,
        workers=workers,
    )


def add_footprints(
    table: pd.DataFrame, plan: ScenePlan, recording: np.ndarray, *, workers: int = 1
) -> pd.DataFrame:
    """Append the footprint columns (:data:`FOOTPRINT_COLUMNS`) to a chirp table."""
    from gridtools.sim.footprint import scene_footprints

    fp = plan.config.footprint
    footprints = scene_footprints(
        plan.model,
        reference_spectrogram(plan.config, workers=workers),
        chirp_ids=table["chirp_id"].tolist(),
        threshold_db=fp.threshold_db,
        visible_snr_db=fp.visible_snr_db,
        margin_s=fp.margin_s,
        harmonic_band=(fp.f_min_hz, fp.f_max_hz),
        recording=recording,
        background=fp.background,
    )
    columns: dict[str, list[Any]] = {name: [] for name in FOOTPRINT_COLUMNS}
    for footprint in footprints:
        for name, value in zip(FOOTPRINT_COLUMNS[:4], footprint.box, strict=True):
            columns[name].append(value)
        columns["snr_db"].append(footprint.snr_db)
        columns["snr_baseline_db"].append(footprint.snr_baseline_db)
        columns["snr_excursion_db"].append(footprint.snr_excursion_db)
        columns["visible_fraction"].append(footprint.visible_fraction)
        columns["visible"].append(footprint.visible)
        for name, value in zip(
            FOOTPRINT_COLUMNS[-4:], footprint.harmonic_box, strict=True
        ):
            columns[name].append(value)
    out = table.copy()
    for name in FOOTPRINT_COLUMNS:
        out[name] = np.asarray(
            columns[name], dtype=bool if name == "visible" else np.float64
        )
    return out


def sign_v_powers(
    recording: np.ndarray,
    samplerate_hz: float,
    frame_times_s: np.ndarray,
    freqs_hz: np.ndarray,
    nfft: int,
    *,
    workers: int = 1,
    chunk: int = 64,
) -> np.ndarray:
    """Per-electrode power at given frequencies as wavetracker's ``sign_v``.

    For every frame: ``nfft`` samples centred at ``round(t * fs)``, mean
    removed per channel, periodic Hann window, one-sided power spectral
    density with matplotlib ``specgram`` scaling (``2 |X|^2 / (fs sum w^2)``,
    V^2/Hz), at the bin nearest each frequency, in dB (floored at -300 dB).
    This is the definition verified on the production tracks (calibration
    v1). Samples outside the recording are zero (wavetracker has no frames
    there). The FFTs run in float32.

    Parameters
    ----------
    recording : numpy.ndarray, shape (n, E)
    samplerate_hz : float
    frame_times_s : numpy.ndarray, shape (T,)
    freqs_hz : numpy.ndarray, shape (F, T)
        Frequency of each of F fish at each frame.
    nfft : int
    workers : int
        Threads for the FFTs.
    chunk : int
        Frames transformed at once.

    Returns
    -------
    numpy.ndarray, shape (F, T, E)
        Power in dB re 1 V^2/Hz.
    """
    import scipy.fft

    n, n_electrodes = recording.shape
    half = nfft // 2
    window = np.hanning(nfft + 1)[:nfft]
    scale = 2.0 / (samplerate_hz * float(np.sum(window**2)))
    window32 = window.astype(np.float32)
    centres = np.round(np.asarray(frame_times_s) * samplerate_hz).astype(np.int64)
    bins = np.round(np.asarray(freqs_hz) * nfft / samplerate_hz).astype(np.int64)
    bins = np.clip(bins, 0, nfft // 2)
    out = np.empty((bins.shape[0], centres.size, n_electrodes), dtype=np.float64)
    offsets = np.arange(nfft)
    length = max(n, int(centres.max(initial=0)) + 1) + nfft
    for e in range(n_electrodes):
        padded = np.zeros(length, dtype=np.float32)
        padded[half : half + n] = recording[:, e]
        for first in range(0, centres.size, chunk):
            frames = centres[first : first + chunk]
            block = padded[frames[:, None] + offsets[None, :]]  # starts at c - half
            block -= block.mean(axis=1, keepdims=True)
            block *= window32
            spectrum = np.asarray(scipy.fft.rfft(block, axis=1, workers=workers))
            rows = np.arange(frames.size)
            for f in range(bins.shape[0]):
                values = spectrum[rows, bins[f, first : first + frames.size]]
                power = scale * (
                    np.real(values).astype(np.float64) ** 2
                    + np.imag(values).astype(np.float64) ** 2
                )
                out[f, first : first + frames.size, e] = 10.0 * np.log10(
                    np.maximum(power, 1e-30)
                )
    return out


def ground_truth_tracks(
    plan: ScenePlan, recording: np.ndarray, *, workers: int = 1
) -> Tracks:
    """Baseline EODf (no chirps) and ``sign_v`` measured on the recording."""
    model = plan.model
    cfg = plan.config.tracks
    fs = model.samplerate_hz
    times = _track_times(model, cfg.step_s)
    n_frames = times.size
    if not model.fish:
        return Tracks(
            times_s=times,
            freqs_hz=np.empty(0),
            idents=np.empty(0),
            time_indices=np.empty(0, dtype=np.int64),
            powers_db=np.empty((0, model.n_electrodes)),
        )
    freqs = np.stack([model.base_frequency_at(f.fish_id, times) for f in model.fish])
    powers = sign_v_powers(
        recording, fs, times, freqs, cfg.power_nfft(fs), workers=workers
    )
    n_fish = len(model.fish)
    index = np.tile(np.arange(n_frames, dtype=np.int64), n_fish)
    ident = np.repeat(np.arange(n_fish, dtype=np.float64), n_frames)
    order = np.lexsort((ident, index))  # frame-major, like wavetracker output
    return Tracks(
        times_s=times,
        freqs_hz=freqs.reshape(-1)[order],
        idents=ident[order],
        time_indices=index[order],
        powers_db=powers.reshape(-1, model.n_electrodes)[order],
    )


# -- writing ----------------------------------------------------------------


def scene_dir(config: SceneConfig, seed: int, root: str | Path = DEFAULT_ROOT) -> Path:
    """Default directory of a scene: ``root / config.name / f"{seed:06d}"``."""
    return Path(root) / config.name / f"{seed:06d}"


@functools.cache
def _versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": platform.python_version()}
    for package in ("gridtools", "numpy", "scipy", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--", "."],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        versions["gridtools_git_sha"] = sha
        versions["gridtools_git_dirty"] = bool(dirty)
    except (OSError, subprocess.SubprocessError):
        versions["gridtools_git_sha"] = None
        versions["gridtools_git_dirty"] = None
    return versions


def _dumps(obj: Any, level: int = 0) -> str:
    """JSON with one key per line but lists of scalars on one line."""
    pad = " " * (level + 1)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [
            f"{pad}{json.dumps(str(k))}: {_dumps(v, level + 1)}" for k, v in obj.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + " " * level + "}"
    if isinstance(obj, list | tuple):
        if all(not isinstance(v, dict | list | tuple) for v in obj):
            return json.dumps(list(obj))
        items = [f"{pad}{_dumps(v, level + 1)}" for v in obj]
        return "[\n" + ",\n".join(items) + "\n" + " " * level + "]"
    return json.dumps(obj)


def _model_dict(model: SceneModel) -> dict[str, Any]:
    return {
        "samplerate_hz": model.samplerate_hz,
        "n_samples": model.n_samples,
        "gain_update_hz": model.gain_update_hz,
        "electrode_xy_m": model.electrode_xy_m.tolist(),
        "reference_xy_m": None
        if model.reference_xy_m is None
        else model.reference_xy_m.tolist(),
        "channel_gain": model.channel_gain.tolist(),
        "shared_drift": model.shared_drift.to_dict(),
        "fish": [
            {
                "fish_id": fish.fish_id,
                "baseline_hz": fish.baseline_hz,
                "amplitude": fish.amplitude,
                "phase0_rad": fish.phase0_rad,
                "harmonics": {
                    "amplitudes": list(fish.harmonics.amplitudes),
                    "phases_rad": list(fish.harmonics.phases_rad),
                },
                "dipole": vars(fish.dipole),
                "drift": fish.drift.to_dict(),
                "ramps": [vars(r) for r in fish.ramps],
            }
            for fish in model.fish
        ],
    }


def scene_info(scene: Scene) -> dict[str, Any]:
    """The content of ``scene.json``."""
    plan = scene.plan
    config = plan.config
    model = plan.model
    streams: dict[str, Any] = {name: list(key) for name, key in STREAMS.items()}
    streams["fish"] = {name: [100, "<fish_id>", k] for name, k in FISH_STREAMS.items()}
    schedule = plan.schedule
    return {
        "format_version": FORMAT_VERSION,
        "name": config.name,
        "seed": plan.seed,
        "duration_s": model.n_samples / model.samplerate_hz,
        "samplerate_hz": model.samplerate_hz,
        "created_utc": datetime.now(UTC).isoformat(),
        "random": {
            "scheme": SEED_SCHEME,
            "spawn_keys": streams,
        },
        "config": config.to_dict(),
        "versions": _versions(),
        "templates": {
            "path": str(plan.pool.path),
            "manifest_sha256": plan.pool.manifest_sha256,
            "samplerate_hz": plan.pool.samplerate_hz,
            "n_samples": plan.pool.n_samples,
            "n_eligible": int(plan.pool.template_ids.size),
            "sampling": plan.pool.sampling,
            "sampling_options": {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in plan.pool.sampling_options.items()
            },
            "split": None if plan.pool.split is None else plan.pool.split.record(),
        },
        "grid": {
            "rows": plan.rows,
            "columns": plan.columns,
            "spacing_m": plan.spacing_m,
            "n_electrodes": plan.layout.n_electrodes,
            "kind": plan.layout.kind,
            "regular": plan.layout.regular,
            "channel_order": (
                "row-major: channel r * columns + c at (c, r) * spacing"
                if plan.layout.regular
                else "electrode_xy_m (also electrodes.csv)"
            ),
            "electrode_xy_m": model.electrode_xy_m.tolist(),
            "arena_m": vars(plan.arena),
            "reference": plan.levels["reference"],
            "reference_xy_m": plan.levels["reference_xy_m"],
        },
        "recording": {
            "file": "recording.wav",
            "samplerate_hz": model.samplerate_hz,
            "n_samples": model.n_samples,
            "n_channels": model.n_electrodes,
            "dtype": "float32",
            "unit": "V at the ADC (electrode potential x 100), as the real raw data"
            + (
                "; the background is converted to it by its scale"
                if plan.background is not None
                else ""
            ),
            "clip_limit": config.noise.clip_limit,
        },
        "levels": plan.levels,
        "channels": plan.channels,
        "fish": plan.fish_info,
        "chirp_schedule": {
            "n_chirps": len(schedule.events),
            "n_spontaneous": schedule.n_spontaneous,
            "n_response_attempts": schedule.n_response_attempts,
            "n_responses": schedule.n_responses,
            "n_dropped_interval": schedule.n_dropped_interval,
            "n_dropped_cap": schedule.n_dropped_cap,
        },
        "clicks": {"rate_hz": scene.click_rate_hz, "n_clicks": len(scene.clicks)},
        "background": (
            {"used": False}
            if plan.background is None
            else {
                **plan.background.record(),
                "real_fish_ident_offset": REAL_FISH_IDENT_OFFSET,
                "n_real_fish": len(scene.real_fish),
            }
        ),
        "real_fish": _records(scene.real_fish),
        "footprint": {
            **dataclasses.asdict(config.footprint),
            "spectrogram": (
                "Hann window, frames centred at nfft/2 + k hop, one-sided PSD "
                "(specgram scaling) summed over electrodes, linear power"
            ),
            "definition": (
                "D = |P(emitter with all chirps) - P(emitter without this chirp, "
                "phase aligned)|; footprint = cells with D >= max(D) * "
                "10**(-threshold_db/10) connected to the chirp trajectory; box at "
                "cell edges; 10 log10(sum D / sum B) over the footprint (snr_db), its "
                "cells in the emitter's baseline band (snr_baseline_db) and outside all "
                "baseline bands (snr_excursion_db); visible = max(baseline, excursion) "
                ">= visible_snr_db; B = "
                "spectrogram of the recording minus the emitter ('others') or minus "
                "this chirp ('recording'), see background"
            ),
        },
        "model": _model_dict(model),
        "timings_s": scene.timings_s,
    }


def _records(table: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-ready rows (NaN -> None, numpy scalars -> Python)."""
    if table.empty:
        return []
    out = []
    for row in table.to_dict("records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            clean[str(key)] = value
        out.append(clean)
    return out


def _write_chirp_traces(model: SceneModel, path: Path) -> None:
    chirps = sorted(model.chirps.values(), key=lambda c: c.chirp_id)
    lengths = np.array([c.delta_f_hz.size for c in chirps], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    def column(name: str, dtype: Any) -> np.ndarray:
        return np.array([getattr(c, name) for c in chirps], dtype=dtype)

    empty32 = np.empty(0, dtype=np.float32)
    np.savez(
        path,
        chirp_id=column("chirp_id", np.int64),
        fish_id=column("fish_id", np.int64),
        template_id=column("template_id", np.int64),
        center_index=column("center_index", np.int64),
        start=column("start", np.int64),
        dip=column("dip", np.float64),
        support_start=column("support_start", np.int64),
        support_stop=column("support_stop", np.int64),
        onset_index=column("onset_index", np.int64),
        peak_index=column("peak_index", np.int64),
        offset_index=column("offset_index", np.int64),
        excursion_hz=column("excursion_hz", np.float64),
        undershoot_hz=column("undershoot_hz", np.float64),
        is_response=column("is_response", bool),
        response_to=column("response_to", np.int64),
        offsets=offsets,
        delta_f_hz=np.concatenate([c.delta_f_hz for c in chirps])
        if chirps
        else empty32,
        amp_factor=np.concatenate([c.amp_factor for c in chirps])
        if chirps
        else empty32,
    )


def _write_paths(model: SceneModel, path: Path) -> None:
    paths = [fish.path for fish in model.fish]
    times = paths[0].times_s if paths else np.empty(0)
    np.savez(
        path,
        times_s=times,
        xy_m=np.stack([p.xy_m for p in paths]) if paths else np.empty((0, 0, 2)),
        heading_rad=np.stack([p.heading_rad for p in paths])
        if paths
        else np.empty((0, 0)),
        velocity_m_s=np.stack([p.velocity_m_s for p in paths])
        if paths
        else np.empty((0, 0)),
        moving=np.stack([p.moving for p in paths])
        if paths
        else np.empty((0, 0), dtype=bool),
        n_wall_reflections=np.array([p.n_wall_reflections for p in paths]),
    )


def write_scene(
    scene: Scene,
    out_dir: str | Path | None = None,
    *,
    plots: bool | None = None,
) -> Path:
    """Write a scene directory (see the module docstring for the files).

    Parameters
    ----------
    scene : Scene
    out_dir : path, optional
        Target directory; default :func:`scene_dir` under :data:`DEFAULT_ROOT`.
        Existing files of the same names are overwritten.
    plots : bool, optional
        Write the annotated spectrograms; default ``config.plot.enabled``.

    Returns
    -------
    pathlib.Path
        The scene directory.
    """
    out = scene_dir(scene.config, scene.seed) if out_dir is None else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = scene.model
    t0 = time.perf_counter()
    write_wav(out / "recording.wav", scene.recording, model.samplerate_hz)
    write_fishgrid_config(
        out / "fishgrid.cfg",
        samplerate_hz=model.samplerate_hz,
        rows=scene.plan.rows,
        columns=scene.plan.columns,
        spacing_m=scene.plan.spacing_m,
    )
    electrodes_csv = out / "electrodes.csv"
    if not scene.plan.layout.regular:
        pd.DataFrame(
            {
                "channel": np.arange(model.n_electrodes),
                "x_m": model.electrode_xy_m[:, 0],
                "y_m": model.electrode_xy_m[:, 1],
            }
        ).to_csv(electrodes_csv, index=False)
    elif electrodes_csv.exists():
        electrodes_csv.unlink()
    save_tracks(scene.tracks, out)
    scene.chirps.to_csv(out / "chirps.csv", index=False)
    scene.chirps.to_parquet(out / "chirps.parquet", index=False)
    scene.clicks.to_csv(out / "clicks.csv", index=False)
    _write_paths(model, out / "paths.npz")
    _write_chirp_traces(model, out / "chirp_traces.npz")
    scene.timings_s["write"] = time.perf_counter() - t0

    do_plots = scene.config.plot.enabled if plots is None else plots
    if do_plots:
        t1 = time.perf_counter()
        try:
            from gridtools.sim.plot import plot_scene
        except ImportError:  # matplotlib missing
            logger.warning("matplotlib not installed; skipping scene plots")
        else:
            plot_scene(scene, out)
        scene.timings_s["plot"] = time.perf_counter() - t1

    info = scene_info(scene)
    (out / "scene.json").write_text(_dumps(info) + "\n")
    logger.info("wrote scene to %s", out)
    return out


# -- loading and re-rendering -----------------------------------------------


def _load_paths(path: Path) -> list[FishPath]:
    with np.load(path) as data:
        times = data["times_s"]
        return [
            FishPath(
                times_s=times,
                xy_m=data["xy_m"][i],
                heading_rad=data["heading_rad"][i],
                velocity_m_s=data["velocity_m_s"][i],
                moving=data["moving"][i],
                n_wall_reflections=int(data["n_wall_reflections"][i]),
            )
            for i in range(data["xy_m"].shape[0])
        ]


def _load_chirps(path: Path) -> list[ChirpPlacement]:
    with np.load(path) as data:
        offsets = data["offsets"]
        delta = data["delta_f_hz"]
        amp = data["amp_factor"]
        chirps = []
        for k in range(data["chirp_id"].size):
            lo, hi = int(offsets[k]), int(offsets[k + 1])
            chirps.append(
                ChirpPlacement(
                    chirp_id=int(data["chirp_id"][k]),
                    fish_id=int(data["fish_id"][k]),
                    template_id=int(data["template_id"][k]),
                    center_index=int(data["center_index"][k]),
                    start=int(data["start"][k]),
                    delta_f_hz=delta[lo:hi].copy(),
                    amp_factor=amp[lo:hi].copy(),
                    dip=float(data["dip"][k]),
                    support_start=int(data["support_start"][k]),
                    support_stop=int(data["support_stop"][k]),
                    onset_index=int(data["onset_index"][k]),
                    peak_index=int(data["peak_index"][k]),
                    offset_index=int(data["offset_index"][k]),
                    excursion_hz=float(data["excursion_hz"][k]),
                    undershoot_hz=float(data["undershoot_hz"][k]),
                    is_response=bool(data["is_response"][k]),
                    response_to=int(data["response_to"][k]),
                )
            )
    return chirps


def _model_from_dir(directory: Path) -> SceneModel:
    info = json.loads((directory / "scene.json").read_text())
    if info.get("format_version") != FORMAT_VERSION:
        msg = f"{directory}: unsupported scene format {info.get('format_version')!r}"
        raise ValueError(msg)
    data = info["model"]
    paths = _load_paths(directory / "paths.npz")
    chirps = _load_chirps(directory / "chirp_traces.npz")
    fish = []
    for entry in data["fish"]:
        fish_id = int(entry["fish_id"])
        own = sorted((c for c in chirps if c.fish_id == fish_id), key=lambda c: c.start)
        fish.append(
            FishModel(
                fish_id=fish_id,
                baseline_hz=float(entry["baseline_hz"]),
                amplitude=float(entry["amplitude"]),
                phase0_rad=float(entry["phase0_rad"]),
                harmonics=Harmonics(
                    amplitudes=tuple(entry["harmonics"]["amplitudes"]),
                    phases_rad=tuple(entry["harmonics"]["phases_rad"]),
                ),
                dipole=DipoleParams(**entry["dipole"]),
                drift=DriftKnots.from_dict(entry["drift"]),
                ramps=tuple(Ramp(**r) for r in entry["ramps"]),
                path=paths[fish_id],
                chirps=tuple(own),
            )
        )
    return SceneModel(
        samplerate_hz=float(data["samplerate_hz"]),
        n_samples=int(data["n_samples"]),
        electrode_xy_m=np.asarray(data["electrode_xy_m"], dtype=np.float64),
        channel_gain=np.asarray(data["channel_gain"], dtype=np.float64),
        shared_drift=DriftKnots.from_dict(data["shared_drift"]),
        fish=tuple(fish),
        gain_update_hz=float(data["gain_update_hz"]),
        reference_xy_m=None
        if data.get("reference_xy_m") is None
        else np.asarray(data["reference_xy_m"], dtype=np.float64),
    )


@functools.lru_cache(maxsize=8)
def _cached_model(directory: str, mtime_ns: int) -> SceneModel:
    del mtime_ns  # part of the cache key only
    return _model_from_dir(Path(directory))


def load_scene_model(directory: str | Path) -> SceneModel:
    """Load the render model of a written scene (cached per directory).

    Needs ``scene.json``, ``paths.npz`` and ``chirp_traces.npz``; not the
    template library and not the recording.
    """
    directory = Path(directory).resolve()
    mtime = (directory / "scene.json").stat().st_mtime_ns
    return _cached_model(str(directory), mtime)


def _as_model(scene: "Scene | SceneModel | str | Path") -> SceneModel:
    if isinstance(scene, Scene):
        return scene.model
    if isinstance(scene, SceneModel):
        return scene
    return load_scene_model(scene)


def _span(
    model: SceneModel, start_s: float | None, stop_s: float | None
) -> tuple[int, int]:
    return _span_samples(model.samplerate_hz, model.n_samples, start_s, stop_s)


def _span_samples(
    fs: float, n_samples: int, start_s: float | None, stop_s: float | None
) -> tuple[int, int]:
    start = 0 if start_s is None else math.ceil(start_s * fs - 1e-9)
    stop = n_samples if stop_s is None else math.ceil(stop_s * fs - 1e-9)
    if not 0 <= start <= stop <= n_samples:
        msg = (
            f"span [{start_s}, {stop_s}) s is outside the scene [0, {n_samples / fs}) s"
        )
        raise IndexError(msg)
    return start, stop


def render_fish(
    scene: "Scene | SceneModel | str | Path",
    fish_id: int,
    start_s: float | None = None,
    stop_s: float | None = None,
    *,
    chirps: ChirpSelection = "all",
    electrodes: Sequence[int] | None = None,
    phase: PhaseMode = "aligned",
) -> Float32Array:
    """Re-render one fish's contribution to the recording.

    Exact: with ``chirps="all"`` the result equals the fish's term in the
    written recording (``recording = sum_i render_fish(i) + background``, to
    float32 rounding of the sum). Sub-spans are bit-identical to the same
    samples of a full-length rendering.

    Parameters
    ----------
    scene : Scene, SceneModel or path
        A simulated scene, its model, or a scene directory.
    fish_id : int
        Fish (= track ident = ``emitter_id`` in ``chirps.csv``).
    start_s, stop_s : float, optional
        Span in seconds; samples ``ceil(start_s * fs) .. ceil(stop_s * fs) - 1``
        (as :meth:`gridtools.io.Recording.read_time`). Default: whole scene.
        Must lie inside the scene.
    chirps : "all", "none" or collection of int
        ``"all"`` renders every chirp, ``"none"`` none of this fish's, and a
        collection of chirp ids renders all except those.
    electrodes : sequence of int, optional
        Channels to return (default all).
    phase : {"aligned", "natural"}
        How omitted chirps affect the phase (see :mod:`gridtools.sim.render`).
        ``"aligned"`` (default): outside an omitted chirp's support the
        signal is identical to the full rendering, so "with minus without"
        is confined to the support. ``"natural"``: the fish that never
        chirped there; its phase lags after the chirp.

    Returns
    -------
    numpy.ndarray of float32, shape (n, len(electrodes))
    """
    model = _as_model(scene)
    if not 0 <= fish_id < len(model.fish):
        msg = f"no fish {fish_id} in this scene ({len(model.fish)} fish)"
        raise KeyError(msg)
    start, stop = _span(model, start_s, stop_s)
    return model.render(
        fish_id, start, stop, chirps=chirps, phase=phase, electrodes=electrodes
    )


def _plan_from_dir(directory: Path) -> ScenePlan:
    info = json.loads((directory / "scene.json").read_text())
    config = SceneConfig.from_dict(info["config"])
    return plan_scene(config, int(info["seed"]))


def render_background(
    scene: "Scene | str | Path",
    start_s: float | None = None,
    stop_s: float | None = None,
    *,
    electrodes: Sequence[int] | None = None,
) -> Float32Array:
    """Re-render everything except the fish: noise, hum and clicks.

    Synthetic background: regenerated from the scene's random streams over
    the whole scene (the coloured noise is synthesised by one FFT), then cut
    to the span, so this costs about as much as the noise of a full
    simulation. For a scene directory the template library must be
    available (the plan is redrawn from config and seed).

    Real background (hybrid scene): ``scale`` times the background excerpt,
    reloaded for a scene directory from the library recorded in
    ``scene.json`` (segment id, offset, channels, resampling, scale; the
    template library is not needed).

    In both cases ``recording = sum_i render_fish(i) + render_background``
    to float32 rounding of the sum (without clipping).

    Returns
    -------
    numpy.ndarray of float32, shape (n, len(electrodes))
    """
    if isinstance(scene, Scene):
        plan = scene.plan
    else:
        directory = Path(scene)
        info = json.loads((directory / "scene.json").read_text())
        record = info.get("background") or {}
        if record.get("used"):
            fs = float(info["samplerate_hz"])
            n_samples = int(info["recording"]["n_samples"])
            background = reproduce(record, fs, n_samples)
            start, stop = _span_samples(fs, n_samples, start_s, stop_s)
            out = background.scaled(start, stop)
            return np.ascontiguousarray(
                out if electrodes is None else out[:, list(electrodes)]
            )
        plan = _plan_from_dir(directory)
    model = plan.model
    start, stop = _span(model, start_s, stop_s)
    if plan.background is not None:
        out = plan.background.scaled(start, stop)
        return np.ascontiguousarray(
            out if electrodes is None else out[:, list(electrodes)]
        )
    components, _, clicks = _background_parts(plan)
    if not components or not isinstance(components[0], np.ndarray):
        zeros = np.zeros((model.n_samples, model.n_electrodes), dtype=np.float32)
        components = [zeros, *components]
    background = mix(
        [],
        [],
        samplerate_hz=model.samplerate_hz,
        additive=components,
        channel_gain=model.channel_gain,
        out=components[0],
    )
    add_clicks(background, clicks)
    out = background[start:stop]
    if electrodes is not None:
        out = out[:, list(electrodes)]
    return np.ascontiguousarray(out)


def chirps_of(scene: "Scene | SceneModel | str | Path", fish_id: int) -> list[int]:
    """Chirp ids of one fish."""
    return _as_model(scene).fish[fish_id].chirp_ids


def omit_all_but(
    scene: "Scene | SceneModel | str | Path", keep: Collection[int]
) -> list[int]:
    """All chirp ids except ``keep`` (for ``render_fish(chirps=...)``)."""
    model = _as_model(scene)
    keep_set = {int(k) for k in keep}
    return [c for c in model.chirps if c not in keep_set]


def replace_config(config: SceneConfig, **sections: Any) -> SceneConfig:
    """Copy of ``config`` with whole sections or nested fields replaced.

    ``replace_config(cfg, noise={"snr_db": (0, 0)})`` replaces fields of a
    section; passing a dataclass replaces the section.
    """
    changes = {}
    for name, value in sections.items():
        if isinstance(value, dict):
            changes[name] = replace(getattr(config, name), **value)
        else:
            changes[name] = value
    return replace(config, **changes)
