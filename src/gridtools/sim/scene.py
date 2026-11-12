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

Randomness. One seed sequence per scene; every component draws from its own
child sequence with a fixed spawn key (:mod:`gridtools.sim.seeding`), so
changing one component's config does not reshuffle the others (e.g. the
noise config leaves the fish untouched).
"""

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
from typing import Any

import numpy as np
import pandas as pd

from gridtools.io import GridGeometry, Tracks, save_tracks
from gridtools.sim._common import Float32Array
from gridtools.sim.chirps import (
    ChirpPlacement,
    ChirpSchedule,
    TemplatePool,
    amplitude_from_frequency,
    library_api,
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
from gridtools.sim.mix import mix
from gridtools.sim.movement import Arena, FishPath, simulate_path
from gridtools.sim.noise import (
    ColoredNoiseParams,
    ElectrodeFaultParams,
    GainMismatchParams,
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

__all__ = [
    "CHIRP_COLUMNS",
    "DEFAULT_ROOT",
    "FISH_STREAMS",
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


def fish_power_db(
    fundamental_amplitude: np.ndarray | float,
    samplerate_hz: float,
    nfft: int,
) -> np.ndarray:
    """Power of an EOD fundamental as wavetracker reports it, in dB.

    wavetracker's ``sign_v`` is the power spectral density at the fish's
    frequency bin (matplotlib ``specgram`` scaling with a Hann window,
    units^2/Hz), in dB re 1 unit^2/Hz in the 2020 production tracks. For a
    sinusoid of amplitude ``a`` centred on a bin this is
    ``(a**2 / 2) / ENBW`` with the Hann window's equivalent noise bandwidth
    ``ENBW = 1.5 * samplerate_hz / nfft``. Here ``a`` is the noise-free
    amplitude of the fundamental on an electrode (channel gain x field
    gain x EOD amplitude x fundamental's relative amplitude), so the value
    is the fish's own power, without noise or other fish, floored at
    -300 dB. The 2020 trackers' ``nfft`` is not recorded; 16384 matches
    their frame grid (first frame at nfft/2, step 0.15 nfft).

    Parameters
    ----------
    fundamental_amplitude : array_like
        Amplitude of the fundamental (any sign).
    samplerate_hz : float
        Sampling rate.
    nfft : int
        Window length of the reference spectrogram.

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
        Grid layout.
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
    """

    config: SceneConfig
    seed: int
    rows: int
    columns: int
    geometry: GridGeometry
    arena: Arena
    model: SceneModel
    schedule: ChirpSchedule
    fish_info: list[dict[str, Any]]
    levels: dict[str, float]
    channels: dict[str, Any]
    pool: TemplatePool


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
    lo, hi = pop.baseline_hz
    baselines: list[float] = []
    ramps: list[list[Ramp]] = []
    info: list[dict[str, Any]] = []
    for i in range(n_fish):
        # fixed number of draws per fish, whatever the branches
        u = rng.random(13)
        uniform_f = lo + u[0] * (hi - lo)
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

        fish_ramps: list[Ramp] = []
        entry: dict[str, Any] = {"close_to": -1, "close_df_hz": None, "crossing": False}
        if close:
            f = baselines[partner] + sign * df
            if not lo <= f <= hi:
                sign = -sign
                f = baselines[partner] + sign * df
            entry.update(close_to=partner, close_df_hz=float(f - baselines[partner]))
            if cross:
                # move towards and past the partner
                fish_ramps.append(
                    _ramp(cross_dur, scene_s, start_u, -sign * (df + overshoot))
                )
                entry["crossing"] = True
        else:
            f = uniform_f
        if random_ramp:
            fish_ramps.append(
                _ramp(random_dur, scene_s, 1.0 - start_u, random_sign * random_df)
            )
        baselines.append(float(f))
        ramps.append(fish_ramps)
        info.append(entry)
    return baselines, ramps, info


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
    fs = config.samplerate_hz
    n_samples = config.n_samples
    duration_s = n_samples / fs
    pool = TemplatePool.open(config.chirps)

    # grid and arena
    grid_rng = stream(seed, "grid")
    rows, columns = config.grid.layouts[
        int(grid_rng.integers(len(config.grid.layouts)))
    ]
    spacing = config.grid.spacing_m
    geometry = GridGeometry.regular(columns, rows, spacing, spacing)
    margin = config.grid.arena_margin_m
    arena = Arena(
        -margin,
        (columns - 1) * spacing + margin,
        -margin,
        (rows - 1) * spacing + margin,
    )

    baselines, ramps, pair_info = _population(config, seed)
    n_fish = len(baselines)
    shared = config.population.shared_drift
    shared_knots = _drift_knots(
        stream(seed, "shared_drift"),
        duration_s,
        sd=shared.sd_rel,
        timescale_s=shared.timescale_s,
        control_rate_hz=shared.control_rate_hz,
    )

    # per-fish body, waveform, drift, path, chirp rate
    field_cfg = config.field
    bodies = []
    for i in range(n_fish):
        body = fish_stream(seed, i, "body")
        amp_lo, amp_hi = field_cfg.amplitude
        amplitude = float(np.exp(body.uniform(np.log(amp_lo), np.log(amp_hi))))
        depth = float(body.uniform(*field_cfg.fish_depth_m))
        phase0 = float(body.uniform(0.0, 2.0 * np.pi))
        harmonics = jitter_harmonics(
            fish_stream(seed, i, "harmonics"),
            APTERONOTUS_LEPTORHYNCHUS,
            config.population.harmonics,
        )
        drift = config.population.drift
        drift_knots = _drift_knots(
            fish_stream(seed, i, "drift"),
            duration_s,
            sd=drift.sd_hz,
            timescale_s=drift.timescale_s,
            control_rate_hz=drift.control_rate_hz,
        )
        path = simulate_path(
            fish_stream(seed, i, "movement"), duration_s, arena, config.movement
        )
        dipole = DipoleParams(
            fish_depth_m=depth,
            electrode_depth_m=field_cfg.electrode_depth_m,
            min_distance_m=field_cfg.min_distance_m,
            distance_exponent=field_cfg.distance_exponent,
            reference_distance_m=field_cfg.reference_distance_m,
        )
        bodies.append((amplitude, depth, phase0, harmonics, drift_knots, path, dipole))

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
    for i, (amplitude, _, phase0, harmonics, drift_knots, path, dipole) in enumerate(
        bodies
    ):
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
    channel_gain = mismatch.astype(np.float64) * faults.signal_gain.astype(np.float64)

    model = SceneModel(
        samplerate_hz=fs,
        n_samples=n_samples,
        electrode_xy_m=geometry.positions_m,
        channel_gain=channel_gain,
        shared_drift=shared_knots,
        fish=tuple(fish_models),
    )

    # levels relative to the median fish at 0.5 m
    amplitudes = np.array([b[0] for b in bodies])
    depths = np.array([b[1] for b in bodies])
    g_ref = abs(
        float(
            dipole_gain(
                [[0.0, 0.0]],
                [0.0],
                [[0.5, 0.0]],
                fish_depth_m=float(np.median(depths)),
                electrode_depth_m=field_cfg.electrode_depth_m,
                min_distance_m=field_cfg.min_distance_m,
                distance_exponent=field_cfg.distance_exponent,
                reference_distance_m=field_cfg.reference_distance_m,
            )[0, 0]
        )
    )
    a_ref = float(np.median(amplitudes)) * g_ref
    levels_rng = stream(seed, "levels")
    snr_db = float(levels_rng.uniform(*noise_cfg.snr_db))
    exponent = float(levels_rng.uniform(*noise_cfg.exponent))
    levels = {
        "a_ref": a_ref,
        "gain_at_0.5m": g_ref,
        "snr_db": snr_db,
        "noise_sd": a_ref * 10.0 ** (-snr_db / 20.0),
        "noise_exponent": exponent,
        "hum_amplitude_low": a_ref * 10.0 ** (noise_cfg.hum_db[0] / 20.0),
        "hum_amplitude_high": a_ref * 10.0 ** (noise_cfg.hum_db[1] / 20.0),
    }
    channels = {
        "gain_mismatch": mismatch.astype(np.float64).tolist(),
        "dead": faults.dead.tolist(),
        "noisy": faults.noisy.tolist(),
        "noise_scale": faults.noise_scale.astype(np.float64).tolist(),
        "channel_gain": channel_gain.tolist(),
    }

    fish_info = []
    for i, fish in enumerate(fish_models):
        n_resp = sum(1 for c in fish.chirps if c.is_response)
        fish_info.append(
            {
                "fish_id": i,
                "baseline_hz": fish.baseline_hz,
                "amplitude": fish.amplitude,
                "depth_m": fish.dipole.fish_depth_m,
                "phase0_rad": fish.phase0_rad,
                "chirp_rate_hz": rates[i],
                "n_chirps": len(fish.chirps),
                "n_responses": n_resp,
                "ramps": [vars(r) for r in fish.ramps],
                **pair_info[i],
            }
        )

    return ScenePlan(
        config=config,
        seed=seed,
        rows=rows,
        columns=columns,
        geometry=geometry,
        arena=arena,
        model=model,
        schedule=schedule,
        fish_info=fish_info,
        levels=levels,
        channels=channels,
        pool=pool,
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
    """

    plan: ScenePlan
    recording: Float32Array
    chirps: pd.DataFrame
    clicks: pd.DataFrame
    tracks: Tracks
    click_rate_hz: float
    timings_s: dict[str, float] = field(default_factory=dict)

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
) -> tuple[Float32Array, Any, float, list[Click]]:
    """Noise array, hum, click rate and clicks of a scene (deterministic)."""
    config = plan.config
    seed = plan.seed
    model = plan.model
    noise_cfg = config.noise
    noise = colored_noise(
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
    hum = powerline_hum(
        stream(seed, "hum"),
        model.n_samples,
        model.n_electrodes,
        model.samplerate_hz,
        HumParams(
            line_freq_hz=noise_cfg.hum_line_freq_hz,
            n_harmonics=noise_cfg.hum_n_harmonics,
            amplitude_range=(
                plan.levels["hum_amplitude_low"],
                plan.levels["hum_amplitude_high"],
            ),
        ),
    )
    rate, clicks = draw_clicks(
        stream(seed, "clicks"),
        config.clicks,
        n_samples=model.n_samples,
        n_electrodes=model.n_electrodes,
        samplerate_hz=model.samplerate_hz,
        noise_sd=plan.levels["noise_sd"],
    )
    return noise, hum, rate, clicks


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
    model = plan.model
    timings["plan"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    eods = [model.waveform(i, 0, model.n_samples) for i in range(len(model.fish))]
    gains = [model.gain_track(i) for i in range(len(model.fish))]
    timings["fish"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    noise, hum, click_rate, clicks = _background_parts(plan, workers=workers)
    timings["noise"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    recording = mix(
        eods,
        gains,
        samplerate_hz=model.samplerate_hz,
        additive=[noise, hum],
        channel_gain=model.channel_gain,
        clip=None,
        gain_update_hz=model.gain_update_hz,
        workers=workers,
        out=noise,
    )
    add_clicks(recording, clicks)
    if config.noise.clip_limit is not None:
        limit = np.float32(config.noise.clip_limit)
        np.clip(recording, -limit, limit, out=recording)
    timings["mix"] = time.perf_counter() - t3

    t4 = time.perf_counter()
    chirps = chirp_table(plan)
    tracks = ground_truth_tracks(plan)
    timings["tables"] = time.perf_counter() - t4
    timings["simulate_total"] = time.perf_counter() - t0
    logger.info(
        "scene %s/%06d: %d fish, %d chirps, %d clicks, %.2f s",
        config.name,
        seed,
        len(model.fish),
        len(chirps),
        len(clicks),
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
    )


def chirp_table(plan: ScenePlan) -> pd.DataFrame:
    """Ground-truth table of all chirps (columns :data:`CHIRP_COLUMNS`)."""
    model = plan.model
    fs = model.samplerate_hz
    chirps = sorted(model.chirps.values(), key=lambda c: c.chirp_id)
    n_fish = len(model.fish)
    rows = []
    centres = np.array([c.center_index / fs for c in chirps])
    emitters = np.array([c.fish_id for c in chirps])
    power_nfft = plan.config.tracks.power_nfft
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


def ground_truth_tracks(plan: ScenePlan) -> Tracks:
    """Baseline EODf (no chirps) and per-electrode power on a wavetracker grid."""
    model = plan.model
    cfg = plan.config.tracks
    fs = model.samplerate_hz
    last_s = (model.n_samples - 1) / fs
    # up to the first frame at or after the last sample, so every chirp lies
    # within the tracks (wavetracker's frames would stop earlier)
    n_frames = math.ceil(last_s / cfg.step_s - 1e-9) + 1
    times = np.arange(n_frames, dtype=np.float64) * cfg.step_s
    freqs, idents, indices, powers = [], [], [], []
    for fish in model.fish:
        freqs.append(model.base_frequency_at(fish.fish_id, times))
        idents.append(np.full(n_frames, float(fish.fish_id)))
        indices.append(np.arange(n_frames, dtype=np.int64))
        gains = model.gain_track(fish.fish_id).evaluate(times)
        fundamental = (
            gains * model.channel_gain * fish.amplitude * fish.harmonics.amplitudes[0]
        )
        powers.append(fish_power_db(fundamental, fs, cfg.power_nfft))
    if not freqs:
        return Tracks(
            times_s=times,
            freqs_hz=np.empty(0),
            idents=np.empty(0),
            time_indices=np.empty(0, dtype=np.int64),
            powers_db=np.empty((0, model.n_electrodes)),
        )
    index = np.concatenate(indices)
    ident = np.concatenate(idents)
    order = np.lexsort((ident, index))  # frame-major, like wavetracker output
    return Tracks(
        times_s=times,
        freqs_hz=np.concatenate(freqs)[order],
        idents=ident[order],
        time_indices=index[order],
        powers_db=np.concatenate(powers)[order],
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
            "api": library_api(),
        },
        "grid": {
            "rows": plan.rows,
            "columns": plan.columns,
            "spacing_m": config.grid.spacing_m,
            "channel_order": "row-major: channel r * columns + c at (c, r) * spacing",
            "electrode_xy_m": model.electrode_xy_m.tolist(),
            "arena_m": vars(plan.arena),
        },
        "recording": {
            "file": "recording.wav",
            "samplerate_hz": model.samplerate_hz,
            "n_samples": model.n_samples,
            "n_channels": model.n_electrodes,
            "dtype": "float32",
            "unit": (
                "EOD potential on the body axis at field.reference_distance_m of a "
                "fish with amplitude 1 (nominally mV)"
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
        "model": _model_dict(model),
        "timings_s": scene.timings_s,
    }


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
        spacing_m=scene.config.grid.spacing_m,
    )
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
    fs = model.samplerate_hz
    start = 0 if start_s is None else math.ceil(start_s * fs - 1e-9)
    stop = model.n_samples if stop_s is None else math.ceil(stop_s * fs - 1e-9)
    if not 0 <= start <= stop <= model.n_samples:
        msg = (
            f"span [{start_s}, {stop_s}) s is outside the scene "
            f"[0, {model.n_samples / fs}) s"
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

    The background is regenerated from the scene's random streams over the
    whole scene (the coloured noise is synthesised by one FFT), then cut to
    the span, so this costs about as much as the noise of a full simulation.
    For a scene directory the template library must be available (the plan
    is redrawn from config and seed).

    Returns
    -------
    numpy.ndarray of float32, shape (n, len(electrodes))
    """
    plan = scene.plan if isinstance(scene, Scene) else _plan_from_dir(Path(scene))
    model = plan.model
    start, stop = _span(model, start_s, stop_s)
    noise, hum, _, clicks = _background_parts(plan)
    background = mix(
        [],
        [],
        samplerate_hz=model.samplerate_hz,
        additive=[noise, hum],
        channel_gain=model.channel_gain,
        out=noise,
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
