"""Calibrated scene defaults: units, field ridge, reference, noise, clicks.

Where possible the calibration's own measurement functions
(:mod:`gridtools.calibrate.spectra`) are run on simulated scenes.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from gridtools.calibrate.spectra import line_amplitude, noise_floor, psd
from gridtools.io import GridGeometry
from gridtools.sim.config import (
    ClickConfig,
    FieldConfig,
    GridConfig,
    MovementConfig,
    NoiseConfig,
    PopulationConfig,
    SceneConfig,
)
from gridtools.sim.field import dipole_gain
from gridtools.sim.movement import Arena
from gridtools.sim.scene import (
    _population,
    calibrated_strongest_gain,
    median_strongest_gain,
    plan_scene,
    render_background,
    render_fish,
    simulate_scene,
)

from .conftest import small_config

FS = 20_000.0


def test_baseline_mixture_follows_the_calibrated_groups(library_dir: Path) -> None:
    population = PopulationConfig(n_fish=(5, 5), close_pair_prob=0.0)
    config = small_config(library_dir, population=population)
    groups: dict[str, list[float]] = {"uniform": [], "female": [], "male": []}
    for seed in range(300):
        baselines, _, info = _population(config, seed)
        for f, entry in zip(baselines, info, strict=True):
            groups[entry["baseline_group"]].append(f)
    n = sum(len(v) for v in groups.values())
    assert abs(len(groups["uniform"]) / n - 0.3) < 0.04
    assert np.mean(groups["female"]) == pytest.approx(664.3, abs=5)
    assert np.std(groups["female"]) == pytest.approx(34.9, abs=4)
    assert np.mean(groups["male"]) == pytest.approx(803.2, abs=6)
    assert min(groups["uniform"]) >= 550
    assert max(groups["uniform"]) <= 1100


def test_field_follows_the_ridge_and_keeps_the_calibrated_amplitude() -> None:
    field = FieldConfig()
    assert field.regulariser_rel(1.5) == pytest.approx(0.05)
    assert field.regulariser_rel(1.875) == pytest.approx(0.225)
    assert field.regulariser_rel(2.0) == pytest.approx(0.3)
    g_cal = calibrated_strongest_gain(field)
    # the calibration's simulated median strongest-electrode gain was 0.215
    assert g_cal == pytest.approx(0.215, rel=0.15)
    gains = {}
    for spacing in (0.15, 0.5):
        electrodes = GridGeometry.regular(5, 3, spacing, spacing).positions_m
        arena = Arena(0.0, 4 * spacing, 0.0, 2 * spacing)
        gains[spacing] = median_strongest_gain(
            np.asarray(electrodes), arena, 1.5, 0.05 * spacing, 0.1
        )
    assert gains[0.5] == pytest.approx(g_cal)
    # closer electrodes see stronger fields: the amplitude is scaled down
    assert g_cal / gains[0.15] < 1.0


def test_plans_draw_spacing_field_and_amplitude(library_dir: Path) -> None:
    config = small_config(
        library_dir, grid=GridConfig(layouts=((3, 5),), real_layout_prob=1.0)
    )
    spacings, exponents = [], []
    for seed in range(40):
        plan = plan_scene(config, seed)
        levels = plan.levels
        spacings.append(levels["spacing_m"])
        exponents.append(levels["field_exponent"])
        assert levels["field_regulariser_m"] == pytest.approx(
            config.field.regulariser_rel(levels["field_exponent"]) * levels["spacing_m"]
        )
        assert plan.model.electrode_xy_m[1, 0] == pytest.approx(levels["spacing_m"])
        for fish in plan.model.fish:
            assert fish.dipole.distance_exponent == levels["field_exponent"]
            assert fish.dipole.min_distance_m == levels["field_regulariser_m"]
            ratio = fish.amplitude / (2.16e-3 * levels["amplitude_scale"])
            assert np.exp(-2.5 * 0.748) - 1e-9 <= ratio <= np.exp(2.5 * 0.748) + 1e-9
            # fish stay over the grid (margin 0)
            xy = fish.path.xy_m
            assert xy.min() >= -1e-9
            assert xy[:, 0].max() <= 4 * levels["spacing_m"] + 1e-9
    assert min(spacings) >= 0.15
    assert max(spacings) <= 0.5
    assert max(spacings) - min(spacings) > 0.2
    assert min(exponents) >= 1.5
    assert max(exponents) <= 2.0


@pytest.mark.parametrize("prob", [0.0, 1.0])
def test_reference_electrode_is_subtracted_per_fish(
    library_dir: Path, prob: float
) -> None:
    config = small_config(
        library_dir,
        duration_s=3.0,
        grid=GridConfig(layouts=((3, 3),), reference_prob=prob),
    )
    scene = simulate_scene(config, 5)
    model = scene.model
    assert (model.reference_xy_m is not None) == (prob == 1.0)
    assert scene.plan.levels["reference"] == ("in tank" if prob else "distant")
    fish = model.fish[0]
    track = model.gain_track(0)
    plain = dipole_gain(
        fish.path.xy_m,
        fish.path.heading_rad,
        model.electrode_xy_m,
        fish_depth_m=0.0,
        electrode_depth_m=0.0,
        min_distance_m=fish.dipole.min_distance_m,
        distance_exponent=fish.dipole.distance_exponent,
        reference_distance_m=fish.dipole.reference_distance_m,
    )
    if model.reference_xy_m is None:
        np.testing.assert_array_equal(track.gains, plain)
    else:
        reference = dipole_gain(
            fish.path.xy_m,
            fish.path.heading_rad,
            model.reference_xy_m[None, :],
            fish_depth_m=0.0,
            electrode_depth_m=0.0,
            min_distance_m=fish.dipole.min_distance_m,
            distance_exponent=fish.dipole.distance_exponent,
            reference_distance_m=fish.dipole.reference_distance_m,
        )
        np.testing.assert_allclose(track.gains, plain - reference, atol=1e-7)
    # linearity holds with or without an in-tank reference
    total = sum(render_fish(scene, i) for i in range(len(model.fish)))
    np.testing.assert_allclose(
        scene.recording,
        total + render_background(scene),
        atol=8 * np.finfo(np.float32).eps * np.abs(scene.recording).max(),
        rtol=0,
    )


def test_noise_floor_and_hum_measure_as_drawn(library_dir: Path) -> None:
    """Closing the loop with the calibration's own estimators."""
    noise = NoiseConfig(
        electrode_floor_sd_db=0.0,
        gain_mismatch_log_sd=0.0,
        dead_prob=0.0,
        noisy_prob=0.0,
        harder_prob=0.0,
        clip_limit=None,
    )
    config = small_config(
        library_dir,
        duration_s=10.0,
        noise=noise,
        clicks=ClickConfig(rate_per_h_median=0.0),
    )
    scene = simulate_scene(config, 2)
    background = render_background(scene)
    freqs, power = psd(background, FS, 8192)
    floor_db = 10 * np.log10(noise_floor(freqs, power, (400.0, 1500.0)))
    np.testing.assert_allclose(
        floor_db, scene.plan.levels["noise_floor_db_v2_hz"], atol=0.5
    )
    hum_only = dataclasses.replace(noise, colored=False)
    quiet = simulate_scene(dataclasses.replace(config, noise=hum_only), 2)
    freqs, power = psd(render_background(quiet), FS, 8192)
    amplitude = line_amplitude(freqs, power, 50.0)
    lo, hi = noise.hum_amplitude_v
    assert lo * 0.8 <= amplitude.max() <= hi * 1.25
    # the strongest electrode carries the drawn amplitude
    assert amplitude.max() == pytest.approx(
        quiet.plan.levels["hum_amplitude_v"], rel=0.15
    )


def test_harder_scenes_raise_the_floor(library_dir: Path) -> None:
    noise = NoiseConfig(harder_prob=1.0, harder_db=(20.0, 20.0))
    plan = plan_scene(small_config(library_dir, noise=noise), 3)
    easy = plan_scene(
        small_config(library_dir, noise=dataclasses.replace(noise, harder_prob=0.0)), 3
    )
    assert plan.levels["noise_floor_db_v2_hz"] == pytest.approx(
        easy.levels["noise_floor_db_v2_hz"] + 20.0
    )


def test_shared_drift_is_a_centred_linear_trend(library_dir: Path) -> None:
    population = PopulationConfig(
        shared_trend_per_h=(-0.0094, 0.0),
        shared_drift=dataclasses.replace(PopulationConfig().shared_drift, sd_rel=0.0),
    )
    plan = plan_scene(
        small_config(library_dir, duration_s=60.0, population=population), 1
    )
    shared = plan.model.shared_drift
    t = np.array([0.0, 30.0, 60.0])
    np.testing.assert_allclose(shared(t), -0.0094 * (t - 30.0) / 3600.0, atol=1e-12)


def test_movement_regimes(library_dir: Path) -> None:
    for per in ("scene", "fish"):
        movement = MovementConfig(light_prob=0.5, per=per)
        mixed = 0
        for seed in range(30):
            plan = plan_scene(
                small_config(library_dir, duration_s=2.0, movement=movement), seed
            )
            regimes = {f["movement"] for f in plan.fish_info}
            mixed += len(regimes) > 1
        assert (mixed == 0) if per == "scene" else (mixed > 0)
    light = plan_scene(
        small_config(
            library_dir, duration_s=60.0, movement=MovementConfig(light_prob=1.0)
        ),
        0,
    )
    rest = np.mean([np.mean(~f.path.moving) for f in light.model.fish])
    assert rest > 0.7


def test_click_patterns(library_dir: Path) -> None:
    clicks = ClickConfig(
        rate_per_h_median=0.0,
        periodic_prob=1.0,
        led_prob=1.0,
    )
    scene = simulate_scene(small_config(library_dir, duration_s=25.0, clicks=clicks), 0)
    table = scene.clicks
    periodic = np.sort(table.loc[table["origin"] == "periodic", "t_start_s"].to_numpy())
    np.testing.assert_allclose(np.diff(periodic[::2]), 10.24, atol=1e-3)
    np.testing.assert_allclose(periodic[1::2] - periodic[::2], 0.03, atol=1e-3)
    led = np.sort(table.loc[table["origin"] == "led", "t_start_s"].to_numpy())
    np.testing.assert_allclose(led[1::2] - led[::2], 0.1, atol=1e-3)
    assert led.size >= 40


def test_click_rate_distribution() -> None:
    from gridtools.sim.clicks import _click_times

    draws = [
        _click_times(np.random.default_rng(seed), ClickConfig(), 1.0)[0] * 3600
        for seed in range(2000)
    ]
    assert np.median(draws) == pytest.approx(792, rel=0.15)
    assert np.quantile(draws, 0.95) > 20_000
    assert max(draws) <= 50_000


def test_default_config_units_are_volts() -> None:
    config = SceneConfig()
    assert config.noise.clip_limit == pytest.approx(0.02048)
    assert config.field.amplitude_median_v == pytest.approx(2.16e-3)
    assert config.tracks.power_nfft(20_000.0) == 32768
    assert config.chirps.max_rate_hz is None
