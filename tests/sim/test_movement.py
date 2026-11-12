"""Behaviour tests for the movement model."""

import math

import numpy as np
import pytest

from gridtools.sim.movement import (
    Arena,
    FishPath,
    MovementParams,
    simulate_path,
    upsample_path,
    wrap_angle,
)

# A 3 x 5 grid at 0.5 m spacing spans 1 m x 2 m; leave 0.25 m around it.
ARENA = Arena(x_min_m=-0.25, x_max_m=1.25, y_min_m=-0.25, y_max_m=2.25)


@pytest.fixture(scope="module")
def long_path() -> FishPath:
    return simulate_path(np.random.default_rng(11), 3 * 3600.0, ARENA)


def test_path_stays_inside_arena(long_path: FishPath) -> None:
    assert np.all(ARENA.contains(long_path.xy_m))
    assert long_path.times_s[-1] >= 3 * 3600.0
    np.testing.assert_allclose(np.diff(long_path.times_s), 1.0 / 30.0)


def test_reflecting_boundary_without_wall_avoidance() -> None:
    params = MovementParams(
        wall_margin_m=0.0,
        mean_rest_s=0.0,
        speed_mean_m_s=0.4,
        speed_fluct_log_sd=0.0,
        turn_rate_sd_rad_s=0.0,
        backward_prob=0.0,
    )
    path = simulate_path(
        np.random.default_rng(3),
        20.0,
        ARENA,
        params,
        start_xy_m=(1.0, 1.0),
        start_heading_rad=0.0,
    )
    assert path.n_wall_reflections > 0
    assert np.all(ARENA.contains(path.xy_m))
    # It hits the wall at x_max and bounces back while its heading stays put.
    assert path.xy_m[:, 0].max() == pytest.approx(ARENA.x_max_m, abs=0.02)
    np.testing.assert_allclose(path.heading_rad, 0.0, atol=1e-12)


def test_wall_avoidance_keeps_reflections_rare(long_path: FishPath) -> None:
    n_bouts_moving = np.count_nonzero(np.diff(long_path.moving.astype(int)) == 1)
    assert long_path.n_wall_reflections < 0.01 * long_path.times_s.size
    assert n_bouts_moving > 100


def test_rest_fraction_matches_dwell_times(long_path: FishPath) -> None:
    params = MovementParams()
    rest = ~long_path.moving
    assert rest.mean() == pytest.approx(params.rest_fraction, abs=0.06)
    # Well into a rest bout (> 2 s after it started) the fish is still.
    settled = rest.copy()
    lag_steps = int(2.0 * params.rate_hz)
    starts = np.flatnonzero(np.diff(rest.astype(int)) == 1) + 1
    for start in starts:
        settled[start : start + lag_steps] = False
    assert np.max(long_path.speed_m_s[settled]) < 0.01
    # Rest and move bouts both last seconds, not single steps.
    assert starts.size < long_path.times_s[-1] / 10.0


def test_speed_bounds_and_distribution(long_path: FishPath) -> None:
    params = MovementParams()
    speed = long_path.speed_m_s
    assert speed.max() <= params.speed_max_m_s + 1e-12
    moving_speed = speed[long_path.moving]
    assert np.median(moving_speed) == pytest.approx(params.speed_mean_m_s, rel=0.4)
    backward = long_path.velocity_m_s[long_path.moving] < 0
    assert 0.02 < backward.mean() < 0.3


def test_heading_is_continuous_with_bounded_turning(long_path: FishPath) -> None:
    params = MovementParams()
    raw_step = np.diff(long_path.heading_rad)
    step = wrap_angle(raw_step)
    assert np.max(np.abs(step)) <= params.turn_rate_max_rad_s / params.rate_hz + 1e-9
    # Wrap-around does occur (raw jumps of ~2 pi), and nothing else jumps.
    wraps = np.abs(raw_step) > np.pi
    assert wraps.any()
    np.testing.assert_allclose(
        np.abs(raw_step[wraps]), 2 * np.pi, atol=params.turn_rate_max_rad_s / 30
    )
    # Heading is unbounded in direction: all quadrants are visited.
    counts, _ = np.histogram(long_path.heading_rad, bins=4, range=(-np.pi, np.pi))
    assert np.all(counts > 0.1 * long_path.heading_rad.size)


def test_displacement_follows_heading_and_velocity(long_path: FishPath) -> None:
    dt = 1.0 / 30.0
    displacement = np.diff(long_path.xy_m, axis=0)
    expected_step = long_path.velocity_m_s[:-1] * dt
    heading_next = long_path.heading_rad[1:]
    along = displacement[:, 0] * np.cos(heading_next) + displacement[:, 1] * np.sin(
        heading_next
    )
    away_from_walls = (
        (long_path.xy_m[1:, 0] > ARENA.x_min_m + 0.03)
        & (long_path.xy_m[1:, 0] < ARENA.x_max_m - 0.03)
        & (long_path.xy_m[1:, 1] > ARENA.y_min_m + 0.03)
        & (long_path.xy_m[1:, 1] < ARENA.y_max_m - 0.03)
    )
    np.testing.assert_allclose(
        along[away_from_walls], expected_step[away_from_walls], atol=1e-12
    )


def test_upsample_path_matches_samples_and_handles_wrap() -> None:
    times = np.arange(4) / 30.0
    heading = wrap_angle(np.array([3.0, 3.1, 3.2, 3.3]))  # crosses +pi
    path = FishPath(
        times_s=times,
        xy_m=np.column_stack([np.linspace(0, 0.03, 4), np.zeros(4)]),
        heading_rad=heading,
        velocity_m_s=np.full(4, 0.3),
        moving=np.array([True, True, False, False]),
    )
    fine = upsample_path(path, np.linspace(0.0, times[-1], 301))
    np.testing.assert_allclose(fine.heading_rad[::100], heading, atol=1e-12)
    np.testing.assert_allclose(fine.xy_m[::100], path.xy_m, atol=1e-12)
    # Between 3.1 and 3.2 rad (wrapped: 3.1 and -3.08) the heading stays near pi.
    unwrapped = np.unwrap(fine.heading_rad)
    assert np.all(np.diff(unwrapped) > 0)
    assert np.all(np.abs(wrap_angle(fine.heading_rad - np.pi)) < 0.2)
    assert fine.moving[150]
    assert not fine.moving[250]


def test_upsampled_long_path_stays_in_arena(long_path: FishPath) -> None:
    span = long_path.times_s[: 30 * 60]
    fine = upsample_path(
        long_path, np.linspace(span[0], span[-1], 20 * span.size, endpoint=True)
    )
    assert np.all(ARENA.contains(fine.xy_m))
    assert np.max(np.abs(wrap_angle(np.diff(fine.heading_rad)))) < 0.05


def test_upsample_rejects_times_outside_path() -> None:
    path = simulate_path(np.random.default_rng(0), 1.0, ARENA)
    with pytest.raises(ValueError, match="time range"):
        upsample_path(path, np.array([0.0, 5.0]))


def test_start_state_is_respected() -> None:
    path = simulate_path(
        np.random.default_rng(0),
        2.0,
        ARENA,
        start_xy_m=(0.5, 1.0),
        start_heading_rad=math.pi / 2,
    )
    np.testing.assert_allclose(path.xy_m[0], [0.5, 1.0])
    assert path.heading_rad[0] == pytest.approx(math.pi / 2)
    with pytest.raises(ValueError, match="outside"):
        simulate_path(np.random.default_rng(0), 2.0, ARENA, start_xy_m=(5.0, 0.0))


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError, match="both be 0"):
        MovementParams(mean_rest_s=0.0, mean_move_s=0.0)
    with pytest.raises(ValueError, match="max > min"):
        Arena(0.0, 0.0, 0.0, 1.0)
