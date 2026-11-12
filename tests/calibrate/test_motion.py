"""Movement proxies of simulated fish."""

import dataclasses

import numpy as np

from gridtools.calibrate.field import GridLayout
from gridtools.calibrate.motion import layout_arena, simulated_dwell_stats
from gridtools.sim.movement import MovementParams


def test_layout_arena_adds_margin() -> None:
    arena = layout_arena(GridLayout("15", 3, 5, 0.5), 0.5)
    assert (arena.x_min_m, arena.x_max_m) == (-0.25, 2.25)
    assert (arena.y_min_m, arena.y_max_m) == (-0.25, 1.25)


def test_resting_fish_switch_less_than_moving_fish() -> None:
    layout = GridLayout("15", 3, 5, 0.5)
    base = MovementParams()
    kwargs = {
        "distance_exponent": 2.0,
        "regulariser_rel": 0.1,
        "margin_rel": 0.25,
        "duration_s": 600.0,
        "frame_dt_s": 0.33,
        "median_peak_snr_db": 50.0,
    }
    resting = simulated_dwell_stats(
        np.random.default_rng(0),
        layout,
        dataclasses.replace(base, mean_move_s=0.0),
        **kwargs,
    )
    moving = simulated_dwell_stats(
        np.random.default_rng(0),
        layout,
        dataclasses.replace(base, mean_rest_s=0.0, speed_mean_m_s=0.2),
        **kwargs,
    )
    # Resting fish still rotate slowly in place (rest_turn_factor), which
    # changes the dipole pattern and causes occasional switches.
    assert resting.switches_per_min < moving.switches_per_min / 3
    assert resting.still_fraction > 0.5
    assert moving.still_fraction < 0.2
