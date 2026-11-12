"""Simulation-based falloff fit: profiles and parameter recovery."""

import numpy as np
import pytest

from gridtools.calibrate.field import (
    GridLayout,
    draw_simulation,
    fit_field_falloff,
    simulate_powers_db,
    simulate_profiles,
)
from gridtools.calibrate.profiles import relative_sorted_profiles
from gridtools.sim.field import dipole_gain


def test_layout_positions_and_missing_electrodes() -> None:
    layout = GridLayout("9-8", 3, 3, 0.5, missing=(8,))
    xy = layout.positions_m()
    assert xy.shape == (8, 2)
    assert not np.any(np.all(xy == [1.0, 1.0], axis=1))
    assert layout.n_electrodes == 8
    random = GridLayout("15-14", 3, 5, 0.5, n_random_missing=1)
    assert (random.n_simulated, random.n_electrodes) == (15, 14)


def test_noise_free_profiles_follow_dipole_gains() -> None:
    layout = GridLayout("3x3", 3, 3, 0.5)
    rng = np.random.default_rng(0)
    draws = draw_simulation(rng, 200, 9, [200.0])  # SNR so high that noise vanishes
    powers = simulate_powers_db(
        layout, draws, distance_exponent=2.0, regulariser_rel=0.1, margin_rel=0.5
    )
    xy = -0.25 + draws.unit_xy * 1.5
    gain = dipole_gain(
        xy,
        draws.heading_rad,
        layout.positions_m(),
        fish_depth_m=0.0,
        electrode_depth_m=0.0,
        min_distance_m=0.05,
    ).astype(np.float64)
    expected = relative_sorted_profiles(20 * np.log10(np.abs(gain)))
    np.testing.assert_allclose(relative_sorted_profiles(powers), expected, atol=1e-3)


def test_noise_floor_limits_the_weakest_electrodes() -> None:
    layout = GridLayout("15", 3, 5, 0.5)
    draws = draw_simulation(np.random.default_rng(1), 5000, 15, [20.0])
    profiles = simulate_profiles(
        layout, draws, distance_exponent=3.0, regulariser_rel=0.1, margin_rel=0.5
    )
    # With 20 dB SNR the weakest electrodes sit near the noise, about 20 dB
    # (plus the spread of an exponentially distributed periodogram bin) down.
    assert -35.0 < np.median(profiles[:, -1]) < -18.0


def test_random_missing_electrodes_shorten_profiles() -> None:
    layout = GridLayout("15-13", 3, 5, 0.5, n_random_missing=2)
    draws = draw_simulation(np.random.default_rng(2), 100, 15, [40.0])
    powers = simulate_powers_db(
        layout, draws, distance_exponent=2.0, regulariser_rel=0.1, margin_rel=0.0
    )
    assert np.all(np.isnan(powers).sum(axis=1) == 2)
    profiles = simulate_profiles(
        layout, draws, distance_exponent=2.0, regulariser_rel=0.1, margin_rel=0.0
    )
    assert profiles.shape == (100, 13)
    assert np.all(np.isfinite(profiles))


def test_fit_recovers_known_falloff() -> None:
    """Profiles simulated with known parameters are fitted back."""
    truth = {"distance_exponent": 2.5, "regulariser_rel": 0.2, "margin_rel": 0.5}
    layouts = {
        "9-8": GridLayout("9-8", 3, 3, 0.5, missing=(8,)),
        "15": GridLayout("15", 3, 5, 0.5),
    }
    snr = np.random.default_rng(3).uniform(35.0, 60.0, 1000)
    data_rng = np.random.default_rng(4)
    observed = {}
    for name, layout in layouts.items():
        trials = []
        for _ in range(4):
            draws = draw_simulation(data_rng, 3000, layout.n_simulated, snr, 2.0)
            trials.append(simulate_profiles(layout, draws, **truth))
        observed[name] = trials
    fit_rng = np.random.default_rng(5)
    draws = {
        name: draw_simulation(fit_rng, 6000, layout.n_simulated, snr, 2.0)
        for name, layout in layouts.items()
    }
    fit = fit_field_falloff(
        observed,
        layouts,
        draws,
        exponents=[1.5, 2.0, 2.5, 3.0, 3.5],
        regularisers_rel=[0.05, 0.2, 0.5],
        margins_rel=[0.0, 0.5, 1.0],
        n_bootstrap=20,
        rng=fit_rng,
    )
    assert fit.best["distance_exponent"] == pytest.approx(2.5)
    assert fit.best["regulariser_rel"] == pytest.approx(0.2)
    assert fit.loss.shape == (5, 3, 3)
    lo, hi = fit.interval("distance_exponent")
    assert lo <= 2.5 <= hi
    assert set(fit.best_quantiles) == {"9-8", "15"}
