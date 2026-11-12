"""Physics tests for the dipole field model."""

import numpy as np
import pytest

from gridtools.sim.field import DipoleParams, GainTrack, dipole_gain, gain_track
from gridtools.sim.movement import Arena, simulate_path, upsample_path

SAME_DEPTH = {"fish_depth_m": 0.1, "electrode_depth_m": 0.1}


def _grid(n_cols: int, n_rows: int, spacing_m: float = 0.5) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(n_cols) * spacing_m, np.arange(n_rows) * spacing_m)
    return np.column_stack([xs.ravel(), ys.ravel()])


def test_on_axis_gain_falls_off_as_inverse_square() -> None:
    distances = np.array([2.0, 4.0, 8.0, 16.0])
    electrodes = np.column_stack([distances, np.zeros_like(distances)])
    gain = dipole_gain(np.zeros((1, 2)), np.zeros(1), electrodes, **SAME_DEPTH)[0]
    # gain = (r_ref / r)^2 with r_ref = 0.1 m, far from the 0.1-m regulariser.
    np.testing.assert_allclose(gain * distances**2, 0.1**2, rtol=5e-3)
    np.testing.assert_allclose(gain[:-1] / gain[1:], 4.0, rtol=5e-3)


def test_distance_exponent_sets_falloff() -> None:
    distances = np.array([2.0, 4.0, 8.0])
    electrodes = np.column_stack([distances, np.zeros_like(distances)])
    gain = dipole_gain(
        np.zeros((1, 2)),
        np.zeros(1),
        electrodes,
        **SAME_DEPTH,
        distance_exponent=3.0,
    )[0]
    np.testing.assert_allclose(gain[:-1] / gain[1:], 8.0, rtol=5e-3)


def test_angular_dependence_is_cosine() -> None:
    angles = np.linspace(-np.pi, np.pi, 13)
    electrodes = 3.0 * np.column_stack([np.cos(angles), np.sin(angles)])
    gain = dipole_gain(np.zeros((1, 2)), np.zeros(1), electrodes, **SAME_DEPTH)[0]
    np.testing.assert_allclose(gain / gain.max(), np.cos(angles), atol=1e-6)


def test_sign_flips_across_perpendicular_plane() -> None:
    electrodes = np.array([[0.5, 0.2], [-0.5, 0.2], [0.0, 0.7]])
    gain = dipole_gain(np.zeros((1, 2)), np.zeros(1), electrodes, **SAME_DEPTH)[0]
    assert gain[0] > 0 > gain[1]
    assert gain[0] == pytest.approx(-gain[1], rel=1e-6)
    assert gain[2] == pytest.approx(0.0, abs=1e-9)


def test_reversing_heading_flips_sign() -> None:
    rng = np.random.default_rng(7)
    fish = rng.uniform(0.0, 1.0, (50, 2))
    heading = rng.uniform(-np.pi, np.pi, 50)
    electrodes = _grid(3, 3)
    forward = dipole_gain(fish, heading, electrodes, **SAME_DEPTH)
    backward = dipole_gain(fish, heading + np.pi, electrodes, **SAME_DEPTH)
    np.testing.assert_allclose(backward, -forward, rtol=1e-5, atol=1e-7)


def test_nearby_electrode_dominates() -> None:
    electrodes = _grid(3, 3)  # 3 x 3 at 0.5 m, like the small grids
    centre, corner = 4, 8
    fish = electrodes[centre] - np.array([[0.08, 0.0]])  # 8 cm behind, facing it
    gain = dipole_gain(fish, np.zeros(1), electrodes, **SAME_DEPTH)[0]
    assert np.argmax(np.abs(gain)) == centre
    assert abs(gain[centre]) > 30 * abs(gain[corner])
    # v1 left all 9 gains within 0.75-1.0 of each other; here they span
    # orders of magnitude.
    assert np.abs(gain).max() / np.median(np.abs(gain)) > 10


def test_spatial_contrast_for_random_positions() -> None:
    rng = np.random.default_rng(8)
    electrodes = _grid(3, 5)
    fish = rng.uniform([-0.25, -0.25], [1.25, 2.25], (2000, 2))
    heading = rng.uniform(-np.pi, np.pi, 2000)
    gain = dipole_gain(
        fish, heading, electrodes, fish_depth_m=0.10, electrode_depth_m=0.15
    )
    power = gain.astype(np.float64) ** 2
    # Strongest electrode vs the median electrode, per sample.
    contrast_db = 10 * np.log10(power.max(axis=1) / np.median(power, axis=1))
    assert np.median(contrast_db) > 10.0


def test_gains_are_not_normalised_per_fish() -> None:
    electrodes = _grid(3, 3)
    near = dipole_gain(np.array([[0.4, 0.0]]), np.zeros(1), electrodes, **SAME_DEPTH)
    far = dipole_gain(np.array([[3.0, 0.0]]), np.zeros(1), electrodes, **SAME_DEPTH)
    assert np.abs(near).max() > 20 * np.abs(far).max()
    # Each sample's gain depends only on that sample, not on the rest of the path.
    path = np.array([[0.4, 0.0], [3.0, 0.0], [10.0, 0.0]])
    joint = dipole_gain(path, np.zeros(3), electrodes, **SAME_DEPTH)
    np.testing.assert_array_equal(joint[0], near[0])
    np.testing.assert_array_equal(joint[1], far[0])


def test_regularisation_bounds_gain_near_fish() -> None:
    offsets = np.linspace(-0.3, 0.3, 601)
    electrodes = np.column_stack([offsets, np.zeros_like(offsets)])
    gain = dipole_gain(np.zeros((1, 2)), np.zeros(1), electrodes, **SAME_DEPTH)[0]
    # On-axis maximum of r / (r^2 + a^2)^1.5 is at r = a / sqrt(2).
    a, r_ref = 0.1, 0.1
    bound = r_ref**2 / a**2 / np.sqrt(2) / 1.5**1.5
    assert np.all(np.isfinite(gain))
    assert np.abs(gain).max() == pytest.approx(bound, rel=1e-3)
    assert offsets[np.argmax(gain)] == pytest.approx(a / np.sqrt(2), abs=1e-3)


def test_vertical_offset_reduces_gain_and_directly_below_is_zero() -> None:
    electrodes = np.array([[1.0, 0.0], [0.0, 0.0]])
    level = dipole_gain(np.zeros((1, 2)), np.zeros(1), electrodes, **SAME_DEPTH)[0]
    deeper = dipole_gain(
        np.zeros((1, 2)),
        np.zeros(1),
        electrodes,
        fish_depth_m=0.1,
        electrode_depth_m=0.6,
    )[0]
    expected = 0.1**2 * 1.0 / (1.0 + 0.5**2 + 0.1**2) ** 1.5
    assert deeper[0] == pytest.approx(expected, rel=1e-6)
    assert deeper[0] < level[0]
    assert deeper[1] == pytest.approx(0.0, abs=1e-12)


def test_gain_track_interpolates_smoothly() -> None:
    arena = Arena(-0.25, 1.25, -0.25, 2.25)
    path = simulate_path(np.random.default_rng(9), 60.0, arena)
    electrodes = _grid(3, 5)
    params = DipoleParams()
    track = gain_track(path.times_s, path.xy_m, path.heading_rad, electrodes, params)
    assert track.gains.dtype == np.float32
    np.testing.assert_allclose(
        track.at(path.times_s), track.gains, rtol=1e-6, atol=1e-7
    )
    fine_times = np.linspace(0.0, 60.0, 60 * 2000)
    fine_path = upsample_path(path, fine_times)
    exact = dipole_gain(
        fine_path.xy_m,
        fine_path.heading_rad,
        electrodes,
        fish_depth_m=params.fish_depth_m,
        electrode_depth_m=params.electrode_depth_m,
    )
    interpolated = track.at(fine_times)
    scale = np.abs(exact).max(axis=0)
    error = np.abs(interpolated - exact) / scale
    assert error.max() < 0.02
    assert np.median(error) < 1e-3
    with pytest.raises(ValueError, match="outside"):
        track.at(np.array([path.times_s[-1] + 1.0]))


def test_invalid_shapes_raise() -> None:
    with pytest.raises(ValueError, match="heading_rad"):
        dipole_gain(np.zeros((2, 2)), np.zeros(3), _grid(3, 3), **SAME_DEPTH)
    with pytest.raises(ValueError, match="electrode_depth_m"):
        dipole_gain(
            np.zeros((1, 2)),
            np.zeros(1),
            _grid(3, 3),
            fish_depth_m=0.1,
            electrode_depth_m=np.zeros(4),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        GainTrack(times_s=np.zeros(5), gains=np.zeros((5, 3), dtype=np.float32))
