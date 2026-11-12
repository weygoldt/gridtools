"""Order-free profile statistics and strongest-electrode dwell times."""

import numpy as np
import pytest

from gridtools.calibrate.profiles import (
    dwell_stats,
    profile_quantiles,
    relative_sorted_profiles,
    strongest_electrode,
)


def test_sorted_profiles_are_invariant_to_channel_order() -> None:
    rng = np.random.default_rng(0)
    powers = rng.normal(-90.0, 10.0, (50, 9))
    profiles = relative_sorted_profiles(powers)
    shuffled = relative_sorted_profiles(powers[:, rng.permutation(9)])
    np.testing.assert_array_equal(profiles, shuffled)
    assert np.all(profiles[:, 0] == 0.0)
    assert np.all(np.diff(profiles, axis=1) <= 0.0)
    q = profile_quantiles(profiles, [0.5])
    assert q.shape == (1, 9)


def test_strongest_electrode_hysteresis_suppresses_flicker() -> None:
    powers = np.array(
        [[0.0, -1.0], [-0.5, 0.0], [0.0, -0.5], [-5.0, 0.0], [-4.0, 0.0], [-1.0, 0.0]]
    )
    assert strongest_electrode(powers).tolist() == [0, 1, 0, 1, 1, 1]
    assert strongest_electrode(powers, 3.0).tolist() == [0, 0, 0, 1, 1, 1]


def test_dwell_stats_counts_switches_and_completed_dwells() -> None:
    times = np.arange(0.0, 20.0, 1.0)
    seq = np.array([0] * 5 + [1] * 5 + [0] * 3 + [2] * 7)
    stats = dwell_stats(times, seq, max_gap_s=2.0, still_min_s=5.0)
    assert stats.n_switches == 3
    assert stats.duration_s == pytest.approx(19.0)
    assert stats.switches_per_min == pytest.approx(3 * 60 / 19)
    # Only the runs entered and left by a switch are complete.
    np.testing.assert_allclose(stats.dwell_times_s, [5.0, 3.0])
    # Runs of >= 5 s: the 0-run lasts 4 s (frames 0..4), the 1-run 4 s, the
    # 2-run 6 s.
    assert stats.still_fraction == pytest.approx(6.0 / 19.0)


def test_dwell_stats_does_not_switch_across_gaps() -> None:
    times = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
    seq = np.array([0, 0, 0, 1, 1])
    stats = dwell_stats(times, seq, max_gap_s=2.0)
    assert stats.n_switches == 0
    assert stats.dwell_times_s.size == 0
    assert stats.duration_s == pytest.approx(3.0)
