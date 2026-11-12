"""Chirp timing statistics on synthetic chirp trains."""

import numpy as np
import pytest

from gridtools.calibrate.chirps import (
    coincidence_fraction,
    cross_fish_stats,
    merge_duplicates,
    next_event_latencies,
)


def test_merge_duplicates_keeps_first_of_close_events() -> None:
    times, n = merge_duplicates([1.0, 1.02, 1.04, 1.2, 5.0, 5.01], 0.05)
    np.testing.assert_allclose(times, [1.0, 1.2, 5.0])
    assert n == 3


def test_next_event_latencies() -> None:
    lat = next_event_latencies([1.0, 2.0, 10.0], [1.5, 2.0, 3.0])
    np.testing.assert_allclose(lat[:2], [0.5, 0.0])
    assert np.isnan(lat[2])


def test_coincidence_fraction_is_symmetric_window() -> None:
    a = np.array([1.0, 5.0, 9.0])
    b = np.array([0.8, 5.3, 20.0])
    assert coincidence_fraction(a, b, 0.25) == pytest.approx(1 / 3)
    assert coincidence_fraction(a, b, 0.35) == pytest.approx(2 / 3)
    assert np.isnan(coincidence_fraction([], b, 0.25))


def test_cross_fish_stats_detects_responses_against_shuffled_null() -> None:
    rng = np.random.default_rng(0)
    duration = 3600.0
    a = np.sort(rng.uniform(0, duration, 300))
    responders = a[rng.random(a.size) < 0.5] + 0.35
    b = np.sort(np.concatenate([responders, rng.uniform(0, duration, 100)]))
    stats = cross_fish_stats(a, b, 0.0, duration, rng, n_shuffles=50)
    peak = int(np.argmax(stats.latency_counts - stats.latency_null_counts))
    assert stats.latency_edges_s[peak] == pytest.approx(0.3)
    assert stats.coincidence_null < 0.1
    # Responses at 0.35 s fall outside +/-0.25 s, so coincidences stay at chance.
    assert stats.coincidence < stats.coincidence_null + 4 * stats.coincidence_null_sd
    wide = cross_fish_stats(a, b, 0.0, duration, rng, window_s=0.4, n_shuffles=50)
    assert wide.coincidence > 0.4
