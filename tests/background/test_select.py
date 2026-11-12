"""Candidate screening and stratified selection."""

from __future__ import annotations

import numpy as np
import pytest

from gridtools.background.select import (
    SelectionConfig,
    candidate_starts,
    choose_stratified,
    count_overlaps,
    expected_missed_chirps,
    frame_coverage,
    local_rate_per_s,
)


def test_candidates_stay_inside_the_edges() -> None:
    starts = candidate_starts(1000.0, 60.0, 15.0, 120.0)
    assert starts[0] == 120.0
    assert starts[-1] + 60.0 <= 1000.0 - 120.0
    assert np.allclose(np.diff(starts), 15.0)
    assert candidate_starts(200.0, 60.0, 15.0, 120.0).size == 0


def test_count_overlaps_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1000, 300)
    b = a + rng.uniform(0, 0.5, a.size)
    starts = np.arange(0.0, 950.0, 7.5)
    margin, length = 2.0, 60.0
    fast = count_overlaps(starts, length, a, b, margin)
    slow = [int(np.sum((a < s + length + margin) & (b > s - margin))) for s in starts]
    np.testing.assert_array_equal(fast, slow)
    with pytest.raises(ValueError, match="differ in length"):
        count_overlaps(starts, length, a, b[:-1])


def test_margin_rejects_detections_just_outside() -> None:
    starts = np.array([100.0])
    # detection ends 1.5 s before the segment: inside a 2-s margin
    assert count_overlaps(starts, 60.0, [98.0], [98.5], 2.0)[0] == 1
    assert count_overlaps(starts, 60.0, [98.0], [98.5], 1.0)[0] == 0
    # detection starts 1.5 s after the end
    assert count_overlaps(starts, 60.0, [161.5], [161.6], 2.0)[0] == 1


def test_local_rate_and_expected_missed_chirps() -> None:
    events = np.array([10.0, 20.0, 30.0, 500.0])
    rate = local_rate_per_s([20.0, 400.0], events, 60.0)
    np.testing.assert_allclose(rate, [3 / 60, 0.0])
    missed = expected_missed_chirps(rate, 60.0, 0.75)
    np.testing.assert_allclose(missed, [3 * 0.25 / 0.75, 0.0])


def test_frame_coverage() -> None:
    times = np.arange(0.0, 100.0, 1.0)
    detected = times >= 50
    cov = frame_coverage([0.0, 40.0, 60.0, 200.0], 20.0, times, detected)
    np.testing.assert_allclose(cov[:3], [0.0, 0.5, 1.0])
    assert np.isnan(cov[3])


def test_choose_stratified_one_per_stratum_lowest_score() -> None:
    starts = np.arange(0.0, 1000.0, 10.0)
    score = np.abs(np.sin(starts))
    valid = np.ones(starts.size, dtype=bool)
    valid[starts < 200] = False
    rng = np.random.default_rng(1)
    chosen = choose_stratified(
        rng, starts, valid, score, segment_s=60.0, n_strata=5, span_s=(0.0, 1000.0)
    )
    assert chosen.size == 4  # first stratum (0-200 s) has no valid candidate
    assert np.all(valid[chosen])
    assert np.all(np.diff(starts[chosen]) >= 60.0)
    for i in chosen:
        k = int(starts[i] // 200)
        same = valid & (starts // 200 == k)
        assert score[i] == score[same].min()


def test_choose_stratified_is_seeded_and_avoids_overlap() -> None:
    starts = np.arange(0.0, 300.0, 5.0)
    score = np.zeros(starts.size)
    valid = np.ones(starts.size, dtype=bool)
    kwargs = {"segment_s": 60.0, "n_strata": 10, "span_s": (0.0, 300.0)}
    a = choose_stratified(np.random.default_rng(7), starts, valid, score, **kwargs)
    b = choose_stratified(np.random.default_rng(7), starts, valid, score, **kwargs)
    np.testing.assert_array_equal(a, b)
    assert np.all(np.diff(starts[a]) >= 60.0)


def test_selection_config_validates() -> None:
    with pytest.raises(ValueError, match="segment_s"):
        SelectionConfig(segment_s=0.0)
    with pytest.raises(ValueError, match="v1_recall"):
        SelectionConfig(v1_recall=0.0)
    with pytest.raises(ValueError, match="segments_per_recording"):
        SelectionConfig(segments_per_recording=0)
