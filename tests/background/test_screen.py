"""Chirp screens: time windows, screen tables, candidate rules and the gate."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridtools.background.screen import (
    RecordingGate,
    Screen,
    TimeWindow,
    covered,
    gate_rate_per_h,
    max_score_near,
    read_screen,
    rejects,
)
from gridtools.background.select import (
    candidate_starts,
    fill_non_overlapping,
    span_starts,
)


def write_screen(
    root: Path,
    dataset: str,
    recording: str,
    boxes: Sequence[tuple[float, float]],
    *,
    covered_s: Sequence[tuple[float, float]] = ((0.0, 1e6),),
    store_threshold: float = 0.05,
    width_s: float = 0.1,
) -> Path:
    """A screen directory with boxes ``(centre time, score)`` for one recording."""
    (root / dataset).mkdir(parents=True, exist_ok=True)
    meta = {
        "name": "test",
        "detector": "synthetic",
        "weights_sha256": "0" * 64,
        "store_threshold": store_threshold,
    }
    (root / "screen.json").write_text(json.dumps(meta))
    pd.DataFrame(
        {
            "t_start_s": [t - width_s / 2 for t, _ in boxes],
            "t_end_s": [t + width_s / 2 for t, _ in boxes],
            "f_low_hz": 700.0,
            "f_high_hz": 900.0,
            "score": [s for _, s in boxes],
        }
    ).to_csv(root / dataset / f"{recording}.csv", index=False)
    (root / dataset / f"{recording}.json").write_text(
        json.dumps({"covered_s": [list(c) for c in covered_s]})
    )
    return root


def test_window_is_measured_from_the_end() -> None:
    w = TimeWindow(last_s=9000.0)
    assert w.span(21600.0) == (12600.0, 21600.0)
    assert w.span(5000.0) == (0.0, 5000.0)  # clipped at the start
    assert TimeWindow().span(100.0) == (0.0, 100.0)


def test_window_never_starts_before_min_start() -> None:
    w = TimeWindow(last_s=9000.0, min_start_s=12600.0, min_duration_s=14400.0)
    assert w.span(21600.0) == (12600.0, 21600.0)
    # a recording stopped early: the window shrinks instead of reaching back
    assert w.span(18000.0) == (12600.0, 18000.0)
    assert w.span(14000.0) is None  # shorter than min_duration_s
    assert TimeWindow(last_s=100.0, min_start_s=500.0).span(400.0) is None
    with pytest.raises(ValueError, match="last_s"):
        TimeWindow(last_s=0.0)
    assert "last 9000 s" in w.describe()


def test_screen_and_gate_validate() -> None:
    with pytest.raises(ValueError, match="identifier"):
        Screen("v-1", Path(), 0.1)
    with pytest.raises(ValueError, match="threshold"):
        Screen("v1", Path(), 1.5)
    with pytest.raises(ValueError, match="max_rate_per_h"):
        RecordingGate(max_rate_per_h=-1.0)


def test_read_screen_round_trip(tmp_path: Path) -> None:
    root = write_screen(tmp_path, "d", "r", [(20.0, 0.3), (10.0, 0.9)])
    table = read_screen(root, "d", "r")
    assert table is not None
    assert table.boxes["t_start_s"].is_monotonic_increasing
    assert table.meta["store_threshold"] == 0.05
    assert len(table.above(0.5)) == 1
    with pytest.raises(ValueError, match="store threshold"):
        table.above(0.01)
    assert read_screen(root, "d", "missing") is None


def test_covered_needs_one_span_with_margin() -> None:
    spans = [(0.0, 100.0), (100.0, 200.0), (300.0, 400.0)]  # first two abut
    ok = covered([0.0, 1.0, 95.0, 250.0, 300.0, 301.0], 10.0, spans, margin_s=1.0)
    np.testing.assert_array_equal(ok, [False, True, True, False, False, True])
    assert not covered([0.0], 1.0, np.empty((0, 2))).any()


def test_screen_rejects_by_threshold_margin_and_coverage(tmp_path: Path) -> None:
    root = write_screen(
        tmp_path,
        "d",
        "r",
        [(50.0, 0.3), (80.0, 0.08)],
        covered_s=[(0.0, 150.0)],
    )
    table = read_screen(root, "d", "r")
    assert table is not None
    starts = np.array([30.0, 38.0, 39.0, 52.0, 60.0, 70.0, 141.0])
    bad, n = rejects(starts, 10.0, table, threshold=0.1, margin_s=1.0)
    # 38 + 10 + 1 = 49 < 49.95: clear; 39 reaches 50; 52 - 1 = 51 > 50.05: clear
    # 70 overlaps the 0.08 box, which is below the threshold; 141 + 11 > 150
    np.testing.assert_array_equal(bad, [False, False, True, False, False, False, True])
    np.testing.assert_array_equal(n, [0, 0, 1, 0, 0, 0, 0])
    bad_low, _ = rejects(starts, 10.0, table, threshold=0.05, margin_s=1.0)
    assert bad_low[5]
    np.testing.assert_allclose(
        max_score_near(starts, 10.0, table.boxes, 1.0),
        [0.0, 0.0, 0.3, 0.0, 0.0, 0.08, 0.0],
    )


def test_gate_rate_counts_per_covered_hour(tmp_path: Path) -> None:
    boxes = [(t, 0.9) for t in np.arange(100.0, 1800.0, 100.0)]  # 17 boxes
    boxes += [(150.0, 0.2), (2500.0, 0.9)]  # the last one outside the coverage
    root = write_screen(tmp_path, "d", "r", boxes, covered_s=[(0.0, 1800.0)])
    table = read_screen(root, "d", "r")
    assert table is not None
    # 17 boxes of >= 0.5 per 0.5 h covered
    assert gate_rate_per_h(table, 0.5, 0.0, 3600.0) == pytest.approx(17 / 0.5)
    assert gate_rate_per_h(table, 0.1, 0.0, 900.0) == pytest.approx(10 / 0.25)
    assert np.isnan(gate_rate_per_h(table, 0.5, 2000.0, 3000.0))


def test_span_starts_generalise_candidate_starts() -> None:
    np.testing.assert_array_equal(
        span_starts(120.0, 1000.0 - 120.0, 60.0, 15.0),
        candidate_starts(1000.0, 60.0, 15.0, 120.0),
    )
    starts = span_starts(12.5, 100.0, 10.0, 5.0)
    assert starts[0] == 12.5
    assert starts[-1] + 10.0 <= 100.0
    assert span_starts(0.0, 5.0, 10.0, 1.0).size == 0


def test_fill_adds_non_overlapping_candidates_up_to_n() -> None:
    rng = np.random.default_rng(0)
    starts = np.arange(0.0, 300.0, 5.0)
    valid = np.ones(starts.size, dtype=bool)
    valid[10:20] = False
    score = rng.random(starts.size)
    chosen = np.array([3])
    out = fill_non_overlapping(
        rng, starts, valid, score, chosen, segment_s=60.0, n_max=10
    )
    assert 3 in out
    picked = starts[out]
    assert np.all(np.diff(np.sort(picked)) >= 60.0)
    assert valid[out].all()
    assert out.size <= 10
    # nothing more fits than floor(300 / 60) with the invalid stretch
    assert out.size >= 3


def test_pack_fits_the_most_segments_in_a_narrow_window() -> None:
    from gridtools.background.select import pack_non_overlapping

    starts = span_starts(0.0, 1740.0, 60.0, 5.0)
    valid = np.ones(starts.size, dtype=bool)
    out = pack_non_overlapping(starts, valid, segment_s=60.0, n_max=30)
    assert out.size == 29
    np.testing.assert_allclose(np.diff(starts[out]), 60.0)
    valid[(starts > 100.0) & (starts < 130.0)] = False  # a screened box
    out = pack_non_overlapping(starts, valid, segment_s=60.0, n_max=30)
    assert valid[out].all()
    assert np.all(np.diff(starts[out]) >= 60.0)
    assert out.size == 28
    few = pack_non_overlapping(
        starts, np.ones(starts.size, bool), segment_s=60.0, n_max=5
    )
    assert few.size == 5
    assert starts[few[0]] == 0.0
