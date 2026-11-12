"""Harvest a few long chirps from a real competition recording."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gridtools.templates.harvest import (
    HarvestConfig,
    harvest_recording,
    read_individuals,
)

ROOT = Path("/mnt/data2/2019_tube_competition")
RECORDING = ROOT / "2020-06-02-10_00"


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.skipif(not RECORDING.exists(), reason="recording not available")
def test_harvest_2020_06_02() -> None:
    before = (RECORDING / "traces-grid1.raw").stat()
    harvest = harvest_recording(
        RECORDING,
        HarvestConfig(),
        individuals=read_individuals(ROOT / "order_meta.csv"),
        max_candidates=8,
    )
    after = (RECORDING / "traces-grid1.raw").stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    assert harvest.summary["n_candidates"] == 8
    assert harvest.summary["sign_v_matches_channels"]
    harvested = [r for r in harvest.results if r.record["status"] == "harvested"]
    assert len(harvested) >= 6
    for result in harvested:
        record, extraction = result.record, result.extraction
        assert extraction is not None
        # one emitter in this trial: track 7, fish 1 of group 6, ~820 Hz
        assert record["emitter_track_id"] == 7
        assert record["individual"] == "g6f1"
        assert 800.0 < record["emitter_hz"] < 840.0
        snippet = extraction.snippet
        assert snippet.analysis.valid
        assert 250.0 < snippet.metrics.excursion_hz < 600.0
        assert snippet.metrics.duration_10_s > 0.1
        d, a = snippet.delta_f_hz, snippet.amp_rel
        assert d[0] == d[-1] == 0.0
        assert a[0] == a[-1] == 1.0
        assert np.all(np.isfinite(d))
    longest = max(r.extraction.snippet.metrics.duration_10_s for r in harvested)  # type: ignore[union-attr]
    assert longest > 0.3  # the 782.5 s chirp lasts about 390 ms
