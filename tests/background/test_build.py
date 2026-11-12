"""Library build on synthetic fishgrid trials."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridtools.background.build import BackgroundConfig, build_background_library
from gridtools.background.levels import REFERENCE_BANDWIDTH_HZ
from gridtools.background.library import read_manifest
from gridtools.background.select import SelectionConfig
from gridtools.background.sources import open_source_recording
from gridtools.io import load_tracks, open_recording

from .conftest import FISH_AMPLITUDE, FS, NOISE_SD, Trial

CONFIG = BackgroundConfig(
    selection=SelectionConfig(
        segment_s=10.0, step_s=2.5, margin_s=1.0, edge_s=5.0, segments_per_recording=4
    ),
    fish_free_per_recording=1,
    workers=1,
)


def build(
    trials: list[Trial], out: Path, config: BackgroundConfig = CONFIG
) -> pd.DataFrame:
    return build_background_library([t.source for t in trials], out, config)


def test_segments_avoid_detections_by_the_margin(
    make_trial: Callable[..., Trial], tmp_path: Path
) -> None:
    trial = make_trial(detection_times_s=(50.0, 51.0, 80.0, 112.0))
    manifest = build([trial], tmp_path / "lib")
    fish = pd.DataFrame(manifest.loc[(manifest["segment_class"] == "fish").to_numpy()])
    assert len(fish) >= 3
    assert bool((fish["n_v1_detections_margin"] == 0).all())
    margin = CONFIG.selection.margin_s
    for start, stop in zip(fish["start_s"], fish["stop_s"], strict=True):
        for d in trial.detection_times_s:
            assert d + 0.05 <= start - margin or d - 0.05 >= stop + margin
    assert bool(fish["screened"].all())
    assert bool((fish["weight"] == 1.0).all())


def test_samples_geometry_and_tracks_are_preserved(
    make_trial: Callable[..., Trial], tmp_path: Path
) -> None:
    trial = make_trial()
    root = tmp_path / "lib"
    manifest = build([trial], root)
    source = open_source_recording(trial.source)
    for row in manifest.to_dict("records"):
        rec = open_recording(root / str(row["path"]))
        assert rec.n_channels == source.n_channels == row["n_electrodes"]
        assert rec.samplerate_hz == FS
        assert rec.geometry is not None
        assert source.geometry is not None
        np.testing.assert_array_equal(
            rec.geometry.positions_m, source.geometry.positions_m
        )
        np.testing.assert_allclose(
            row["electrode_positions_m"], source.geometry.positions_m
        )
        start = round(row["start_s"] * FS)
        np.testing.assert_array_equal(
            rec.read(0, rec.n_samples), trial.samples[start : start + rec.n_samples]
        )
        if row["segment_class"] == "fish":
            tracks = load_tracks(root / str(row["path"]))
            full = load_tracks(trial.source.recording_path)
            inside = (full.times_s >= row["start_s"]) & (full.times_s < row["stop_s"])
            np.testing.assert_allclose(
                tracks.times_s, full.times_s[inside] - row["start_s"]
            )
            assert sorted(tracks.ids.tolist()) == [3, 7] == sorted(row["fish_ids"])
            assert tracks.powers_db is not None
            assert tracks.powers_db.shape[1] == rec.n_channels  # padded 5 -> 6
            assert np.all(np.isnan(tracks.powers_db[:, -1]))


def test_levels_and_real_fish_snr(
    make_trial: Callable[..., Trial], tmp_path: Path
) -> None:
    trial = make_trial()
    manifest = build([trial], tmp_path / "lib")
    noise_psd = NOISE_SD**2 / (FS / 2)
    np.testing.assert_allclose(manifest["noise_psd"], noise_psd, rtol=0.05)
    np.testing.assert_allclose(
        manifest["level"], noise_psd * REFERENCE_BANDWIDTH_HZ, rtol=0.05
    )
    expected_snr = 10 * math.log10(
        (FISH_AMPLITUDE**2 / 2) / (noise_psd * REFERENCE_BANDWIDTH_HZ)
    )
    fish = pd.DataFrame(manifest.loc[(manifest["segment_class"] == "fish").to_numpy()])
    for snrs in fish["fish_snr_db"]:
        np.testing.assert_allclose(snrs, expected_snr, atol=0.5)
    assert bool(fish["tracks_complete"].all())
    assert all(d == [] for d in manifest["dead_channels"])
    assert bool((manifest["units"].str.contains("ADC")).all())


def test_fish_free_segments_come_from_fish_free_stretches(
    make_trial: Callable[..., Trial], tmp_path: Path
) -> None:
    trial = make_trial(fish_from_s=30.0)
    root = tmp_path / "lib"
    manifest = build([trial], root)
    free = pd.DataFrame(
        manifest.loc[(manifest["segment_class"] == "fish_free").to_numpy()]
    )
    assert len(free) == 1
    row = free.to_dict("records")[0]
    assert row["stop_s"] <= trial.fish_from_s
    assert row["n_fish"] == 0
    assert row["screened"]
    assert not (root / str(row["path"]) / "ident_v.npy").exists()


def test_build_is_deterministic_and_independent_of_workers(
    make_trial: Callable[..., Trial], tmp_path: Path
) -> None:
    trials = [make_trial("trial-a", seed=1), make_trial("trial-b", seed=2)]
    a = build(trials, tmp_path / "a")
    b = build(trials, tmp_path / "b")
    parallel = BackgroundConfig(**{**CONFIG.__dict__, "workers": 2})
    c = build(trials, tmp_path / "c", parallel)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)
    # another seed picks other segments
    other = BackgroundConfig(**{**CONFIG.__dict__, "seed": 99})
    d = build(trials, tmp_path / "d", other)
    assert set(d["segment_id"]) != set(a["segment_id"])


def test_manifest_round_trip(make_trial: Callable[..., Trial], tmp_path: Path) -> None:
    root = tmp_path / "lib"
    manifest = build([make_trial()], root)
    from_csv = read_manifest(root / "manifest.csv")
    from_parquet = read_manifest(root / "manifest.parquet")
    assert list(from_parquet.columns) == list(manifest.columns)
    for column in ("segment_id", "start_s", "level", "fish_ids", "fish_snr_db"):
        left, right = from_csv[column].tolist(), from_parquet[column].tolist()
        if column in ("start_s", "level"):
            np.testing.assert_allclose(left, right, rtol=1e-9)
        elif column == "fish_snr_db":
            for x, y in zip(left, right, strict=True):
                np.testing.assert_allclose(x, y, rtol=1e-9)
        else:
            assert left == right
    assert (root / "library.json").exists()
    assert (root / "recordings.csv").exists()


def test_labelled_recordings_are_skipped_or_their_spans_avoided(
    make_trial: Callable[..., Trial], tmp_path: Path
) -> None:
    trial = make_trial(labels=True)
    assert build([trial], tmp_path / "skip").empty
    keep = BackgroundConfig(**{**CONFIG.__dict__, "exclude_labelled": False})
    manifest = build([trial], tmp_path / "keep", keep)
    assert not manifest.empty
    margin = CONFIG.selection.margin_s
    for start, stop in zip(manifest["start_s"], manifest["stop_s"], strict=True):
        assert stop + margin <= 100.0 or start - margin >= 110.0


def test_partial_trailing_frame_is_dropped(
    make_trial: Callable[..., Trial], tmp_path: Path
) -> None:
    trial = make_trial(partial_frame=True, duration_s=60.0)
    rec = open_source_recording(trial.source)
    assert rec.n_samples == trial.samples.shape[0]
    with pytest.raises(ValueError, match="whole number"):
        open_recording(trial.source.recording_path)


def test_refine_untracked_drops_recurring_lines() -> None:
    from gridtools.background.build import refine_untracked

    rows = []
    for k in range(20):
        lines = [585.9, 1955.6] if k % 3 else [585.9]
        if k == 0:
            lines = [*lines, 733.0]  # a line only once: maybe a fish
        rows.append(
            {
                "dataset": "d",
                "segment_class": "fish",
                "untracked_peaks_hz": lines,
                "fish_coverage": [1.0, 0.95],
                "tracks_complete": False,
            }
        )
    out = refine_untracked(pd.DataFrame(rows))
    assert out.loc[0, "untracked_peaks_hz"] == [733.0]
    assert not out.loc[0, "tracks_complete"]
    assert all(out.loc[1:, "untracked_peaks_hz"].map(len) == 0)
    assert bool(out.loc[1:, "tracks_complete"].all())
