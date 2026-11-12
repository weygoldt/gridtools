"""Library build with a time window, chirp screens and the recording gate."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridtools.background.build import (
    BackgroundConfig,
    build_background_library,
    carry_over,
)
from gridtools.background.library import read_manifest
from gridtools.background.screen import RecordingGate, Screen, TimeWindow
from gridtools.background.select import SelectionConfig

from .conftest import Trial
from .test_screen import write_screen

SELECTION = SelectionConfig(
    segment_s=10.0, step_s=2.5, margin_s=1.0, edge_s=5.0, segments_per_recording=4
)


def config(**kwargs: object) -> BackgroundConfig:
    base: dict[str, object] = {
        "selection": SELECTION,
        "fish_free_per_recording": 0,
        "workers": 1,
        "use_saved_detections": False,
    }
    base.update(kwargs)
    return BackgroundConfig(**base)  # type: ignore[arg-type]


def fish(manifest: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(manifest.loc[(manifest["segment_class"] == "fish").to_numpy()])


def test_window_and_screen_restrict_segments(
    make_trial: Callable[..., Trial], tmp_path: Path
) -> None:
    trial = make_trial(duration_s=150.0, fish_from_s=10.0)
    boxes = [(100.0, 0.3), (120.0, 0.07)]
    root = write_screen(tmp_path / "screen", "synthetic", trial.source.name, boxes)
    cfg = config(
        windows={"synthetic": TimeWindow(last_s=70.0)},
        screens=(Screen("v1", root, threshold=0.1, margin_s=1.0),),
        selection=dataclasses.replace(SELECTION, segments_per_recording=6, fill=True),
    )
    manifest = build_background_library([trial.source], tmp_path / "lib", cfg)
    f = fish(manifest)
    assert len(f) >= 3
    assert bool((f["start_s"] >= 80.0 - 1e-9).all())  # window [80, 150]
    assert bool((f["stop_s"] <= 145.0 + 1e-9).all())  # edge 5 s
    for start, stop in zip(f["start_s"], f["stop_s"], strict=True):
        assert stop + 1.0 <= 100.0 - 0.05 or start - 1.0 >= 100.0 + 0.05
    # the 0.07 box is below the threshold: segments may contain it
    assert bool((f["screen_v1_n_near"] == 0).all())
    assert bool((f["screen_v1_threshold"] == 0.1).all())
    assert bool((f["screen_v1_max_score_near"] < 0.1).all())
    assert set(f["screen_v1_weights_sha256"]) == {"0" * 64}
    assert bool((f["window_start_s"] == 80.0).all())
    assert bool(f["screened"].all())
    starts = np.sort(f["start_s"].to_numpy())
    assert np.all(np.diff(starts) >= 10.0)
    recs = pd.read_csv(tmp_path / "lib" / "recordings.csv")
    row = recs.to_dict("records")[0]
    assert row["n_outside_window"] > 0
    assert row["n_rejected_v1"] > 0
    assert row["n_fish_segments"] == len(f)


def test_uncovered_spans_are_not_screened(
    make_trial: Callable[..., Trial], tmp_path: Path
) -> None:
    trial = make_trial(duration_s=150.0, fish_from_s=10.0)
    root = write_screen(
        tmp_path / "screen",
        "synthetic",
        trial.source.name,
        [],
        covered_s=[(60.0, 150.0)],
    )
    cfg = config(screens=(Screen("v1", root, threshold=0.1),))
    f = fish(build_background_library([trial.source], tmp_path / "lib", cfg))
    assert len(f) > 0
    assert bool((f["start_s"] >= 61.0 - 1e-9).all())


def test_gate_drops_busy_recordings(
    make_trial: Callable[..., Trial], tmp_path: Path
) -> None:
    quiet = make_trial("quiet", duration_s=150.0, fish_from_s=10.0)
    busy = make_trial("busy", duration_s=150.0, fish_from_s=10.0, seed=3)
    root = tmp_path / "screen"
    write_screen(root, "synthetic", "quiet", [(60.0, 0.9)])
    write_screen(root, "synthetic", "busy", [(t, 0.9) for t in (20.0, 60.0, 120.0)])
    # 1 box in 150 s = 24/h passes 50/h; 3 boxes = 72/h do not
    cfg = config(
        screens=(Screen("v1", root, threshold=0.1),),
        gate=RecordingGate("v1", threshold=0.5, max_rate_per_h=50.0),
    )
    manifest = build_background_library(
        [quiet.source, busy.source], tmp_path / "lib", cfg
    )
    assert set(manifest["recording"]) == {"quiet"}
    np.testing.assert_allclose(manifest["gate_rate_per_h"], 24.0)
    recs = pd.read_csv(tmp_path / "lib" / "recordings.csv").set_index("recording")
    assert str(recs.loc["busy", "skipped"]).startswith("recording gate")
    assert recs.loc["busy", "gate_rate_per_h"] == pytest.approx(72.0)
    assert recs.loc["busy", "n_gated"] > 0
    assert str(recs.at["busy", "gate_passed"]) == "False"


def test_missing_required_screen_skips_the_recording(
    make_trial: Callable[..., Trial], tmp_path: Path
) -> None:
    trial = make_trial()
    root = write_screen(tmp_path / "screen", "synthetic", "other", [])
    cfg = config(screens=(Screen("v1", root, threshold=0.1),))
    assert build_background_library([trial.source], tmp_path / "lib", cfg).empty
    optional = config(screens=(Screen("v1", root, threshold=0.1, required=False),))
    manifest = build_background_library([trial.source], tmp_path / "lib2", optional)
    assert not manifest.empty


def test_short_recordings_are_skipped(
    make_trial: Callable[..., Trial], tmp_path: Path
) -> None:
    trial = make_trial(duration_s=60.0)
    cfg = config(windows={"synthetic": TimeWindow(last_s=30.0, min_duration_s=100.0)})
    assert build_background_library([trial.source], tmp_path / "lib", cfg).empty
    recs = pd.read_csv(tmp_path / "lib" / "recordings.csv")
    assert "too short" in str(recs.loc[0, "skipped"])


def test_config_round_trip_and_validation(tmp_path: Path) -> None:
    cfg = config(
        windows={"a": TimeWindow(last_s=9000.0, min_start_s=12600.0)},
        selection_by_dataset={"a": dataclasses.replace(SELECTION, step_s=1.0)},
        screens=(Screen("v1", tmp_path, 0.1), Screen("v2", tmp_path, 0.2)),
        gate=RecordingGate("v1"),
    )
    again = BackgroundConfig.from_dict(cfg.to_dict())
    assert again == cfg
    assert again.selection_for("a").step_s == 1.0
    assert again.selection_for("b") == SELECTION
    with pytest.raises(ValueError, match="gate screen"):
        config(gate=RecordingGate("v1"))
    with pytest.raises(ValueError, match="unique"):
        config(screens=(Screen("v1", tmp_path, 0.1), Screen("v1", tmp_path, 0.2)))
    with pytest.raises(ValueError, match="unknown"):
        BackgroundConfig.from_dict({"windowz": {}})
    toml = tmp_path / "c.toml"
    toml.write_text(
        "[windows.a]\nlast_s = 1800.0\n[selection_by_dataset.a]\nstep_s = 5.0\n"
        f'[[screens]]\nname = "v1"\nroot = "{tmp_path}"\nthreshold = 0.1\n'
        '[gate]\nscreen = "v1"\n'
    )
    loaded = BackgroundConfig.from_toml(toml)
    assert loaded.windows["a"].last_s == 1800.0
    assert loaded.selection_for("a").step_s == 5.0
    assert loaded.selection_for("a").segment_s == loaded.selection.segment_s
    assert loaded.gate == RecordingGate("v1")


def test_carry_over_links_segments_unchanged(
    make_trial: Callable[..., Trial], tmp_path: Path
) -> None:
    a = make_trial("trial-a", seed=1)
    src = tmp_path / "src"
    built = build_background_library(
        [a.source], src, BackgroundConfig(selection=SELECTION, workers=1)
    )
    dst = tmp_path / "dst"
    out = carry_over(src, dst, ["synthetic"])
    assert list(out["segment_id"]) == list(built["segment_id"])
    for column in ("start_s", "level", "fish_ids", "accept"):
        assert out[column].tolist() == built[column].tolist()
    for path in built["path"]:
        for f in (src / str(path)).iterdir():
            assert os.path.samefile(f, dst / str(path) / f.name)
    assert (dst / "review.csv").exists()
    assert (dst / "recordings.csv").exists()
    with pytest.raises(ValueError, match="no segments"):
        carry_over(src, dst, ["nothing"])
    # a later build of another dataset keeps the carried rows
    b = make_trial("trial-b", seed=2)
    other = dataclasses.replace(b.source, dataset="other")
    manifest = build_background_library(
        [other], dst, BackgroundConfig(selection=SELECTION, workers=1)
    )
    assert set(manifest["dataset"]) == {"synthetic", "other"}
    kept = set(read_manifest(dst)["segment_id"]) & set(built["segment_id"])
    assert len(kept) == len(built)


def test_report_and_diagnostics_of_a_screened_build(
    make_trial: Callable[..., Trial], tmp_path: Path
) -> None:
    pytest.importorskip("matplotlib")
    from gridtools.background.diagnostics import write_diagnostics
    from gridtools.background.report import write_report

    quiet = make_trial("quiet", duration_s=150.0, fish_from_s=10.0)
    busy = make_trial("busy", duration_s=150.0, fish_from_s=10.0, seed=3)
    root = tmp_path / "screen"
    write_screen(root, "synthetic", "quiet", [(60.0, 0.9), (100.0, 0.2)])
    write_screen(root, "synthetic", "busy", [(t, 0.9) for t in (20.0, 60.0, 120.0)])
    cfg = config(
        windows={"synthetic": TimeWindow(last_s=120.0)},
        screens=(Screen("v1", root, threshold=0.1),),
        gate=RecordingGate("v1", threshold=0.5, max_rate_per_h=50.0),
    )
    lib = tmp_path / "lib"
    build_background_library([quiet.source, busy.source], lib, cfg)
    splits = {"train": ["synthetic/quiet"], "test": ["synthetic/busy"]}
    report = write_report(lib, n_demo_per_dataset=1, splits=splits).read_text()
    assert "## Chirp screening" in report
    assert "- test (1 of 1): `synthetic/busy`" in report
    assert "Dropped by the gate (1): synthetic/busy" in report
    counts = pd.read_csv(lib / "screen_counts.csv").set_index("recording")
    assert counts.loc["busy", "status"] == "gated"
    assert counts.loc["quiet", "n_rejected_v1"] > 0
    out = write_diagnostics(
        lib, screen="v1", snippet_recordings=["quiet"], n_snippets=2
    )
    diag = lib / "figures" / "diagnostics"
    assert (diag / "rates" / "synthetic_quiet.png").exists()
    assert (diag / "rates" / "synthetic_busy.png").exists()
    assert (diag / "gate.png").exists()
    assert len(out["snippets"]["quiet"]) == 2
    assert (diag / "snippets" / "quiet.html").exists()
    rates = pd.read_csv(diag / "rates.csv").set_index("recording")
    # window [30, 150] s: the boxes at 60 and 120 s
    assert rates.loc["busy", "window_rate_0.5_per_h"] == pytest.approx(2 / (120 / 3600))
