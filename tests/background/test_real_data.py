"""A small library from the tube competition recordings (opt-in: data, slow)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from gridtools.background import (
    BackgroundConfig,
    BackgroundLibrary,
    SelectionConfig,
    competition_2019_sources,
    competition_2022_sources,
)
from gridtools.background.build import build_background_library
from gridtools.background.sources import COMPETITION_2022_ROOT
from gridtools.io import read_v1_detections

pytestmark = [
    pytest.mark.data,
    pytest.mark.slow,
    pytest.mark.skipif(
        not COMPETITION_2022_ROOT.exists(), reason=f"{COMPETITION_2022_ROOT} missing"
    ),
]

RECORDINGS_2019 = ["2019-12-02-10_00", "2020-03-16-10_00", "2020-05-21-10_00"]


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("background")
    sources = competition_2019_sources(recordings=RECORDINGS_2019)
    sources += competition_2022_sources(recordings=["2022-06-24-10_00"])
    config = BackgroundConfig(
        selection=SelectionConfig(segments_per_recording=2),
        fish_free_per_recording=1,
        workers=4,
    )
    build_background_library(sources, root, config)
    return root


def test_no_v1_detection_near_any_screened_fish_segment(built: Path) -> None:
    lib = BackgroundLibrary.load(built)
    m = lib.select(datasets=["competition2019"], segment_class="fish").manifest
    assert len(m) == 2 * len(RECORDINGS_2019)
    margin = SelectionConfig().margin_s
    for recording, rows in m.groupby("recording"):
        det = read_v1_detections(
            Path("/mnt/data2/2019_tube_competition") / str(recording)
        )
        for start, stop in zip(rows["start_s"], rows["stop_s"], strict=True):
            near = (det["t_start_s"] < stop + margin) & (
                det["t_end_s"] > start - margin
            )
            assert not bool(near.any())
    assert bool((m["n_v1_detections_margin"] == 0).all())


def test_levels_and_real_fish_are_plausible(built: Path) -> None:
    m = BackgroundLibrary.load(built).manifest
    # calibration: floors -135 to -119 dB re V^2/Hz (2019/2020); 2022 is quieter
    floors = m["noise_psd_db"].to_numpy(float)
    assert np.all((floors > -150) & (floors < -110))
    snrs = [
        s for lst in m["fish_snr_db"] for s in lst if s is not None and math.isfinite(s)
    ]
    assert len(snrs) >= 10
    assert 35 < float(np.median(snrs)) < 65
    # 2019 trials: channel 9 carries no fish signal
    dead = m.loc[m["recording"] == "2019-12-02-10_00", "dead_channels"]
    assert all(d == [8] for d in dead)


def test_geometry_and_screening_flags(built: Path) -> None:
    m = BackgroundLibrary.load(built).manifest
    c2022 = m[m["dataset"] == "competition2022"]
    assert len(c2022) >= 1
    assert not bool(c2022["geometry_known"].any())
    assert not bool(c2022[c2022["segment_class"] == "fish"]["screened"].any())
    assert set(c2022["n_electrodes"]) == {11}
    c2019 = m[m["recording"] == "2019-12-02-10_00"]
    assert set(c2019["n_electrodes"]) == {9}
    assert bool(c2019["geometry_known"].all())
