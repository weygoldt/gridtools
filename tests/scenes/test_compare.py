"""Simulated scenes measured with the calibration's estimators."""

from pathlib import Path

import numpy as np
import pytest

from gridtools.sim.config import GridConfig
from gridtools.sim.scene import simulate_scene

from .conftest import small_config

CALIBRATION = Path("/mnt/data1/chirp_v2/calibration/v1/calibration.json")


def test_measure_scene_runs_the_calibration_estimators(
    scene_path: Path, library_dir: Path, tmp_path: Path
) -> None:
    from gridtools.calibrate.scenes import compare_scenes, measure_scene
    from gridtools.sim.scene import write_scene

    stats = measure_scene(scene_path)
    assert stats.n_electrodes >= 1
    assert np.isfinite(stats.floor_db)
    assert len(stats.baselines_hz) == 3
    single = simulate_scene(
        small_config(
            library_dir,
            duration_s=3.0,
            grid=GridConfig(real_layout_prob=0.0, n_electrodes=(1, 1)),
        ),
        1,
    )
    one = measure_scene(write_scene(single, tmp_path / "single"))
    assert one.max_minus_median_db == []  # no profiles from one electrode
    if not CALIBRATION.exists():
        pytest.skip("calibration.json not available")
    table = compare_scenes([scene_path, tmp_path / "single"], CALIBRATION)
    assert list(table.columns) == ["quantity", "simulated", "real"]
    assert len(table) > 10
