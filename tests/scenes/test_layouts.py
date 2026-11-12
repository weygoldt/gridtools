"""Electrode counts from 1 to 16: layouts, writers, re-rendering, footprints."""

import collections
from pathlib import Path

import numpy as np
import pytest

from gridtools.io import load_tracks, open_recording, read_fishgrid_config
from gridtools.sim.config import GridConfig, SceneConfig
from gridtools.sim.footprint import HannSpectrogram, chirp_footprint
from gridtools.sim.scene import (
    _rectangle,
    draw_layout,
    layout_arena,
    render_background,
    render_fish,
    simulate_scene,
    write_scene,
)

from .conftest import small_config

FS = 20_000.0


def test_layout_distribution() -> None:
    config = SceneConfig()
    counts: collections.Counter[int] = collections.Counter()
    kinds: collections.Counter[str] = collections.Counter()
    n = 1500
    for seed in range(n):
        layout = draw_layout(config, seed)
        counts[layout.n_electrodes] += 1
        kinds[layout.kind] += 1
        assert layout.rows * layout.columns == layout.n_electrodes
        assert layout.positions_m.min() >= -1e-12
    assert abs(kinds["real"] / n - 0.3) < 0.05
    small = sum(counts[k] for k in range(1, 5)) / n
    assert abs(small - 0.7 * 0.4) < 0.05
    assert counts[1] > 0.05 * n  # single electrodes well represented
    assert set(counts) == set(range(1, 17))


def test_rectangles_and_gaps() -> None:
    assert _rectangle(12, 3.0) == (3, 4)
    assert _rectangle(16, 3.0) == (4, 4)
    assert _rectangle(14, 3.0) is None  # 2 x 7 is too elongated
    assert _rectangle(7, 3.0) is None
    grid = GridConfig(
        real_layout_prob=0.0, n_electrodes=(7, 7), jitter_rel=0.0, rotation_prob=0.0
    )
    layout = draw_layout(SceneConfig(grid=grid), 0)
    assert layout.kind == "irregular"
    assert (layout.rows, layout.columns) == (1, 7)
    assert not layout.regular
    spacing = layout.spacing_m
    # the positions are on a 3 x 3 lattice with two gaps
    lattice = layout.positions_m / spacing
    np.testing.assert_allclose(lattice, np.round(lattice), atol=1e-9)
    assert np.unique(np.round(lattice), axis=0).shape[0] == 7


def test_single_electrode_arena_has_extent() -> None:
    grid = GridConfig(real_layout_prob=0.0, n_electrodes=(1, 1))
    layout = draw_layout(SceneConfig(grid=grid), 3)
    arena = layout_arena(layout, 0.0)
    assert arena.width_m == pytest.approx(layout.spacing_m)
    assert arena.height_m == pytest.approx(layout.spacing_m)


@pytest.mark.parametrize("n_electrodes", [1, 2, 4, 7, 16])
def test_scenes_with_any_electrode_count(
    library_dir: Path, tmp_path: Path, n_electrodes: int
) -> None:
    grid = GridConfig(real_layout_prob=0.0, n_electrodes=(n_electrodes, n_electrodes))
    config = small_config(library_dir, duration_s=3.0, grid=grid)
    scene = simulate_scene(config, 11)
    model = scene.model
    assert scene.recording.shape == (config.n_samples, n_electrodes)
    assert scene.tracks.powers_db is not None
    assert scene.tracks.powers_db.shape[1] == n_electrodes
    # exact re-rendering
    total = sum(render_fish(scene, i) for i in range(len(model.fish)))
    np.testing.assert_allclose(
        scene.recording,
        total + render_background(scene),
        atol=8 * np.finfo(np.float32).eps * np.abs(scene.recording).max(),
        rtol=0,
    )
    span = render_fish(scene, 0, 1.0, 1.5)
    assert span.shape == (10_000, n_electrodes)
    np.testing.assert_array_equal(span, model.render(0, 20_000, 30_000))
    # footprints
    inner = ~scene.chirps["truncated"].to_numpy(dtype=bool)
    assert np.all(np.isfinite(scene.chirps["box_t0_s"].to_numpy(dtype=float)[inner]))
    ids = scene.chirps["chirp_id"].to_numpy(dtype=int)[inner]
    if ids.size:
        fp = chirp_footprint(
            scene, int(ids[0]), HannSpectrogram(window_s=0.0256, hop_s=0.0016)
        )
        assert np.isfinite(fp.box[0])
    # writers
    path = write_scene(scene, tmp_path / "scene")
    recording = open_recording(path)
    assert recording.n_channels == n_electrodes
    assert recording.geometry is not None
    np.testing.assert_allclose(recording.geometry.positions_m, model.electrode_xy_m)
    cfg = read_fishgrid_config(path)
    assert cfg.n_rows * cfg.n_columns == n_electrodes
    assert (path / "electrodes.csv").exists() == (not scene.plan.layout.regular)
    tracks = load_tracks(path)
    assert tracks.powers_db is not None
    assert tracks.powers_db.shape[1] == n_electrodes


def test_plots_with_one_electrode(library_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from gridtools.sim.plot import plot_scene_dir

    grid = GridConfig(real_layout_prob=0.0, n_electrodes=(1, 1))
    scene = simulate_scene(small_config(library_dir, duration_s=3.0, grid=grid), 2)
    path = write_scene(scene, tmp_path / "scene", plots=True)
    assert (path / "spectrogram_zooms.png").exists()
    assert len(plot_scene_dir(path, tmp_path / "again")) == 2
