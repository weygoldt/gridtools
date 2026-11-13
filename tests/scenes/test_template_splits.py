"""Scenes restricted to one fish-disjoint template split."""

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridtools.sim.chirps import TemplatePool, _open_pool
from gridtools.sim.config import ChirpConfig, SceneConfig, default_config_path
from gridtools.sim.scene import Scene, simulate_scene, write_scene
from gridtools.templates import load_library
from gridtools.templates.splits import (
    TemplateSplits,
    assign_template_splits,
    load_splits,
    write_splits,
)

from .conftest import FS, N_TEMPLATE, gaussian_template, small_config

#: Draws of ``TemplatePool.cycler(default_rng(3))`` and the template ids of the
#: ``scene`` fixture (seed 7) on the synthetic library, recorded with gridtools
#: v2 ``3c72542`` before template splits existed.
V2_DRAWS = [0, 17, 20, 3, 19, 3, 14, 17, 18, 9, 10, 17, 9, 4, 6, 3, 23, 15, 3, 12]
V2_DRAWS += [7, 16, 17, 6, 9, 14, 0, 9, 17, 21, 9, 3, 17, 12, 8, 17, 13, 4, 4, 12]
V2_SCENE_IDS = [2, 16, 20, 8, 4, 17, 4, 17, 0, 15, 6, 11, 21, 22, 17, 15, 12, 15]

N_GROUPS = 20


def write_grouped_library(path: Path) -> Path:
    """60 Gaussian templates in 20 groups; groups 0-2 are one fish (g6f5)."""
    path.mkdir(parents=True, exist_ok=True)
    excursions = [40.0, 60.0, 80.0, 120.0, 160.0, 250.0, 320.0, 400.0]
    widths = [0.006, 0.012, 0.025]
    combos = [(e, w) for e in excursions for w in widths]
    deltas, amps, rows = [], [], []
    for i in range(3 * N_GROUPS):
        exc, fwhm = combos[(7 * i) % len(combos)]
        delta, amp = gaussian_template(exc, fwhm, undershoot_hz=0.05 * exc)
        deltas.append(delta)
        amps.append(amp)
        g = i % N_GROUPS
        if g < 3:
            group = {
                "source": "competition",
                "group_label": f"2020-06-0{g + 1}-10_00/track4",
                "individual": "g6f5",
            }
        else:
            group = {
                "source": "chirp_chamber",
                "group_label": f"chirp_chamber/{g}",
                "individual": None,
            }
        rows.append(
            {
                "template_id": i,
                "excursion_hz": float(delta.max()),
                "duration_50_s": fwhm,
                "duration_10_s": fwhm * np.sqrt(np.log(10) / np.log(2)),
                "family": "big" if exc >= 200 else "small",
                "valid": True,
                "has_nan": False,
                "duplicate_of": -1,
                "truncated": False,
                "step_flag": False,
                "too_short": False,
                "multi_peak": False,
                "eodf_atypical": False,
                "ripple_flag": False,
                "amp_ripple_flag": False,
                "amp_usable": True,
                "inferred_group": g,
                **group,
            }
        )
    np.save(path / "delta_f_hz.npy", np.stack(deltas))
    np.save(path / "amp_rel.npy", np.stack(amps))
    pd.DataFrame(rows).to_parquet(path / "templates.parquet", index=False)
    manifest = {
        "library_format_version": 1,
        "samplerate_hz": FS,
        "n_templates": len(rows),
        "n_samples": N_TEMPLATE,
        "centre_index": N_TEMPLATE // 2,
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


@pytest.fixture(scope="module")
def split_setup(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, TemplateSplits]:
    library = write_grouped_library(tmp_path_factory.mktemp("grouped"))
    splits = assign_template_splits(
        load_library(library), seed=0, pins={"competition/g6f5": "train"}
    )
    path = write_splits(splits, library / "splits_v1.json", library)
    return library, path, splits


def split_config(library: Path, splits: Path, split: str) -> ChirpConfig:
    return ChirpConfig(
        library=str(library),
        rate_hz=(0.8, 1.0),
        response_prob=0.5,
        template_split=split,  # type: ignore[arg-type]
        template_splits=str(splits),
    )


def test_test_split_scene_uses_only_test_templates(
    split_setup: tuple[Path, Path, TemplateSplits], tmp_path: Path
) -> None:
    library, path, splits = split_setup
    test_ids = set(splits.template_ids["test"])
    assert test_ids
    assert not test_ids & set(splits.template_ids["train"])
    config = small_config(library, chirps=split_config(library, path, "test"))
    scene = simulate_scene(config, 7)
    used = set(scene.chirps["template_id"].astype(int))
    assert used
    assert used <= test_ids
    record = json.loads((write_scene(scene, tmp_path / "s") / "scene.json").read_text())
    split = record["templates"]["split"]
    assert split["name"] == "test"
    assert split["n_templates"] == record["templates"]["n_eligible"] == len(test_ids)
    assert split["n_eligible_all"] == 60
    assert len(split["sha256"]) == 64
    assert record["config"]["chirps"]["template_split"] == "test"


@pytest.mark.parametrize("name", ["train", "val", "test"])
def test_pool_draws_only_its_split_on_the_whole_pools_bins(
    split_setup: tuple[Path, Path, TemplateSplits],
    name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    library, path, splits = split_setup
    whole = TemplatePool.open(ChirpConfig(library=str(library)))
    _open_pool.cache_clear()  # log again
    with caplog.at_level(logging.WARNING, logger="gridtools.sim.chirps"):
        pool = TemplatePool.open(split_config(library, path, name))
    assert pool.split is not None
    assert set(pool.template_ids.tolist()) == set(splits.template_ids[name])
    draws = pool.cycler(np.random.default_rng(0)).draw(3000)
    assert set(draws.tolist()) == set(splits.template_ids[name])  # all reached
    eq = whole.view.equalization_bins(pool.bins)
    assert pool.split.edges is not None
    assert np.array_equal(pool.split.edges[0], eq.excursion_edges_hz)
    assert np.array_equal(pool.split.edges[1], eq.duration_edges_s)
    assert pool.split.n_bins_all == eq.n_nonempty
    assert pool.split.n_bins + len(pool.split.empty_bins) == eq.n_nonempty
    if pool.split.empty_bins:  # skipped, logged, and their share reported
        assert 0.0 < pool.split.lost_share < 1.0
        assert "empty and skipped" in caplog.text
    assert f"template split {name!r}" in pool.sampling


def test_val_and_test_empty_bins_do_not_crash(
    split_setup: tuple[Path, Path, TemplateSplits],
) -> None:
    library, path, _ = split_setup
    pools = [TemplatePool.open(split_config(library, path, s)) for s in ("val", "test")]
    assert any(p.split is not None and p.split.empty_bins for p in pools)


def test_unset_split_keeps_the_v2_draws(library_dir: Path, scene: Scene) -> None:
    pool = TemplatePool.open(ChirpConfig(library=str(library_dir)))
    assert pool.split is None
    assert pool.template_ids.size == 23
    assert pool.cycler(np.random.default_rng(3)).draw(40).tolist() == V2_DRAWS
    assert scene.chirps["template_id"].astype(int).tolist() == V2_SCENE_IDS
    assert "split" not in pool.sampling
    assert "edges" not in pool.sampling_options


def test_unset_split_is_recorded_as_none(scene_path: Path) -> None:
    record = json.loads((scene_path / "scene.json").read_text())
    assert record["templates"]["split"] is None


def test_splits_file_of_another_library_is_refused(
    split_setup: tuple[Path, Path, TemplateSplits], tmp_path: Path
) -> None:
    library, path, _ = split_setup
    document = json.loads(path.read_text())
    document["provenance"]["library"]["manifest_sha256"] = "0" * 64
    other = tmp_path / "other.json"
    other.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="another library build"):
        TemplatePool.open(split_config(library, other, "test"))
    with pytest.raises(FileNotFoundError):
        TemplatePool.open(split_config(library, tmp_path / "missing.json", "test"))
    assert load_splits(path).units["competition/g6f5"] == "train"


def test_split_config_keys(tmp_path: Path) -> None:
    default = SceneConfig.from_toml(default_config_path())
    assert default == SceneConfig()
    assert default.chirps.template_split is None
    assert default.chirps.template_splits is None
    toml = tmp_path / "scenes.toml"
    toml.write_text(
        '[chirps]\ntemplate_split = "val"\n'
        'template_splits = "/mnt/data1/chirp_v2/templates/v4/splits_v1.json"\n'
    )
    config = SceneConfig.from_toml(toml)
    assert config.chirps.template_split == "val"
    assert SceneConfig.from_json(config.to_json()) == config
    assert replace(config, name="x").chirps == config.chirps
    with pytest.raises(ValueError, match="go together"):
        ChirpConfig(template_split="test")
    with pytest.raises(ValueError, match="go together"):
        ChirpConfig(template_splits="/x.json")
    with pytest.raises(ValueError, match="template_split"):
        ChirpConfig(template_split="holdout", template_splits="/x.json")  # type: ignore[arg-type]
    toml.write_text('[chirps]\ntemplate_split = "holdout"\ntemplate_splits = "x"\n')
    with pytest.raises((TypeError, ValueError)):
        SceneConfig.from_toml(toml)
