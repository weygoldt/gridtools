"""Building, loading, selecting and sampling a library from a synthetic source."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridtools.templates import (
    BuildConfig,
    TemplateLibrary,
    build_library,
    load_library,
)
from gridtools.templates.build import AMP_FILE, DELTA_FILE, MANIFEST_FILE

from .conftest import FS, SourceRow


@pytest.fixture
def built(synthetic_h5: tuple[Path, list[SourceRow]], tmp_path: Path) -> Path:
    h5_path, _ = synthetic_h5
    out = tmp_path / "library"
    build_library(h5_path, out, BuildConfig(chunk_rows=3))
    return out


def test_build_writes_consistent_library(
    built: Path, synthetic_h5: tuple[Path, list[SourceRow]]
) -> None:
    _, rows = synthetic_h5
    lib = load_library(built)
    assert lib.n_templates == len(rows)
    assert lib.samplerate_hz == FS
    assert lib.n_samples == BuildConfig().n_samples(FS)
    np.testing.assert_array_equal(lib.template_ids, np.arange(len(rows)))
    np.testing.assert_array_equal(lib.metadata["source_row"], np.arange(len(rows)))
    assert np.all(lib.delta_f_hz[:, 0] == 0.0)
    assert np.all(lib.delta_f_hz[:, -1] == 0.0)
    assert np.all(lib.amp_rel[:, 0] == 1.0)
    assert np.all(lib.amp_rel[:, -1] == 1.0)
    meta = lib.metadata
    for i, row in enumerate(rows[:6]):
        assert meta.loc[i, "baseline_hz"] == pytest.approx(
            row.baseline_eodf_hz, abs=0.5
        )
    assert list(meta["family"][:6]) == [
        "small",
        "small",
        "small",
        "big",
        "big",
        "small",
    ]
    # row 6 repeats row 1 exactly, row 7 is row 5 with a little added noise
    assert meta.loc[6, "duplicate_of"] == 1
    assert meta.loc[6, "duplicate_kind"] == "exact"
    assert meta.loc[7, "duplicate_of"] == 5
    assert meta.loc[7, "duplicate_kind"] == "near"
    assert (meta.loc[:5, "duplicate_of"] == -1).all()


def test_manifest_contents(
    built: Path, synthetic_h5: tuple[Path, list[SourceRow]]
) -> None:
    h5_path, _ = synthetic_h5
    manifest = json.loads((built / MANIFEST_FILE).read_text())
    assert manifest["library_format_version"] == 1
    assert manifest["samplerate_hz"] == FS
    assert manifest["config"] == BuildConfig(chunk_rows=3).to_dict()
    assert BuildConfig.from_dict(manifest["config"]) == BuildConfig(chunk_rows=3)
    src = manifest["source"]["h5"]
    stat = h5_path.stat()
    assert src["size_bytes"] == stat.st_size
    assert src["mtime_ns"] == stat.st_mtime_ns
    assert len(src["sha256"]) == 64
    assert "gridtools_version" in manifest["code"]
    assert "git_sha" in manifest["code"]
    for name in (DELTA_FILE, AMP_FILE):
        assert manifest["outputs"][name]["size_bytes"] == (built / name).stat().st_size
    counts = manifest["counts"]
    assert counts["flagged"]["exact_duplicate"] == 1
    assert counts["flagged"]["near_duplicate"] == 1


def test_build_is_deterministic(
    synthetic_h5: tuple[Path, list[SourceRow]], tmp_path: Path
) -> None:
    h5_path, _ = synthetic_h5
    a, b = tmp_path / "a", tmp_path / "b"
    build_library(h5_path, a, BuildConfig())
    build_library(h5_path, b, BuildConfig(chunk_rows=5))
    for name in (DELTA_FILE, AMP_FILE):
        np.testing.assert_array_equal(np.load(a / name), np.load(b / name))
    pd.testing.assert_frame_equal(load_library(a).metadata, load_library(b).metadata)


def test_select_by_flags_and_family(built: Path) -> None:
    lib = load_library(built)
    clean = lib.select()
    assert set(clean.template_ids) <= set(range(6))
    assert 6 not in clean.template_ids
    assert 7 not in clean.template_ids
    big = lib.select(families=["big"])
    assert set(big.metadata["family"]) == {"big"}
    everything = lib.select(exclude=())
    assert everything.n_templates == lib.n_templates
    with pytest.raises(KeyError, match="unknown flag"):
        lib.select(exclude=("no_such_flag",))


def make_library(n: int, duplicate_of: list[int] | None = None) -> TemplateLibrary:
    """Tiny in-memory library whose template i has value i everywhere."""
    values = np.arange(n, dtype=np.float32)[:, None] * np.ones((1, 8), dtype=np.float32)
    meta = pd.DataFrame(
        {
            "template_id": np.arange(n),
            "duplicate_of": duplicate_of if duplicate_of is not None else [-1] * n,
            "family": ["small"] * n,
        }
    )
    return TemplateLibrary(values, np.ones_like(values), meta, FS)


def test_sample_without_duplicate_inflation() -> None:
    # templates 3 and 4 duplicate template 0: the pool has 5 distinct chirps, not 7
    lib = make_library(7, duplicate_of=[-1, -1, -1, 0, 0, -1, -1])
    np.testing.assert_array_equal(lib.unique_ids(), [0, 1, 2, 5, 6])
    draw = lib.sample(np.random.default_rng(0), 23)
    counts = np.bincount(draw.template_ids, minlength=7)
    assert counts[3] == 0
    assert counts[4] == 0
    used = counts[[0, 1, 2, 5, 6]]
    assert used.max() - used.min() <= 1  # 23 = 4 full cycles + 3
    # every block of 5 consecutive draws is a permutation of the pool
    for k in range(4):
        assert sorted(draw.template_ids[5 * k : 5 * k + 5]) == [0, 1, 2, 5, 6]
    np.testing.assert_array_equal(draw.delta_f_hz[:, 0], draw.template_ids)


def test_sample_no_repeats_when_pool_suffices() -> None:
    lib = make_library(50)
    ids = lib.sample(np.random.default_rng(3), 50).template_ids
    assert len(set(ids.tolist())) == 50


def test_sample_is_deterministic_under_seed() -> None:
    lib = make_library(20)
    a = lib.sample(np.random.default_rng(123), 45).template_ids
    b = lib.sample(np.random.default_rng(123), 45).template_ids
    c = lib.sample(np.random.default_rng(124), 45).template_ids
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_cycler_balances_across_calls() -> None:
    lib = make_library(6)
    cycler = lib.cycler(np.random.default_rng(5))
    ids = np.concatenate([cycler.draw(1) for _ in range(18)])
    np.testing.assert_array_equal(np.bincount(ids, minlength=6), [3] * 6)


def test_get_and_rows_of() -> None:
    lib = make_library(5).select(exclude=(), template_ids=[1, 3, 4])
    got = lib.get([4, 1])
    np.testing.assert_array_equal(got.template_ids, [4, 1])
    np.testing.assert_array_equal(got.delta_f_hz[:, 0], [4, 1])
    with pytest.raises(KeyError, match="not in this library"):
        lib.get([0])


def test_library_validates_shapes() -> None:
    meta = pd.DataFrame({"template_id": [0, 1]})
    with pytest.raises(ValueError, match="rows"):
        TemplateLibrary(
            np.zeros((3, 4), np.float32), np.ones((3, 4), np.float32), meta, FS
        )
