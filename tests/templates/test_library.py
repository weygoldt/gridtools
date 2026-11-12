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


def make_library(
    n: int,
    duplicate_of: list[int] | None = None,
    excursion_hz: list[float] | None = None,
    duration_s: list[float] | None = None,
) -> TemplateLibrary:
    """Tiny in-memory library whose template i has value i everywhere."""
    values = np.arange(n, dtype=np.float32)[:, None] * np.ones((1, 8), dtype=np.float32)
    meta = pd.DataFrame(
        {
            "template_id": np.arange(n),
            "duplicate_of": duplicate_of if duplicate_of is not None else [-1] * n,
            "family": ["small"] * n,
            "excursion_hz": excursion_hz if excursion_hz is not None else [60.0] * n,
            "duration_10_s": duration_s if duration_s is not None else [0.015] * n,
        }
    )
    return TemplateLibrary(values, np.ones_like(values), meta, FS)


def test_sample_without_duplicate_inflation() -> None:
    # templates 3 and 4 duplicate template 0: the pool has 5 distinct chirps, not 7
    lib = make_library(7, duplicate_of=[-1, -1, -1, 0, 0, -1, -1])
    np.testing.assert_array_equal(lib.unique_ids(), [0, 1, 2, 5, 6])
    draw = lib.sample(np.random.default_rng(0), 23, equalize=False)
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
    ids = lib.sample(np.random.default_rng(3), 50, equalize=False).template_ids
    assert len(set(ids.tolist())) == 50


@pytest.mark.parametrize("equalize", [False, True])
def test_sample_is_deterministic_under_seed(equalize: bool) -> None:
    lib = skewed_library()
    a = lib.sample(np.random.default_rng(123), 45, equalize=equalize).template_ids
    b = lib.sample(np.random.default_rng(123), 45, equalize=equalize).template_ids
    c = lib.sample(np.random.default_rng(124), 45, equalize=equalize).template_ids
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_cycler_balances_across_calls() -> None:
    lib = make_library(6)
    cycler = lib.cycler(np.random.default_rng(5), equalize=False)
    ids = np.concatenate([cycler.draw(1) for _ in range(18)])
    np.testing.assert_array_equal(np.bincount(ids, minlength=6), [3] * 6)


def skewed_library() -> TemplateLibrary:
    """100 short small chirps, 5 mid-sized, 1 long big chirp, 1 exact duplicate."""
    excursion = [50.0 + 0.1 * i for i in range(100)] + [250.0] * 5 + [500.0, 500.0]
    duration = [0.012 + 1e-5 * i for i in range(100)] + [0.03] * 5 + [0.2, 0.2]
    duplicate_of = [-1] * 106 + [105]
    return make_library(107, duplicate_of, excursion, duration)


def test_equalization_bins_equal_width() -> None:
    lib = skewed_library()
    eq = lib.equalization_bins((10, 10), spacing="linear")
    assert eq.template_ids.size == 106  # the duplicate is not in the pool
    np.testing.assert_allclose(np.diff(eq.excursion_edges_hz), (500.0 - 50.0) / 10)
    np.testing.assert_allclose(np.diff(eq.duration_edges_s), (0.2 - 0.012) / 10)
    assert eq.counts.sum() == 106
    assert eq.n_nonempty == 3
    # the largest and longest template falls into the last bin
    assert eq.counts[-1, -1] == 1
    table = eq.table()
    assert table.shape == (10, 10)
    assert int(table.to_numpy().sum()) == 106


def test_equalized_sampling_balances_bins_and_members() -> None:
    # the library v2 scheme: equal-width bins, every non-empty bin equally likely
    lib = skewed_library()
    n = 3000
    ids = lib.sample(
        np.random.default_rng(0), n, equalize=True, spacing="linear", alpha=1.0
    ).template_ids
    counts = np.bincount(ids, minlength=107)
    assert counts[106] == 0  # duplicate never drawn
    # each of the 3 non-empty bins gets n / 3 draws
    assert counts[:100].sum() == n // 3
    assert counts[100:105].sum() == n // 3
    assert counts[105] == n // 3
    # within a bin, members are used evenly (successive permutations)
    assert counts[:100].max() - counts[:100].min() <= 1
    assert counts[100:105].max() - counts[100:105].min() <= 1
    eq = lib.equalization_bins(spacing="linear")
    expected = dict(
        zip(eq.template_ids.tolist(), eq.expected_draws(n).tolist(), strict=True)
    )
    assert expected[105] == pytest.approx(1000.0)
    assert expected[0] == pytest.approx(10.0)


def test_equalized_cycler_balances_across_calls() -> None:
    lib = skewed_library()
    cycler = lib.cycler(
        np.random.default_rng(9), equalize=True, spacing="linear", alpha=1.0
    )
    assert cycler.n_groups == 3
    ids = np.concatenate([cycler.draw(1) for _ in range(30)])
    bins = np.where(ids < 100, 0, np.where(ids < 105, 1, 2))
    np.testing.assert_array_equal(np.bincount(bins), [10, 10, 10])


def test_natural_sampling_follows_the_pool() -> None:
    lib = skewed_library()
    ids = lib.sample(np.random.default_rng(1), 106, equalize=False).template_ids
    assert sorted(ids.tolist()) == list(range(106))


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


def test_default_exclusions_follow_author_decisions() -> None:
    from gridtools.templates import DEFAULT_EXCLUDE

    assert "eodf_atypical" not in DEFAULT_EXCLUDE  # kept (possibly Eigenmannia)
    assert "amp_ripple_flag" not in DEFAULT_EXCLUDE  # measured amplitude only
    # only clean samples (library v4)
    for flag in (
        "too_short",
        "brief_cycles",
        "ripple_flag",
        "multi_peak",
        "line_in_span",
        "baseline_drift",
        "low_snr",
        "decay_interrupted",
    ):
        assert flag in DEFAULT_EXCLUDE


def test_too_short_flag_falls_back_to_duration() -> None:
    # libraries built before the flag existed get it from the width at 10%
    lib = make_library(4, duration_s=[0.003, 0.0049, 0.005, 0.2])
    np.testing.assert_array_equal(
        lib.flag_mask("too_short"), [True, True, False, False]
    )
    kept = lib.select(exclude=("too_short",))
    np.testing.assert_array_equal(kept.template_ids, [2, 3])


def test_log_bins_are_geometric() -> None:
    lib = skewed_library()
    eq = lib.equalization_bins((10, 10), spacing="log")
    assert eq.spacing == "log"
    np.testing.assert_allclose(eq.excursion_edges_hz[[0, -1]], [50.0, 500.0])
    ratios = eq.excursion_edges_hz[1:] / eq.excursion_edges_hz[:-1]
    np.testing.assert_allclose(ratios, 10**0.1)
    np.testing.assert_allclose(eq.duration_edges_s[[0, -1]], [0.012, 0.2])
    assert eq.counts.sum() == 106
    with pytest.raises(ValueError, match="spacing"):
        lib.equalization_bins(spacing="cubic")  # type: ignore[arg-type]


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.6, 1.0])
def test_tempered_bin_probabilities(alpha: float) -> None:
    lib = skewed_library()
    eq = lib.equalization_bins((10, 10), spacing="linear")
    p = eq.bin_probabilities(alpha)
    counts = eq.counts.ravel().astype(float)
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p[counts == 0] == 0)
    expected = np.where(counts > 0, counts ** (1 - alpha), 0.0)
    np.testing.assert_allclose(p, expected / expected.sum())
    per_template = eq.template_probabilities(alpha)
    assert per_template.sum() == pytest.approx(1.0)
    # templates of one bin share its probability equally
    first = eq.bin_of == eq.bin_of[0]
    np.testing.assert_allclose(per_template[first], per_template[first][0])
    np.testing.assert_allclose(eq.expected_draws(1000, alpha), 1000 * per_template)
    with pytest.raises(ValueError, match="alpha"):
        eq.bin_probabilities(1.5)


def test_tempered_sampling_matches_probabilities_and_cycles_members() -> None:
    lib = skewed_library()
    n = 20_000
    alpha = 0.5
    ids = lib.sample(
        np.random.default_rng(2), n, spacing="linear", alpha=alpha
    ).template_ids
    counts = np.bincount(ids, minlength=107)
    assert counts[106] == 0
    eq = lib.equalization_bins((10, 10), spacing="linear")
    p = eq.bin_probabilities(alpha)
    per_bin = np.array(
        [counts[:100].sum(), counts[100:105].sum(), counts[105]], dtype=float
    )
    expected = n * p[p > 0]
    # systematic blocks of 1024 draws: each bin is within one draw of its
    # expectation per block, far tighter than independent draws (sd ~ 60 here)
    n_blocks = int(np.ceil(n / 1024))
    assert np.all(np.abs(per_bin - expected) <= n_blocks)
    # members of a bin are used evenly (successive permutations)
    assert counts[:100].max() - counts[:100].min() <= 1
    assert counts[100:105].max() - counts[100:105].min() <= 1


def test_tempered_cycler_balances_across_calls() -> None:
    lib = skewed_library()
    cycler = lib.cycler(np.random.default_rng(4), spacing="linear", alpha=0.5)
    ids = np.concatenate([cycler.draw(1) for _ in range(1024)])
    bins = np.where(ids < 100, 0, np.where(ids < 105, 1, 2))
    p = lib.equalization_bins(spacing="linear").bin_probabilities(0.5)
    # one full systematic block, drawn one at a time
    assert np.all(np.abs(np.bincount(bins, minlength=3) - 1024 * p[p > 0]) <= 1)


def test_alpha_zero_gives_natural_bin_shares() -> None:
    lib = skewed_library()
    n = 10_600
    ids = lib.sample(np.random.default_rng(6), n, spacing="linear", alpha=0.0)
    counts = np.bincount(ids.template_ids, minlength=107)
    # natural: every distinct template about equally often
    assert abs(counts[:106].mean() - 100) < 1e-9
    assert counts[:106].max() - counts[:106].min() <= 6


def test_cycler_validates_weights() -> None:
    from gridtools.templates import TemplateCycler

    groups = [np.array([0, 1]), np.array([2])]
    with pytest.raises(ValueError, match="one weight per group"):
        TemplateCycler(groups, np.random.default_rng(0), weights=[1.0])
    with pytest.raises(ValueError, match="non-negative"):
        TemplateCycler(groups, np.random.default_rng(0), weights=[1.0, -1.0])
    only_first = TemplateCycler(groups, np.random.default_rng(0), weights=[1.0, 0.0])
    assert set(only_first.draw(10).tolist()) == {0, 1}


# ---- library v4: clean samples and fish-first sampling ---------------------


def flagged_library() -> TemplateLibrary:
    """Eight templates, each of templates 1-7 carrying one quality flag."""
    n = 8
    values = np.arange(n, dtype=np.float32)[:, None] * np.ones((1, 8), np.float32)
    meta = pd.DataFrame(
        {
            "template_id": np.arange(n),
            "valid": True,
            "has_nan": False,
            "duplicate_of": -1,
            "truncated": False,
            "step_flag": False,
            "family": "small",
            "excursion_hz": 80.0,
            "duration_10_s": 0.02,
            # template 1: FWHM 2.4 ms at 800 Hz = 1.9 cycles; template 2: 3 ms
            # at 400 Hz = 1.2 cycles; template 0: 12 ms at 800 Hz
            "duration_50_s": [0.012, 0.0024, 0.003, 0.012, 0.012, 0.012, 0.012, 0.012],
            "baseline_hz": [800.0, 800.0, 400.0, 800.0, 800.0, 800.0, 800.0, np.nan],
            "ripple_flag": [False, False, False, True, False, False, False, False],
            "multi_peak": [False, False, False, False, True, False, False, False],
            "line_in_span": [None, None, None, None, None, True, False, None],
            "low_snr": [False, False, False, False, False, False, True, False],
        }
    )
    return TemplateLibrary(values, np.ones_like(values), meta, FS)


def test_quality_flags_are_excluded_by_default() -> None:
    lib = flagged_library()
    np.testing.assert_array_equal(
        lib.flag_mask("brief_cycles"), [0, 1, 1, 0, 0, 0, 0, 0]
    )
    # a missing value is not a flag; a missing harvest-only column neither
    np.testing.assert_array_equal(
        lib.flag_mask("line_in_span"), [0, 0, 0, 0, 0, 1, 0, 0]
    )
    assert not lib.flag_mask("baseline_drift").any()
    assert not lib.flag_mask("decay_interrupted").any()
    # unknown EODf: cycles cannot be counted, so not flagged
    assert not lib.flag_mask("brief_cycles")[7]
    np.testing.assert_array_equal(lib.select().template_ids, [0, 7])
    assert lib.select(exclude=()).n_templates == 8


def test_individual_labels_prefer_group_label() -> None:
    from gridtools.templates.library import individual_labels

    meta = pd.DataFrame(
        {
            "template_id": [0, 1, 2, 3],
            "inferred_group": [5, 5, 6, 7],
            "group_label": [None, "", "recA/track1", "recA/track1"],
        }
    )
    assert individual_labels(meta).tolist() == [
        "group/5",
        "group/5",
        "recA/track1",
        "recA/track1",
    ]
    bare = pd.DataFrame({"template_id": [3, 4]})
    assert individual_labels(bare).tolist() == ["template/3", "template/4"]


def fish_library() -> TemplateLibrary:
    """One bin: fish A with 1 template, B with 3, C with 30 (34 templates)."""
    owners = ["A"] + ["B"] * 3 + ["C"] * 30
    n = len(owners)
    values = np.arange(n, dtype=np.float32)[:, None] * np.ones((1, 8), np.float32)
    meta = pd.DataFrame(
        {
            "template_id": np.arange(n),
            "duplicate_of": -1,
            "family": "small",
            "excursion_hz": 60.0,
            "duration_10_s": 0.015,
            "group_label": owners,
        }
    )
    return TemplateLibrary(values, np.ones_like(values), meta, FS)


def test_fish_first_sampling_balances_individuals_within_a_bin() -> None:
    lib = fish_library()
    n = 3000
    ids = lib.sample(np.random.default_rng(0), n).template_ids
    counts = np.bincount(ids, minlength=34)
    per_fish = [counts[:1].sum(), counts[1:4].sum(), counts[4:].sum()]
    # individuals come in successive permutations: exactly n / 3 each
    assert per_fish == [1000, 1000, 1000]
    # each fish's templates are used evenly (successive permutations)
    assert counts[1:4].max() - counts[1:4].min() <= 1
    assert counts[4:].max() - counts[4:].min() <= 1
    # template balance gives every template the same share instead
    flat = lib.sample(np.random.default_rng(0), n, balance="template").template_ids
    flat_counts = np.bincount(flat, minlength=34)
    assert flat_counts.max() - flat_counts.min() <= 1
    with pytest.raises(ValueError, match="balance"):
        lib.cycler(np.random.default_rng(0), balance="fish")  # type: ignore[arg-type]


def test_fish_first_probabilities_and_determinism() -> None:
    lib = fish_library()
    eq = lib.equalization_bins()
    p = dict(
        zip(
            eq.template_ids.tolist(),
            eq.template_probabilities(balance="individual").tolist(),
            strict=True,
        )
    )
    assert sum(p.values()) == pytest.approx(1.0)
    assert p[0] == pytest.approx(1 / 3)
    assert p[1] == pytest.approx(1 / 9)
    assert p[10] == pytest.approx(1 / 90)
    flat = eq.template_probabilities(balance="template")
    np.testing.assert_allclose(flat, 1 / 34)
    a = lib.sample(np.random.default_rng(7), 200).template_ids
    b = lib.sample(np.random.default_rng(7), 200).template_ids
    np.testing.assert_array_equal(a, b)
    # drawing one at a time keeps the balance across calls
    cycler = lib.cycler(np.random.default_rng(8))
    one = np.concatenate([cycler.draw(1) for _ in range(300)])
    per_fish = [np.isin(one, [0]).sum(), np.isin(one, [1, 2, 3]).sum()]
    assert per_fish == [100, 100]


def test_fish_first_equals_template_balance_without_individuals() -> None:
    # every template its own individual: identical draws to template balance
    lib = skewed_library()
    a = lib.sample(np.random.default_rng(3), 500, balance="individual").template_ids
    b = lib.sample(np.random.default_rng(3), 500, balance="template").template_ids
    np.testing.assert_array_equal(a, b)
