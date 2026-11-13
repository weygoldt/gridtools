"""Fish-disjoint train/val/test splits of the template library."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridtools.templates import TemplateLibrary
from gridtools.templates.splits import (
    SPLIT_NAMES,
    assign_template_splits,
    fish_units,
    load_splits,
    splits_report,
    write_splits,
)

PINS = {"competition/g6f5": "train"}

#: Harvested competition tracks: (recording, track, individual, n, n_long).
COMPETITION = [
    ("2020-05-29-10_00", 10, "g6f5", 6, 1),
    ("2020-06-08-10_00", 4, "g6f5", 20, 14),
    ("2020-06-11-10_00", 10, "g6f5", 10, 6),
    ("2020-05-27-10_00", 17, "g6f1", 3, 1),
    ("2020-06-02-10_00", 7, "g6f1", 12, 9),
    ("2020-05-19-10_00", 7, "g5f4", 6, 3),
    ("2020-05-12-10_00", 6, "g5f3", 2, 0),
    ("2020-05-18-10_00", 6, "g5f3", 3, 2),
    ("2020-03-16-10_00", 3, "g4f3", 4, 1),
    ("2020-03-10-09_56", 35, "", 1, 0),
]


def grouped_metadata(n_chamber_groups: int = 60, seed: int = 0) -> pd.DataFrame:
    """Template metadata with chirp-chamber groups and competition fish."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    def add(exc: float, dur: float, **extra: object) -> None:
        rows.append(
            {
                "template_id": len(rows),
                "excursion_hz": exc,
                "duration_10_s": dur,
                "duration_50_s": 0.5 * dur,
                "baseline_hz": 800.0,
                "family": "big" if exc >= 200 else "small",
                "valid": True,
                "has_nan": False,
                "duplicate_of": -1,
                "truncated": False,
                "step_flag": False,
                "ripple_flag": False,
                "multi_peak": False,
                "eodf_atypical": False,
                "amp_ripple_flag": False,
                "amp_usable": True,
                **extra,
            }
        )

    for g in range(n_chamber_groups):
        big = g % 4 == 0
        for _ in range(int(rng.integers(1, 9))):
            exc = rng.uniform(250, 500) if big else rng.uniform(30, 180)
            dur = rng.uniform(0.01, 0.12) if big else rng.uniform(0.008, 0.03)
            add(
                exc,
                dur,
                source="chirp_chamber",
                group_label=f"chirp_chamber/{g}",
                inferred_group=g,
                individual=None,
            )
    for recording, track, individual, n, n_long in COMPETITION:
        for k in range(n):
            dur = rng.uniform(0.16, 0.4) if k < n_long else rng.uniform(0.06, 0.14)
            add(
                rng.uniform(300, 500),
                dur,
                source="competition",
                group_label=f"{recording}/track{track}",
                inferred_group=1000 + track,
                individual=individual,
                recording=recording,
            )
    add(  # flagged: outside the default selection, still assigned
        60.0,
        0.012,
        source="chirp_chamber",
        group_label="chirp_chamber/999",
        inferred_group=999,
        individual=None,
        step_flag=True,
    )
    return pd.DataFrame(rows)


def library_of(meta: pd.DataFrame) -> TemplateLibrary:
    n = len(meta)
    return TemplateLibrary(
        delta_f_hz=np.empty((n, 0), dtype=np.float32),
        amp_rel=np.empty((n, 0), dtype=np.float32),
        metadata=meta,
        samplerate_hz=20_000.0,
    )


@pytest.fixture(scope="module")
def library() -> TemplateLibrary:
    return library_of(grouped_metadata())


def test_units_are_fish_across_recordings(library: TemplateLibrary) -> None:
    units = fish_units(library.metadata)
    meta = library.metadata
    g6f5 = meta["individual"] == "g6f5"
    assert set(units[g6f5.to_numpy()]) == {"competition/g6f5"}
    assert meta.loc[g6f5, "group_label"].nunique() == 3
    unknown = (meta["group_label"] == "2020-03-10-09_56/track35").to_numpy()
    assert set(units[unknown]) == {"2020-03-10-09_56/track35"}
    chamber = (meta["source"] == "chirp_chamber").to_numpy()
    assert (units[chamber] == meta.loc[chamber, "group_label"].to_numpy()).all()


def test_assignment_is_deterministic(library: TemplateLibrary) -> None:
    first = assign_template_splits(library, seed=3, pins=PINS)
    second = assign_template_splits(library, seed=3, pins=PINS)
    assert first.units == second.units
    assert first.template_ids == second.template_ids
    others = [assign_template_splits(library, seed=s, pins=PINS) for s in (4, 5)]
    assert any(o.units != first.units for o in others)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_splits_are_disjoint_by_group_and_fish(
    library: TemplateLibrary, seed: int
) -> None:
    splits = assign_template_splits(library, seed=seed, pins=PINS)
    meta = library.metadata
    split = splits.split_of(meta).tolist()
    by_group: dict[str, set[str]] = {}
    by_fish: dict[str, set[str]] = {}
    groups = meta["group_label"].astype(str).tolist()
    fish = meta["individual"].fillna("").astype(str).tolist()
    for g, f, s in zip(groups, fish, split, strict=True):
        by_group.setdefault(g, set()).add(s)
        if f:
            by_fish.setdefault(f, set()).add(s)
    assert all(len(v) == 1 for v in by_group.values())
    assert len(by_fish) == 5
    assert all(len(v) == 1 for v in by_fish.values())
    split_of_id = dict(zip(meta["template_id"].tolist(), split, strict=True))
    listed = [t for s in SPLIT_NAMES for t in splits.template_ids[s]]
    assert sorted(listed) == sorted(split_of_id)  # every template exactly once
    for s in SPLIT_NAMES:
        assert all(split_of_id[t] == s for t in splits.template_ids[s])


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_pinned_fish_stays_in_train(library: TemplateLibrary, seed: int) -> None:
    splits = assign_template_splits(library, seed=seed, pins=PINS)
    assert splits.units["competition/g6f5"] == "train"
    meta = library.metadata
    rows = (meta["individual"] == "g6f5").to_numpy()
    assert set(splits.split_of(meta)[rows]) == {"train"}
    assert splits.provenance["pins"] == PINS


def test_val_and_test_cover_small_big_and_long(library: TemplateLibrary) -> None:
    for seed in range(4):
        splits = assign_template_splits(library, seed=seed, pins=PINS)
        sel = library.select()
        meta = sel.metadata
        split = splits.split_of(meta)
        long = meta["duration_10_s"].to_numpy() >= 0.15
        big = (meta["family"] == "big").to_numpy()
        for s in SPLIT_NAMES:
            mask = split == s
            assert (mask & big).any(), (seed, s)
            assert (mask & ~big).any(), (seed, s)
            assert (mask & long).any(), (seed, s)
        # the non-pinned fish with most long chirps is the test's long-chirp fish
        assert splits.units["competition/g6f1"] == "test"


def test_unit_fractions_follow_the_targets(library: TemplateLibrary) -> None:
    splits = assign_template_splits(library, seed=0, pins=PINS)
    sel_units = set(fish_units(library.select().metadata).tolist())
    counts = [sum(splits.units[u] == s for u in sel_units) for s in SPLIT_NAMES]
    expected = np.array([0.8, 0.1, 0.1]) * len(sel_units)
    assert np.all(np.abs(np.array(counts) - expected) <= 1)
    assert splits.units["chirp_chamber/999"] in SPLIT_NAMES  # unselected unit


def test_pins_and_fractions_are_checked(library: TemplateLibrary) -> None:
    with pytest.raises(KeyError, match="not in the library"):
        assign_template_splits(library, pins={"competition/g9f9": "train"})
    with pytest.raises(ValueError, match="unknown split"):
        assign_template_splits(library, pins={"competition/g6f5": "holdout"})
    with pytest.raises(ValueError, match="three"):
        assign_template_splits(library, fractions=(0.9, 0.1))


def test_json_round_trip(library: TemplateLibrary, tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"library_format_version": 1}')
    library.metadata.to_parquet(tmp_path / "templates.parquet")
    splits = assign_template_splits(library, seed=0, pins=PINS)
    path = write_splits(splits, tmp_path / "splits.json", tmp_path)
    loaded = load_splits(path)
    assert loaded.units == splits.units
    assert loaded.groups == splits.groups
    assert loaded.template_ids == splits.template_ids
    document = json.loads(path.read_text())
    assert document["groups"]["2020-06-02-10_00/track7"] == "test"
    assert set(document["units"]["competition/g6f5"]["groups"]) == {
        "2020-05-29-10_00/track10",
        "2020-06-08-10_00/track4",
        "2020-06-11-10_00/track10",
    }
    prov = loaded.provenance
    assert prov["seed"] == 0
    assert prov["fractions"] == {"train": 0.8, "val": 0.1, "test": 0.1}
    assert len(prov["library"]["manifest_sha256"]) == 64
    assert np.array_equal(
        loaded.split_of(library.metadata), splits.split_of(library.metadata)
    )
    other = library.metadata.assign(group_label="chirp_chamber/elsewhere")
    with pytest.raises(KeyError, match="not in the split assignment"):
        loaded.split_of(other)


def test_report_lists_splits_and_gaps(library: TemplateLibrary) -> None:
    splits = assign_template_splits(library, seed=0, pins=PINS)
    text = splits_report(library, splits)
    for s in SPLIT_NAMES:
        assert f"| {s} |" in text
        assert f"### Units per bin: {s}" in text
    assert "## Bins empty in val or test" in text
    assert "`competition/g6f5` | train |" in text


def test_subset_bins_on_parent_edges() -> None:
    lib = library_of(grouped_metadata())
    full = lib.equalization_bins()
    small = lib.select(families=["small"])
    edges = (full.excursion_edges_hz, full.duration_edges_s)
    own = small.equalization_bins(edges=edges)
    assert np.array_equal(own.excursion_edges_hz, full.excursion_edges_hz)
    assert own.counts.sum() == small.unique_ids().size
    assert own.n_nonempty < full.n_nonempty  # big-chirp bins are empty
    ids = small.cycler(np.random.default_rng(0), edges=edges).draw(500)
    assert np.isin(ids, small.template_ids).all()
    with pytest.raises(ValueError, match="increasing"):
        small.equalization_bins(edges=(full.excursion_edges_hz[:5], edges[1]))


def test_assignment_does_not_depend_on_string_hashing(tmp_path: Path) -> None:
    """The split file is written once; a fresh process must reproduce it."""
    import os
    import subprocess
    import sys

    grouped_metadata().to_parquet(tmp_path / "meta.parquet")
    script = (
        "import json, sys, numpy as np, pandas as pd\n"
        "from gridtools.templates import TemplateLibrary\n"
        "from gridtools.templates.splits import assign_template_splits\n"
        "m = pd.read_parquet(sys.argv[1])\n"
        "z = np.empty((len(m), 0), dtype=np.float32)\n"
        "lib = TemplateLibrary(delta_f_hz=z, amp_rel=z, metadata=m, samplerate_hz=1.0)\n"
        "s = assign_template_splits(lib, seed=0, pins={'competition/g6f5': 'train'})\n"
        "print(json.dumps(s.units, sort_keys=True))\n"
    )
    outputs = [
        subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / "meta.parquet")],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        ).stdout
        for hash_seed in ("0", "12345")
    ]
    assert outputs[0] == outputs[1]
    here = assign_template_splits(library_of(grouped_metadata()), seed=0, pins=PINS)
    assert json.loads(outputs[0]) == here.units
