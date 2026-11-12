"""Harvest synthetic grid recordings and merge them into a library."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridtools.templates import BuildConfig, build_library, load_library
from gridtools.templates.harvest import (
    HarvestConfig,
    read_review,
    run_harvest,
)
from gridtools.templates.merge import (
    build_merged_library,
    merged_length,
    resize_centred,
    support_half_width,
)

from .conftest import FS, SourceRow, SyntheticGrid, write_grid_directory


def test_resize_centred_pads_and_crops_exactly() -> None:
    rows = np.zeros((2, 8), dtype=np.float32)
    rows[0, 3:6] = [1.0, 2.0, 3.0]
    rows[1, 4] = 5.0
    padded = resize_centred(rows, 14, 0.0)
    assert padded.shape == (2, 14)
    assert padded[0, 7 - 1] == 1.0  # sample 3 (centre 4) -> 6 (centre 7)
    assert padded[1, 7] == 5.0
    assert padded.sum() == rows.sum()
    back = resize_centred(padded, 8, 0.0)
    np.testing.assert_array_equal(back, rows)
    with pytest.raises(ValueError, match="cut template support"):
        resize_centred(rows, 2, 0.0)  # would drop sample 5 (3.0)
    ones = resize_centred(np.ones((1, 4), np.float32), 10, 1.0)
    assert np.all(ones == 1.0)


def test_merged_length_fits_the_widest_support() -> None:
    delta = np.zeros((2, 1000))
    delta[0, 500 + 300] = 1.0
    amp = np.ones_like(delta)
    amp[1, 500 - 420] = 0.5
    assert support_half_width(delta, amp) == 420
    assert merged_length(400, 420, 1000.0) == 900  # 2 * 420 + 2 -> 0.9 s
    assert merged_length(1200, 420, 1000.0) == 1200


@pytest.fixture
def harvested(
    tmp_path: Path, synthetic_grid: Callable[..., SyntheticGrid]
) -> tuple[Path, list[Path]]:
    grids = [
        synthetic_grid(seed=11, excursion_hz=450.0, plateau_s=0.15, decay_tau_s=0.05),
        synthetic_grid(seed=12, excursion_hz=380.0, plateau_s=0.25, decay_tau_s=0.04),
        synthetic_grid(seed=13, other_hz=1050.0),  # rejected: fish in span
    ]
    recordings = [
        write_grid_directory(tmp_path / f"rec{k}", grid) for k, grid in enumerate(grids)
    ]
    out = tmp_path / "long_chirps"
    run_harvest(recordings, out, HarvestConfig(), workers=1)
    return out, recordings


def test_run_harvest_writes_candidates_templates_and_review(
    harvested: tuple[Path, list[Path]],
) -> None:
    out, _ = harvested
    candidates = pd.read_csv(out / "candidates.csv")
    assert candidates["status"].tolist() == ["harvested", "harvested", "rejected"]
    assert candidates["reason"].fillna("").tolist()[2] == "fish_in_span"
    per_recording = pd.read_csv(out / "recordings.csv")
    assert per_recording["n_candidates"].tolist() == [1, 1, 1]
    assert per_recording["n_harvested"].tolist() == [1, 1, 0]
    review = read_review(out / "review.csv")
    assert review["accept"].tolist() == [True, True]
    meta = pd.read_parquet(out / "harvest" / "templates.parquet")
    assert set(meta["source"].tolist()) == {"competition"}
    assert meta["group_label"].tolist() == ["rec0/track1", "rec1/track1"]
    assert np.all(meta["duration_10_s"].to_numpy(dtype=float) > 0.25)
    delta = np.load(out / "harvest" / "delta_f_hz.npy")
    assert delta.shape == (2, HarvestConfig().template.n_samples(FS))
    assert np.all(delta[:, [0, -1]] == 0.0)


def test_merge_adds_long_templates_and_respects_vetoes(
    harvested: tuple[Path, list[Path]],
    synthetic_h5: tuple[Path, list[SourceRow]],
    tmp_path: Path,
) -> None:
    out, _ = harvested
    h5_path, _ = synthetic_h5
    base = tmp_path / "base"
    build_library(h5_path, base, BuildConfig())
    base_lib = load_library(base)
    merged = tmp_path / "merged"
    build_merged_library(base, out / "harvest", merged)
    lib = load_library(merged, mmap=True)
    n_base = base_lib.n_templates
    assert lib.n_templates == n_base + 2
    # long templates need more than the 0.4 s of the base library
    assert lib.n_samples > base_lib.n_samples
    assert lib.n_samples % round(0.1 * FS) == 0
    assert np.all(lib.delta_f_hz[:, [0, -1]] == 0.0)
    assert np.all(lib.amp_rel[:, [0, -1]] == 1.0)
    # base templates are unchanged, only padded around the same centre
    np.testing.assert_array_equal(
        resize_centred(np.asarray(lib.delta_f_hz[:n_base]), base_lib.n_samples, 0.0),
        base_lib.delta_f_hz,
    )
    meta = lib.metadata
    np.testing.assert_array_equal(meta["template_id"], np.arange(n_base + 2))
    assert meta["source"].tolist() == ["chirp_chamber"] * n_base + ["competition"] * 2
    np.testing.assert_array_equal(
        meta["excursion_hz"].to_numpy()[:n_base],
        base_lib.metadata["excursion_hz"].to_numpy(),
    )
    groups = meta["inferred_group"].to_numpy(dtype=np.int64)
    assert groups[n_base:].min() > groups[:n_base].max()
    # the unchanged API: select, sample and get give 2-D arrays of the new length
    clean = lib.select()
    assert set(clean.template_ids) >= {n_base, n_base + 1}
    drawn = clean.sample(np.random.default_rng(0), 50)
    assert drawn.delta_f_hz.shape == (50, lib.n_samples)
    assert drawn.amp_rel.shape == (50, lib.n_samples)
    long_ids = clean.get([n_base + 1])
    assert np.count_nonzero(long_ids.delta_f_hz) / FS > 0.3
    # a veto in the review sheet removes the template from the next build
    review = pd.read_csv(out / "review.csv")
    review.loc[0, "accept"] = False
    review.to_csv(out / "review.csv", index=False)
    vetoed = tmp_path / "vetoed"
    manifest = build_merged_library(base, out / "harvest", vetoed)
    lib = load_library(vetoed)
    assert lib.n_templates == n_base + 1
    assert manifest["source"]["review"]["vetoed"] == [review.loc[0, "harvest_key"]]
    assert (vetoed / "review.csv").exists()


def test_merge_is_deterministic(
    harvested: tuple[Path, list[Path]],
    synthetic_h5: tuple[Path, list[SourceRow]],
    tmp_path: Path,
) -> None:
    out, _ = harvested
    h5_path, _ = synthetic_h5
    base = tmp_path / "base"
    build_library(h5_path, base, BuildConfig())
    a, b = tmp_path / "a", tmp_path / "b"
    build_merged_library(base, out / "harvest", a)
    build_merged_library(base, out / "harvest", b)
    for name in ("delta_f_hz.npy", "amp_rel.npy"):
        np.testing.assert_array_equal(np.load(a / name), np.load(b / name))
    pd.testing.assert_frame_equal(load_library(a).metadata, load_library(b).metadata)


def test_merged_report(
    harvested: tuple[Path, list[Path]],
    synthetic_h5: tuple[Path, list[SourceRow]],
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from gridtools.templates.report_merged import (
        DURATION_CLASSES_MS,
        Scheme,
        sampling_summary,
        write_merged_report,
    )

    out, _ = harvested
    h5_path, _ = synthetic_h5
    base = tmp_path / "base"
    build_library(h5_path, base, BuildConfig())
    merged = tmp_path / "merged"
    build_merged_library(base, out / "harvest", merged)
    path = write_merged_report(merged, out)
    text = path.read_text()
    for heading in (
        "## 1. Long chirps from competition recordings",
        "## 2. Template length",
        "## 3. Minimum duration",
        "## 5. Histogram equalization",
    ):
        assert heading in text
    assert (merged / "figures" / "long_chirps.png").stat().st_size > 0
    assert (merged / "figures" / "sampling.png").stat().st_size > 0
    lib = load_library(merged).select()
    natural = sampling_summary(lib, Scheme("natural", False))
    assert natural["distinct"] <= lib.n_templates
    shares = [natural[label] for label, _, _ in DURATION_CLASSES_MS]
    assert sum(shares) == pytest.approx(1.0)
    assert natural["big"] + natural["small"] == pytest.approx(1.0)
    tempered = sampling_summary(lib, Scheme("log", True, "log", 0.6))
    assert tempered["big"] + tempered["small"] == pytest.approx(1.0)
    assert tempered["max_repeats"] >= natural["max_repeats"]
