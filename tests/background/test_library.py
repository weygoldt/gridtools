"""Loading, drawing, adding and merging background segments; the veto table."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridtools.background.build import BackgroundConfig, build_background_library
from gridtools.background.levels import REFERENCE_BANDWIDTH_HZ
from gridtools.background.library import (
    BackgroundLibrary,
    add_background,
    choose_channels,
    clip_tracks,
    merge_tracks,
)
from gridtools.background.review import read_review
from gridtools.background.select import SelectionConfig
from gridtools.io import GridGeometry, Tracks

from .conftest import COLUMNS, FS, ROWS, Trial

CONFIG = BackgroundConfig(
    selection=SelectionConfig(
        segment_s=10.0, step_s=2.5, margin_s=1.0, edge_s=5.0, segments_per_recording=4
    ),
    fish_free_per_recording=1,
    workers=1,
)


@pytest.fixture
def library(make_trial: Callable[..., Trial], tmp_path: Path) -> BackgroundLibrary:
    trials = [
        make_trial("trial-a", seed=1),
        make_trial("trial-b", seed=2, noisy_channel=2),
    ]
    build_background_library([t.source for t in trials], tmp_path / "lib", CONFIG)
    return BackgroundLibrary.load(tmp_path / "lib")


def test_select_filters(library: BackgroundLibrary) -> None:
    assert len(library) >= 6
    fish = library.select(segment_class="fish")
    assert set(fish.manifest["segment_class"]) == {"fish"}
    free = library.select(segment_class="fish_free")
    assert len(fish) + len(free) == len(library)
    one = library.select(include_groups=["synthetic/trial-a"])
    assert set(one.manifest["recording"]) == {"trial-a"}
    other = library.select(exclude_groups=["synthetic/trial-a"])
    assert set(other.manifest["recording"]) == {"trial-b"}
    assert len(library.select(n_electrodes=99)) == 0


def test_sample_is_seeded_and_keeps_the_geometry(library: BackgroundLibrary) -> None:
    geometry = GridGeometry.regular(COLUMNS, ROWS, 0.5, 0.5)
    a = library.sample(np.random.default_rng(3), 4.0, geometry=geometry)
    b = library.sample(np.random.default_rng(3), 4.0, geometry=geometry)
    assert a.segment_id == b.segment_id
    assert a.offset_s == b.offset_s
    np.testing.assert_array_equal(a.read(), b.read())
    assert a.n_samples == round(4.0 * FS)
    assert a.n_electrodes == ROWS * COLUMNS
    np.testing.assert_allclose(a.geometry.positions_m, geometry.positions_m)
    full = library.segment(a.segment_id)
    start = round(a.offset_s * FS)
    np.testing.assert_array_equal(a.read(), full.read(start, start + a.n_samples))
    if a.tracks is not None:
        assert a.tracks.times_s.min() >= 0
        assert a.tracks.times_s.max() < 4.0
    with pytest.raises(ValueError, match="no background segment"):
        library.sample(np.random.default_rng(0), 4.0, n_electrodes=ROWS * COLUMNS + 1)
    with pytest.raises(ValueError, match="no background segment"):
        library.sample(np.random.default_rng(0), 100.0)


def test_sample_with_resampling(library: BackgroundLibrary) -> None:
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="no background segment"):
        library.sample(rng, 2.0, samplerate_hz=8000.0)
    seg = library.sample(rng, 2.0, samplerate_hz=8000.0, resample=True)
    assert seg.samplerate_hz == 8000.0
    assert seg.n_samples == 16000
    # a 712 Hz fish survives resampling with its power
    spectrum = np.abs(np.fft.rfft(seg.read()[:, 0])) ** 2
    freqs = np.fft.rfftfreq(seg.n_samples, 1 / 8000.0)
    assert abs(freqs[np.argmax(spectrum)] - 712.0) < 1.0


def test_add_background(library: BackgroundLibrary) -> None:
    seg = library.sample(np.random.default_rng(5), 3.0)
    scene = np.random.default_rng(6).normal(size=seg.data.shape).astype(np.float32)
    out = add_background(
        scene, seg, 2.5, geometry=seg.geometry, samplerate_hz=seg.samplerate_hz
    )
    np.testing.assert_allclose(out, scene + 2.5 * seg.read(), rtol=1e-6, atol=1e-9)
    inplace = scene.copy()
    add_background(inplace, seg, 2.5, out=inplace, chunk_samples=1000)
    np.testing.assert_allclose(inplace, out, rtol=1e-6, atol=1e-9)
    with pytest.raises(ValueError, match="shape"):
        add_background(scene[:-1], seg, 1.0)
    with pytest.raises(ValueError, match="geometry"):
        add_background(scene, seg, 1.0, geometry=GridGeometry.regular(6, 1, 0.5, 0.5))
    with pytest.raises(ValueError, match="rate"):
        add_background(scene, seg, 1.0, samplerate_hz=20_000.0)
    with pytest.raises(ValueError, match="scale"):
        add_background(scene, seg, -1.0)


def _tracks(times: np.ndarray, ids: list[int], n_el: int | None) -> Tracks:
    idx = np.repeat(np.arange(times.size), len(ids))
    ident = np.tile(np.asarray(ids, dtype=float), times.size)
    return Tracks(
        times_s=times,
        freqs_hz=600.0 + 10 * ident,
        idents=ident,
        time_indices=idx,
        powers_db=None if n_el is None else np.zeros((idx.size, n_el)),
    )


def test_merge_tracks_keeps_detections_and_separates_identities() -> None:
    scene = _tracks(np.arange(0.0, 10.0, 0.1229), [0, 1, 2], 6)
    real = _tracks(np.arange(0.2, 10.0, 0.3277), [7, 16], 6)
    merged, mapping = merge_tracks(scene, real, power_offset_db=-20.0)
    assert merged is not None
    assert mapping == {1000: 7, 1001: 16}
    assert sorted(merged.ids.tolist()) == [0, 1, 2, 1000, 1001]
    for ident, source in ((1000, 7), (0, 0)):
        track = merged.track(ident)
        original = (real if ident >= 1000 else scene).track(source)
        np.testing.assert_allclose(track.times_s, original.times_s)
        np.testing.assert_allclose(track.freqs_hz, original.freqs_hz)
    real_powers = merged.track(1000).powers_db
    assert real_powers is not None
    np.testing.assert_allclose(real_powers, -20.0)
    # without scene tracks, or with a custom offset
    alone, ids = merge_tracks(None, real, ident_offset=50)
    assert alone is not None
    assert sorted(alone.ids.tolist()) == [50, 51]
    assert ids == {50: 7, 51: 16}
    with pytest.raises(ValueError, match="collides"):
        merge_tracks(scene, real, ident_offset=2)
    assert merge_tracks(scene, None) == (scene, {})


def test_clip_tracks_pads_powers() -> None:
    tracks = _tracks(np.arange(0.0, 20.0, 0.5), [3], 5)
    clipped = clip_tracks(tracks, 5.0, 7.0, n_electrodes=6)
    assert clipped is not None
    np.testing.assert_allclose(clipped.times_s, [0.0, 0.5, 1.0, 1.5])
    assert clipped.powers_db is not None
    assert clipped.powers_db.shape == (4, 6)
    assert np.all(np.isnan(clipped.powers_db[:, 5]))
    assert clip_tracks(tracks, 50.0, 60.0) is None


def test_real_fish_table(library: BackgroundLibrary) -> None:
    fish = library.select(segment_class="fish")
    seg = fish.segment(str(fish.manifest["segment_id"].iloc[0]))
    _, mapping = merge_tracks(None, seg.tracks)
    table = seg.real_fish(mapping)
    assert bool(table["real_fish"].all())
    assert sorted(table["source_ident"]) == [3, 7]
    assert sorted(table["ident"]) == [1000, 1001]


def test_review_vetoes_survive_rebuilds(
    make_trial: Callable[..., Trial], tmp_path: Path
) -> None:
    trial = make_trial("trial-a", seed=1)
    root = tmp_path / "lib"
    first = build_background_library([trial.source], root, CONFIG)
    review = read_review(root)
    assert review is not None
    assert bool(review["accept"].all())
    assert set(review["segment_id"]) == set(first["segment_id"])
    fish_ids = first.loc[first["segment_class"] == "fish", "segment_id"].tolist()
    vetoed = fish_ids[0]
    review.loc[review["segment_id"] == vetoed, ["accept", "note"]] = [
        False,
        "chirp at 3 s",
    ]
    review.to_csv(root / "review.csv", index=False)

    # the library drops it at once
    assert vetoed not in set(BackgroundLibrary.load(root).manifest["segment_id"])
    kept = BackgroundLibrary.load(root, include_rejected=True).manifest
    assert kept.loc[kept["segment_id"] == vetoed, "note"].item() == "chirp at 3 s"

    # a rebuild avoids the vetoed span and keeps the author's edit
    second = build_background_library([trial.source], root, CONFIG)
    assert vetoed not in set(second["segment_id"])
    start = float(vetoed.rsplit("_", 1)[1])
    for s, e in zip(second["start_s"], second["stop_s"], strict=True):
        assert e + 1.0 <= start or s - 1.0 >= start + 10.0
    after = read_review(root)
    assert after is not None
    row = after[after["segment_id"] == vetoed]
    assert not bool(row["accept"].item())
    assert row["note"].item() == "chirp at 3 s"


def test_incremental_build_keeps_other_recordings(
    make_trial: Callable[..., Trial], tmp_path: Path
) -> None:
    a, b = make_trial("trial-a", seed=1), make_trial("trial-b", seed=2)
    root = tmp_path / "lib"
    both = build_background_library([a.source, b.source], root, CONFIG)
    only_b = build_background_library([b.source], root, CONFIG)
    pd.testing.assert_frame_equal(
        both.drop(columns=["accept"]), only_b.drop(columns=["accept"])
    )


def test_noisy_channel_is_recorded(library: BackgroundLibrary) -> None:
    m = library.manifest
    for recording, noisy in zip(m["recording"], m["noisy_channels"], strict=True):
        assert noisy == ([2] if recording == "trial-b" else [])


def test_choose_channels() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        run = choose_channels(rng, 9, 3, bad=[4])
        assert run is not None
        assert 4 not in run
        assert list(run) == list(range(run[0], run[0] + 3))
        pick = choose_channels(rng, 9, 4, bad=[0, 8], mode="random")
        assert pick is not None
        assert sorted(set(pick)) == list(pick)
        assert not {0, 8} & set(pick)
    assert choose_channels(rng, 9, 9, bad=[8]) is None
    assert choose_channels(rng, 5, 3, bad=[2]) is None  # no run of 3 avoids 2
    assert choose_channels(rng, 5, 2, bad=[2], mode="random") is not None
    assert choose_channels(rng, 3, 4) is None
    with pytest.raises(ValueError, match="mode"):
        choose_channels(rng, 3, 1, mode="corners")


@pytest.mark.parametrize("mode", ["contiguous", "random"])
@pytest.mark.parametrize("k", [1, 3])
def test_channel_subsets(library: BackgroundLibrary, k: int, mode: str) -> None:
    for seed in range(8):
        seg = library.sample(
            np.random.default_rng(seed), 4.0, n_electrodes=k, channels=mode
        )
        assert seg.n_electrodes == k
        assert len(seg.channels) == k
        if mode == "contiguous":
            assert list(seg.channels) == list(
                range(seg.channels[0], seg.channels[0] + k)
            )
        if seg.info["recording"] == "trial-b":
            assert 2 not in seg.channels  # noisy channel avoided
        full = library.segment(
            seg.segment_id, offset_s=seg.offset_s, duration_s=seg.duration_s
        )
        np.testing.assert_array_equal(seg.read(), full.read()[:, list(seg.channels)])
        np.testing.assert_allclose(
            seg.geometry.positions_m, full.geometry.positions_m[list(seg.channels)]
        )
        assert seg.positions_nominal
        assert seg.info["source_channels"] == list(seg.channels)
        floors = full.info["noise_psd"]
        assert seg.level == pytest.approx(
            seg.info["noise_psd"] * REFERENCE_BANDWIDTH_HZ
        )
        assert seg.level > 0.0
        assert floors > 0.0
        if seg.tracks is not None and full.tracks is not None:
            assert seg.tracks.powers_db is not None
            assert full.tracks.powers_db is not None
            assert seg.tracks.powers_db.shape[1] == k
            np.testing.assert_array_equal(
                seg.tracks.powers_db, full.tracks.powers_db[:, list(seg.channels)]
            )


def test_single_channel_level_is_that_channels_floor(
    library: BackgroundLibrary,
) -> None:
    seg = library.sample(np.random.default_rng(1), 4.0, n_electrodes=1)
    full = library.segment(seg.segment_id)
    details = (library.segment_dir(seg.segment_id) / "segment.json").read_text()
    floors = __import__("json").loads(details)["noise_psd_per_electrode"]
    assert seg.info["noise_psd"] == pytest.approx(floors[seg.channels[0]])
    assert full.n_electrodes == ROWS * COLUMNS


def test_bad_channels_can_be_allowed(library: BackgroundLibrary) -> None:
    b = library.select(include_groups=["synthetic/trial-b"])
    with pytest.raises(ValueError, match="no background segment"):
        b.sample(np.random.default_rng(0), 4.0, n_electrodes=ROWS * COLUMNS)
    seg = b.sample(
        np.random.default_rng(0),
        4.0,
        n_electrodes=ROWS * COLUMNS,
        avoid_bad_channels=False,
    )
    assert seg.noisy_channels == [2]
    assert seg.channels == tuple(range(ROWS * COLUMNS))


def test_incremental_build_upgrades_an_old_manifest(
    make_trial: Callable[..., Trial], tmp_path: Path
) -> None:
    a, b = make_trial("trial-a", seed=1), make_trial("trial-b", seed=2)
    root = tmp_path / "lib"
    build_background_library([a.source, b.source], root, CONFIG)
    old = pd.read_parquet(root / "manifest.parquet").drop(
        columns=["noisy_channels", "spacing_known", "accept"]
    )
    old.to_parquet(root / "manifest.parquet", index=False)
    merged = build_background_library([b.source], root, CONFIG)
    assert set(merged["recording"]) == {"trial-a", "trial-b"}
    assert merged.loc[merged["recording"] == "trial-b", "noisy_channels"].notna().all()
