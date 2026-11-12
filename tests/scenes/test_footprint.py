"""Chirp footprints: boxes from with/without differences, visibility."""

import dataclasses
import itertools
import math
import time
from pathlib import Path

import numpy as np
import pytest

from gridtools.sim.config import (
    ChirpConfig,
    ClickConfig,
    NoiseConfig,
    PopulationConfig,
    SceneConfig,
)
from gridtools.sim.footprint import (
    HannSpectrogram,
    _in_box,
    chirp_footprint,
    scene_footprints,
)
from gridtools.sim.render import SceneModel
from gridtools.sim.scene import Scene, simulate_scene

from .conftest import small_config

FS = 20_000.0
CLEAN = NoiseConfig(
    colored=False, hum=False, clip_limit=None, dead_prob=0.0, noisy_prob=0.0
)
NO_CLICKS = ClickConfig(rate_per_h_median=0.0)


def _single_fish(
    library: Path,
    noise: NoiseConfig = CLEAN,
    baseline_hz: float = 800.0,
) -> SceneConfig:
    """One fish at a fixed baseline (800 Hz: 2nd harmonic above the band)."""
    population = PopulationConfig(
        n_fish=(1, 1),
        baseline_uniform_hz=(baseline_hz, baseline_hz),
        baseline_uniform_prob=1.0,
    )
    return small_config(
        library,
        duration_s=4.0,
        population=population,
        noise=noise,
        clicks=NO_CLICKS,
        chirps=ChirpConfig(library=str(library), rate_hz=(1.0, 1.0)),
    )


@pytest.fixture(scope="module")
def clean_scene(library_dir: Path) -> Scene:
    return simulate_scene(_single_fish(library_dir), 3)


def _inner_chirps(scene: Scene) -> list[int]:
    table = scene.chirps
    return table.loc[~table["truncated"], "chirp_id"].astype(int).tolist()


def test_box_holds_the_difference_energy(clean_scene: Scene) -> None:
    ids = _inner_chirps(clean_scene)
    assert len(ids) >= 3
    for chirp_id in ids:
        for nfft, hop in ((4096, 100), (512, 32)):
            fp = chirp_footprint(
                clean_scene,
                chirp_id,
                HannSpectrogram(window_s=nfft / FS, hop_s=hop / FS),
                keep_arrays=True,
            )
            assert fp.energy_fraction >= 0.99
            d = fp.arrays["d"]
            inside = _in_box(fp.box, fp.arrays["times_s"], fp.arrays["freqs_hz"])
            assert d[inside].sum() >= 0.99 * d.sum()
            # the box contains the chirp's peak and its baseline at the centre
            row = clean_scene.chirps.set_index("chirp_id").loc[chirp_id]
            t0, t1, f0, f1 = fp.box
            assert t0 <= row["t_peak_s"] <= t1
            assert f0 <= row["baseline_hz"] <= f1


def test_box_edges_lie_at_cell_edges(clean_scene: Scene) -> None:
    fn = HannSpectrogram(window_s=1024 / FS, hop_s=50 / FS)
    fp = chirp_footprint(
        clean_scene, _inner_chirps(clean_scene)[0], fn, keep_arrays=True
    )
    times, freqs, mask = fp.arrays["times_s"], fp.arrays["freqs_hz"], fp.arrays["mask"]
    cols = np.flatnonzero(mask.any(axis=0))
    rows = np.flatnonzero(mask.any(axis=1))
    dt, df = 50 / FS, FS / 1024
    assert fp.box[0] == pytest.approx(times[cols[0]] - dt / 2)
    assert fp.box[1] == pytest.approx(times[cols[-1]] + dt / 2)
    assert fp.box[2] == pytest.approx(freqs[rows[0]] - df / 2)
    assert fp.box[3] == pytest.approx(freqs[rows[-1]] + df / 2)


def test_long_windows_smear_boxes_in_time(clean_scene: Scene) -> None:
    """At nfft 4096 a box is wider than at 512 by the window's extent above the
    threshold: for a short event the difference follows the squared Hann
    window, which stays within ``threshold_db`` of its peak over ~0.86 of its
    length at 25 dB (not the full window length)."""
    hop = 32
    threshold = 25.0
    extent = {}
    for nfft in (4096, 512):
        w = np.hanning(nfft + 1)[:nfft]
        extent[nfft] = np.mean(w**2 >= 10 ** (-threshold / 10)) * nfft / FS
    expected = extent[4096] - extent[512]
    for chirp_id in _inner_chirps(clean_scene):
        boxes = {
            nfft: chirp_footprint(
                clean_scene,
                chirp_id,
                HannSpectrogram(window_s=nfft / FS, hop_s=hop / FS),
            ).box
            for nfft in (4096, 512)
        }
        widths = {n: b[1] - b[0] for n, b in boxes.items()}
        assert widths[4096] > widths[512]
        # bounded by the window lengths, and close to the thresholded extent
        assert widths[4096] - widths[512] <= (4096 - 512) / FS + 2 * hop / FS
        assert widths[4096] - widths[512] == pytest.approx(expected, abs=0.035)


def test_clean_chirps_are_visible_and_drowned_chirps_are_not(
    library_dir: Path, clean_scene: Scene
) -> None:
    assert np.all(clean_scene.chirps["visible"].to_numpy(dtype=bool))
    assert np.all(np.isinf(clean_scene.chirps["snr_db"].to_numpy(dtype=float)))
    loud = NoiseConfig(
        floor_modes=((-60.0, 0.0, 1.0),),
        harder_prob=0.0,
        hum=False,
        clip_limit=None,
        dead_prob=0.0,
        noisy_prob=0.0,
    )
    drowned = simulate_scene(_single_fish(library_dir, loud), 3)
    np.testing.assert_array_equal(
        drowned.chirps["box_t0_s"], clean_scene.chirps["box_t0_s"]
    )
    assert not np.any(drowned.chirps["visible"].to_numpy(dtype=bool))
    assert np.all(drowned.chirps["snr_db"].to_numpy(dtype=float) < -20)
    moderate = dataclasses.replace(loud, floor_modes=((-120.0, 0.0, 1.0),))
    audible = simulate_scene(_single_fish(library_dir, moderate), 3)
    assert np.all(audible.chirps["visible"].to_numpy(dtype=bool))


def test_overlapping_chirps_of_two_fish_get_their_own_boxes(library_dir: Path) -> None:
    config = small_config(
        library_dir,
        duration_s=8.0,
        noise=CLEAN,
        clicks=NO_CLICKS,
        population=PopulationConfig(n_fish=(2, 2), close_pair_prob=0.0),
        chirps=ChirpConfig(
            library=str(library_dir),
            rate_hz=(1.0, 1.0),
            response_prob=1.0,
            simultaneous_fraction=1.0,
            simultaneous_window_s=0.05,
        ),
    )
    scene = simulate_scene(config, 1)
    rows = [dict(r) for r in scene.chirps.to_dict("records")]
    pairs = [
        (a, b)
        for a, b in itertools.combinations(rows, 2)
        if a["emitter_id"] != b["emitter_id"]
        and abs(a["t_center_s"] - b["t_center_s"]) < 0.05
    ]
    assert pairs
    keys = ("box_t0_s", "box_t1_s", "box_f0_hz", "box_f1_hz")
    for a, b in pairs:
        box_a = tuple(a[k] for k in keys)
        box_b = tuple(b[k] for k in keys)
        assert box_a != box_b
        for row, box in ((a, box_a), (b, box_b)):
            assert box[2] <= row["baseline_hz"] <= box[3]
            assert box[0] <= row["t_peak_s"] <= box[1]


def test_box_is_independent_of_other_fish(library_dir: Path) -> None:
    config = small_config(library_dir, duration_s=5.0, noise=CLEAN, clicks=NO_CLICKS)
    scene = simulate_scene(config, 4)
    model = scene.model
    emitter = int(scene.chirps["emitter_id"].iloc[0])
    alone = SceneModel(
        samplerate_hz=model.samplerate_hz,
        n_samples=model.n_samples,
        electrode_xy_m=model.electrode_xy_m,
        channel_gain=model.channel_gain,
        shared_drift=model.shared_drift,
        fish=tuple(
            f if f.fish_id == emitter else dataclasses.replace(f, chirps=())
            for f in model.fish
        ),
        gain_update_hz=model.gain_update_hz,
        reference_xy_m=model.reference_xy_m,
    )
    for chirp_id in model.fish[emitter].chirp_ids:
        with_others = chirp_footprint(model, chirp_id)
        without_others = chirp_footprint(alone, chirp_id)
        assert with_others.box == without_others.box
        assert math.isnan(with_others.snr_db)  # no recording given: no visibility


def test_second_harmonic_box_when_in_band(library_dir: Path) -> None:
    scene = simulate_scene(_single_fish(library_dir, baseline_hz=600.0), 3)
    harmonic = scene.chirps["harmonic_box_f0_hz"].to_numpy(dtype=float)
    top = scene.chirps["box_f1_hz"].to_numpy(dtype=float)
    assert np.all(np.isfinite(harmonic))
    assert np.all(harmonic >= 1.5 * 600.0)
    assert np.all(harmonic > top)


def test_scene_footprints_match_the_table(clean_scene: Scene, tmp_path: Path) -> None:
    from gridtools.sim.scene import reference_spectrogram, write_scene

    path = write_scene(clean_scene, tmp_path / "scene")
    footprints = scene_footprints(path, reference_spectrogram(clean_scene.config))
    keys = ("box_t0_s", "box_t1_s", "box_f0_hz", "box_f1_hz")
    for fp, row in zip(footprints, clean_scene.chirps.to_dict("records"), strict=True):
        assert fp.box == pytest.approx(tuple(row[k] for k in keys))
        assert fp.visible == row["visible"]


@pytest.mark.slow
def test_footprints_of_a_60s_scene_take_seconds(library_dir: Path) -> None:
    config = small_config(
        library_dir,
        duration_s=60.0,
        population=PopulationConfig(n_fish=(4, 4)),
        chirps=ChirpConfig(library=str(library_dir), rate_hz=(0.4, 0.5)),
        footprint=dataclasses.replace(SceneConfig().footprint, enabled=False),
    )
    scene = simulate_scene(config, 0)
    start = time.perf_counter()
    footprints = scene_footprints(scene, HannSpectrogram())
    elapsed = time.perf_counter() - start
    assert len(footprints) >= 80
    assert elapsed < 10.0, f"{len(footprints)} footprints took {elapsed:.1f} s"
