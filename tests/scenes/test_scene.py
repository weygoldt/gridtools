"""Scene assembly: determinism, seeding, writers, ground-truth tables."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gridtools.io import load_tracks, open_recording
from gridtools.sim.config import (
    ChirpConfig,
    ClickConfig,
    NoiseConfig,
    PlotConfig,
    PopulationConfig,
    SceneConfig,
)
from gridtools.sim.scene import (
    CHIRP_COLUMNS,
    Scene,
    fish_power_db,
    plan_scene,
    render_fish,
    simulate_scene,
    write_scene,
)

from .conftest import small_config

FS = 20_000.0
TRACK_FILES = ("times.npy", "fund_v.npy", "ident_v.npy", "idx_v.npy", "sign_v.npy")


def _records(table: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): v for k, v in r.items()} for r in table.to_dict("records")]


def test_same_seed_gives_bit_identical_files(
    config: SceneConfig, scene_path: Path, tmp_path: Path
) -> None:
    again = write_scene(simulate_scene(config, 7), tmp_path / "again")
    for name in ("recording.wav", "chirps.csv", "clicks.csv", *TRACK_FILES):
        assert (again / name).read_bytes() == (scene_path / name).read_bytes(), name


def test_different_seed_differs(config: SceneConfig, scene: Scene) -> None:
    other = simulate_scene(config, 8)
    assert not np.array_equal(other.recording, scene.recording)
    assert not other.chirps.equals(scene.chirps)


def test_noise_config_leaves_fish_untouched(config: SceneConfig, scene: Scene) -> None:
    noisy = replace(
        config,
        noise=NoiseConfig(snr_db=(-20.0, -20.0), exponent=(0.0, 0.0), dead_prob=0.0),
        clicks=ClickConfig(rate_hz=(5.0, 5.0)),
    )
    other = simulate_scene(noisy, 7)
    pd.testing.assert_frame_equal(other.chirps, scene.chirps)
    assert not np.array_equal(other.recording, scene.recording)
    for i in range(len(scene.model.fish)):
        # channel gains include dead electrodes, so compare before them
        a, b = other.model, scene.model
        np.testing.assert_array_equal(
            a.waveform(i, 0, a.n_samples), b.waveform(i, 0, b.n_samples)
        )
        np.testing.assert_array_equal(a.gains(i, 0, 5000), b.gains(i, 0, 5000))


def test_chirp_config_leaves_paths_and_noise_untouched(
    config: SceneConfig, scene: Scene
) -> None:
    other = simulate_scene(
        replace(config, chirps=replace(config.chirps, rate_hz=(0.1, 0.1))), 7
    )
    for a, b in zip(other.model.fish, scene.model.fish, strict=True):
        np.testing.assert_array_equal(a.path.xy_m, b.path.xy_m)
        assert a.baseline_hz == b.baseline_hz
    assert other.plan.levels == scene.plan.levels
    assert len(other.chirps) < len(scene.chirps)


def test_recording_round_trips_through_gridtools_io(
    scene: Scene, scene_path: Path
) -> None:
    recording = open_recording(scene_path)
    assert recording.samplerate_hz == FS
    assert recording.n_channels == 9
    assert recording.data.dtype == np.float32
    np.testing.assert_array_equal(
        recording.read(0, recording.n_samples), scene.recording
    )
    assert recording.geometry is not None
    np.testing.assert_array_equal(
        recording.geometry.positions_m, scene.model.electrode_xy_m
    )


def test_wav_opens_in_soundfile_and_audioio(scene: Scene, scene_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    data, rate = soundfile.read(scene_path / "recording.wav", dtype="float32")
    assert rate == FS
    np.testing.assert_array_equal(data, scene.recording)
    info = soundfile.info(scene_path / "recording.wav")
    assert info.format == "WAVEX"
    assert info.subtype == "FLOAT"
    audioio = pytest.importorskip("audioio")
    data, rate = audioio.load_audio(str(scene_path / "recording.wav"))
    assert rate == FS
    assert data.shape == scene.recording.shape
    np.testing.assert_array_equal(data.astype(np.float32), scene.recording)


def test_tracks_round_trip_and_follow_the_baselines(
    scene: Scene, scene_path: Path
) -> None:
    tracks = load_tracks(scene_path)
    np.testing.assert_array_equal(tracks.freqs_hz, scene.tracks.freqs_hz)
    assert tracks.powers_db is not None
    assert tracks.powers_db.shape[1] == 9
    assert list(tracks.ids) == list(range(len(scene.model.fish)))
    step = np.diff(tracks.times_s)
    np.testing.assert_allclose(step, 0.1229)
    assert tracks.times_s[0] == 0.0
    assert tracks.times_s[-1] >= (scene.model.n_samples - 1) / FS
    for row in _records(scene.chirps):
        f = tracks.track(row["emitter_id"]).frequency_at(row["t_center_s"])[0]
        # linear interpolation of a smooth drift between 8-Hz frames
        assert f == pytest.approx(row["baseline_hz"], abs=0.05)


def test_sign_v_is_the_fundamental_power(scene: Scene) -> None:
    model = scene.model
    tracks = scene.tracks
    assert tracks.powers_db is not None
    track = tracks.track(0)
    assert track.powers_db is not None
    t = track.times_s[10]
    fish = model.fish[0]
    gains = model.gain_track(0).evaluate([t])[0]
    amplitude = (
        model.channel_gain * gains * fish.amplitude * fish.harmonics.amplitudes[0]
    )
    np.testing.assert_allclose(track.powers_db[10], fish_power_db(amplitude, FS, 16384))
    # a sinusoid of amplitude 1 centred on a bin: 1/2 / ENBW
    assert fish_power_db(1.0, FS, 16384) == pytest.approx(
        10 * np.log10(0.5 / (1.5 * FS / 16384))
    )


def test_chirp_table_is_consistent(scene: Scene) -> None:
    table = scene.chirps
    assert tuple(table.columns) == CHIRP_COLUMNS
    assert table["chirp_id"].tolist() == list(range(len(table)))
    assert np.all(np.diff(table["t_center_s"]) >= 0)
    np.testing.assert_allclose(
        table["f_high_hz"], table["baseline_hz"] + table["excursion_hz"]
    )
    np.testing.assert_allclose(
        table["f_low_hz"], table["baseline_hz"] - table["undershoot_hz"]
    )
    assert np.all(table["t_onset_s"] <= table["t_peak_s"])
    assert np.all(table["t_peak_s"] <= table["t_offset_s"])
    assert np.all(table["t_support_start_s"] <= table["t_onset_s"])
    assert np.all(table["t_offset_s"] <= table["t_support_stop_s"])
    assert set(table["family"]) <= {"small", "big"}
    model = scene.model
    records = _records(table)
    for row in records:
        chirp = model.chirps[row["chirp_id"]]
        assert chirp.fish_id == row["emitter_id"]
        assert row["dip"] == pytest.approx(chirp.dip)
        others = [r for r in records if r["emitter_id"] != row["emitter_id"]]
        dt = min(abs(r["t_center_s"] - row["t_center_s"]) for r in others)
        assert row["dt_nearest_other_chirp_s"] == pytest.approx(dt)
        assert row["overlaps_other_chirp"] == (dt <= 0.25)
        f_all = [
            model.base_frequency_at(j, [row["t_center_s"]])[0]
            for j in range(len(model.fish))
        ]
        df = [
            abs(f - row["baseline_hz"])
            for j, f in enumerate(f_all)
            if j != row["emitter_id"]
        ]
        assert row["min_df_hz"] == pytest.approx(min(df))
        if row["is_response"]:
            trigger = records[row["response_to"]]
            assert trigger["emitter_id"] != row["emitter_id"]
            assert 0 <= row["t_center_s"] - trigger["t_center_s"] <= 0.5 + 1 / FS
    assert any(r["overlaps_other_chirp"] for r in records)
    assert any(r["is_response"] for r in records)


def test_chirp_files_round_trip(scene: Scene, scene_path: Path) -> None:
    pd.testing.assert_frame_equal(
        pd.read_parquet(scene_path / "chirps.parquet"), scene.chirps
    )
    from_csv = pd.read_csv(scene_path / "chirps.csv")
    pd.testing.assert_frame_equal(from_csv, scene.chirps, check_dtype=False)


def test_scene_json_is_complete(scene: Scene, scene_path: Path) -> None:
    info = json.loads((scene_path / "scene.json").read_text())
    for key in (
        "seed",
        "config",
        "random",
        "versions",
        "templates",
        "grid",
        "recording",
        "levels",
        "channels",
        "fish",
        "model",
    ):
        assert key in info, key
    assert info["seed"] == 7
    assert info["duration_s"] == 6.0  # read by chirpdetector.evaluation
    assert "visible" not in scene.chirps.columns  # reserved for visibility labels
    assert SceneConfig.from_dict(info["config"]) == scene.config
    assert len(info["templates"]["manifest_sha256"]) == 64
    assert info["grid"]["electrode_xy_m"] == scene.model.electrode_xy_m.tolist()
    assert info["versions"]["gridtools"]
    assert len(info["fish"]) == len(scene.model.fish)
    for fish in info["fish"]:
        assert {"baseline_hz", "amplitude", "depth_m", "close_to", "n_chirps"} <= set(
            fish
        )
    paths = np.load(scene_path / "paths.npz")
    assert paths["xy_m"].shape[0] == len(scene.model.fish)


@pytest.mark.parametrize("seed", range(3))
def test_rendering_from_a_dir_needs_no_library(
    library_dir: Path, tmp_path: Path, seed: int
) -> None:
    scene = simulate_scene(small_config(library_dir, duration_s=3.0), seed)
    path = write_scene(scene, tmp_path / "scene")
    (path / "scene.json").write_text(
        (path / "scene.json").read_text().replace(str(library_dir), "/nonexistent")
    )
    for i in range(len(scene.model.fish)):
        np.testing.assert_array_equal(render_fish(path, i), render_fish(scene, i))


def test_close_pairs_and_crossings_occur_at_configured_rates(library_dir: Path) -> None:
    population = PopulationConfig(n_fish=(4, 4), close_pair_prob=0.5, crossing_prob=0.5)
    config = small_config(library_dir, duration_s=5.0, population=population)
    close = crossing = later = 0
    for seed in range(120):
        plan = plan_scene(config, seed)
        for fish in plan.fish_info[1:]:
            later += 1
            if fish["close_to"] >= 0:
                close += 1
                assert abs(fish["close_df_hz"]) <= 20.0
                crossing += fish["crossing"]
    assert abs(close / later - 0.5) < 4 * np.sqrt(0.25 / later)
    assert abs(crossing / close - 0.5) < 4 * np.sqrt(0.25 / close)


def test_crossing_ramps_cross_the_partner(library_dir: Path) -> None:
    population = PopulationConfig(n_fish=(2, 2), close_pair_prob=1.0, crossing_prob=1.0)
    config = small_config(library_dir, duration_s=30.0, population=population)
    for seed in range(10):
        plan = plan_scene(config, seed)
        model = plan.model
        t = np.linspace(0, 30.0, 301)
        df = model.base_frequency_at(1, t) - model.base_frequency_at(0, t)
        assert np.sign(df[0]) != np.sign(df[-1])


def test_spacing_and_rate_cap_hold_in_scenes(library_dir: Path) -> None:
    config = small_config(
        library_dir,
        duration_s=20.0,
        chirps=ChirpConfig(
            library=str(library_dir), rate_hz=(3.0, 3.0), response_prob=0.8
        ),
    )
    for seed in range(5):
        plan = plan_scene(config, seed)
        for fish in plan.model.fish:
            centres = np.array([c.center_index for c in fish.chirps])
            assert centres.size <= 20
            assert np.all(np.diff(np.sort(centres)) >= 0.2 * FS)


def test_edge_chirps_are_truncated_and_render(library_dir: Path) -> None:
    config = small_config(
        library_dir,
        duration_s=2.0,
        chirps=ChirpConfig(library=str(library_dir), rate_hz=(3.0, 3.0)),
    )
    for seed in range(50):
        scene = simulate_scene(config, seed)
        truncated = [r for r in _records(scene.chirps) if r["truncated"]]
        if truncated:
            break
    else:
        pytest.fail("no truncated chirp in 50 seeds")
    row = truncated[0]
    full = render_fish(scene, int(row["emitter_id"]))
    without = render_fish(scene, int(row["emitter_id"]), chirps=[int(row["chirp_id"])])
    assert np.abs(full - without).max() > 0


def test_clipping(library_dir: Path) -> None:
    config = small_config(
        library_dir, duration_s=2.0, noise=NoiseConfig(clip_limit=0.01)
    )
    scene = simulate_scene(config, 1)
    assert np.abs(scene.recording).max() <= 0.01


def test_measured_amplitude_mode(library_dir: Path) -> None:
    chirps = ChirpConfig(
        library=str(library_dir), amplitude_mode="measured", rate_hz=(1, 1)
    )
    scene = simulate_scene(small_config(library_dir, duration_s=4.0, chirps=chirps), 2)
    assert len(scene.chirps)
    assert np.all(scene.chirps["template_id"] % 2 == 0)  # amp_usable ones
    for chirp in scene.model.chirps.values():
        assert chirp.dip == pytest.approx(1.0 - chirp.amp_factor.min())
    assert set(scene.chirps["amplitude_mode"]) == {"measured"}


def test_seed_must_be_non_negative(config: SceneConfig) -> None:
    with pytest.raises(ValueError, match="seed"):
        plan_scene(config, -1)


@pytest.mark.slow
def test_default_config_with_real_library(tmp_path: Path) -> None:
    config = SceneConfig(duration_s=10.0, plot=PlotConfig(enabled=False))
    if not Path(config.chirps.library).exists():
        pytest.skip("template library not available")
    scene = simulate_scene(config, 0)
    fish = sum(render_fish(scene, i) for i in range(len(scene.model.fish)))
    from gridtools.sim.scene import render_background

    np.testing.assert_allclose(
        scene.recording,
        fish + render_background(scene),
        atol=8 * np.finfo(np.float32).eps * np.abs(scene.recording).max(),
        rtol=0,
    )
    write_scene(scene, tmp_path / "real")
