"""Hybrid scenes: simulated fish on a real background segment.

Needs a background library (``$GRIDTOOLS_TEST_BACKGROUND``, default the
development machine's ``/mnt/data1/chirp_v2/background/v1``); skipped
without it. The chirp templates are the synthetic test library.
"""

import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gridtools.io import load_tracks, open_recording
from gridtools.sim.config import GridConfig, HybridConfig, SceneConfig
from gridtools.sim.hybrid import (
    REAL_FISH_IDENT_OFFSET,
    background_library,
    candidates,
    load_segment,
    silent_channels,
)
from gridtools.sim.scene import (
    Scene,
    plan_scene,
    render_background,
    render_fish,
    simulate_scene,
    write_scene,
)

from .conftest import small_config

LIBRARY = Path(
    os.environ.get("GRIDTOOLS_TEST_BACKGROUND", "/mnt/data1/chirp_v2/background/v1")
)


def _hybrid(**kwargs: object) -> HybridConfig:
    return HybridConfig(library=str(LIBRARY), **kwargs)  # type: ignore[arg-type]


pytestmark = pytest.mark.skipif(
    not (LIBRARY / "manifest.parquet").exists(),
    reason=f"background library {LIBRARY} not available",
)


def hybrid_config(
    library: Path,
    *,
    n_electrodes: int | None = None,
    datasets: tuple[str, ...] = ("competition2019",),
    **background: object,
) -> SceneConfig:
    """Always hybrid; ``n_electrodes=None`` takes the segment's whole grid."""
    grid = (
        GridConfig(real_layout_prob=1.0)
        if n_electrodes is None
        else GridConfig(real_layout_prob=0.0, n_electrodes=(n_electrodes, n_electrodes))
    )
    config = small_config(library, duration_s=4.0, grid=grid)
    return replace(
        config,
        background=_hybrid(prob=1.0, datasets=datasets, **background),
    )


def _manifest_row(segment_id: str) -> dict[str, object]:
    m = background_library(str(LIBRARY)).manifest
    return m[m["segment_id"] == segment_id].iloc[0].to_dict()


@pytest.fixture(scope="module")
def hybrid_scene(library_dir: Path) -> Scene:
    return simulate_scene(hybrid_config(library_dir, n_electrodes=4), 3)


def test_single_electrode_uses_one_segment_channel(library_dir: Path) -> None:
    for seed in range(3):
        plan = plan_scene(hybrid_config(library_dir, n_electrodes=1), seed)
        background = plan.background
        assert background is not None
        segment = background.segment
        (channel,) = segment.channels
        row = _manifest_row(segment.segment_id)
        assert plan.layout.kind == "background"
        assert plan.model.n_electrodes == 1
        assert segment.data.shape == (plan.model.n_samples, 1)
        np.testing.assert_array_equal(
            plan.model.electrode_xy_m[0],
            np.asarray(row["electrode_positions_m"])[channel],
        )
        bad = set(row["dead_channels"]) | set(row["noisy_channels"])  # type: ignore[arg-type]
        assert channel not in bad


def test_subset_keeps_segment_channels_and_positions(
    hybrid_scene: Scene, tmp_path: Path
) -> None:
    plan = hybrid_scene.plan
    background = plan.background
    assert background is not None
    channels = background.segment.channels
    row = _manifest_row(background.segment.segment_id)
    assert len(channels) == 4 < int(row["n_electrodes"])  # type: ignore[arg-type]
    assert list(channels) == list(range(channels[0], channels[0] + 4))  # contiguous
    expected = np.asarray(row["electrode_positions_m"])[list(channels)]
    np.testing.assert_allclose(plan.model.electrode_xy_m, expected)
    assert not plan.layout.regular  # a subset is written with electrodes.csv
    out = write_scene(hybrid_scene, tmp_path / "scene", plots=False)
    rec = open_recording(out / "recording.wav")
    assert rec.geometry is not None
    np.testing.assert_allclose(rec.geometry.positions_m, expected)
    info = json.loads((out / "scene.json").read_text())
    assert info["background"]["channels"] == list(channels)
    assert info["background"]["segment_id"] == background.segment.segment_id
    assert info["background"]["positions_nominal"] is True  # competition grid


def test_whole_grid_when_the_layout_is_real(library_dir: Path) -> None:
    plan = plan_scene(hybrid_config(library_dir), 0)
    assert plan.background is not None
    row = _manifest_row(plan.background.segment.segment_id)
    assert plan.model.n_electrodes == int(row["n_electrodes"])  # type: ignore[arg-type]
    assert plan.background.record()["whole_grid"]
    assert (plan.rows, plan.columns) == (int(row["n_rows"]), int(row["n_columns"]))  # type: ignore[arg-type]


def test_electrode_count_is_capped_by_the_segment(library_dir: Path) -> None:
    plan = plan_scene(hybrid_config(library_dir, n_electrodes=16), 1)
    assert plan.background is not None
    row = _manifest_row(plan.background.segment.segment_id)
    assert plan.model.n_electrodes == int(row["n_electrodes"]) < 16  # type: ignore[arg-type]


def test_synthetic_noise_hum_and_clicks_are_off(hybrid_scene: Scene) -> None:
    assert len(hybrid_scene.clicks) == 0
    levels = hybrid_scene.plan.levels
    assert levels["background"] == "real"
    assert levels["noise_sd"] is None
    assert levels["hum_amplitude_v"] is None


def test_field_segment_is_resampled_to_the_scene_rate(library_dir: Path) -> None:
    config = hybrid_config(library_dir, n_electrodes=3, datasets=("field_iriri2026",))
    scene = simulate_scene(config, 2)
    background = scene.plan.background
    assert background is not None
    assert background.source_samplerate_hz == 48_000.0
    assert background.segment.samplerate_hz == config.samplerate_hz
    record = background.record()
    assert record["resampled"]
    assert record["source_n_samples"] == round(4.0 * 48_000)
    # the field logger's int16 full scale is mapped onto volts
    assert record["natural_scale"] != 1.0
    assert scene.recording.shape == (scene.model.n_samples, 3)
    # fish-free: no real fish
    assert len(scene.real_fish) == 0
    assert set(scene.tracks.ids.tolist()) == set(range(len(scene.model.fish)))
    # the excerpt matches the source spectrum below the old Nyquist rate:
    # same power in 500-1500 Hz as the source excerpt (within 0.5 dB)
    lib = background_library(str(LIBRARY))
    source = lib.segment(
        background.segment.segment_id,
        offset_s=record["offset_s"],
        duration_s=4.0,
        channels=list(background.segment.channels),
    )

    def band_power(x: np.ndarray, fs: float) -> float:
        spectrum = np.abs(np.fft.rfft(x - x.mean(axis=0), axis=0)) ** 2 / fs
        f = np.fft.rfftfreq(x.shape[0], 1.0 / fs)
        return float(spectrum[(f >= 500) & (f <= 1500)].sum() / x.shape[0])

    p_scene = band_power(
        np.asarray(background.segment.data, dtype=np.float64), 20_000.0
    )
    p_source = band_power(np.asarray(source.data, dtype=np.float64), 48_000.0)
    assert 10 * math.log10(p_scene / p_source) == pytest.approx(0.0, abs=0.5)


def test_real_fish_join_the_tracks(hybrid_scene: Scene, tmp_path: Path) -> None:
    scene = hybrid_scene
    n_sim = len(scene.model.fish)
    real = scene.real_fish
    assert len(real) >= 1
    assert np.all(real["ident"].to_numpy() >= REAL_FISH_IDENT_OFFSET)
    ids = set(scene.tracks.ids.tolist())
    assert ids == set(range(n_sim)) | set(real["ident"].tolist())
    # only simulated fish chirp
    assert np.all(scene.chirps["emitter_id"].to_numpy() < n_sim)
    out = write_scene(scene, tmp_path / "scene", plots=False)
    info = json.loads((out / "scene.json").read_text())
    assert [f["real_fish"] for f in info["real_fish"]] == [True] * len(real)
    assert [f["ident"] for f in info["real_fish"]] == real["ident"].tolist()
    assert all(not f["real_fish"] for f in info["fish"])
    assert info["background"]["n_real_fish"] == len(real)
    tracks = load_tracks(out)
    assert set(tracks.ids.tolist()) == ids


def test_real_fish_power_matches_the_recording(hybrid_scene: Scene) -> None:
    """The merged real-fish sign_v (tracker's, offset) agrees with the scene's
    definition measured on the recording at the tracked frequencies."""
    from gridtools.sim.scene import sign_v_powers

    scene = hybrid_scene
    fs = scene.samplerate_hz
    nfft = scene.config.tracks.power_nfft(fs)
    diffs = []
    for ident in scene.real_fish["ident"]:
        track = scene.tracks.track(int(ident))
        keep = (track.times_s > 0.5) & (
            track.times_s < scene.model.n_samples / fs - 0.5
        )
        if not keep.any():
            continue
        measured = sign_v_powers(
            scene.recording,
            fs,
            track.times_s[keep],
            track.freqs_hz[keep][None, :],
            nfft,
        )[0]
        assert track.powers_db is not None
        merged = track.powers_db[keep]
        best = np.argmax(np.nanmedian(merged, axis=0))
        diffs.append(float(np.nanmedian(measured[:, best] - merged[:, best])))
    assert diffs
    assert np.median(diffs) == pytest.approx(0.0, abs=3.0)


def test_recording_is_fish_plus_scaled_background(
    hybrid_scene: Scene, tmp_path: Path
) -> None:
    scene = hybrid_scene
    background = render_background(scene)
    total = background.astype(np.float64)
    for i in range(len(scene.model.fish)):
        total += render_fish(scene, i).astype(np.float64)
    scale = float(np.max(np.abs(scene.recording)))
    np.testing.assert_allclose(scene.recording, total, rtol=0, atol=2e-6 * scale)
    assert scene.plan.background is not None
    np.testing.assert_array_equal(background, scene.plan.background.scaled())
    # from the directory: the segment is reloaded from the library record
    out = write_scene(scene, tmp_path / "scene", plots=False)
    np.testing.assert_array_equal(
        render_background(out, 1.0, 2.0), background[20_000:40_000]
    )
    np.testing.assert_array_equal(
        render_background(out, electrodes=[2]), background[:, [2]]
    )


def test_snr_sets_the_background_scale(library_dir: Path) -> None:
    def plan(snr: float, **kw: Any):
        config = hybrid_config(
            library_dir,
            n_electrodes=4,
            snr_hard_db=(snr, snr),
            snr_typical_db=(snr, snr),
            max_real_gap_db=None,
            **kw,
        )
        return plan_scene(config, 5)

    high, low = plan(70.0), plan(20.0)
    assert high.background is not None
    assert low.background is not None
    # same draws apart from the scaling
    assert high.background.segment.segment_id == low.background.segment.segment_id
    ratio_high = high.background.scale / high.background.fish_gain
    ratio_low = low.background.scale / low.background.fish_gain
    assert 20 * math.log10(ratio_low / ratio_high) == pytest.approx(50.0, abs=1e-6)
    # nothing is amplified beyond its own level
    for p in (high, low):
        assert p.background is not None
        assert p.background.scale <= p.background.natural_scale * (1 + 1e-12)
        assert p.background.fish_gain <= 1.0
    # unless asked to
    amplified = plan(20.0, amplify_background=True)
    assert amplified.background is not None
    assert amplified.background.fish_gain == 1.0
    assert amplified.background.scale == pytest.approx(ratio_low)


def test_real_fish_gap_is_capped(library_dir: Path) -> None:
    config = hybrid_config(
        library_dir, n_electrodes=4, snr_hard_db=(10.0, 10.0), snr_hard_fraction=1.0
    )
    for seed in range(4):
        background = plan_scene(config, seed).background
        assert background is not None
        record = background.record()
        if math.isfinite(record["real_snr_max_db"]):
            assert record["min_snr_db"] == pytest.approx(
                record["real_snr_max_db"] - 30.0
            )
            assert background.snr_db >= record["min_snr_db"] - 1e-9


def test_visibility_is_measured_against_the_real_background(library_dir: Path) -> None:
    """A louder background (lower SNR) lowers every chirp's footprint SNR."""

    def scene(snr: float) -> Scene:
        config = hybrid_config(
            library_dir,
            n_electrodes=4,
            snr_hard_db=(snr, snr),
            snr_typical_db=(snr, snr),
            max_real_gap_db=None,
            amplify_background=True,  # same fish, only the background changes
        )
        return simulate_scene(config, 5)

    quiet, loud = scene(70.0), scene(15.0)
    assert len(quiet.chirps) > 0
    np.testing.assert_array_equal(
        quiet.chirps["chirp_id"].to_numpy(), loud.chirps["chirp_id"].to_numpy()
    )
    drop = quiet.chirps["snr_db"].to_numpy() - loud.chirps["snr_db"].to_numpy()
    assert np.all(drop > 0)
    assert np.median(drop) > 10.0
    n_quiet = int(quiet.chirps["visible"].to_numpy().sum())
    assert n_quiet >= int(loud.chirps["visible"].to_numpy().sum())


def test_reproducible_under_the_seed(library_dir: Path) -> None:
    config = hybrid_config(library_dir, n_electrodes=5)
    a, b = simulate_scene(config, 11), simulate_scene(config, 11)
    np.testing.assert_array_equal(a.recording, b.recording)
    assert a.plan.background is not None
    assert b.plan.background is not None
    assert a.plan.background.record() == b.plan.background.record()
    c = simulate_scene(config, 12)
    assert c.plan.background is not None
    assert c.plan.background.record() != a.plan.background.record()


def test_hybrid_share_follows_prob(library_dir: Path) -> None:
    config = replace(
        hybrid_config(library_dir, n_electrodes=1),
        background=_hybrid(prob=0.5, datasets=("competition2019",)),
    )
    hybrid = [plan_scene(config, seed).background is not None for seed in range(16)]
    assert 0 < sum(hybrid) < 16


def test_longer_scenes_are_refused(library_dir: Path) -> None:
    config = replace(hybrid_config(library_dir, n_electrodes=1), duration_s=61.0)
    with pytest.raises(ValueError, match="limited to the background segments"):
        plan_scene(config, 0)
    synthetic = replace(config, background=HybridConfig(prob=0.0))
    assert plan_scene(synthetic, 0).background is None


def test_source_weights() -> None:
    m, w = candidates(_hybrid())
    unscreened = ~m["screened"].to_numpy(dtype=bool)
    assert unscreened.any()
    assert np.all(w[unscreened] == pytest.approx(0.3))
    assert np.all(w[~unscreened] == 1.0)
    m2, w2 = candidates(_hybrid(dataset_weights=(("field_iriri2026", 0.5),)))
    field = (m2["dataset"] == "field_iriri2026").to_numpy()
    assert np.all(w2[field] == 0.5)
    with pytest.raises(ValueError, match="not in"):
        candidates(_hybrid(datasets=("nope",)))


def test_silent_channels_are_never_chosen_and_not_in_the_level() -> None:
    lib = background_library(str(LIBRARY))
    flona = lib.manifest[lib.manifest["dataset"] == "field_flona2025"]
    segment_id = next(
        (s for s in flona["segment_id"] if silent_channels(lib, str(s))), None
    )
    if segment_id is None:
        pytest.skip("no segment with silent channels")
    silent = silent_channels(lib, segment_id)
    config = _hybrid(datasets=("field_flona2025",))
    rng = np.random.default_rng(0)
    from gridtools.sim.hybrid import _choose

    row = lib.manifest[lib.manifest["segment_id"] == segment_id].iloc[0].to_dict()
    for k in (1, 4, 8, 12):
        chosen = _choose(rng, row, k, config, silent)
        assert chosen is not None
        assert not set(chosen) & set(silent)
    whole = load_segment(
        lib,
        segment_id,
        offset_s=0.0,
        source_n_samples=4800,
        channels=None,
        samplerate_hz=48_000.0,
        n_samples=4800,
    )
    assert set(silent) <= set(whole.dead_channels)
    assert whole.level > float(row["level"])  # zeros no longer pull the median down
