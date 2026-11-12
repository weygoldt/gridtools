"""Sampling-rate agnostic scenes: the same seed at 20, 40 and 48 kHz."""

from pathlib import Path

import numpy as np
import pytest

from gridtools.sim.config import SceneConfig
from gridtools.sim.footprint import HannSpectrogram
from gridtools.sim.scene import Scene, plan_scene, simulate_scene

from .conftest import small_config


def _scene(library: Path, rate: float) -> Scene:
    return simulate_scene(small_config(library, samplerate_hz=rate), 7)


@pytest.fixture(scope="module")
def reference(library_dir: Path) -> Scene:
    return _scene(library_dir, 20_000.0)


@pytest.mark.parametrize("rate", [40_000.0, 48_000.0])
def test_same_seed_same_scene_at_any_rate(
    library_dir: Path, reference: Scene, rate: float
) -> None:
    other = _scene(library_dir, rate)
    assert other.recording.shape[0] == round(6.0 * rate)
    a, b = reference.chirps, other.chirps
    assert len(a) == len(b)
    np.testing.assert_array_equal(a["template_id"], b["template_id"])
    np.testing.assert_array_equal(a["emitter_id"], b["emitter_id"])
    tolerance_s = 1 / 20_000 + 1 / rate  # centres snap to each rate's samples
    np.testing.assert_allclose(a["t_center_s"], b["t_center_s"], atol=tolerance_s)
    np.testing.assert_allclose(a["excursion_hz"], b["excursion_hz"], rtol=5e-3)
    np.testing.assert_allclose(a["baseline_hz"], b["baseline_hz"], atol=1e-3)
    # footprint boxes agree within one frame and one frequency bin (plus the
    # small offset between the two rates' grids: round(0.2048 * fs) samples),
    # except for chirps whose footprint is cut by the scene's ends
    frame = SceneConfig().footprint.hop_s
    bin_hz = 1.0 / SceneConfig().footprint.window_s
    reach = SceneConfig().footprint.window_s + 0.05
    inside = (a["t_support_start_s"].to_numpy(dtype=float) - reach >= 0.0) & (
        a["t_support_stop_s"].to_numpy(dtype=float) + reach <= 6.0
    )
    assert inside.sum() >= 5
    for edge in ("box_t0_s", "box_t1_s"):
        np.testing.assert_allclose(
            a[edge].to_numpy()[inside], b[edge].to_numpy()[inside], atol=frame + 1e-4
        )
    for edge in ("box_f0_hz", "box_f1_hz"):
        np.testing.assert_allclose(
            a[edge].to_numpy()[inside], b[edge].to_numpy()[inside], atol=bin_hz + 0.1
        )
    # tracks and their frame grid are the same; sign_v nearly so
    np.testing.assert_allclose(
        reference.tracks.freqs_hz, other.tracks.freqs_hz, atol=1e-6
    )
    assert reference.tracks.powers_db is not None
    assert other.tracks.powers_db is not None
    fish = reference.tracks.powers_db > -85  # where fish dominate the noise
    np.testing.assert_allclose(
        reference.tracks.powers_db[fish], other.tracks.powers_db[fish], atol=1.5
    )
    # the noise floor is per Hz: the same at every rate
    assert other.plan.levels["noise_floor_db_v2_hz"] == pytest.approx(
        reference.plan.levels["noise_floor_db_v2_hz"]
    )
    assert other.plan.levels["noise_sd"] == pytest.approx(
        reference.plan.levels["noise_sd"] * np.sqrt(rate / 20_000.0)
    )


def test_spectrogram_resolution_is_physical() -> None:
    fn = HannSpectrogram(window_s=0.2048, hop_s=0.005)
    for rate in (20_000.0, 48_000.0):
        x = np.sin(2 * np.pi * 800.0 * np.arange(round(rate)) / rate)[:, None]
        power, freqs, times = fn(x, rate, 0.0)
        assert freqs[1] - freqs[0] == pytest.approx(1 / 0.2048, rel=1e-3)
        assert times[1] - times[0] == pytest.approx(0.005, rel=1e-3)
        assert abs(freqs[np.argmax(power[:, 0])] - 800.0) < 5.0


def test_rate_drawn_from_choices(library_dir: Path) -> None:
    config = small_config(library_dir, samplerate_choices_hz=(20_000.0, 48_000.0))
    rates = {plan_scene(config, seed).config.samplerate_hz for seed in range(12)}
    assert rates == {20_000.0, 48_000.0}
    plan = plan_scene(config, 0)
    assert plan.config.samplerate_choices_hz == ()
    assert plan.model.samplerate_hz == plan.config.samplerate_hz
