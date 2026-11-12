"""Tests for chunked mixing into a multi-electrode recording."""

import numpy as np
import pytest

from gridtools.sim.eod import synthesize_eod
from gridtools.sim.field import DipoleParams, GainTrack, gain_track
from gridtools.sim.mix import Renderable, mix
from gridtools.sim.movement import Arena, simulate_path
from gridtools.sim.noise import (
    ClipParams,
    ColoredNoiseParams,
    colored_noise,
    powerline_hum,
)

FS_HZ = 20_000.0
N = 40_000  # 2 s
ARENA = Arena(-0.25, 1.25, -0.25, 1.25)
ELECTRODES = np.array([[x, y] for y in (0.0, 0.5, 1.0) for x in (0.0, 0.5, 1.0)])


def _fish(seed: int, baseline_hz: float) -> tuple[np.ndarray, GainTrack]:
    rng = np.random.default_rng(seed)
    path = simulate_path(rng, N / FS_HZ, ARENA)
    track = gain_track(path.times_s, path.xy_m, path.heading_rad, ELECTRODES)
    eod = synthesize_eod(np.full(N, baseline_hz), 1.0, FS_HZ)
    return eod, track


@pytest.fixture(scope="module")
def scene() -> dict[str, object]:
    eod_a, track_a = _fish(1, 650.0)
    eod_b, track_b = _fish(2, 820.0)
    noise = colored_noise(
        np.random.default_rng(3), N, 9, FS_HZ, ColoredNoiseParams(sd=1e-3)
    )
    hum = powerline_hum(np.random.default_rng(4), N, 9, FS_HZ)
    return {
        "eod_a": eod_a,
        "eod_b": eod_b,
        "track_a": track_a,
        "track_b": track_b,
        "noise": noise,
        "hum": hum,
    }


def test_single_fish_with_full_rate_gains_is_outer_product() -> None:
    eod = np.sin(np.arange(1000) / 7.0).astype(np.float32)
    gains = np.random.default_rng(0).normal(size=(1000, 3)).astype(np.float32)
    out = mix([eod], [gains], samplerate_hz=FS_HZ)
    assert out.dtype == np.float32
    assert out.shape == (1000, 3)
    np.testing.assert_allclose(out, eod[:, None] * gains, rtol=1e-6, atol=1e-7)


def test_mix_is_linear(scene: dict) -> None:
    a, b = scene["eod_a"], scene["eod_b"]
    ta, tb = scene["track_a"], scene["track_b"]
    both = mix([a, b], [ta, tb], samplerate_hz=FS_HZ)
    only_a = mix([a], [ta], samplerate_hz=FS_HZ)
    only_b = mix([b], [tb], samplerate_hz=FS_HZ)
    np.testing.assert_allclose(both, only_a + only_b, rtol=1e-5, atol=1e-7)
    summed = mix([a + b], [ta], samplerate_hz=FS_HZ)
    b_on_a = mix([b], [ta], samplerate_hz=FS_HZ)
    np.testing.assert_allclose(summed, only_a + b_on_a, rtol=1e-5, atol=1e-7)
    doubled = mix([2.0 * a], [ta], samplerate_hz=FS_HZ)
    np.testing.assert_allclose(doubled, 2.0 * only_a, rtol=1e-6, atol=1e-7)
    with_noise = mix([a], [ta], samplerate_hz=FS_HZ, additive=[scene["noise"]])
    np.testing.assert_allclose(
        with_noise, only_a + scene["noise"], rtol=1e-5, atol=1e-7
    )


@pytest.mark.parametrize(
    ("chunk", "n_samples"), [(1, 500), (7, 5000), (999, N), (4096, N), (N, N)]
)
def test_chunked_equals_unchunked(scene: dict, chunk: int, n_samples: int) -> None:
    kwargs = {
        "samplerate_hz": FS_HZ,
        "additive": [scene["noise"][:n_samples], scene["noise"][::-1][:n_samples]],
        "channel_gain": np.linspace(0.8, 1.2, 9),
        "clip": ClipParams(limit=0.05),
    }
    eods = [scene["eod_a"][:n_samples], scene["eod_b"][:n_samples]]
    tracks = [scene["track_a"], scene["track_b"]]
    reference = mix(eods, tracks, chunk_samples=n_samples, **kwargs)
    chunked = mix(eods, tracks, chunk_samples=chunk, **kwargs)
    np.testing.assert_array_equal(chunked, reference)
    if n_samples == N:  # hum (a Renderable) is rendered per chunk as well
        with_hum = {**kwargs, "additive": [scene["hum"]]}
        np.testing.assert_array_equal(
            mix(eods, tracks, chunk_samples=chunk, **with_hum),
            mix(eods, tracks, chunk_samples=N, **with_hum),
        )


def test_gain_track_matches_full_rate_gains(scene: dict) -> None:
    track = scene["track_a"]
    exact = track.evaluate(np.arange(N) / FS_HZ)
    via_array = mix([scene["eod_a"]], [exact], samplerate_hz=FS_HZ)
    per_sample = mix(
        [scene["eod_a"]], [track], samplerate_hz=FS_HZ, gain_update_hz=None
    )
    np.testing.assert_array_equal(per_sample, via_array)
    # Default: exact spline every 20 samples (1 kHz), linear in between.
    knots = mix([scene["eod_a"]], [track], samplerate_hz=FS_HZ)
    scale = np.abs(via_array).max(axis=0)
    # Linear interpolation over 1 ms: error ~1e-5 of each electrode's peak.
    assert np.max(np.abs(knots - via_array) / scale) < 1e-4


@pytest.mark.parametrize("gain_update_hz", [None, 1000.0, 333.0])
def test_threads_give_identical_results(
    scene: dict, gain_update_hz: float | None
) -> None:
    kwargs = {
        "samplerate_hz": FS_HZ,
        "additive": [scene["noise"], scene["hum"]],
        "channel_gain": np.linspace(0.8, 1.2, 9),
        "gain_update_hz": gain_update_hz,
        "chunk_samples": 3001,
    }
    eods = [scene["eod_a"], scene["eod_b"]]
    tracks = [scene["track_a"], scene["track_b"]]
    serial = mix(eods, tracks, workers=1, **kwargs)
    threaded = mix(eods, tracks, workers=4, **kwargs)
    np.testing.assert_array_equal(threaded, serial)


def test_channel_gain_and_clipping(scene: dict) -> None:
    base = mix([scene["eod_a"]], [scene["track_a"]], samplerate_hz=FS_HZ)
    gains = np.arange(1, 10, dtype=np.float64)
    scaled = mix(
        [scene["eod_a"]], [scene["track_a"]], samplerate_hz=FS_HZ, channel_gain=gains
    )
    np.testing.assert_allclose(scaled, base * gains, rtol=1e-6, atol=1e-8)
    limit = 0.5 * float(np.abs(scaled).max())
    clipped = mix(
        [scene["eod_a"]],
        [scene["track_a"]],
        samplerate_hz=FS_HZ,
        channel_gain=gains,
        clip=ClipParams(limit=limit),
    )
    assert np.abs(clipped).max() <= limit
    inside = np.abs(scaled) < limit
    np.testing.assert_array_equal(clipped[inside], scaled[inside])


def test_out_may_alias_an_additive_array(scene: dict) -> None:
    noise = scene["noise"].copy()
    expected = mix(
        [scene["eod_a"]],
        [scene["track_a"]],
        samplerate_hz=FS_HZ,
        additive=[noise.copy()],
    )
    result = mix(
        [scene["eod_a"]],
        [scene["track_a"]],
        samplerate_hz=FS_HZ,
        additive=[noise],
        out=noise,
        chunk_samples=3000,
    )
    assert result is noise
    np.testing.assert_array_equal(result, expected)


def test_hum_is_renderable_and_noise_only_mix_works(scene: dict) -> None:
    assert isinstance(scene["hum"], Renderable)
    only_hum = mix([], [], samplerate_hz=FS_HZ, additive=[scene["hum"]])
    np.testing.assert_array_equal(only_hum, scene["hum"].render(0, N))


def test_shape_errors(scene: dict) -> None:
    with pytest.raises(ValueError, match="EODs but"):
        mix([scene["eod_a"]], [], samplerate_hz=FS_HZ)
    with pytest.raises(ValueError, match="eods\\[1\\]"):
        mix(
            [scene["eod_a"], scene["eod_b"][:-1]],
            [scene["track_a"], scene["track_b"]],
            samplerate_hz=FS_HZ,
        )
    with pytest.raises(ValueError, match="does not cover"):
        mix([scene["eod_a"]], [scene["track_a"]], samplerate_hz=FS_HZ, start_time_s=5.0)
    with pytest.raises(ValueError, match="channel_gain"):
        mix(
            [scene["eod_a"]],
            [scene["track_a"]],
            samplerate_hz=FS_HZ,
            channel_gain=[1.0],
        )
    with pytest.raises(ValueError, match="cannot infer"):
        mix([], [], samplerate_hz=FS_HZ)


def test_dipole_params_reach_the_mix() -> None:
    path = simulate_path(np.random.default_rng(5), 1.0, ARENA)
    near = gain_track(path.times_s, path.xy_m, path.heading_rad, ELECTRODES)
    weaker = gain_track(
        path.times_s,
        path.xy_m,
        path.heading_rad,
        ELECTRODES,
        DipoleParams(reference_distance_m=0.05),
    )
    np.testing.assert_allclose(weaker.gains, near.gains / 4.0, rtol=1e-5)


def test_track_ending_at_last_sample_is_handled() -> None:
    # Knots past the last sample are extrapolated from the spline's end piece.
    n = 1003
    times = np.linspace(0.0, (n - 1) / FS_HZ, 8)
    gains = np.linspace(1.0, 2.0, 8 * 3).reshape(8, 3).astype(np.float32)
    track = GainTrack(times_s=times, gains=gains)
    eod = np.ones(n, dtype=np.float32)
    result = mix([eod], [track], samplerate_hz=FS_HZ)
    exact = track.evaluate(np.arange(n) / FS_HZ)
    np.testing.assert_allclose(result, exact, rtol=1e-6)
