"""Exact re-rendering: linearity, sub-spans, chirp omission, ground truth."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import butter, hilbert, sosfiltfilt

from gridtools.sim.render import BLOCK
from gridtools.sim.scene import Scene, render_background, render_fish

FS = 20_000.0


def _tolerance(x: np.ndarray) -> float:
    """A few float32 ulps of the largest value."""
    return 8 * float(np.finfo(np.float32).eps) * float(np.max(np.abs(x)))


def test_recording_is_sum_of_fish_plus_background(scene: Scene) -> None:
    fish = sum(render_fish(scene, i) for i in range(len(scene.model.fish)))
    background = render_background(scene)
    assert len(scene.clicks) > 0  # clicks are part of the background
    np.testing.assert_allclose(
        scene.recording, fish + background, rtol=0, atol=_tolerance(scene.recording)
    )


def test_fish_contribution_matches_mix_of_that_fish_alone(scene: Scene) -> None:
    from gridtools.sim.mix import mix

    model = scene.model
    for i in range(len(model.fish)):
        alone = mix(
            [model.waveform(i, 0, model.n_samples)],
            [model.gain_track(i)],
            samplerate_hz=FS,
            channel_gain=model.channel_gain,
        )
        np.testing.assert_array_equal(render_fish(scene, i), alone)


@pytest.mark.parametrize(
    ("start", "stop"), [(0, 17), (12_345, 54_321), (BLOCK - 3, BLOCK + 5)]
)
def test_sub_span_is_bit_identical_to_full_rendering(
    scene: Scene, start: int, stop: int
) -> None:
    model = scene.model
    for i in range(len(model.fish)):
        full = model.render(i, 0, model.n_samples)
        np.testing.assert_array_equal(model.render(i, start, stop), full[start:stop])
        chirp_ids = model.fish[i].chirp_ids
        if chirp_ids:
            omit = chirp_ids[:1]
            full = model.render(i, 0, model.n_samples, chirps=omit)
            part = model.render(i, start, stop, chirps=omit)
            np.testing.assert_array_equal(part, full[start:stop])


def test_render_from_directory_equals_render_from_memory(
    scene: Scene, scene_path: Path
) -> None:
    for i in range(len(scene.model.fish)):
        np.testing.assert_array_equal(
            render_fish(scene_path, i, 1.0, 2.5), render_fish(scene, i, 1.0, 2.5)
        )
        ids = scene.model.fish[i].chirp_ids
        np.testing.assert_array_equal(
            render_fish(scene_path, i, chirps=ids[:2]),
            render_fish(scene, i, chirps=ids[:2]),
        )


def test_seconds_map_to_samples_like_recording_read_time(scene: Scene) -> None:
    out = render_fish(scene, 0, 0.5, 0.75)
    assert out.shape == (5000, 9)
    np.testing.assert_array_equal(out, scene.model.render(0, 10_000, 15_000))
    with pytest.raises(IndexError):
        render_fish(scene, 0, -0.1, 1.0)
    with pytest.raises(IndexError):
        render_fish(scene, 0, 1.0, 100.0)
    with pytest.raises(KeyError):
        render_fish(scene, 99)
    with pytest.raises(KeyError):
        render_fish(scene, 0, chirps=[10_000])


def test_electrode_subset_equals_columns(scene: Scene) -> None:
    full = render_fish(scene, 1, 2.0, 3.0)
    np.testing.assert_array_equal(
        render_fish(scene, 1, 2.0, 3.0, electrodes=[4, 0]), full[:, [4, 0]]
    )


def _chirps_with_neighbours_apart(scene: Scene):
    model = scene.model
    for fish in model.fish:
        for k, chirp in enumerate(fish.chirps):
            yield fish.fish_id, k, chirp


def test_omitting_a_chirp_changes_only_its_support(scene: Scene) -> None:
    model = scene.model
    n_checked = 0
    for fish_id, _, chirp in _chirps_with_neighbours_apart(scene):
        full = model.render(fish_id, 0, model.n_samples)
        without = model.render(fish_id, 0, model.n_samples, chirps=[chirp.chirp_id])
        diff = np.abs(full.astype(np.float64) - without)
        lo = max(chirp.support_start, 0)
        hi = min(chirp.support_stop + 1, model.n_samples)
        # before the chirp: same operations, bit-identical
        assert np.array_equal(full[:lo], without[:lo])
        # after it: phase differs by whole cycles only
        assert diff[hi:].max(initial=0.0) <= _tolerance(full)
        # inside it: the chirp is there
        if chirp.excursion_hz > 30 and hi - lo > 100:
            assert diff[lo:hi].max() > 0.05 * np.abs(full[lo:hi]).max()
        n_checked += 1
    assert n_checked >= 5


def test_omitting_all_chirps_aligned_differs_only_inside_supports(scene: Scene) -> None:
    model = scene.model
    for fish in model.fish:
        full = model.render(fish.fish_id, 0, model.n_samples)
        none = model.render(fish.fish_id, 0, model.n_samples, chirps="none")
        outside = np.ones(model.n_samples, dtype=bool)
        for chirp in fish.chirps:
            outside[max(chirp.support_start, 0) : chirp.support_stop + 1] = False
        diff = np.abs(full.astype(np.float64) - none)[outside]
        assert diff.max(initial=0.0) <= _tolerance(full)


def test_natural_phase_lags_after_an_omitted_chirp(scene: Scene) -> None:
    model = scene.model
    fish_id, chirp = next(
        (f.fish_id, c)
        for f in model.fish
        for c in f.chirps
        if c.support_stop + 4000 < model.n_samples and c.excursion_hz > 40
    )
    _, cycles = chirp.local_cycles
    turns = cycles[-1] / FS
    assert abs(turns - round(turns)) > 0.02  # otherwise the test says nothing
    span = (chirp.support_stop + 100, chirp.support_stop + 4000)
    full = model.render(fish_id, *span)
    natural = model.render(fish_id, *span, chirps=[chirp.chirp_id], phase="natural")
    aligned = model.render(fish_id, *span, chirps=[chirp.chirp_id], phase="aligned")
    assert np.abs(full - aligned).max() <= _tolerance(full)
    assert np.abs(full - natural).max() > 0.05 * np.abs(full).max()
    # the lag is the chirp's phase advance
    theta_full = model.phase(fish_id, *span)
    theta_natural = model.phase(
        fish_id, *span, chirps=[chirp.chirp_id], phase="natural"
    )
    np.testing.assert_allclose(theta_full - theta_natural, 2 * np.pi * turns, atol=1e-6)


def test_phase_is_continuous(scene: Scene) -> None:
    model = scene.model
    for fish in model.fish:
        for mode in ("aligned", "natural"):
            theta = model.phase(
                fish.fish_id, 0, model.n_samples, chirps="none", phase=mode
            )
            step = np.diff(theta) * FS / (2 * np.pi)  # instantaneous frequency
            assert step.min() > 300.0
            assert step.max() < 2_000.0
        freq = model.frequency(fish.fish_id, 0, model.n_samples)
        theta = model.phase(fish.fish_id, 0, model.n_samples)
        step = np.diff(theta) * FS / (2 * np.pi)
        np.testing.assert_allclose(step, 0.5 * (freq[:-1] + freq[1:]), atol=1e-4)


@dataclass(frozen=True)
class _Row:
    t_peak_s: float
    baseline_hz: float
    excursion_hz: float
    emitter_id: int


def _rows(table: pd.DataFrame) -> list[_Row]:
    return [
        _Row(float(t), float(b), float(e), int(i))
        for t, b, e, i in zip(
            table["t_peak_s"],
            table["baseline_hz"],
            table["excursion_hz"],
            table["emitter_id"],
            strict=True,
        )
    ]


def _instantaneous_frequency(
    x: np.ndarray, base_hz: float, top_hz: float
) -> np.ndarray:
    """Hilbert frequency of the fundamental: band-pass wide around the chirp
    (narrow edges cut the sidebands of fast, deeply modulated chirps) but
    below the second harmonic."""
    f_lo = base_hz - 250.0
    f_hi = min(2.0 * base_hz - 250.0, top_hz + 300.0)
    sos = butter(8, [f_lo, f_hi], btype="bandpass", fs=FS, output="sos")
    phase = np.unwrap(np.angle(np.asarray(hilbert(sosfiltfilt(sos, x)))))
    return np.gradient(phase) * FS / (2 * np.pi)


def test_emitter_shows_the_template_excursion_and_others_do_not(scene: Scene) -> None:
    model = scene.model
    table = scene.chirps
    half = 3000
    n_checked = 0
    for row in _rows(table):
        peak = round(row.t_peak_s * FS)
        if peak - half < 0 or peak + half > model.n_samples:
            continue
        base = row.baseline_hz
        top = base + row.excursion_hz
        emitter = model.render(row.emitter_id, peak - half, peak + half)
        channel = int(np.argmax(np.std(emitter, axis=0)))
        freq = _instantaneous_frequency(
            emitter[:, channel].astype(np.float64), base, top
        )
        assert freq[half] == pytest.approx(top, abs=max(2.0, 0.02 * row.excursion_hz))
        # 120-140 ms before the peak the emitter sits at its baseline (averaged
        # over 25 ms to cancel the ripple from the attenuated 2nd harmonic)
        before = slice(half - 2800, half - 2300)
        expected = model.base_frequency(row.emitter_id, peak - 2800, peak - 2300)
        assert np.mean(freq[before]) == pytest.approx(np.mean(expected), abs=1.0)
        for other in model.fish:
            if other.fish_id == row.emitter_id:
                continue
            if any(
                c.support_start - 400 <= peak <= c.support_stop + 400
                for c in other.chirps
            ):
                continue  # the other fish chirps here itself
            signal = model.render(other.fish_id, peak - half, peak + half)
            channel = int(np.argmax(np.std(signal, axis=0)))
            f_other = model.base_frequency(other.fish_id, peak, peak + 1)[0]
            freq_other = _instantaneous_frequency(
                signal[:, channel].astype(np.float64), f_other, f_other
            )
            assert np.mean(freq_other[half - 250 : half + 250]) == pytest.approx(
                f_other, abs=1.0
            )
            assert np.abs(freq_other[half - 250 : half + 250] - f_other).max() < 10.0
        n_checked += 1
    assert n_checked >= 5
