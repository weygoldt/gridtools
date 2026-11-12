"""Chirp timing, template sampling and placement."""

import itertools
from pathlib import Path

import numpy as np
import pytest

from gridtools.sim.chirps import (
    TemplatePool,
    amplitude_from_frequency,
    place_chirp,
    random_dips,
    resample_template,
    schedule_chirps,
)
from gridtools.sim.config import ChirpConfig

FS = 20_000.0


def _draw(before: int = 400, after: int = 1000):
    """Template draw with a fixed support (20 ms before, 50 ms after the centre)."""

    def draw(fish_id: int) -> tuple[int, int, int]:
        return fish_id, before, after

    return draw


def _schedule(
    config: ChirpConfig,
    rates: list[float],
    seed: int,
    duration_s: float,
    draw=None,
):
    rngs = [np.random.default_rng([seed, i]) for i in range(len(rates))]
    return schedule_chirps(
        config,
        rates,
        rngs,
        np.random.default_rng([seed, 99]),
        _draw() if draw is None else draw,
        n_samples=round(duration_s * FS),
        samplerate_hz=FS,
    )


@pytest.mark.parametrize("seed", range(5))
def test_schedule_respects_interval_cap_and_response_lags(seed: int) -> None:
    config = ChirpConfig(rate_hz=(2.0, 2.0), response_prob=0.6, max_rate_hz=1.0)
    duration_s = 30.0
    schedule = _schedule(config, [3.0, 2.0, 0.5], seed, duration_s)
    events = schedule.events
    centres = np.array([e.center_index for e in events])
    assert np.all(np.diff(centres) >= 0)  # sorted
    for fish in range(3):
        own = [e for e in events if e.fish_id == fish]
        for first, second in itertools.pairwise(own):
            assert second.center_index - first.center_index >= 0.2 * FS
            assert second.start >= first.stop
        assert len(own) <= int(1.0 * duration_s)  # max_rate_hz
    for event in events:
        assert event.template_id == event.fish_id  # from the fake draw
        if event.is_response:
            trigger = events[event.trigger]
            assert not trigger.is_response
            assert trigger.fish_id != event.fish_id
            lag_s = (event.center_index - trigger.center_index) / FS
            in_delay = (
                config.response_delay_s[0] - 1 / FS
                <= lag_s
                <= config.response_delay_s[1] + 1 / FS
            )
            near = abs(lag_s) <= config.simultaneous_window_s + 1 / FS
            assert in_delay or near
        else:
            assert event.trigger == -1
    assert schedule.n_responses <= schedule.n_response_attempts
    assert schedule.n_spontaneous + schedule.n_responses == len(events)


def test_no_rate_cap_by_default() -> None:
    config = ChirpConfig(burst_size_mean=3.0)
    assert config.max_rate_hz is None
    schedule = _schedule(config, [1.6], 0, 60.0)
    assert len(schedule.events) > 60  # above one per second
    assert schedule.n_dropped_cap == 0


def test_near_simultaneous_responses_follow_their_fraction() -> None:
    config = ChirpConfig(response_prob=1.0, simultaneous_fraction=0.25)
    near = total = 0
    for seed in range(20):
        schedule = _schedule(config, [0.3, 0.3], seed, 120.0)
        for event in schedule.events:
            if event.is_response:
                lag = event.center_index - schedule.events[event.trigger].center_index
                near += abs(lag) <= 0.25 * FS
                total += 1
    # the response lags 0.45-2.2 s never fall in the +-0.25 s window
    assert abs(near / total - 0.25) < 4 * np.sqrt(0.25 * 0.75 / total)


def test_within_burst_intervals_follow_the_lognormal() -> None:
    config = ChirpConfig(burst_size_mean=50.0, burst_interval_s=(0.75, 0.45))
    rng = np.random.default_rng(0)
    from gridtools.sim.chirps import _spontaneous

    events = _spontaneous(0, rng, 0.5, config, _draw(10, 10), round(600 * FS), FS)
    intervals = np.diff([e.center_index for e in events]) / FS
    intervals = intervals[intervals < 5.0]  # within bursts
    assert np.median(intervals) == pytest.approx(0.75, rel=0.1)
    assert np.std(np.log(intervals)) == pytest.approx(0.45, rel=0.15)


def test_spacing_uses_the_drawn_supports() -> None:
    """Long chirps take more room: their supports never overlap."""
    rng = np.random.default_rng(3)

    def draw(fish_id: int) -> tuple[int, int, int]:
        long = rng.random() < 0.5
        return int(long), (2000 if long else 300), (6000 if long else 900)

    config = ChirpConfig(burst_size_mean=4.0, burst_interval_s=(0.3, 0.3))
    schedule = _schedule(config, [1.0], 0, 120.0, draw)
    events = schedule.events
    assert any(e.template_id == 1 for e in events)
    assert schedule.n_dropped_interval > 0
    for a, b in itertools.pairwise(events):
        assert b.start >= a.stop
        assert b.center_index - a.center_index >= 0.2 * FS


def test_response_attempts_follow_response_prob() -> None:
    config = ChirpConfig(response_prob=0.3)
    attempts = spontaneous = 0
    for seed in range(40):
        schedule = _schedule(config, [0.3, 0.3], seed, 60.0)
        attempts += schedule.n_response_attempts
        spontaneous += schedule.n_spontaneous
    rate = attempts / spontaneous
    sd = np.sqrt(0.3 * 0.7 / spontaneous)
    assert abs(rate - 0.3) < 4 * sd


def test_single_fish_has_no_responses() -> None:
    schedule = _schedule(ChirpConfig(response_prob=1.0), [0.5], 0, 60.0)
    assert schedule.n_responses == 0
    assert schedule.n_spontaneous > 0


def test_zero_rate_gives_no_chirps() -> None:
    schedule = _schedule(ChirpConfig(), [0.0, 0.0], 0, 60.0)
    assert schedule.events == ()


def test_pool_excludes_flags_and_equalizes(library_dir: Path) -> None:
    flat = ChirpConfig(
        library=str(library_dir), equalize_spacing="linear", equalize_alpha=1.0
    )
    pool = TemplatePool.open(flat)
    assert 5 not in pool.template_ids  # step_flag
    assert pool.template_ids.size == 23
    assert pool.n_samples == 4000
    assert "spacing='linear'" in pool.sampling
    assert pool.sampling_options["alpha"] == 1.0
    cycler = pool.cycler(np.random.default_rng(1))
    draws = np.concatenate([cycler.draw(1) for _ in range(4600)])
    groups = pool.view.equalization_bins((10, 10), spacing="linear").groups()
    big_groups = sum(pool.family(int(g[0])) == "big" for g in groups)
    big = np.isin(draws, [i for i in pool.template_ids if pool.family(i) == "big"])
    # alpha = 1: every non-empty (excursion x width) bin equally likely
    assert abs(big.mean() - big_groups / len(groups)) < 0.03
    tempered = TemplatePool.open(ChirpConfig(library=str(library_dir)))
    assert tempered.options == ()  # the library's own defaults
    from gridtools.templates.library import DEFAULT_ALPHA, DEFAULT_SPACING

    assert tempered.sampling_options["spacing"] == DEFAULT_SPACING
    assert tempered.sampling_options["alpha"] == DEFAULT_ALPHA
    tempered_cycler = tempered.cycler(np.random.default_rng(2))
    ids = np.concatenate([tempered_cycler.draw(1) for _ in range(200)])
    assert np.isin(ids, tempered.template_ids).all()
    measured = TemplatePool.open(
        ChirpConfig(library=str(library_dir), amplitude_mode="measured")
    )
    assert np.all(measured.template_ids % 2 == 0)


def test_support_is_measured_on_the_trace(library_dir: Path) -> None:
    pool = TemplatePool.open(ChirpConfig(library=str(library_dir)))
    delta, _ = pool.traces(0)
    nonzero = np.flatnonzero(delta)
    before, after = pool.support(0, FS)
    assert before == pool.centre_index - nonzero[0]
    assert after == nonzero[-1] + 1 - pool.centre_index
    half_before, half_after = pool.support(0, FS / 2)
    assert abs(half_before - before / 2) <= 1
    assert abs(half_after - after / 2) <= 1


def test_random_dips_range() -> None:
    dips = random_dips(np.random.default_rng(0), 1000, (0.1, 0.3))
    assert dips.min() >= 0.1
    assert dips.max() <= 0.3
    with pytest.raises(ValueError, match="dip_range"):
        random_dips(np.random.default_rng(0), 3, (0.5, 1.2))


def test_amplitude_from_frequency_is_the_inverted_trace() -> None:
    delta = np.array([0.0, 10.0, 50.0, 100.0, 50.0, -5.0, 0.0])
    amp = amplitude_from_frequency(delta, 0.6)
    np.testing.assert_allclose(amp, [1, 0.94, 0.7, 0.4, 0.7, 1, 1])
    np.testing.assert_array_equal(amplitude_from_frequency(delta, 0.0), np.ones(7))


def test_resample_template_identity_and_rate_change() -> None:
    trace = np.sin(np.linspace(0, np.pi, 401)).astype(np.float32)
    same, centre = resample_template(trace, FS, 200, FS, 0.0)
    assert centre == 200
    np.testing.assert_array_equal(same, trace)
    half, centre_half = resample_template(trace, FS, 200, FS / 2, 0.0)
    assert half[centre_half] == pytest.approx(1.0)
    assert half.size == 201


def test_place_chirp_crops_to_support_and_scene() -> None:
    delta = np.zeros(1000, dtype=np.float32)
    delta[400:600] = np.hanning(200).astype(np.float32) * 100.0
    delta[600:650] = -5.0 * np.hanning(50).astype(np.float32)
    amp = np.ones(1000, dtype=np.float32)
    inside = place_chirp(
        chirp_id=0,
        fish_id=1,
        template_id=3,
        delta_f_hz=delta,
        amp_factor=amp,
        centre_index_template=500,
        center_index=10_000,
        dip=0.0,
        n_samples=100_000,
    )
    assert inside is not None
    assert inside.support_start == 10_000 - 500 + 401  # first nonzero sample
    assert inside.support_stop == 10_000 - 500 + 649
    assert inside.start == inside.support_start
    assert inside.stop == inside.support_stop
    assert inside.excursion_hz == pytest.approx(100.0, rel=1e-3)
    assert inside.undershoot_hz == pytest.approx(5.0, rel=1e-2)
    assert inside.onset_index < inside.peak_index < inside.offset_index
    first, cycles = inside.local_cycles
    assert first == inside.start - 1
    assert cycles[0] == 0.0
    assert cycles[-1] == pytest.approx(float(np.sum(delta, dtype=np.float64)))
    edge = place_chirp(
        chirp_id=1,
        fish_id=1,
        template_id=3,
        delta_f_hz=delta,
        amp_factor=amp,
        centre_index_template=500,
        center_index=0,
        dip=0.0,
        n_samples=100_000,
    )
    assert edge is not None
    assert edge.support_start < 0
    assert edge.start == 0
    assert edge.delta_f_hz.size == edge.support_stop
    assert edge.local_cycles[0] == 0
    gone = place_chirp(
        chirp_id=2,
        fish_id=1,
        template_id=3,
        delta_f_hz=delta,
        amp_factor=amp,
        centre_index_template=500,
        center_index=100_500,
        dip=0.0,
        n_samples=100_000,
    )
    assert gone is None
