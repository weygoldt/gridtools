"""Chirp timing, template sampling and placement."""

import itertools
from pathlib import Path

import numpy as np
import pytest

from gridtools.sim.chirps import (
    TemplatePool,
    _local_groups,
    amplitude_from_frequency,
    library_api,
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
            assert second.start - first.stop >= round(config.min_interval_s * FS)
        assert len(own) <= int(config.max_rate_hz * duration_s)
    for event in events:
        assert event.template_id == event.fish_id  # from the fake draw
        if event.is_response:
            trigger = events[event.trigger]
            assert not trigger.is_response
            assert trigger.fish_id != event.fish_id
            lag_s = (event.center_index - trigger.center_index) / FS
            assert config.response_delay_s[0] - 1 / FS <= lag_s
            assert lag_s <= config.response_delay_s[1] + 1 / FS
        else:
            assert event.trigger == -1
    assert schedule.n_responses <= schedule.n_response_attempts
    assert schedule.n_spontaneous + schedule.n_responses == len(events)


def test_spacing_uses_the_drawn_supports() -> None:
    """Long chirps take more room: 300-ms supports still keep 200 ms of silence."""
    rng = np.random.default_rng(3)

    def draw(fish_id: int) -> tuple[int, int, int]:
        long = rng.random() < 0.5
        return int(long), (2000 if long else 300), (4000 if long else 900)

    config = ChirpConfig(burst_size_mean=4.0, burst_gap_s=(0.2, 0.3))
    schedule = _schedule(config, [1.0], 0, 120.0, draw)
    events = schedule.events
    assert any(e.template_id == 1 for e in events)
    gaps = [b.start - a.stop for a, b in itertools.pairwise(events)]
    assert min(gaps) >= 0.2 * FS
    # within bursts the gap is drawn from burst_gap_s, so many are short
    assert np.mean(np.array(gaps) < 0.3 * FS) > 0.2


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
    pool = TemplatePool.open(ChirpConfig(library=str(library_dir)))
    assert 5 not in pool.template_ids  # step_flag
    assert pool.template_ids.size == 23
    assert pool.n_samples == 4000
    draws = pool.cycler(np.random.default_rng(1)).draw(4600)
    big = np.isin(draws, [i for i in pool.template_ids if pool.family(i) == "big"])
    # every non-empty (excursion x width) bin is equally likely: linear bins
    # over 40-400 Hz put 40 and 60 Hz into one bin; 4 small and 3 big
    # excursion bins times 3 widths give 21 groups, 9 of them big
    groups = _local_groups(pool.view, (10, 10))
    assert len(groups) == 21
    big_groups = sum(pool.family(int(g[0])) == "big" for g in groups)
    assert big_groups == 9
    assert abs(big.mean() - 9 / 21) < 0.03
    natural = pool.cycler(np.random.default_rng(1))
    assert natural.draw(1).size == 1
    measured = TemplatePool.open(
        ChirpConfig(library=str(library_dir), amplitude_mode="measured")
    )
    assert np.all(measured.template_ids % 2 == 0)


def test_local_copy_matches_the_library_cycler(library_dir: Path) -> None:
    from gridtools.sim.chirps import _LocalCycler

    pool = TemplatePool.open(ChirpConfig(library=str(library_dir)))
    if library_api() != "gridtools.templates":
        pytest.skip("installed gridtools.templates predates library v2")
    for equalize in (True, False):
        library = pool.view.cycler(
            np.random.default_rng(5),
            equalize=equalize,  # type: ignore[call-arg]
            bins=(10, 10),  # type: ignore[call-arg]
            spacing="linear",  # type: ignore[call-arg]
            alpha=1.0,  # type: ignore[call-arg]
        )
        groups = (
            _local_groups(pool.view, (10, 10)) if equalize else [pool.view.unique_ids()]
        )
        local = _LocalCycler(groups, np.random.default_rng(5))
        # scenes draw one template at a time; compare in that mode
        expected = np.concatenate([library.draw(1) for _ in range(300)])
        got = np.concatenate([local.draw(1) for _ in range(300)])
        np.testing.assert_array_equal(got, expected)


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
