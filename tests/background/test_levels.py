"""Level definitions, SNR scaling and the measurements behind them."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gridtools.background.levels import (
    REFERENCE_BANDWIDTH_HZ,
    SnrDistribution,
    background_level,
    count_clicks,
    fish_signal_level,
    fish_signal_powers,
    line_amplitude,
    scale_for_snr,
    snr_db,
    tonal_peaks,
    tracked_line_power,
    welch_psd,
)

FS = 20_000.0


def white(rng: np.random.Generator, n: int, e: int, sd: float) -> np.ndarray:
    return rng.normal(0.0, sd, (n, e))


def test_reference_bandwidth_is_the_v1_spectrogram_bin() -> None:
    assert pytest.approx(1.5 * 20_000 / 4096) == REFERENCE_BANDWIDTH_HZ


def test_level_of_white_noise() -> None:
    rng = np.random.default_rng(0)
    sd = 3e-3
    x = white(rng, int(20 * FS), 3, sd)
    level = background_level(x, FS)
    expected_psd = sd**2 / (FS / 2)
    assert level.noise_psd == pytest.approx(expected_psd, rel=0.03)
    assert level.level == pytest.approx(expected_psd * REFERENCE_BANDWIDTH_HZ, rel=0.03)
    assert level.noise_psd_per_electrode.shape == (3,)
    assert level.level_db == pytest.approx(10 * math.log10(level.level))


def test_level_ignores_lines_and_hum() -> None:
    rng = np.random.default_rng(1)
    n = int(20 * FS)
    noise = white(rng, n, 4, 1e-3)
    t = np.arange(n) / FS
    lines = np.sum(
        [
            a * np.sin(2 * np.pi * f * t)
            for f, a in ((712.3, 0.3), (1424.6, 0.1), (843.0, 0.05), (1050.0, 0.02))
        ],
        axis=0,
    )
    noisy = noise + lines[:, None] * np.array([1.0, 0.5, 0.2, 0.1])
    clean = background_level(noise, FS).noise_psd
    assert background_level(noisy, FS).noise_psd == pytest.approx(clean, rel=0.03)


def test_level_is_robust_to_a_dead_electrode() -> None:
    rng = np.random.default_rng(2)
    x = white(rng, int(10 * FS), 5, 1e-3)
    x[:, 4] *= 1e-3
    assert background_level(x, FS).noise_psd == pytest.approx(1e-6 / (FS / 2), rel=0.05)


def test_line_amplitude_of_a_sinusoid() -> None:
    rng = np.random.default_rng(3)
    n = int(10 * FS)
    t = np.arange(n) / FS
    x = 0.02 * np.sin(2 * np.pi * 703.7 * t)[:, None] + white(rng, n, 1, 1e-4)
    freqs, psd = welch_psd(x, FS)
    assert line_amplitude(freqs, psd, 703.7)[0] == pytest.approx(0.02, rel=0.02)


@pytest.mark.parametrize("target_db", [10.0, 25.0, 40.0, 55.0])
def test_scale_for_snr_reaches_the_target(target_db: float) -> None:
    """Scaled background + fish, measured with the same definitions."""
    rng = np.random.default_rng(4)
    n = int(30 * FS)
    t = np.arange(n) / FS
    gains = np.array([0.1, 1.0, 0.4, -0.6])
    amplitude = 2.0  # scene unit
    f0 = 812.4
    fish = amplitude * np.sin(2 * np.pi * f0 * t)[:, None] * gains
    background = white(rng, n, gains.size, 1.7e-4)  # another unit
    signal = fish_signal_level([np.tile(amplitude * gains, (10, 1))])
    scale = scale_for_snr(background_level(background, FS).level, signal, target_db)
    mixture = fish + scale * background

    measured = background_level(mixture, FS)
    _, power = tracked_line_power(
        mixture,
        FS,
        lambda times: np.full(times.shape, f0),
        measured.noise_psd_per_electrode,
    )
    fish_power = float(np.median(np.max(power, axis=1)))
    assert fish_power == pytest.approx(signal, rel=0.05)
    assert snr_db(fish_power, measured.level) == pytest.approx(target_db, abs=0.5)


def test_scale_for_snr_formula_and_other_noise() -> None:
    assert scale_for_snr(4.0, 400.0, 20.0) == pytest.approx(1.0)
    assert scale_for_snr(1.0, 1.0, 0.0) == pytest.approx(1.0)
    # other noise takes part of the allowed noise
    s = scale_for_snr(1.0, 100.0, 10.0, other_noise_level=5.0)
    assert s**2 * 1.0 + 5.0 == pytest.approx(100.0 / 10.0)
    with pytest.raises(ValueError, match="exceeds"):
        scale_for_snr(1.0, 100.0, 10.0, other_noise_level=20.0)
    for bad in ((0.0, 1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, math.nan)):
        with pytest.raises(ValueError, match="must be finite"):
            scale_for_snr(*bad)


def test_fish_signal_powers_use_the_strongest_electrode() -> None:
    a = np.array([[1.0, -3.0], [2.0, 0.5], [1.0, 1.0]])  # median of max a^2/2
    b = np.array([[0.1, 0.2]])
    powers = fish_signal_powers([a, b])
    assert powers[0] == pytest.approx(2.0)  # max^2/2 per row: 4.5, 2.0, 0.5
    assert powers[1] == pytest.approx(0.02)
    assert fish_signal_level([a, b, b]) == pytest.approx(0.02)
    with pytest.raises(ValueError, match="at least one fish"):
        fish_signal_level([])


def test_snr_distribution_is_seeded_and_in_range() -> None:
    dist = SnrDistribution()
    a = dist.sample(np.random.default_rng(5), 20_000)
    b = dist.sample(np.random.default_rng(5), 20_000)
    np.testing.assert_array_equal(a, b)
    assert isinstance(a, np.ndarray)
    low, high = dist.range_db
    assert a.min() >= low
    assert a.max() <= high
    hard = np.mean(a < dist.hard_db[1])
    assert hard == pytest.approx(dist.hard_fraction, abs=0.02)
    assert isinstance(dist.sample(np.random.default_rng(0)), float)
    with pytest.raises(ValueError, match="hard_db"):
        SnrDistribution(hard_db=(10.0, 0.0))
    with pytest.raises(ValueError, match="hard_fraction"):
        SnrDistribution(hard_fraction=1.5)


def test_tonal_peaks_skip_mains_harmonics() -> None:
    rng = np.random.default_rng(6)
    n = int(10 * FS)
    t = np.arange(n) / FS
    x = white(rng, n, 2, 1e-4)
    x += (1e-2 * np.sin(2 * np.pi * 650.0 * t) + 1e-2 * np.sin(2 * np.pi * 777.0 * t))[
        :, None
    ]
    freqs, psd = welch_psd(x, FS)
    found, excess = tonal_peaks(freqs, psd)
    assert any(abs(f - 777.0) < 3 for f in found)
    assert not any(abs(f - 650.0) < 3 for f in found)  # 13th mains harmonic
    assert np.all(excess > 10)


def test_count_clicks() -> None:
    rng = np.random.default_rng(7)
    n = int(5 * FS)
    x = white(rng, n, 3, 1e-4)
    for onset_s, channel in ((1.0, 0), (2.5, 1), (4.0, 2)):
        i = int(onset_s * FS)
        x[i : i + 2, channel] += [0.02, -0.02]  # broadband impulse
    counts, times = count_clicks(x, FS)
    assert counts.tolist() == [1, 1, 1]
    np.testing.assert_allclose(times, [1.0, 2.5, 4.0], atol=0.002)


def test_tracked_line_power_drops_crowded_frames() -> None:
    rng = np.random.default_rng(8)
    n = int(10 * FS)
    t = np.arange(n) / FS
    x = (0.01 * np.sin(2 * np.pi * 700.0 * t))[:, None] + white(rng, n, 1, 1e-4)
    times, power = tracked_line_power(
        x,
        FS,
        lambda s: np.where(s < 5.0, 700.0, np.nan),
        np.array([2e-8 / (FS / 2)]),
        others_hz=[lambda s: np.full(s.shape, 705.0)],
    )
    assert np.all(np.isnan(power))
    times, power = tracked_line_power(
        x, FS, lambda s: np.where(s < 5.0, 700.0, np.nan), np.array([0.0])
    )
    early = times < 4.7
    assert np.all(np.isfinite(power[early]))
    assert np.all(np.isnan(power[times > 5.3]))
    assert np.median(power[early]) == pytest.approx(0.01**2 / 2, rel=0.02)


def test_snr_distribution_truncation() -> None:
    dist = SnrDistribution(hard_db=(0.0, 30.0), typical_db=(30.0, 80.0))
    rng = np.random.default_rng(9)
    draws = np.asarray(dist.sample(rng, 20_000, min_db=25.0))
    assert draws.min() >= 25.0
    assert draws.max() <= 80.0
    # mass left: hard 0.4 * 5/30, typical 0.6 -> P(< 30) = 0.0667 / 0.6667
    assert np.mean(draws < 30.0) == pytest.approx(0.1, abs=0.01)
    assert np.all(np.asarray(dist.sample(rng, 5, min_db=90.0)) == 90.0)
    plain = np.asarray(dist.sample(np.random.default_rng(1), 100))
    same = np.asarray(dist.sample(np.random.default_rng(1), 100, min_db=-10.0))
    np.testing.assert_allclose(plain, same)
