"""PSD scaling, line amplitudes, noise floor and click detection."""

import numpy as np
import pytest

from gridtools.calibrate.spectra import (
    carrier_to_noise_db_hz,
    count_clicks,
    floor_spectrum,
    hum_lines,
    line_amplitude,
    noise_floor,
    power_law_fit,
    psd,
    segment_starts,
)
from gridtools.sim.noise import ColoredNoiseParams, colored_noise

FS = 20_000.0


def test_segment_starts_are_stratified_and_reproducible() -> None:
    a = segment_starts(
        np.random.default_rng(5), 1_000_000, 10_000, 20, margin_samples=5000
    )
    b = segment_starts(
        np.random.default_rng(5), 1_000_000, 10_000, 20, margin_samples=5000
    )
    np.testing.assert_array_equal(a, b)
    assert np.all(np.diff(a) >= 10_000)
    assert a[0] >= 5000
    assert a[-1] + 10_000 <= 1_000_000 - 5000
    with pytest.raises(ValueError, match="cannot place"):
        segment_starts(np.random.default_rng(0), 100_000, 10_000, 20)


def test_white_noise_floor_is_variance_over_nyquist() -> None:
    rng = np.random.default_rng(0)
    sd = 3e-5
    x = rng.normal(0.0, sd, (200_000, 2))
    freqs, power = psd(x, FS, 8192)
    floor = noise_floor(freqs, power, (400.0, 1500.0))
    np.testing.assert_allclose(floor, sd**2 / (FS / 2), rtol=0.05)


@pytest.mark.parametrize("freq_hz", [700.0, 700.0 + FS / 8192 / 2, 1234.5])
def test_line_amplitude_is_independent_of_bin_position(freq_hz: float) -> None:
    rng = np.random.default_rng(1)
    t = np.arange(200_000) / FS
    x = 2e-4 * np.cos(2 * np.pi * freq_hz * t) + rng.normal(0.0, 1e-5, t.size)
    freqs, power = psd(x[:, None], FS, 8192)
    assert line_amplitude(freqs, power, freq_hz)[0] == pytest.approx(2e-4, rel=0.02)


def test_carrier_to_noise_matches_definition() -> None:
    """sign_v-style single-bin PSD + floor give (A^2/2)/N0 for a centred line."""
    rng = np.random.default_rng(2)
    nfft = 16384
    freq = 40 * FS / nfft * 10  # exactly on a bin
    sd, amp = 1e-5, 5e-4
    n = 400_000
    t = np.arange(n) / FS
    x = amp * np.cos(2 * np.pi * freq * t) + rng.normal(0.0, sd, n)
    freqs, power = psd(x[:, None], FS, nfft)
    peak_db = 10 * np.log10(power[round(freq * nfft / FS), 0])
    floor_db = 10 * np.log10(noise_floor(freqs, power, (400.0, 1500.0))[0])
    cn0 = carrier_to_noise_db_hz(peak_db, floor_db, FS, nfft)
    expected = 10 * np.log10((amp**2 / 2) / (sd**2 / (FS / 2)))
    assert cn0 == pytest.approx(expected, abs=0.3)


def test_power_law_fit_recovers_noise_exponent() -> None:
    rng = np.random.default_rng(3)
    edges = np.geomspace(100.0, 5000.0, 15)
    for exponent in (0.0, 1.0):
        x = colored_noise(rng, 400_000, 1, FS, ColoredNoiseParams(1e-3, exponent, 1.0))
        freqs, power = psd(x, FS, 8192)
        centres, floor = floor_spectrum(freqs, power, edges)
        alpha, _ = power_law_fit(centres, floor[:, 0])
        assert alpha == pytest.approx(exponent, abs=0.1)


def test_hum_lines_measure_odd_harmonics() -> None:
    rng = np.random.default_rng(4)
    t = np.arange(200_000) / FS
    x = (
        1e-4 * np.cos(2 * np.pi * 50 * t)
        + 3e-5 * np.cos(2 * np.pi * 150 * t + 1.0)
        + rng.normal(0.0, 1e-6, t.size)
    )
    freqs, power = psd(x[:, None], FS, 8192)
    hum = hum_lines(freqs, power, n_harmonics=4)
    np.testing.assert_allclose(hum.amplitudes[[0, 2], 0], [1e-4, 3e-5], rtol=0.03)
    assert hum.detected[:, 0].tolist() == [True, False, True, False]


def test_count_clicks_finds_impulses_and_ignores_tones() -> None:
    rng = np.random.default_rng(5)
    n = 200_000
    t = np.arange(n) / FS
    tone = 1e-3 * np.cos(2 * np.pi * 750 * t) + 1e-4 * np.cos(2 * np.pi * 6000 * t)
    x = np.column_stack([tone, tone]) + rng.normal(0.0, 1e-5, (n, 2))
    counts, onsets = count_clicks(x, FS)
    assert counts.tolist() == [0, 0]
    assert onsets.size == 0
    for at in (20_000, 80_000, 150_000):
        x[at : at + 5, 0] += 5e-3 * np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    counts, onsets = count_clicks(x, FS)
    assert counts.tolist() == [3, 0]
    np.testing.assert_allclose(onsets, [1.0, 4.0, 7.5], atol=0.003)
