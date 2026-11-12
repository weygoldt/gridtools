"""Statistical tests for the recording-chain noise models."""

import numpy as np
import pytest
from scipy.signal import welch

from gridtools.sim.noise import (
    ClipParams,
    ColoredNoiseParams,
    ElectrodeFaultParams,
    GainMismatchParams,
    HumParams,
    clip,
    colored_noise,
    electrode_faults,
    gain_mismatch,
    powerline_hum,
)

FS_HZ = 20_000.0


@pytest.mark.parametrize("exponent", [0.0, 1.0, 2.0])
def test_colored_noise_psd_slope_matches_exponent(exponent: float) -> None:
    params = ColoredNoiseParams(sd=1.0, exponent=exponent, f_min_hz=1.0)
    noise = colored_noise(np.random.default_rng(1), 1 << 18, 4, FS_HZ, params)
    freqs, psd = welch(noise, fs=FS_HZ, nperseg=8192, axis=0)
    band = (freqs >= 20.0) & (freqs <= 2000.0)
    slope = np.polyfit(np.log10(freqs[band]), np.log10(psd[band].mean(axis=1)), 1)[0]
    assert slope == pytest.approx(-exponent, abs=0.1)


def test_white_noise_has_configured_sd_and_no_cross_correlation() -> None:
    params = ColoredNoiseParams(sd=0.3, exponent=0.0)
    noise = colored_noise(np.random.default_rng(2), 1 << 16, 8, FS_HZ, params)
    assert noise.dtype == np.float32
    assert noise.shape == (1 << 16, 8)
    np.testing.assert_allclose(noise.std(axis=0), 0.3, rtol=0.03)
    corr = np.asarray(np.corrcoef(noise.T))
    off_diagonal = corr[np.triu_indices(8, 1)]
    assert np.max(np.abs(off_diagonal)) < 0.03


def test_pink_noise_is_independent_across_electrodes() -> None:
    params = ColoredNoiseParams(sd=1.0, exponent=1.0, f_min_hz=1.0)
    noise = colored_noise(np.random.default_rng(3), 1 << 18, 6, FS_HZ, params)
    # Expected (ensemble) SD, not forced on each realisation.
    assert noise.std(axis=0).mean() == pytest.approx(1.0, rel=0.15)
    corr = np.asarray(np.corrcoef(noise.T))
    assert np.max(np.abs(corr[np.triu_indices(6, 1)])) < 0.1


def test_noise_sd_scale_per_electrode() -> None:
    params = ColoredNoiseParams(sd=1.0, exponent=0.0)
    scale = np.array([1.0, 5.0, 0.0])
    noise = colored_noise(
        np.random.default_rng(4), 1 << 15, 3, FS_HZ, params, sd_scale=scale
    )
    np.testing.assert_allclose(noise.std(axis=0), scale, rtol=0.05, atol=1e-6)
    with pytest.raises(ValueError, match="sd_scale"):
        colored_noise(
            np.random.default_rng(4), 100, 3, FS_HZ, params, sd_scale=np.ones(2)
        )


def test_odd_and_tiny_lengths_work() -> None:
    noise = colored_noise(np.random.default_rng(5), 1001, 2, FS_HZ)
    assert noise.shape == (1001, 2)
    assert np.all(np.isfinite(noise))


def test_hum_peaks_at_line_harmonics() -> None:
    params = HumParams(line_freq_sd_hz=0.0, harmonic_log_sd=0.0, n_harmonics=3)
    hum = powerline_hum(np.random.default_rng(6), 1 << 17, 4, FS_HZ, params)
    assert hum.frequencies_hz == (50.0, 100.0, 150.0)
    data = hum.render(0, 1 << 17)
    assert data.shape == (1 << 17, 4)
    freqs, psd = welch(
        data[:, 0] + 1e-9 * np.random.default_rng(0).standard_normal(1 << 17),
        fs=FS_HZ,
        nperseg=1 << 14,
    )
    floor = np.median(psd)
    for f_hz in (50.0, 100.0, 150.0):
        peak = psd[np.argmin(np.abs(freqs - f_hz))]
        assert peak > 1e6 * floor
    assert psd[np.argmin(np.abs(freqs - 200.0))] < 1e3 * floor


def test_hum_is_one_source_with_per_electrode_coupling() -> None:
    hum = powerline_hum(np.random.default_rng(7), 20_000, 5, FS_HZ)
    data = hum.render(0, 20_000)
    corr = np.corrcoef(data.T)
    np.testing.assert_allclose(np.abs(corr), 1.0, atol=1e-5)
    np.testing.assert_allclose(
        data.std(axis=0) / hum.source.std(), hum.coupling, rtol=1e-4
    )
    np.testing.assert_array_equal(hum.render(100, 200), data[100:200])


def test_hum_drops_components_above_nyquist() -> None:
    hum = powerline_hum(
        np.random.default_rng(8), 1000, 2, 400.0, HumParams(n_harmonics=8)
    )
    assert max(hum.frequencies_hz) < 200.0


def test_gain_mismatch_distribution() -> None:
    gains = gain_mismatch(np.random.default_rng(9), 20_000, GainMismatchParams(0.1))
    assert gains.dtype == np.float32
    assert np.all(gains > 0)
    assert np.log(gains).std() == pytest.approx(0.1, rel=0.05)
    assert np.log(gains).mean() == pytest.approx(0.0, abs=0.005)


def test_electrode_faults_rates_and_factors() -> None:
    params = ElectrodeFaultParams(
        dead_prob=0.2, noisy_prob=0.1, dead_gain=1e-3, noisy_factor=5.0
    )
    faults = electrode_faults(np.random.default_rng(10), 50_000, params)
    assert faults.dead.mean() == pytest.approx(0.2, abs=0.01)
    assert faults.noisy.mean() == pytest.approx(0.8 * 0.1, abs=0.01)
    assert not np.any(faults.dead & faults.noisy)
    np.testing.assert_array_equal(faults.signal_gain[faults.dead], np.float32(1e-3))
    np.testing.assert_array_equal(faults.signal_gain[~faults.dead], 1.0)
    np.testing.assert_array_equal(faults.noise_scale[faults.noisy], 5.0)
    np.testing.assert_array_equal(faults.noise_scale[~faults.noisy], 1.0)


def test_clipping_bounds_are_respected() -> None:
    x = np.random.default_rng(11).normal(0.0, 10.0, (1000, 3))
    clipped = clip(x, ClipParams(limit=5.0))
    assert clipped.dtype == np.float32
    assert np.abs(clipped).max() <= 5.0
    inside = np.abs(x) < 5.0
    np.testing.assert_allclose(clipped[inside], x[inside], rtol=1e-6)
    np.testing.assert_array_equal(clipped[x >= 5.0], 5.0)
    np.testing.assert_array_equal(clipped[x <= -5.0], -5.0)
