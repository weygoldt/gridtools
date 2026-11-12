"""Physics tests for EOD synthesis and baseline drift."""

import math

import numpy as np
import pytest
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.special import erf

from gridtools.sim.eod import (
    APTERONOTUS_LEPTORHYNCHUS,
    DriftParams,
    HarmonicJitterParams,
    Harmonics,
    SharedDriftParams,
    baseline_frequency,
    eod_phase,
    harmonic_waveform,
    jitter_harmonics,
    shared_frequency_drift,
    synthesize_eod,
)

FS_HZ = 20_000.0
F0_HZ = 800.0
CHIRP_HZ = 100.0
CHIRP_FWHM_S = 0.020
CHIRP_SD_S = CHIRP_FWHM_S / (2.0 * math.sqrt(2.0 * math.log(2.0)))
CHIRP_T_S = 0.25


def _chirp_frequency(times_s: np.ndarray) -> np.ndarray:
    return F0_HZ + CHIRP_HZ * np.exp(-0.5 * ((times_s - CHIRP_T_S) / CHIRP_SD_S) ** 2)


def _chirp_phase_exact(times_s: np.ndarray, phase0_rad: float) -> np.ndarray:
    """Closed-form integral of the Gaussian chirp frequency."""
    scale = math.sqrt(2.0) * CHIRP_SD_S
    area = (
        CHIRP_HZ
        * CHIRP_SD_S
        * math.sqrt(math.pi / 2.0)
        * (erf((times_s - CHIRP_T_S) / scale) - erf(-CHIRP_T_S / scale))
    )
    return phase0_rad + 2.0 * np.pi * (F0_HZ * times_s + area)


def _reference_waveform(phase: np.ndarray, harmonics: Harmonics) -> np.ndarray:
    out = np.zeros_like(phase)
    for k, (a_k, phi_k) in enumerate(
        zip(harmonics.amplitudes, harmonics.phases_rad, strict=True), start=1
    ):
        out += a_k * np.cos(k * phase + phi_k)
    return out


def test_constant_frequency_phase_is_exact_ramp() -> None:
    n = 100_000
    freq = np.full(n, 812.345)
    phase = eod_phase(freq, FS_HZ, phase0_rad=0.7)
    expected = 0.7 + 2.0 * np.pi * 812.345 * np.arange(n) / FS_HZ
    np.testing.assert_allclose(phase, expected, rtol=0, atol=1e-9)


def test_waveform_matches_closed_form_through_sharp_chirp() -> None:
    times_s = np.arange(int(0.5 * FS_HZ)) / FS_HZ
    eod = synthesize_eod(
        _chirp_frequency(times_s), 1.0, FS_HZ, APTERONOTUS_LEPTORHYNCHUS, 0.4
    )
    expected = _reference_waveform(
        _chirp_phase_exact(times_s, 0.4), APTERONOTUS_LEPTORHYNCHUS
    )
    assert eod.dtype == np.float32
    # The trapezoidal rule's phase error is (h^2 / 12) f'(t) cycles, at most
    # ~1.5e-6 cycles (1e-5 rad) on the chirp flanks and zero once f' = 0.
    np.testing.assert_allclose(eod, expected, rtol=0, atol=3e-5)
    after = times_s > CHIRP_T_S + 6 * CHIRP_SD_S
    np.testing.assert_allclose(eod[after], expected[after], rtol=0, atol=2e-6)


def test_instantaneous_frequency_follows_chirp_without_phase_jump() -> None:
    times_s = np.arange(int(0.5 * FS_HZ)) / FS_HZ
    freq = _chirp_frequency(times_s)
    eod = synthesize_eod(freq, 1.0, FS_HZ).astype(np.float64)
    sos = butter(8, [500.0, 1300.0], btype="bandpass", fs=FS_HZ, output="sos")
    fundamental = sosfiltfilt(sos, eod)
    analytic = np.asarray(hilbert(fundamental), dtype=np.complex128)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.gradient(phase) * FS_HZ / (2.0 * np.pi)
    core = slice(int(0.05 * FS_HZ), int(0.45 * FS_HZ))
    error_hz = np.abs(inst_freq[core] - freq[core])
    assert error_hz.max() < 1.0
    # The chirp peak itself is reproduced, not smoothed away.
    peak = int(CHIRP_T_S * FS_HZ)
    assert inst_freq[peak] == pytest.approx(F0_HZ + CHIRP_HZ, abs=1.0)
    # Phase advance from the chirp matches its integral (no lost cycles).
    phase_offset = phase - 2.0 * np.pi * np.cumsum(freq) / FS_HZ
    drift_rad = phase_offset[core].max() - phase_offset[core].min()
    assert drift_rad < 0.05


def test_harmonic_amplitudes_and_phases_recovered_by_fft() -> None:
    n = int(FS_HZ)  # 1 s: 1-Hz bins, F0 on a bin
    phase0 = 0.3
    eod = synthesize_eod(np.full(n, F0_HZ), 2.0, FS_HZ, phase0_rad=phase0)
    spectrum = np.fft.rfft(eod.astype(np.float64)) / n
    for k, (a_k, phi_k) in enumerate(
        zip(
            APTERONOTUS_LEPTORHYNCHUS.amplitudes,
            APTERONOTUS_LEPTORHYNCHUS.phases_rad,
            strict=True,
        ),
        start=1,
    ):
        coeff = spectrum[int(k * F0_HZ)]
        assert 2.0 * abs(coeff) == pytest.approx(2.0 * a_k, rel=1e-4)
        expected_phase = np.angle(np.exp(1j * (phi_k + k * phase0)))
        assert np.angle(coeff * np.exp(-1j * expected_phase)) == pytest.approx(
            0.0, abs=1e-3
        )


def test_amplitude_envelope_is_multiplied_in() -> None:
    n = 4000
    freq = np.full(n, F0_HZ)
    envelope = np.linspace(0.2, 1.0, n)
    eod = synthesize_eod(freq, envelope, FS_HZ)
    unit = synthesize_eod(freq, 1.0, FS_HZ)
    np.testing.assert_allclose(eod, unit * envelope, rtol=1e-6, atol=1e-7)


def test_span_rendering_from_global_phase_is_exact() -> None:
    times_s = np.arange(int(0.5 * FS_HZ)) / FS_HZ
    freq = _chirp_frequency(times_s)
    full = synthesize_eod(freq, 1.0, FS_HZ)
    phase = eod_phase(freq, FS_HZ)
    span = slice(3000, 6000)
    part = harmonic_waveform(phase[span], APTERONOTUS_LEPTORHYNCHUS)
    np.testing.assert_array_equal(part, full[span])


def test_phase_stays_precise_over_long_recordings() -> None:
    n = 2_000_000  # 100 s at 20 kHz
    t = np.arange(n) / FS_HZ
    freq = 800.0 + 3.0 * np.sin(2.0 * np.pi * t / 37.0) + 0.5 * np.cos(t / 3.0)
    phase = eod_phase(freq, FS_HZ)
    exact_freq = freq.astype(np.longdouble)
    increments = (exact_freq[:-1] + exact_freq[1:]) / 2
    cycles = np.concatenate([[np.longdouble(0)], np.cumsum(increments)])
    reference = (2 * np.pi * cycles / np.longdouble(FS_HZ)).astype(np.float64)
    assert np.max(np.abs(phase - reference)) < 1e-6


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="freq_hz"):
        eod_phase(np.array([800.0, np.nan]), FS_HZ)
    with pytest.raises(ValueError, match="amplitude"):
        synthesize_eod(np.full(10, F0_HZ), np.ones(9), FS_HZ)
    with pytest.raises(ValueError, match="equally long"):
        Harmonics(amplitudes=(1.0, 0.5), phases_rad=(0.0,))


def test_jitter_harmonics_keeps_fundamental_and_perturbs_others() -> None:
    rng = np.random.default_rng(1)
    jittered = jitter_harmonics(rng, APTERONOTUS_LEPTORHYNCHUS)
    base = APTERONOTUS_LEPTORHYNCHUS
    assert jittered.amplitudes[0] == base.amplitudes[0]
    assert jittered.phases_rad[0] == base.phases_rad[0]
    assert jittered.amplitudes[1:] != base.amplitudes[1:]
    ratios = np.array(jittered.amplitudes[1:]) / np.array(base.amplitudes[1:])
    assert np.all((ratios > 0.3) & (ratios < 3.0))
    none = jitter_harmonics(
        rng, base, HarmonicJitterParams(amplitude_log_sd=0.0, phase_sd_rad=0.0)
    )
    np.testing.assert_allclose(none.amplitudes, base.amplitudes)
    np.testing.assert_allclose(none.phases_rad, base.phases_rad, atol=1e-12)


def test_individual_drift_has_configured_ensemble_sd() -> None:
    params = DriftParams(sd_hz=0.5, timescale_s=5.0, control_rate_hz=1.0)
    rng = np.random.default_rng(2)
    samplerate_hz = 10.0
    values = np.array(
        [
            baseline_frequency(rng, 600, samplerate_hz, 700.0, params)[[0, 300, 599]]
            for _ in range(400)
        ]
    )
    assert values.mean() == pytest.approx(700.0, abs=0.1)
    assert values.std() == pytest.approx(0.5, rel=0.1)


def test_drift_is_smooth_and_slow() -> None:
    rng = np.random.default_rng(3)
    params = DriftParams(sd_hz=1.0, timescale_s=10.0, control_rate_hz=1.0)
    freq = baseline_frequency(rng, 200_000, 2_000.0, 650.0, params)
    # Largest per-sample change is tiny compared with the drift SD.
    assert np.max(np.abs(np.diff(freq))) < 1e-3
    assert freq.std() > 0.1


def test_shared_drift_scales_all_fish_by_the_same_fraction() -> None:
    n, fs = 50_000, 1_000.0
    shared = shared_frequency_drift(
        np.random.default_rng(4),
        n,
        fs,
        SharedDriftParams(sd_rel=0.01, timescale_s=20.0, control_rate_hz=0.5),
    )
    still = DriftParams(sd_hz=0.0)
    f_low = baseline_frequency(np.random.default_rng(5), n, fs, 600.0, still, shared)
    f_high = baseline_frequency(np.random.default_rng(6), n, fs, 900.0, still, shared)
    np.testing.assert_allclose(f_low / 600.0, f_high / 900.0, rtol=1e-12)
    np.testing.assert_allclose(f_low / 600.0 - 1.0, shared, atol=1e-12)


def test_drift_grid_must_resolve_timescale() -> None:
    with pytest.raises(ValueError, match="control grid"):
        DriftParams(timescale_s=1.0, control_rate_hz=1.0)
