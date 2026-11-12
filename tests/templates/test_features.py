"""Measurements on synthetic chirps with known shape."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from gridtools.templates.features import (
    detrended_rms,
    dominant_frequency,
    extraction_filter_gain,
    find_knots,
    knot_eodf,
    level_crossings,
    mask_segments,
    measure_template,
    step_score,
)

from .conftest import FS, SourceRow, gaussian_chirp

N = 4000
CENTRE = N // 2


def fw_at(level: float, fwhm_s: float) -> float:
    """Full width of a Gaussian at ``level`` of its peak, given its FWHM."""
    sigma = fwhm_s / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return 2.0 * sigma * np.sqrt(-2.0 * np.log(level))


@pytest.mark.parametrize(
    ("excursion", "fwhm"), [(60.0, 0.008), (400.0, 0.02), (35.0, 0.004)]
)
def test_crossings_of_gaussian(excursion: float, fwhm: float) -> None:
    delta = gaussian_chirp(N, CENTRE + 0.3, excursion, fwhm)
    peak = int(np.argmax(delta))
    c50 = level_crossings(delta, peak, 0.5)
    c10 = level_crossings(delta, peak, 0.1)
    assert c50.width / FS == pytest.approx(fwhm, abs=1.5 / FS)
    assert c10.width / FS == pytest.approx(fw_at(0.1, fwhm), abs=1.5 / FS)
    assert c50.midpoint == pytest.approx(CENTRE + 0.3, abs=0.2)
    assert not (c10.onset_clipped)
    assert not (c10.offset_clipped)


def test_crossings_clipped_at_edge() -> None:
    delta = gaussian_chirp(N, 5.0, 100.0, 0.01)
    c = level_crossings(delta, int(np.argmax(delta)), 0.1)
    assert c.onset_clipped
    assert c.onset == 0.0
    assert not c.offset_clipped


def test_crossings_need_positive_peak() -> None:
    with pytest.raises(ValueError, match="positive"):
        level_crossings(np.zeros(10), 3, 0.5)


def test_measure_template_known_shape() -> None:
    excursion, fwhm, dip = 90.0, 0.01, 0.25
    delta = gaussian_chirp(N, CENTRE, excursion, fwhm)
    # undershoot: negative Gaussian 22 ms after the peak
    delta -= gaussian_chirp(N, CENTRE + 0.022 * FS, 12.0, 0.006)
    amp = 1.0 - dip * gaussian_chirp(N, CENTRE, 1.0, 0.012)
    m = measure_template(delta, amp, FS)
    assert m.excursion_hz == pytest.approx(excursion, rel=1e-3)
    assert m.duration_50_s == pytest.approx(fwhm, abs=2 / FS)
    assert m.duration_10_s == pytest.approx(fw_at(0.1, fwhm), abs=2 / FS)
    assert m.peak_time_s == pytest.approx(0.0, abs=1 / FS)
    assert m.onset_50_s == pytest.approx(-fwhm / 2, abs=2 / FS)
    assert m.undershoot_hz == pytest.approx(12.0, rel=0.05)
    assert m.amp_dip == pytest.approx(dip, rel=1e-3)
    assert m.amp_peak == pytest.approx(0.0, abs=1e-9)
    assert m.asymmetry == pytest.approx(0.0, abs=0.02)
    assert m.n_peaks == 1
    assert not m.clipped


def test_measure_template_asymmetry_sign() -> None:
    t = np.arange(N, dtype=np.float64)
    rise = np.exp(-0.5 * ((t - CENTRE) / 20.0) ** 2)
    fall = np.exp(-0.5 * ((t - CENTRE) / 80.0) ** 2)
    delta = 100.0 * np.where(t < CENTRE, rise, fall)
    m = measure_template(delta, np.ones(N), FS)
    assert m.asymmetry == pytest.approx(0.6, abs=0.01)  # (80 - 20) / (80 + 20)


def test_measure_template_counts_two_peaks() -> None:
    delta = gaussian_chirp(N, CENTRE - 300, 80.0, 0.006) + gaussian_chirp(
        N, CENTRE + 300, 70.0, 0.006
    )
    m = measure_template(delta, np.ones(N), FS)
    assert m.n_peaks == 1
    assert m.n_extra_events == 1


def test_measure_template_small_wiggle_is_one_peak() -> None:
    delta = gaussian_chirp(N, CENTRE, 80.0, 0.01) + 2.0 * np.sin(np.arange(N) / 15.0)
    delta[:200] = 0.0
    m = measure_template(delta, np.ones(N), FS)
    assert m.n_peaks == 1
    assert m.n_extra_events == 0


def test_measure_template_counts_internal_structure() -> None:
    # a long chirp with a notch: one event, two internal peaks
    delta = gaussian_chirp(N, CENTRE - 60, 300.0, 0.006) + gaussian_chirp(
        N, CENTRE + 60, 300.0, 0.006
    )
    m = measure_template(delta, np.ones(N), FS)
    assert m.n_peaks == 2
    assert m.n_extra_events == 0


def test_extraction_filter_gain() -> None:
    from scipy.signal import butter, sosfreqz

    eodf = 800.0
    f = np.array([700.0, 800.0, 860.0, 1100.0])
    sos = butter(2, [eodf - 100, 1.9 * eodf], btype="bandpass", output="sos", fs=40_000)
    _, h = sosfreqz(sos, worN=f, fs=40_000)
    np.testing.assert_allclose(
        extraction_filter_gain(f, eodf), np.abs(h) ** 2, rtol=5e-3
    )
    # a 60 Hz rise raises the measured amplitude by about 12%
    ratio = extraction_filter_gain(860.0, eodf) / extraction_filter_gain(800.0, eodf)
    assert float(ratio) == pytest.approx(1.125, abs=0.01)


def test_knot_eodf_recovers_absolute_baseline(
    source_row: Callable[..., SourceRow],
) -> None:
    row = source_row(
        seed=3, baseline_eodf_hz=812.3, excursion_hz=120.0, chirp_centre=4000.0
    )
    orig = np.roll(row.traces["orig_freq"].astype(np.float64), -row.shift)
    knots = find_knots(orig)
    # one knot per EOD cycle
    assert knots.size == pytest.approx(0.4 * 812.3, abs=60)
    baseline = np.ones(orig.size, dtype=bool)
    baseline[3500:4600] = False
    chirp = ~baseline
    est = knot_eodf(orig, baseline, chirp, FS)
    eodf = est.offset_hz + np.median(orig[baseline])
    assert eodf == pytest.approx(812.3, abs=0.5)
    assert est.mad_hz < 0.5
    # the identity also holds inside the chirp, so the relative trace is in Hz
    assert est.chirp_mad_hz < 2.0


def test_detrended_rms_of_sinusoid() -> None:
    t = np.arange(8000) / FS
    trace = 3.0 * np.sin(2 * np.pi * 40.0 * t) + 5.0 * t + 2.0
    mask = np.ones(t.size, dtype=bool)
    mask[3000:5000] = False
    rms, longest = detrended_rms(trace, mask, min_samples=400)
    assert rms == pytest.approx(3.0 / np.sqrt(2.0), rel=0.03)
    assert dominant_frequency(longest, FS) == pytest.approx(40.0, abs=1.0)


def test_mask_segments() -> None:
    mask = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert mask_segments(mask) == [(0, 2), (4, 5), (6, 9)]


def test_step_score_detects_jump_not_smooth_chirp() -> None:
    smooth = 1.0 - 0.4 * gaussian_chirp(8000, 4000, 1.0, 0.01)
    assert step_score(smooth, min_step=0.005) < 5.0
    stepped = smooth.copy()
    stepped[6000:] *= 0.9
    assert step_score(stepped, min_step=0.005) > 20.0


def test_crossings_bridge_brief_internal_dips() -> None:
    # a 60 ms plateau with a 1 ms dip to zero in the middle (a phase slip)
    delta = np.zeros(N)
    delta[CENTRE - 600 : CENTRE + 600] = 300.0
    delta[CENTRE + 100 : CENTRE + 120] = 0.0
    peak = CENTRE - 300
    unbridged = level_crossings(delta, peak, 0.5)
    bridged = level_crossings(delta, peak, 0.5, max_gap=100)
    assert unbridged.offset < CENTRE + 101
    assert bridged.offset == pytest.approx(CENTRE + 599.5, abs=1.0)
    # gaps longer than max_gap still split events
    delta[CENTRE + 100 : CENTRE + 300] = 0.0
    assert level_crossings(delta, peak, 0.5, max_gap=100).offset < CENTRE + 101


def test_bridging_does_not_merge_baseline_ripple() -> None:
    from gridtools.templates.features import chirp_crossings

    t = np.arange(N) / FS
    chirp = gaussian_chirp(N, CENTRE, 60.0, 0.008)
    ripple = 12.0 * np.sin(2 * np.pi * 120.0 * t)  # above 10% of 60 Hz, gaps < 5 ms
    delta = chirp + ripple
    peak = int(np.argmax(delta))
    c10, c50 = chirp_crossings(delta, peak, max_gap=100)
    assert c10.width / FS < 0.03
    assert c50.width / FS == pytest.approx(0.008, abs=0.003)
    # a plateau with a brief dip below 10% stays one event
    plateau = np.zeros(N)
    plateau[CENTRE - 600 : CENTRE + 600] = 300.0
    plateau[CENTRE + 100 : CENTRE + 120] = 0.0
    c10, c50 = chirp_crossings(plateau, CENTRE - 300, max_gap=100)
    assert c50.width == pytest.approx(1199, abs=2)
    assert c10.width == pytest.approx(1199, abs=2)
