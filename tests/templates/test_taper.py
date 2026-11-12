"""Taper boundary conditions, smoothness and copy semantics."""

from __future__ import annotations

import numpy as np
import pytest

from gridtools.templates.taper import (
    apply_taper_delta,
    apply_taper_ratio,
    crop_centred,
    taper_window,
)


def test_window_exact_values_and_smoothness() -> None:
    w = taper_window(4000, 1700.0, 2300.0, 200.0)
    assert not w.truncated
    assert w.weights[0] == 0.0
    assert w.weights[-1] == 0.0
    assert np.all(w.weights[:1500] == 0.0)
    assert np.all(w.weights[2501:] == 0.0)
    assert np.all(w.weights[1700:2301] == 1.0)
    assert np.all((w.weights >= 0.0) & (w.weights <= 1.0))
    # raised cosine: first difference continuous (no kinks larger than the ramp curvature)
    d1 = np.diff(w.weights)
    assert np.max(np.abs(d1)) <= np.pi / (2 * 200) + 1e-12
    assert np.max(np.abs(np.diff(d1))) <= (np.pi / 200) ** 2 / 2 + 1e-12


def test_window_shrinks_support_when_truncated() -> None:
    w = taper_window(4000, -50.0, 2300.0, 200.0, valid_start=100)
    assert w.truncated
    assert w.support_start == pytest.approx(300.0)
    assert np.all(w.weights[:101] == 0.0)
    assert w.weights[300] == 1.0


def test_window_rejects_bad_ramp() -> None:
    with pytest.raises(ValueError, match="ramp"):
        taper_window(100, 40.0, 60.0, 0.0)


def test_taper_ends_exact_and_inputs_untouched() -> None:
    rng = np.random.default_rng(1)
    delta = rng.normal(0, 5, 4000) - 3.0
    ratio = 1.0 + rng.normal(0, 0.05, 4000)
    delta_before, ratio_before = delta.copy(), ratio.copy()
    w = taper_window(4000, 1800.0, 2200.0, 100.0)
    out_delta = apply_taper_delta(delta, w.weights)
    out_ratio = apply_taper_ratio(ratio, w.weights)
    assert out_delta[0] == 0.0
    assert out_delta[-1] == 0.0
    assert not np.signbit(out_delta[0])
    assert out_ratio[0] == 1.0
    assert out_ratio[-1] == 1.0
    np.testing.assert_array_equal(out_delta[1800:2201], delta[1800:2201])
    np.testing.assert_array_equal(delta, delta_before)
    np.testing.assert_array_equal(ratio, ratio_before)
    assert not np.shares_memory(out_delta, delta)


def test_crop_centred_pads_and_copies() -> None:
    trace = np.arange(100, dtype=np.float64)
    crop, lo, hi = crop_centred(trace, 10, 40, fill=-1.0)
    assert crop.size == 40
    assert crop[20] == 10.0
    assert (lo, hi) == (10, 40)
    assert np.all(crop[:10] == -1.0)
    crop[25] = 999.0
    assert trace[15] == 15.0
