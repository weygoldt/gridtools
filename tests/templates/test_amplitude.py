"""Amplitude envelope from the inverted frequency excursion (thesis model)."""

from __future__ import annotations

import numpy as np
import pytest

from gridtools.templates import DEFAULT_DIP_RANGE, amplitude_from_frequency, random_dips

from .conftest import gaussian_chirp


def test_amplitude_inverts_and_scales_the_excursion() -> None:
    delta = gaussian_chirp(2000, 1000, 120.0, 0.01)
    delta[1300:1400] -= 15.0  # undershoot
    delta[0] = delta[-1] = 0.0
    amp = amplitude_from_frequency(delta, 0.4)
    assert amp[1000] == pytest.approx(0.6)
    assert amp.min() == pytest.approx(0.6)
    assert amp[0] == 1.0
    assert amp[-1] == 1.0
    # the undershoot is clipped: amplitude never exceeds 1
    assert amp.max() == 1.0
    np.testing.assert_array_equal(amp[1300:1400], 1.0)
    # proportional to the positive excursion
    np.testing.assert_allclose(1 - amp[800:1200], 0.4 * delta[800:1200] / 120.0)


def test_amplitude_row_wise_dips() -> None:
    delta = np.stack(
        [gaussian_chirp(500, 250, 60.0, 0.005), gaussian_chirp(500, 250, 400.0, 0.008)]
    )
    amp = amplitude_from_frequency(delta, np.array([0.0, 0.8]))
    np.testing.assert_array_equal(amp[0], 1.0)
    assert amp[1].min() == pytest.approx(0.2)
    flat = amplitude_from_frequency(np.zeros((2, 10)), 0.5)
    np.testing.assert_array_equal(flat, 1.0)


def test_amplitude_rejects_bad_dip() -> None:
    with pytest.raises(ValueError, match="dip"):
        amplitude_from_frequency(np.ones(5), 1.5)


def test_random_dips_default_range_and_seed() -> None:
    assert DEFAULT_DIP_RANGE == (0.0, 0.8)
    a = random_dips(np.random.default_rng(0), 1000)
    b = random_dips(np.random.default_rng(0), 1000)
    np.testing.assert_array_equal(a, b)
    assert a.min() >= 0.0
    assert a.max() <= 0.8
    with pytest.raises(ValueError, match="dip_range"):
        random_dips(np.random.default_rng(0), 3, (0.5, 1.2))
