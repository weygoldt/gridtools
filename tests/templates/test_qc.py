"""Duplicate detection and robust statistics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from gridtools.templates.qc import (
    exact_duplicates,
    max_lag_correlation,
    near_duplicates,
    robust_z,
)

from .conftest import gaussian_chirp

L = 2000


def test_exact_duplicates_point_to_first() -> None:
    hashes = [b"a", b"b", b"a", b"c", b"b", b"a"]
    np.testing.assert_array_equal(exact_duplicates(hashes), [-1, -1, 0, -1, 1, 0])


def test_max_lag_correlation_finds_shift() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    b = np.roll(a, 3)
    corr, lag = max_lag_correlation(a, b, 5)
    assert corr > 0.99
    assert lag == -3


def test_near_duplicates_need_shape_and_fingerprint() -> None:
    rng = np.random.default_rng(7)
    chirp = gaussian_chirp(L, L // 2, 80.0, 0.008)
    # cycle-to-cycle jitter is spline-interpolated, so it is correlated over ~1 cycle
    jitter = [gaussian_filter1d(rng.normal(0, 1.0, L), 6) for _ in range(4)]
    shape = np.stack(
        [
            chirp + 0.2 * jitter[0],  # 0: original
            np.roll(chirp + 0.2 * jitter[0], 1)
            + 1e-4 * rng.normal(size=L),  # 1: re-extraction
            chirp + 0.2 * jitter[2],  # 2: stereotyped twin, independent jitter
            gaussian_chirp(L, L // 2, 300.0, 0.02),  # 3: different chirp
        ]
    )
    fingerprint = np.stack(
        [
            jitter[0],
            np.roll(jitter[0], 1) + 0.01 * rng.normal(size=L),
            jitter[2],
            jitter[3],
        ]
    )
    result = near_duplicates(
        shape, fingerprint, shape_corr_min=0.999, fingerprint_corr_min=0.9
    )
    np.testing.assert_array_equal(result.duplicate_of, [-1, 0, -1, -1])
    # the stereotyped twin reaches the shape threshold on its own but is kept
    assert result.max_shape_corr[2] > 0.999
    assert np.isnan(result.max_fingerprint_corr[2])
    assert len(result.pairs) == 1


def test_near_duplicates_respects_exclude() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=(1, L))
    shape = np.repeat(base, 3, axis=0)
    result = near_duplicates(
        shape,
        shape,
        shape_corr_min=0.999,
        fingerprint_corr_min=0.9,
        exclude=np.array([False, True, False]),
    )
    np.testing.assert_array_equal(result.duplicate_of, [-1, -1, 0])


def test_robust_z() -> None:
    z = robust_z([1.0, 2.0, 3.0, 4.0, 100.0, np.nan])
    assert z[2] == pytest.approx(0.0)
    assert z[4] > 10
    assert np.isnan(z[5])


def test_inferred_groups_split_on_eodf_jumps() -> None:
    from gridtools.templates.build import inferred_groups

    eodf = np.array([800.0, 800.5, 801.0, 850.0, 851.0, np.nan, 852.0, 700.0])
    np.testing.assert_array_equal(inferred_groups(eodf, 10.0), [0, 0, 0, 1, 1, 2, 3, 4])
    assert inferred_groups(np.empty(0), 10.0).size == 0
