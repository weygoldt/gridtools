"""Unrolling, alignment and template construction on synthetic source rows."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from gridtools.templates.build import process_snippet
from gridtools.templates.config import BuildConfig
from gridtools.templates.source import find_wrap_index, unroll_rows

from .conftest import FS, N_SOURCE, SourceRow


def test_wrap_index_and_unroll(source_row: Callable[..., SourceRow]) -> None:
    row = source_row(seed=1, chirp_centre=4300.0)
    stored = row.traces["smooth_freq"].astype(np.float64)
    wrap = find_wrap_index(stored)
    assert wrap == row.shift % N_SOURCE
    contiguous = unroll_rows(stored, wrap)
    assert contiguous[0] == 0.0
    # contiguous again: no jump larger than the chirp's own slope
    assert np.max(np.abs(np.diff(contiguous))) < 5.0
    assert not np.shares_memory(contiguous, stored)


def test_wrap_index_missing_or_ambiguous() -> None:
    assert find_wrap_index(np.ones(10)) == -1
    assert find_wrap_index(np.array([0.0, 1.0, 0.0])) == -1


@pytest.mark.parametrize("centre", [3700.0, 4000.0, 4450.0])
def test_process_snippet_alignment_and_metadata(
    source_row: Callable[..., SourceRow], centre: float
) -> None:
    config = BuildConfig()
    row = source_row(
        seed=2, excursion_hz=85.0, fwhm_s=0.009, amp_dip=0.3, chirp_centre=centre
    )
    before = {k: v.copy() for k, v in row.traces.items()}
    traces = {k: v.astype(np.float64) for k, v in row.traces.items()}
    out = process_snippet(traces, FS, config)
    n = config.n_samples(FS)
    assert out.delta_f_hz.shape == (n,)
    # aligned: the 50% midpoint sits at the template centre
    assert abs(out.metrics.onset_50_s + out.metrics.offset_50_s) / 2 < 1.0 / FS
    # smooth_freq is the trace convolved with a Gaussian (sigma = 1 ms), which
    # lowers and widens a Gaussian chirp in a known way
    sigma_chirp = 0.009 / (2 * np.sqrt(2 * np.log(2)))
    sigma_total = np.hypot(sigma_chirp, 0.001)
    assert out.metrics.excursion_hz == pytest.approx(
        85.0 * sigma_chirp / sigma_total, rel=0.01
    )
    fwhm_total = sigma_total * 2 * np.sqrt(2 * np.log(2))
    assert out.metrics.duration_50_s == pytest.approx(fwhm_total, rel=0.02)
    # the extraction-filter gain is undone: the true dip comes back
    assert out.metrics.amp_dip == pytest.approx(0.3, rel=0.05)
    assert out.amp_dip_uncorrected < 0.3 - 0.05
    assert out.filter_gain_max > 1.05
    assert out.analysis.baseline_eodf_hz == pytest.approx(800.0, abs=0.5)
    assert out.delta_f_hz[0] == 0.0
    assert out.delta_f_hz[-1] == 0.0
    assert out.amp_rel[0] == 1.0
    assert out.amp_rel[-1] == 1.0
    assert not out.truncated
    for k, v in row.traces.items():
        np.testing.assert_array_equal(v, before[k])


def test_process_snippet_flags_truncated(source_row: Callable[..., SourceRow]) -> None:
    row = source_row(seed=4, chirp_centre=7900.0)
    traces = {k: v.astype(np.float64) for k, v in row.traces.items()}
    out = process_snippet(traces, FS, BuildConfig())
    assert out.truncated
    assert out.delta_f_hz[0] == 0.0
    assert out.delta_f_hz[-1] == 0.0
    assert out.amp_rel[0] == 1.0
    assert out.amp_rel[-1] == 1.0
