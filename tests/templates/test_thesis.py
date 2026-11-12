"""The ported thesis baseline cross-fade."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from gridtools.templates.thesis import thesis_crossfade, thesis_support

from .conftest import gaussian_chirp


def test_thesis_support_includes_the_undershoot() -> None:
    rng = np.random.default_rng(0)
    trace = 800.0 + gaussian_chirp(8000, 4000, 100.0, 0.01)
    trace -= gaussian_chirp(8000, 4000 + 400, 15.0, 0.01)  # undershoot 20 ms later
    trace += rng.normal(0, 0.2, trace.size)
    trace -= trace[0]  # relative to the window start, as stored
    support = thesis_support(trace)
    assert support is not None
    assert abs(support.baseline) < 1.0
    assert support.start < 4000 - 100
    assert support.stop > 4000 + 400  # past the undershoot minimum


def test_thesis_crossfade_flattens_the_baseline() -> None:
    rng = np.random.default_rng(1)
    # smoothed jitter, as in smooth_freq (the thesis worked on smoothed traces)
    jitter = gaussian_filter1d(rng.normal(0, 5.0, 8000), 20)
    trace = 800.0 + gaussian_chirp(8000, 4000, 100.0, 0.01) + jitter
    out = thesis_crossfade(trace - trace[0])
    assert out is not None
    assert np.max(np.abs(out[:2000])) < 0.5  # smoothed baseline
    assert abs(out[4000] - 100.0) < 3.0  # chirp kept
    assert thesis_crossfade(np.sin(np.arange(8000) / 50.0)) is None  # many crossings
