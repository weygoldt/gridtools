"""Decay extension of the template support for long chirps."""

from __future__ import annotations

import numpy as np
import pytest

from gridtools.templates import BuildConfig
from gridtools.templates.build import template_from_traces
from gridtools.templates.features import decay_end

from .conftest import FS, gaussian_chirp, plateau_decay_chirp


def test_decay_end_reaches_level_or_stops_at_a_trough() -> None:
    t = np.arange(4000) / FS
    decay = 400.0 * np.exp(-t / 0.05)
    end = decay_end(decay, 0.0, 8.0)
    expected = 0.05 * np.log(400.0 / 8.0) * FS
    assert end.position == pytest.approx(expected, abs=2)
    assert not end.interrupted
    assert not end.clipped
    # a following chirp starts before the decay has reached the level
    followed = decay + 300.0 * (t > 0.1)
    end = decay_end(followed, 0.0, 8.0)
    assert end.interrupted
    assert end.position == pytest.approx(0.1 * FS, abs=2)
    end = decay_end(400.0 * np.exp(-t / 1.0), 0.0, 8.0)
    assert end.clipped


def _traces(delta: np.ndarray, baseline_hz: float = 800.0) -> dict[str, np.ndarray]:
    freq = baseline_hz + delta
    return {
        "orig_freq": freq,
        "smooth_freq": freq,
        "orig_amp": np.ones_like(freq),
        "smooth_amp": np.ones_like(freq),
    }


@pytest.mark.parametrize(
    ("plateau_s", "tau_s"), [(0.10, 0.05), (0.15, 0.06), (0.25, 0.05)]
)
def test_decay_extension_covers_the_slow_decay(plateau_s: float, tau_s: float) -> None:
    t = np.arange(round(1.4 * FS)) / FS
    delta = plateau_decay_chirp(t, 0.5, 450.0, plateau_s, tau_s)
    end_2pct = 0.5 + plateau_s + tau_s * np.log(50.0)
    extended = BuildConfig(template_duration_s=1.2, decay_level_rel=0.02)
    snippet = template_from_traces(_traces(delta), FS, extended, absolute_freq=True)
    a = snippet.analysis
    assert a.decay is not None
    assert not a.decay.interrupted
    assert a.chirp_end / FS == pytest.approx(end_2pct, abs=0.003)
    assert a.support_stop / FS >= end_2pct
    # inside the support the template is the chirp, untouched (up to the
    # baseline estimate, which sees the far end of the exponential tail)
    centre = snippet.delta_f_hz.size // 2
    first = a.align_index - centre
    support = np.arange(int(np.ceil(a.support_start)), int(a.support_stop))
    np.testing.assert_allclose(
        snippet.delta_f_hz[support - first], delta[support], atol=0.01
    )
    # exact ends
    assert snippet.delta_f_hz[0] == 0.0
    assert snippet.delta_f_hz[-1] == 0.0
    assert snippet.amp_rel[0] == 1.0
    assert snippet.amp_rel[-1] == 1.0
    # the v2 support stops 30 ms after the 10% offset and cuts the decay
    v2 = template_from_traces(
        _traces(delta), FS, BuildConfig(template_duration_s=1.2), absolute_freq=True
    )
    assert v2.analysis.decay is None
    assert v2.analysis.support_stop == pytest.approx(
        v2.analysis.crossing_10.offset + 0.03 * FS
    )
    assert v2.analysis.support_stop / FS < end_2pct - 0.05


def test_short_chirps_keep_the_v2_support() -> None:
    # the library default (v2) has no decay extension: the support ends exactly
    # 30 ms after the 10% offset, as before
    delta = gaussian_chirp(8000, 4000.0, 80.0, 0.01)
    v2 = template_from_traces(_traces(delta), FS, BuildConfig(), absolute_freq=True)
    assert BuildConfig().decay_level_rel is None
    assert v2.analysis.decay is None
    assert v2.analysis.chirp_end == v2.analysis.crossing_10.offset
    assert v2.analysis.support_stop == v2.analysis.crossing_10.offset + 0.03 * FS
    # switched on (harvest only), a short chirp's support moves by less than
    # the 5 ms smoothing of the decay walk, and nothing else changes
    on = template_from_traces(
        _traces(delta), FS, BuildConfig(decay_level_rel=0.02), absolute_freq=True
    )
    shift_s = (on.analysis.support_stop - v2.analysis.support_stop) / FS
    assert 0.0 <= shift_s < 0.01
    assert on.analysis.support_start == v2.analysis.support_start
    assert on.metrics.duration_10_s == v2.metrics.duration_10_s
    assert on.metrics.excursion_hz == v2.metrics.excursion_hz
