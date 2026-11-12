"""Checks on the real tube-competition recordings (skipped without the data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gridtools.calibrate.measure import MeasureConfig, measure_trial
from gridtools.calibrate.trials import (
    ADC_STEP_V,
    open_raw,
    read_trial_summary,
    resolve_trial_dir,
)

ROOT = Path("/mnt/data2/2019_tube_competition")

pytestmark = pytest.mark.data


def test_partial_frame_recordings_open() -> None:
    rec = open_raw(resolve_trial_dir(ROOT, "2020-03-19"))
    assert rec.n_channels == 15
    assert rec.duration_s > 5 * 3600


def test_raw_units_are_adc_volts() -> None:
    rec = open_raw(resolve_trial_dir(ROOT, "2020-05-11"))
    x = rec.read(2_000_000, 2_100_000, [0]).astype(np.float64).ravel()
    steps = np.diff(np.unique(x))
    assert np.median(steps) == pytest.approx(ADC_STEP_V, rel=0.01)


@pytest.mark.slow
def test_2019_sign_v_is_raw_channels_1_to_8_and_channel_9_is_dead() -> None:
    trial = resolve_trial_dir(ROOT, "2019-12-09")
    meta = read_trial_summary(ROOT)[trial.name]
    config = MeasureConfig(n_segments=4, n_profile_samples=200, n_shuffles=5)
    stats = measure_trial(trial, meta, np.random.default_rng(0), config)
    assert stats.layout == "9-8"
    assert stats.sign_v_nfft == 32768
    assert stats.noise.dead.tolist() == [False] * 8 + [True]
    assert stats.noise.sign_v_best_channel.tolist() == list(range(8))
    np.testing.assert_allclose(
        np.nanmedian(stats.noise.sign_v_check_db, axis=0), 0.0, atol=0.05
    )
