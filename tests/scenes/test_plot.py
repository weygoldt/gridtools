"""Annotated scene spectrograms."""

import itertools

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from gridtools.sim.plot import _zoom_windows, summed_spectrogram


def test_summed_spectrogram_peaks_at_the_tone() -> None:
    fs = 20_000.0
    t = np.arange(40_000) / fs
    x = np.stack([np.sin(2 * np.pi * 800 * t), 0.5 * np.sin(2 * np.pi * 800 * t)], 1)
    times, freqs, power = summed_spectrogram(
        x, fs, nfft=4096, hop=512, f_min_hz=500, f_max_hz=1500
    )
    assert freqs[0] >= 500
    assert freqs[-1] <= 1500
    assert power.shape == (freqs.size, times.size)
    assert abs(freqs[np.argmax(power[:, 10])] - 800) < fs / 4096
    assert times[0] == pytest.approx(2048 / fs)


def test_zoom_windows_prefer_hard_cases_and_do_not_overlap() -> None:
    chirps = pd.DataFrame(
        {
            "t_center_s": [1.0, 10.0, 10.2, 30.0, 50.0],
            "overlaps_other_chirp": [False, True, True, False, False],
            "min_df_hz": [100.0, 50.0, 50.0, 3.0, np.nan],
            "family": ["small"] * 5,
        }
    )
    windows = _zoom_windows(chirps, 60.0, 2, 5.0)
    assert len(windows) == 2
    assert windows[0][0] <= 10.0 <= windows[0][1]
    assert windows[1][0] <= 30.0 <= windows[1][1]
    windows = _zoom_windows(chirps, 60.0, 10, 5.0)
    for (_, b), (c, _) in itertools.pairwise(windows):
        assert b <= c
