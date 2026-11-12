"""A small synthetic fishgrid trial for the calibration tests.

Two fish with constant EOD frequencies (712 and 772 Hz) on a 3x3 grid whose
9th channel carries only noise, as in the 2019 recordings. ``sign_v`` is
computed from the raw data exactly as wavetracker did (Hann PSD at the bin
nearest the fundamental, 16384-point frames) and holds 8 columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from gridtools.calibrate.trials import TrialMeta

FS = 20_000.0
DURATION_S = 40.0
NFFT = 16384
HOP = 2458
NOISE_SD = 1e-5
# Away from 50-Hz multiples: fish near a hum harmonic are skipped for amplitudes.
FISH_HZ = (712.0, 772.0)
# Fundamental amplitude (V) of each fish on channels 0..8 (channel 8 dead).
AMPLITUDES = (
    np.array([4e-4, 2e-4, 1e-4, 2e-4, 1e-4, 5e-5, 1e-4, 5e-5, 0.0]),
    np.array([3e-5, 6e-5, 1e-4, 6e-5, 1e-4, 2e-4, 1e-4, 3e-4, 0.0]),
)
CONFIG = """*FishGrid
  Grid &1:
      Columns1         : 3
      Rows1            : 3
      ColumnDistance1  : 50.0cm
      RowDistance1     : 50.0cm
  Hardware Settings:
      DAQ board:
          AISampleRate: 20.000kHz
          AIMaxVolt   : 20.0mV
"""


@dataclass(frozen=True)
class SyntheticTrial:
    path: Path
    meta: TrialMeta
    chirp_times_s: dict[int, np.ndarray]


def _write_trial(path: Path) -> SyntheticTrial:
    rng = np.random.default_rng(42)
    n = round(DURATION_S * FS)
    t = np.arange(n) / FS
    data = rng.normal(0.0, NOISE_SD, (n, 9))
    for f, amps in zip(FISH_HZ, AMPLITUDES, strict=True):
        data += np.cos(2 * np.pi * f * t)[:, None] * amps[None, :]
    data = data.astype(np.float32)
    path.mkdir(parents=True)
    (path / "fishgrid.cfg").write_text(CONFIG, encoding="utf-8")
    data.tofile(path / "traces-grid1.raw")

    centres = np.arange(NFFT // 2, n - NFFT // 2, HOP)
    times = centres / FS
    window = np.hanning(NFFT + 1)[:NFFT]
    fund, ident, idx, sign = [], [], [], []
    for k, c in enumerate(centres):
        x = data[c - NFFT // 2 : c + NFFT // 2].astype(np.float64)
        x -= x.mean(axis=0)
        power = 2 * np.abs(np.fft.rfft(x * window[:, None], axis=0)) ** 2
        power /= FS * np.sum(window**2)
        for i, f in enumerate(FISH_HZ, start=1):
            b = round(f * NFFT / FS)
            fund.append(f)
            ident.append(float(i))
            idx.append(k)
            sign.append(10 * np.log10(power[b, :8]))
    np.save(path / "times.npy", times)
    np.save(path / "fund_v.npy", np.asarray(fund))
    np.save(path / "ident_v.npy", np.asarray(ident))
    np.save(path / "idx_v.npy", np.asarray(idx, dtype=np.int64))
    np.save(path / "sign_v.npy", np.asarray(sign))
    (path / "temperatures.csv").write_text(
        "time/s; T/C    \n     0;  25.000\n    20;  24.900\n    40;  24.800\n",
        encoding="utf-8",
    )
    chirps = {1: np.array([5.0, 12.0, 12.02, 30.0]), 2: np.array([5.35, 20.0])}
    rows = ["recording,t1,f1,t2,f2,score,emitter_eodf,track_id"]
    for ident_, ts in chirps.items():
        for c in ts:
            rows.append(
                f"x,{c - 0.07},700,{c + 0.07},900,0.9,{FISH_HZ[ident_ - 1]},{ident_}"
            )
    rows.append("x,25.0,1000,25.1,1200,0.7,1100,")
    (path / "chirpdetector_bboxes.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    meta = TrialMeta(path.name, 1, 2, "f", "m", 12.0, 11.0)
    return SyntheticTrial(path, meta, chirps)


@pytest.fixture(scope="session")
def synthetic_trial(tmp_path_factory: pytest.TempPathFactory) -> SyntheticTrial:
    """A 40-s, 3x3-grid trial with two fish and a dead 9th channel."""
    return _write_trial(tmp_path_factory.mktemp("trials") / "2019-01-01-10_00")
