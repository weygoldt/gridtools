"""Synthetic chirps and source files for the template tests.

The synthetic source rows mimic ``extracted_chirps.h5``: a contiguous
window whose ``orig_freq`` is a cubic spline through one value per EOD cycle
(placed at the zero crossings), ``smooth_freq`` its Gaussian-smoothed
version, amplitudes divided by the window maximum, frequencies relative to
the first window sample, and the whole row circularly shifted so the chirp
centre lands at the middle sample.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from gridtools.templates.features import extraction_filter_gain

FS = 20_000.0
N_SOURCE = 8000


def gaussian_chirp(
    n: int, centre: float, excursion_hz: float, fwhm_s: float, fs: float = FS
) -> npt.NDArray[np.float64]:
    """Gaussian frequency excursion with a known peak and full width at half maximum."""
    sigma = fwhm_s * fs / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    t = np.arange(n, dtype=np.float64)
    return excursion_hz * np.exp(-0.5 * ((t - centre) / sigma) ** 2)


def plateau_chirp(
    n: int, centre: float, excursion_hz: float, duration_s: float, fs: float = FS
) -> npt.NDArray[np.float64]:
    """Long chirp: a plateau of ``duration_s`` (at half height) with 1 ms edges."""
    t = (np.arange(n, dtype=np.float64) - centre) / fs
    edge = 0.001
    rise = 1.0 / (1.0 + np.exp(-(t + duration_s / 2) / edge))
    fall = 1.0 / (1.0 + np.exp((t - duration_s / 2) / edge))
    return excursion_hz * rise * fall


@dataclass(frozen=True)
class SourceRow:
    """One synthetic source row plus its ground truth."""

    traces: dict[str, npt.NDArray[np.float32]]
    baseline_eodf_hz: float
    excursion_hz: float
    fwhm_s: float
    amp_dip: float
    chirp_centre: float  # in the contiguous window
    shift: int  # roll applied when storing


def make_source_row(
    rng: np.random.Generator,
    *,
    baseline_eodf_hz: float = 800.0,
    excursion_hz: float = 80.0,
    fwhm_s: float = 0.008,
    amp_dip: float = 0.2,
    chirp_centre: float = 4100.0,
    jitter_s: float = 2e-6,
    ripple_hz: float = 0.0,
    ripple_freq_hz: float = 30.0,
    plateau_s: float = 0.0,
) -> SourceRow:
    """Synthesise one stored row the way the thesis extraction produced it.

    With ``plateau_s > 0`` the chirp is a plateau of that length (1 ms
    logistic edges) instead of a Gaussian, and ``fwhm_s`` is ignored.
    """
    n = N_SOURCE
    t = np.arange(n) / FS
    # absolute instantaneous frequency on a fine grid, integrate to get zero crossings
    if plateau_s > 0:
        excursion = plateau_chirp(n, chirp_centre, excursion_hz, plateau_s)
    else:
        excursion = gaussian_chirp(n, chirp_centre, excursion_hz, fwhm_s)
    ripple = ripple_hz * np.sin(
        2 * np.pi * ripple_freq_hz * t + rng.uniform(0, 2 * np.pi)
    )
    f_abs = baseline_eodf_hz + excursion + ripple
    upsample = 20
    t_fine = np.arange(n * upsample + 1) / (FS * upsample)
    f_fine = np.interp(
        t_fine, np.append(t, t[-1] + 1 / FS), np.append(f_abs, f_abs[-1])
    )
    phase = np.concatenate([[0.0], np.cumsum(f_fine[:-1]) / (FS * upsample)])
    phase += rng.uniform(0, 1)
    cycles = np.arange(np.ceil(phase[0]), np.floor(phase[-1]) + 1)
    t_zero = np.interp(cycles, phase, t_fine)
    # zero-crossing jitter, as in a real recording; it makes the spline knots visible
    t_zero = t_zero + rng.normal(0.0, jitter_s, t_zero.size)
    per_cycle = 1.0 / np.diff(t_zero)
    per_cycle = np.append(per_cycle, per_cycle[-1])
    orig = CubicSpline(t_zero, per_cycle)(t)
    smooth = gaussian_filter1d(orig, 20)
    dip = 1.0 - amp_dip * excursion / max(excursion_hz, 1e-9)
    # the extraction band-pass raises the measured amplitude as the frequency rises
    gain = extraction_filter_gain(f_abs, baseline_eodf_hz) / extraction_filter_gain(
        baseline_eodf_hz, baseline_eodf_hz
    )
    amp = 0.6 * dip * gain
    smooth_amp = gaussian_filter1d(amp, 20)
    window = {
        "orig_freq": orig - orig[0],
        "smooth_freq": smooth - smooth[0],
        "orig_amp": amp / amp.max(),
        "smooth_amp": smooth_amp / smooth_amp.max(),
    }
    shift = n // 2 - round(chirp_centre)
    traces = {k: np.roll(v, shift).astype(np.float32) for k, v in window.items()}
    # the rolled zero must stay exactly zero after the float32 cast
    for k in ("orig_freq", "smooth_freq"):
        assert np.count_nonzero(traces[k] == 0.0) >= 1
    return SourceRow(
        traces, baseline_eodf_hz, excursion_hz, fwhm_s, amp_dip, chirp_centre, shift
    )


def write_source_file(path: Path, rows: list[SourceRow]) -> Path:
    """Write rows into an HDF5 file with the layout of ``extracted_chirps.h5``."""
    with h5py.File(path, "w") as f:
        for name in ("orig_freq", "smooth_freq", "orig_amp", "smooth_amp"):
            f.create_dataset(name, data=np.stack([r.traces[name] for r in rows]))
        f["samplerate"] = int(FS)
    return path


@pytest.fixture
def source_row() -> Callable[..., SourceRow]:
    """Factory for synthetic source rows (seeded per call via ``seed=``)."""

    def factory(seed: int = 0, **kwargs: float) -> SourceRow:
        return make_source_row(np.random.default_rng(seed), **kwargs)

    return factory


@pytest.fixture
def synthetic_h5(tmp_path: Path) -> tuple[Path, list[SourceRow]]:
    """A small source file with known chirps, one exact and one near duplicate."""
    rng = np.random.default_rng(42)
    rows = [
        make_source_row(rng, excursion_hz=60.0, fwhm_s=0.008, chirp_centre=4050.0),
        make_source_row(rng, excursion_hz=90.0, fwhm_s=0.010, chirp_centre=3900.0),
        make_source_row(rng, excursion_hz=120.0, fwhm_s=0.006, chirp_centre=4200.0),
        make_source_row(
            rng, excursion_hz=350.0, fwhm_s=0.012, chirp_centre=4000.0, amp_dip=0.5
        ),
        make_source_row(
            rng, excursion_hz=420.0, fwhm_s=0.015, chirp_centre=4300.0, amp_dip=0.6
        ),
        make_source_row(
            rng, excursion_hz=70.0, fwhm_s=0.009, chirp_centre=4000.0, jitter_s=5e-6
        ),
    ]
    rows.append(rows[1])  # exact duplicate of row 1
    near = rows[5]
    noisy = {
        k: (v + (rng.normal(0, 1e-3, v.size) if "freq" in k else 0)).astype(np.float32)
        for k, v in near.traces.items()
    }
    # keep the wrap marker exactly zero in both frequency traces
    for k in ("orig_freq", "smooth_freq"):
        noisy[k][near.traces[k] == 0.0] = 0.0
    rows.append(
        SourceRow(
            noisy, near.baseline_eodf_hz, near.excursion_hz, near.fwhm_s,
            near.amp_dip, near.chirp_centre, near.shift,
        )
    )  # fmt: skip
    path = write_source_file(tmp_path / "extracted_chirps.h5", rows)
    return path, rows
