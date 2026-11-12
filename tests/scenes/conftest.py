"""Fixtures for the scene tests: a small synthetic template library and scenes.

The library mimics the on-disk format of ``gridtools.templates`` (manifest,
``delta_f_hz.npy``, ``amp_rel.npy``, ``templates.parquet``) with Gaussian
chirps of known excursion and width, so the tests do not need
``/mnt/data1``.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from gridtools.sim.config import (
    ChirpConfig,
    ClickConfig,
    GridConfig,
    HybridConfig,
    NoiseConfig,
    PlotConfig,
    PopulationConfig,
    SceneConfig,
)

if TYPE_CHECKING:
    from gridtools.sim.scene import Scene

FS = 20_000.0
N_TEMPLATE = 4000  # 0.2 s, centre at 2000
TAPER = 200  # samples of raised-cosine taper at the support edges


def gaussian_template(
    excursion_hz: float, fwhm_s: float, undershoot_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian chirp centred on sample N/2, undershoot after it, tapered."""
    t = (np.arange(N_TEMPLATE) - N_TEMPLATE // 2) / FS
    sigma = fwhm_s / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    delta = excursion_hz * np.exp(-0.5 * (t / sigma) ** 2)
    delta -= undershoot_hz * np.exp(-0.5 * ((t - 3 * sigma - 0.01) / 0.006) ** 2)
    weight = np.zeros(N_TEMPLATE)
    support = (t > -6 * sigma - 0.005) & (t < 3 * sigma + 0.04)
    idx = np.flatnonzero(support)
    weight[idx] = 1.0
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(TAPER) / TAPER))
    weight[idx[0] - TAPER : idx[0]] = ramp
    weight[idx[-1] + 1 : idx[-1] + 1 + TAPER] = ramp[::-1]
    delta *= weight
    amp = 1.0 - 0.3 * np.clip(delta, 0, None) / excursion_hz
    return delta.astype(np.float32), amp.astype(np.float32)


def write_synthetic_library(path: Path) -> Path:
    """Write a 24-template library: small and big chirps, a few flagged."""
    path.mkdir(parents=True, exist_ok=True)
    excursions = [40.0, 60.0, 80.0, 120.0, 160.0, 250.0, 320.0, 400.0]
    widths = [0.006, 0.012, 0.025]
    deltas, amps, rows = [], [], []
    for i, (exc, fwhm) in enumerate((e, w) for e in excursions for w in widths):
        delta, amp = gaussian_template(exc, fwhm, undershoot_hz=0.05 * exc)
        deltas.append(delta)
        amps.append(amp)
        rows.append(
            {
                "template_id": i,
                "excursion_hz": float(delta.max()),
                "duration_50_s": fwhm,
                "duration_10_s": fwhm * np.sqrt(np.log(10) / np.log(2)),
                "family": "big" if exc >= 200 else "small",
                "valid": True,
                "has_nan": False,
                "duplicate_of": -1,
                "truncated": False,
                "step_flag": i == 5,  # one flagged template
                "too_short": False,
                "multi_peak": False,
                "outlier": False,
                "eodf_atypical": False,
                "ripple_flag": False,
                "amp_ripple_flag": False,
                "amp_usable": i % 2 == 0,
            }
        )
    np.save(path / "delta_f_hz.npy", np.stack(deltas))
    np.save(path / "amp_rel.npy", np.stack(amps))
    pd.DataFrame(rows).to_parquet(path / "templates.parquet", index=False)
    manifest = {
        "library_format_version": 1,
        "samplerate_hz": FS,
        "n_templates": len(rows),
        "n_samples": N_TEMPLATE,
        "centre_index": N_TEMPLATE // 2,
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


@pytest.fixture(scope="session")
def library_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_synthetic_library(tmp_path_factory.mktemp("library"))


def small_config(library: Path, **changes: object) -> SceneConfig:
    """A short scene that still exercises every component.

    Clipping is off so that ``recording = sum(fish) + background`` holds
    exactly.
    """
    config = SceneConfig(
        name="test",
        duration_s=6.0,
        grid=GridConfig(layouts=((3, 3),), real_layout_prob=1.0),
        population=PopulationConfig(n_fish=(3, 3)),
        chirps=ChirpConfig(library=str(library), rate_hz=(0.8, 1.0), response_prob=0.5),
        clicks=ClickConfig(rate_per_h_median=3600.0, rate_log10_sd=0.0),
        noise=NoiseConfig(clip_limit=None),
        plot=PlotConfig(enabled=False, n_zooms=2, zoom_s=2.0, hop_s=0.0512),
        background=HybridConfig(prob=0.0),  # synthetic; hybrid: test_hybrid.py
    )
    return replace(config, **changes) if changes else config


@pytest.fixture(scope="session")
def config(library_dir: Path) -> SceneConfig:
    return small_config(library_dir)


@pytest.fixture(scope="session")
def scene(config: SceneConfig) -> "Scene":
    from gridtools.sim.scene import simulate_scene

    return simulate_scene(config, 7)


@pytest.fixture(scope="session")
def scene_path(scene: "Scene", tmp_path_factory: pytest.TempPathFactory) -> Path:
    from gridtools.sim.scene import write_scene

    return write_scene(scene, tmp_path_factory.mktemp("scene") / "000007")
