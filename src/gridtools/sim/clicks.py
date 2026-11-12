"""Rare impulsive artefacts ("clicks") on single or few electrodes.

Clicks are added to the recording after the channel gains (they arise in
the acquisition chain, so a dead electrode does not silence them). See
:class:`gridtools.sim.config.ClickConfig` for the model.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridtools.sim._common import Float32Array, IntArray
from gridtools.sim.config import ClickConfig

__all__ = ["Click", "add_clicks", "click_table", "draw_clicks"]


@dataclass(frozen=True, eq=False)
class Click:
    """One click.

    Attributes
    ----------
    click_id : int
        Index in the scene.
    start : int
        First sample.
    kind : str
        ``"burst"`` (Hann-windowed white noise) or ``"spike"`` (instant rise,
        exponential decay).
    amplitude : float
        Peak absolute value on the first electrode (recording units).
    amplitude_rel : float
        ``amplitude`` in units of the scene's noise SD.
    electrodes : numpy.ndarray of int64
        Electrodes hit, first = strongest.
    weights : numpy.ndarray of float64
        Relative amplitude per electrode (first is 1).
    waveform : numpy.ndarray of float32
        Unit-peak waveform (signed), length = duration in samples.
    """

    click_id: int
    start: int
    kind: str
    amplitude: float
    amplitude_rel: float
    electrodes: IntArray
    weights: np.ndarray
    waveform: Float32Array

    @property
    def n_samples(self) -> int:
        """Duration in samples."""
        return int(self.waveform.size)


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def draw_clicks(
    rng: np.random.Generator,
    config: ClickConfig,
    *,
    n_samples: int,
    n_electrodes: int,
    samplerate_hz: float,
    noise_sd: float,
) -> tuple[float, list[Click]]:
    """Draw the clicks of a scene.

    Returns
    -------
    rate_hz : float
        The scene's click rate.
    clicks : list of Click
        Sorted by start.
    """
    duration_s = n_samples / samplerate_hz
    rate_hz = float(rng.uniform(*config.rate_hz))
    n_clicks = int(rng.poisson(rate_hz * duration_s))
    starts = np.sort(rng.integers(0, n_samples, n_clicks))
    clicks = []
    for click_id, start in enumerate(starts.tolist()):
        duration_s = _log_uniform(rng, *config.duration_s)
        amplitude_rel = _log_uniform(rng, *config.amplitude_rel)
        is_burst = bool(rng.random() < config.burst_prob)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        n_hit = min(int(rng.geometric(1.0 / config.electrodes_mean)), n_electrodes)
        electrodes = rng.choice(n_electrodes, size=n_hit, replace=False)
        weights = np.concatenate([[1.0], rng.uniform(*config.spread, n_hit - 1)])
        length = max(1, round(duration_s * samplerate_hz))
        noise = rng.standard_normal(length)
        k = np.arange(length, dtype=np.float64)
        if is_burst:
            shape = noise * np.sin(np.pi * (k + 0.5) / length) ** 2
        else:
            tau = max(length / 5.0, 0.5)
            shape = np.exp(-k / tau)
            tail = max(1, length // 10)
            shape[-tail:] *= np.linspace(1.0, 0.0, tail + 1)[1:]
        peak = float(np.max(np.abs(shape)))
        shape = sign * shape / (peak if peak > 0 else 1.0)
        clicks.append(
            Click(
                click_id=click_id,
                start=int(start),
                kind="burst" if is_burst else "spike",
                amplitude=amplitude_rel * noise_sd,
                amplitude_rel=amplitude_rel,
                electrodes=electrodes.astype(np.int64),
                weights=weights,
                waveform=shape.astype(np.float32),
            )
        )
    return rate_hz, clicks


def add_clicks(recording: np.ndarray, clicks: list[Click], start: int = 0) -> None:
    """Add clicks in place to ``recording`` holding samples ``start:start+n``."""
    n = recording.shape[0]
    for click in clicks:
        lo = max(click.start, start)
        hi = min(click.start + click.n_samples, start + n)
        if lo >= hi:
            continue
        piece = click.waveform[lo - click.start : hi - click.start].astype(np.float64)
        for electrode, weight in zip(click.electrodes, click.weights, strict=True):
            column = recording[lo - start : hi - start, electrode]
            column += (click.amplitude * weight * piece).astype(recording.dtype)


def click_table(clicks: list[Click], samplerate_hz: float) -> pd.DataFrame:
    """One row per click (for ``clicks.csv``)."""
    return pd.DataFrame(
        {
            "click_id": [c.click_id for c in clicks],
            "t_start_s": [c.start / samplerate_hz for c in clicks],
            "t_end_s": [(c.start + c.n_samples) / samplerate_hz for c in clicks],
            "duration_s": [c.n_samples / samplerate_hz for c in clicks],
            "kind": [c.kind for c in clicks],
            "amplitude": [c.amplitude for c in clicks],
            "amplitude_rel_noise_sd": [c.amplitude_rel for c in clicks],
            "electrodes": [";".join(str(e) for e in c.electrodes) for c in clicks],
            "weights": [
                ";".join(f"{w:.4g}" for w in c.weights.tolist()) for c in clicks
            ],
        }
    )
