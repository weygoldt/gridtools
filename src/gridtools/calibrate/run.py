"""Measure many trials in parallel, with a per-trial cache.

Every trial gets its own random stream, derived from one seed and the
trial's position in the sorted list of dates, so results do not depend on
the number of workers or the order in which trials finish.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from gridtools.calibrate.measure import MeasureConfig, TrialStats, measure_trial
from gridtools.calibrate.trials import read_trial_summary, resolve_trial_dir

__all__ = ["measure_trials", "trial_rng"]

logger = logging.getLogger(__name__)


def trial_rng(seed: int, index: int) -> np.random.Generator:
    """Random generator of the ``index``-th trial for a given seed."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(index,)))


def _measure_one(
    root: str,
    date: str,
    index: int,
    seed: int,
    config: MeasureConfig,
    cache: str | None,
) -> TrialStats:
    trial_dir = resolve_trial_dir(root, date)
    if cache is not None:
        path = Path(cache) / f"{trial_dir.name}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                return pickle.load(handle)
    meta = read_trial_summary(root)[trial_dir.name]
    stats = measure_trial(trial_dir, meta, trial_rng(seed, index), config)
    if cache is not None:
        path = Path(cache) / f"{trial_dir.name}.pkl"
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(stats, handle)
        tmp.replace(path)
    return stats


def measure_trials(
    root: str | Path,
    dates: Sequence[str],
    *,
    seed: int = 0,
    workers: int = 1,
    config: MeasureConfig | None = None,
    cache_dir: str | Path | None = None,
) -> list[TrialStats]:
    """Measure trials, optionally in parallel processes.

    Parameters
    ----------
    root : str or Path
        Directory holding the trial directories and ``trial_summary.csv``.
    dates : sequence of str
        Date prefixes of the trials.
    seed : int
        Base seed; trial ``i`` of ``sorted(dates)`` uses
        :func:`trial_rng` ``(seed, i)``.
    workers : int
        Worker processes (1 = in this process).
    config : MeasureConfig, optional
    cache_dir : str or Path, optional
        If given, results are pickled there per trial and reused.

    Returns
    -------
    list of TrialStats
        In the order of ``sorted(dates)``.
    """
    config = MeasureConfig() if config is None else config
    ordered = sorted(dates)
    cache = None
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache = str(cache_dir)
    args = [(str(root), d, i, seed, config, cache) for i, d in enumerate(ordered)]
    if workers <= 1:
        return [_measure_one(*a) for a in args]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_measure_one, *a) for a in args]
        results = []
        for date, future in zip(ordered, futures, strict=True):
            results.append(future.result())
            logger.info("measured %s", date)
    return results
