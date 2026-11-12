"""Command line: measure trials, aggregate, write calibration.json and report.md.

Example::

    python -m gridtools.calibrate /mnt/data2/2019_tube_competition \\
        /mnt/data1/chirp_v2/calibration/v1 --workers 8 --seed 0
"""

from __future__ import annotations

import argparse
import logging
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from gridtools.calibrate.aggregate import AggregateConfig, aggregate
from gridtools.calibrate.measure import MeasureConfig
from gridtools.calibrate.report import write_calibration_json, write_report
from gridtools.calibrate.run import measure_trials
from gridtools.calibrate.trials import ANALYSED_TRIAL_DATES

logger = logging.getLogger("gridtools.calibrate")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "."],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    """Run the calibration."""
    parser = argparse.ArgumentParser(
        prog="python -m gridtools.calibrate", description=__doc__
    )
    parser.add_argument("root", type=Path, help="directory with the trial directories")
    parser.add_argument("out", type=Path, help="output directory")
    parser.add_argument("--dates", nargs="*", default=list(ANALYSED_TRIAL_DATES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--no-cache", action="store_true", help="re-measure every trial"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    cache = None if args.no_cache else args.out / "cache"
    measure_config = MeasureConfig()
    aggregate_config = AggregateConfig(seed=args.seed)
    t0 = time.perf_counter()
    trials = measure_trials(
        args.root,
        args.dates,
        seed=args.seed,
        workers=args.workers,
        config=measure_config,
        cache_dir=cache,
    )
    t1 = time.perf_counter()
    logger.info("measured %d trials in %.0f s", len(trials), t1 - t0)
    calibration = aggregate(trials, aggregate_config)
    t2 = time.perf_counter()
    logger.info("aggregated in %.0f s", t2 - t1)
    with (args.out / "calibration.pkl").open("wb") as handle:
        pickle.dump(calibration, handle)
    provenance = {
        "gridtools_commit": _git_sha(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "data_root": str(args.root),
        "trials": [t.name for t in trials],
        "seed": args.seed,
        "spacing_m": aggregate_config.spacing_m,
        "measure_config": repr(measure_config),
        "aggregate_config": repr(aggregate_config),
        "runtime_measure_s": round(t1 - t0, 1),
        "runtime_aggregate_s": round(t2 - t1, 1),
        "command": "python -m gridtools.calibrate "
        + " ".join(sys.argv[1:] if argv is None else argv),
    }
    write_calibration_json(calibration, args.out / "calibration.json", provenance)
    report = write_report(calibration, args.out, provenance)
    logger.info("wrote %s", report)


if __name__ == "__main__":
    main()
