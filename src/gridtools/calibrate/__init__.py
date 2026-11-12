"""Measure grid recordings and turn them into simulator parameters.

Typical use (also available as ``python -m gridtools.calibrate``)::

    from gridtools.calibrate import measure_trials, aggregate, write_report

    trials = measure_trials(root, dates, seed=0, workers=8, cache_dir=out / "cache")
    calibration = aggregate(trials)
    write_calibration_json(calibration, out / "calibration.json", provenance)
    write_report(calibration, out, provenance)

Modules:

- :mod:`~gridtools.calibrate.trials`: trial discovery, metadata, tolerant raw access;
- :mod:`~gridtools.calibrate.frequency`: baselines, delta F, drift variograms;
- :mod:`~gridtools.calibrate.profiles`: order-free ``sign_v`` profile statistics;
- :mod:`~gridtools.calibrate.field`: simulation-based fit of the dipole falloff;
- :mod:`~gridtools.calibrate.spectra`: noise floor, hum, line amplitudes, clicks;
- :mod:`~gridtools.calibrate.chirps`: chirp timing statistics;
- :mod:`~gridtools.calibrate.motion`: movement proxies of simulated fish;
- :mod:`~gridtools.calibrate.measure`: everything for one trial;
- :mod:`~gridtools.calibrate.run`: many trials in parallel, cached;
- :mod:`~gridtools.calibrate.aggregate`: pooling and simulator parameters;
- :mod:`~gridtools.calibrate.report`: ``calibration.json`` and the Markdown report.
"""

from gridtools.calibrate.aggregate import AggregateConfig, Calibration, aggregate
from gridtools.calibrate.field import FieldFit, GridLayout, fit_field_falloff
from gridtools.calibrate.measure import MeasureConfig, TrialStats, measure_trial
from gridtools.calibrate.report import write_calibration_json, write_report
from gridtools.calibrate.run import measure_trials
from gridtools.calibrate.trials import ANALYSED_TRIAL_DATES

__all__ = [
    "ANALYSED_TRIAL_DATES",
    "AggregateConfig",
    "Calibration",
    "FieldFit",
    "GridLayout",
    "MeasureConfig",
    "TrialStats",
    "aggregate",
    "fit_field_falloff",
    "measure_trial",
    "measure_trials",
    "write_calibration_json",
    "write_report",
]
