"""Pool per-trial measurements and map them to simulator parameters.

:func:`aggregate` turns a list of :class:`~gridtools.calibrate.measure.TrialStats`
into a :class:`Calibration`: summaries per measured quantity, the field fit,
and :attr:`Calibration.parameters`, a mapping from simulator parameter names
(``"gridtools.sim.eod.DriftParams.sd_hz"``, ...) to proposed values with
uncertainty, method and the current default.

Units: "recording units" of the simulator are taken to be the raw ADC units
of the grid recordings, volts at the ADC input (electrode potential times the
amplifier gain of 100).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from gridtools.calibrate.field import (
    FieldFit,
    GridLayout,
    draw_simulation,
    fit_field_falloff,
)
from gridtools.calibrate.frequency import fit_drift_variogram
from gridtools.calibrate.measure import TrialStats
from gridtools.calibrate.motion import simulated_dwell_stats
from gridtools.sim.eod import APTERONOTUS_LEPTORHYNCHUS, DriftParams, SharedDriftParams
from gridtools.sim.field import DipoleParams, dipole_gain
from gridtools.sim.movement import MovementParams
from gridtools.sim.noise import (
    ClipParams,
    ColoredNoiseParams,
    ElectrodeFaultParams,
    GainMismatchParams,
    HumParams,
)

__all__ = ["AggregateConfig", "Calibration", "aggregate", "layout_for"]

FloatArray = npt.NDArray[np.float64]
Json = dict[str, Any]


@dataclass(frozen=True)
class AggregateConfig:
    """Settings of :func:`aggregate`.

    Attributes
    ----------
    spacing_m : float
        Electrode spacing from ``fishgrid.cfg`` (0.5 m for all trials). The
        paper's methods describe a 100 cm x 50 cm tank, in which a 3x5 grid
        at 0.5 m does not fit; lengths of the field fit are therefore also
        reported relative to the spacing.
    missing_2019 : tuple of int
        Row-major index of the 3x3 electrode without signal in 2019 (channel
        9; a corner under row-major, column-major and serpentine orders).
    exponents, regularisers_rel, margins_rel : tuple of float
        Grid of the field fit.
    n_field_samples : int
        Simulated detections per layout and parameter set.
    n_bootstrap : int
    movement_duration_s : float
        Simulated duration per movement-model variant.
    seed : int
    """

    spacing_m: float = 0.5
    missing_2019: tuple[int, ...] = (8,)
    exponents: tuple[float, ...] = tuple(np.round(np.arange(1.0, 4.01, 0.25), 2))
    regularisers_rel: tuple[float, ...] = (
        0.02,
        0.05,
        0.1,
        0.15,
        0.2,
        0.3,
        0.4,
        0.5,
        0.7,
        1.0,
    )
    margins_rel: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    n_field_samples: int = 20000
    n_bootstrap: int = 200
    movement_duration_s: float = 3600.0
    seed: int = 0


def layout_for(name: str, config: AggregateConfig) -> GridLayout:
    """Grid layout of a ``TrialStats.layout`` name (``"9-8"``, ``"15-14"``)."""
    total, live = (int(v) for v in name.split("-"))
    if total == 9:
        missing = config.missing_2019 if live <= 8 else ()
        return GridLayout(
            name,
            3,
            3,
            config.spacing_m,
            missing,
            n_random_missing=8 - live if live < 8 else 0,
        )
    if total == 15:
        return GridLayout(name, 3, 5, config.spacing_m, (), n_random_missing=15 - live)
    msg = f"unknown layout {name!r}"
    raise ValueError(msg)


@dataclass(eq=False)
class Calibration:
    """Pooled measurements, fits and proposed simulator parameters.

    Attributes
    ----------
    summary : dict
        JSON-able summaries per measured quantity.
    parameters : dict
        Simulator parameter name -> proposal (``value``, ``uncertainty``,
        ``current_default``, ``method``, ``n``).
    proposed_new : dict
        Proposed parameters for which the simulator has no field yet.
    field_fit : FieldFit
    field_sensitivity : dict
    figures : dict
        Arrays used by the report figures.
    """

    summary: Json
    parameters: Json
    proposed_new: Json
    field_fit: FieldFit
    field_sensitivity: Json = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)


def _q(
    values: npt.ArrayLike, qs: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95)
) -> Json:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"q{round(100 * q):02d}": None for q in qs} | {"n": 0}
    out = {
        f"q{round(100 * q):02d}": float(x)
        for q, x in zip(qs, np.quantile(v, qs), strict=True)
    }
    out["n"] = int(v.size)
    return out


def _round(value: float | None, digits: int) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _default(cls: type, name: str) -> Any:
    return {f.name: f.default for f in dataclasses.fields(cls)}[name]


def _param(
    value: Any,
    *,
    current: Any,
    method: str,
    uncertainty: Any = None,
    n: int | None = None,
    note: str | None = None,
) -> Json:
    out: Json = {
        "value": value,
        "current_default": current,
        "uncertainty": uncertainty,
        "method": method,
    }
    if n is not None:
        out["n"] = n
    if note:
        out["note"] = note
    return out


def _fish_table(trials: Sequence[TrialStats]) -> list[Json]:
    rows = []
    for t in trials:
        for f in t.fish:
            rows.append(
                {
                    "trial": t.name,
                    "ident": f.ident,
                    "role": f.role,
                    "sex": f.sex,
                    "baseline_hz": f.baseline_hz,
                    "baseline_dark_hz": f.baseline_dark_hz,
                    "baseline_light_hz": f.baseline_light_hz,
                    "baseline_25c_hz": f.baseline_25c_hz,
                    "slope_hz_per_h": f.slope_hz_per_h,
                    "q10": f.q10,
                    "drift_sd_hz": f.drift_gaussian.sd,
                    "drift_timescale_s": f.drift_gaussian.timescale_s,
                    "drift_nugget_hz2": f.drift_gaussian.nugget,
                    "drift_rms_log": f.drift_gaussian.rms_log_residual,
                    "drift_exp_sd_hz": f.drift_exponential.sd,
                    "drift_exp_timescale_s": f.drift_exponential.timescale_s,
                    "drift_exp_rms_log": f.drift_exponential.rms_log_residual,
                    "coverage_h": f.coverage_h,
                    "median_peak_amplitude_v": float(np.median(f.peak_amplitude_v)),
                    "median_raw_peak_amplitude_v": float(
                        np.median(f.raw_peak_amplitude_v)
                    )
                    if f.raw_peak_amplitude_v.size
                    else float("nan"),
                    "median_cn0_db_hz": float(np.median(f.cn0_db_hz)),
                    "switches_per_min_3db": f.dwell[3.0]["switches_per_min"],
                    "switches_per_min_3db_dark": f.dwell[3.0]["switches_per_min_dark"],
                    "switches_per_min_3db_light": f.dwell[3.0][
                        "switches_per_min_light"
                    ],
                    "still_fraction_3db": f.dwell[3.0]["still_fraction"],
                    "still_fraction_3db_dark": f.dwell[3.0]["still_fraction_dark"],
                    "still_fraction_3db_light": f.dwell[3.0]["still_fraction_light"],
                    "switches_per_min_0db": f.dwell[0.0]["switches_per_min"],
                }
            )
    return rows


def _eodf_summary(rows: Sequence[Json]) -> Json:
    base = np.array([r["baseline_hz"] for r in rows])
    b25 = np.array([r["baseline_25c_hz"] for r in rows])
    sex = np.array([r["sex"] for r in rows])
    role = np.array([r["role"] for r in rows])
    out: Json = {"all": _q(base), "all_25c": _q(b25)}
    for s in ("f", "m"):
        sel = sex == s
        out[f"sex_{s}"] = _q(base[sel]) | {
            "mean": float(np.mean(base[sel])),
            "sd": float(np.std(base[sel], ddof=1)),
        }
        out[f"sex_{s}_25c"] = _q(b25[sel]) | {
            "mean": float(np.nanmean(b25[sel])),
            "sd": float(np.nanstd(b25[sel], ddof=1)),
        }
    for r in ("winner", "loser"):
        out[f"role_{r}"] = _q(base[role == r])
    try:
        from scipy.stats import mannwhitneyu

        out["sex_test_25c"] = {
            "test": "Mann-Whitney U, two-sided, fish x trial as units (fish recur across trials)",
            "p": float(
                np.asarray(
                    mannwhitneyu(
                        b25[(sex == "f") & np.isfinite(b25)],
                        b25[(sex == "m") & np.isfinite(b25)],
                    )[1]
                )
            ),
        }
    except ValueError:
        pass
    return out


def _delta_f_summary(trials: Sequence[TrialStats]) -> Json:
    pooled = np.concatenate([t.pair.abs_delta_f_hz for t in trials])
    pooled = pooled[np.isfinite(pooled)]
    thresholds = (5.0, 10.0, 20.0, 50.0)
    hours = sum(t.pair.delta_f.duration_h for t in trials)
    crossings = sum(t.pair.delta_f.n_crossings for t in trials)
    hours_b = sum(t.pair.delta_f_baseline.duration_h for t in trials)
    crossings_b = sum(t.pair.delta_f_baseline.n_crossings for t in trials)
    per_trial = {
        t.name: {
            "median_abs_hz": t.pair.delta_f.quantiles_hz[0.5],
            "fraction_below": {
                str(k): v for k, v in t.pair.delta_f.fraction_below.items()
            },
            "crossings": t.pair.delta_f.n_crossings,
            "crossings_baseline": t.pair.delta_f_baseline.n_crossings,
        }
        for t in trials
    }
    return {
        "abs_delta_f_hz": _q(pooled, (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)),
        "fraction_below_pooled": {
            str(th): float(np.mean(pooled < th)) for th in thresholds
        },
        "trials_with_time_below": {
            str(th): int(sum(t.pair.delta_f.fraction_below[th] > 0 for t in trials))
            for th in thresholds
        },
        "crossings_per_h": crossings / hours if hours else float("nan"),
        "n_crossings": int(crossings),
        "crossings_per_h_baseline": crossings_b / hours_b if hours_b else float("nan"),
        "n_crossings_baseline": int(crossings_b),
        "hours": hours,
        "per_trial": per_trial,
    }


def _noise_summary(trials: Sequence[TrialStats]) -> tuple[Json, Json]:
    per_trial = {}
    spreads = []
    variation = []
    alphas = []
    hum1 = []
    coupling_sd = []
    ratios: dict[int, list[float]] = {}
    n_detected = []
    dead_total = 0
    electrodes_total = 0
    noisy_total = 0
    noisy_factor = []
    clicks_per_h = []
    click_intervals = []
    ref_floors: list[float] = []
    for t in trials:
        n = t.noise
        live = ~n.dead
        floors_db = 10.0 * np.log10(np.median(n.floor_v2_hz, axis=0))
        ref = float(np.median(floors_db[live]))
        spreads.append(float(np.std(floors_db[live] - ref, ddof=1)))
        variation.append(
            float(np.median(np.std(10 * np.log10(n.floor_v2_hz[:, live]), axis=0)))
        )
        alphas.extend(n.alpha[live].tolist())
        dead_total += int(n.dead.sum())
        electrodes_total += n.dead.size
        noisy_total += int(n.noisy.sum())
        if n.noisy.any():
            noisy_factor.extend((10 ** ((floors_db[n.noisy] - ref) / 20)).tolist())
        amp = np.median(n.hum_amplitude_v, axis=0)  # (H, E)
        det = np.mean(n.hum_detected, axis=0)  # (H, E)
        top = int(np.argmax(amp[0] * live))
        hum1.append(float(amp[0, top]))
        positive = amp[0, live][amp[0, live] > 0]
        if positive.size > 2:
            coupling_sd.append(float(np.std(np.log(positive), ddof=1)))
        for h in range(1, amp.shape[0]):
            if amp[0, top] > 0:
                ratios.setdefault(h + 1, []).append(float(amp[h, top] / amp[0, top]))
        n_detected.append(int(np.sum(det[:, top] >= 0.5)))
        seconds = n.click_events.size * n.segment_s
        clicks_per_h.append(float(n.click_events.sum() / seconds * 3600.0))
        times = np.sort(n.click_times_s)
        gaps = np.diff(times)
        click_intervals.extend(gaps[gaps < n.segment_s].tolist())
        ref_floors.append(ref)
        per_trial[t.name] = {
            "floor_db_v2_hz": ref,
            "floor_db_per_electrode": floors_db.round(2).tolist(),
            "dead": np.flatnonzero(n.dead).tolist(),
            "noisy": np.flatnonzero(n.noisy).tolist(),
            "alpha_median": float(np.median(n.alpha[live])),
            "hum_50hz_max_v": float(amp[0, top]),
            "hum_harmonics_detected": int(np.sum(det[:, top] >= 0.5)),
            "max_abs_v": n.max_abs_v,
            "clipped_fraction": n.clipped_fraction,
            "clicks_per_h": clicks_per_h[-1],
            "sign_v_check_max_abs_db": float(
                np.nanmax(np.abs(np.nanmedian(n.sign_v_check_db, axis=0)))
            ),
            "sign_v_columns_match_raw_channels": bool(
                np.array_equal(
                    n.sign_v_best_channel, np.arange(n.sign_v_best_channel.size)
                )
            ),
        }
    names = [t.name for t in trials]
    floors = np.array([ref_floors[i] for i in range(len(trials))])
    intervals = np.asarray(click_intervals)
    summary = {
        "floor_db_v2_hz": _q(floors),
        "floor_by_year": {
            year: _q(floors[[k.startswith(year) for k in names]])
            for year in ("2019", "2020")
        },
        "electrode_spread_db": _q(spreads),
        "segment_variation_db": _q(variation),
        "alpha": _q(alphas),
        "dead_electrodes": dead_total,
        "electrodes": electrodes_total,
        "noisy_electrodes": noisy_total,
        "noisy_factor": _q(noisy_factor),
        "hum_50hz_max_electrode_v": _q(hum1),
        "hum_50hz_by_year_v": {
            year: _q(np.asarray(hum1)[[k.startswith(year) for k in names]])
            for year in ("2019", "2020")
        },
        "hum_coupling_log_sd": _q(coupling_sd),
        "hum_harmonic_ratio": {
            str(h): _q(v) for h, v in sorted(ratios.items()) if h <= 10
        },
        "hum_harmonics_detected": _q(n_detected),
        "clicks_per_h": _q(clicks_per_h),
        "click_interval_fractions": {
            "n_intervals": int(intervals.size),
            "near_0.03s": float(np.mean(np.abs(intervals - 0.03) < 0.005))
            if intervals.size
            else None,
            "near_0.1s": float(np.mean(np.abs(intervals - 0.1) < 0.01))
            if intervals.size
            else None,
            "near_1s": float(np.mean(np.abs(intervals - 1.0) < 0.05))
            if intervals.size
            else None,
        },
        "max_abs_v": _q([p["max_abs_v"] for p in per_trial.values()]),
        "clipped_fraction_max": float(
            max(p["clipped_fraction"] for p in per_trial.values())
        ),
        "per_trial": per_trial,
    }
    extra = {
        "floors": floors,
        "alphas": np.asarray(alphas),
        "hum1": np.asarray(hum1),
        "ratios": ratios,
        "clicks_per_h": np.asarray(clicks_per_h),
        "click_intervals": intervals,
        "years": [t.name[:4] for t in trials],
    }
    return summary, extra


def _chirp_summary(trials: Sequence[TrialStats]) -> tuple[Json, Json]:
    rates: dict[str, list[float]] = {"winner": [], "loser": []}
    rates_dark: dict[str, list[float]] = {"winner": [], "loser": []}
    rates_light: dict[str, list[float]] = {"winner": [], "loser": []}
    icis = []
    n_raw = n_merged = 0
    min_ici = []
    dark_s = 3 * 3600.0
    for t in trials:
        for c in t.chirps:
            n_raw += c.n_raw
            n_merged += c.n_merged
            hours = t.duration_s / 3600.0
            rates[c.role].append(c.times_s.size / hours)
            rates_dark[c.role].append(
                float(np.sum(c.times_s < dark_s)) / (dark_s / 3600.0)
            )
            rates_light[c.role].append(
                float(np.sum(c.times_s >= dark_s)) / ((t.duration_s - dark_s) / 3600.0)
            )
            d = np.diff(c.times_s)
            icis.append(d)
            if d.size:
                min_ici.append(float(d.min()))
    ici = np.concatenate(icis) if icis else np.zeros(0)
    edges = None
    obs = null = None
    coinc: dict[str, list[tuple[float, float, int]]] = {}
    for t in trials:
        for key, cs in t.cross.items():
            if edges is None:
                edges = cs.latency_edges_s
                obs = np.zeros(edges.size - 1)
                null = np.zeros(edges.size - 1)
            assert obs is not None
            assert null is not None
            obs += cs.latency_counts
            null += cs.latency_null_counts
            coinc.setdefault(key, []).append(
                (cs.coincidence, cs.coincidence_null, cs.n_a)
            )
    coincidence = {}
    for key, values in coinc.items():
        arr = np.array(values)
        weights = arr[:, 2]
        if weights.sum() == 0:
            continue
        coincidence[key] = {
            "observed": float(np.nansum(arr[:, 0] * weights) / weights.sum()),
            "null": float(np.nansum(arr[:, 1] * weights) / weights.sum()),
            "n_chirps": int(weights.sum()),
        }
    all_c = [v for key in coinc for v in coinc[key]]
    arr = np.array(all_c) if all_c else np.zeros((0, 3))
    w = arr[:, 2] if arr.size else np.zeros(0)
    summary = {
        "bias_note": "v1 detections (chirpdetector_bboxes.csv, YOLO, threshold 0.5) with v1 emitter assignment: biased toward detectable, well-separated chirps; assignment unvalidated",
        "n_detections_assigned": int(n_raw),
        "n_duplicates_merged": int(n_merged),
        "rate_per_h": {role: _q(v) for role, v in rates.items()},
        "rate_per_h_dark": {role: _q(v) for role, v in rates_dark.items()},
        "rate_per_h_light": {role: _q(v) for role, v in rates_light.items()},
        "ici_s": _q(ici, (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95)),
        "ici_fraction_below": {
            str(x): float(np.mean(ici < x)) if ici.size else None
            for x in (0.2, 0.5, 1.0, 2.0, 5.0)
        },
        "min_ici_per_fish_s": _q(min_ici),
        "coincidence_0.25s": coincidence
        | {
            "pooled": {
                "observed": float(np.nansum(arr[:, 0] * w) / w.sum())
                if w.size and w.sum()
                else None,
                "null": float(np.nansum(arr[:, 1] * w) / w.sum())
                if w.size and w.sum()
                else None,
            }
        },
    }
    if edges is not None and obs is not None and null is not None:
        centres = 0.5 * (edges[:-1] + edges[1:])
        ratio = np.where(null > 0, obs / np.maximum(null, 1e-12), np.nan)
        summary["latency_ratio"] = {
            f"{edges[i]:.1f}-{edges[i + 1]:.1f}s": float(ratio[i]) for i in range(10)
        }
        summary["latency_excess_0_1s"] = float(
            obs[:10].sum() / max(null[:10].sum(), 1e-12)
        )
        summary["latency_excess_1_3s"] = float(
            obs[10:30].sum() / max(null[10:30].sum(), 1e-12)
        )
    else:
        centres = ratio = np.zeros(0)
    extra = {
        "ici": ici,
        "rates": rates,
        "latency_centres": centres,
        "latency_obs": obs,
        "latency_null": null,
        "latency_ratio": ratio,
    }
    return summary, extra


def _movement_summary(rows: Sequence[Json], trials: Sequence[TrialStats]) -> Json:
    out: Json = {}
    layouts = {t.name: t.layout for t in trials}
    for layout in sorted(set(layouts.values())):
        sel = [r for r in rows if layouts[r["trial"]] == layout]
        out[layout] = {
            key: _q([r[key] for r in sel])
            for key in (
                "switches_per_min_0db",
                "switches_per_min_3db",
                "switches_per_min_3db_dark",
                "switches_per_min_3db_light",
                "still_fraction_3db_dark",
                "still_fraction_3db_light",
            )
        }
    dwell = np.concatenate(
        [np.asarray(f.dwell[3.0]["dwell_times_s"]) for t in trials for f in t.fish]
    )
    out["dwell_times_3db_s"] = _q(dwell, (0.05, 0.25, 0.5, 0.75, 0.95, 0.99))
    return out


def _gain_quantiles(
    fit: FieldFit, config: AggregateConfig, rng: np.random.Generator, n: int = 20000
) -> FloatArray:
    """Gain at the strongest electrode of the 15-electrode grid, simulator geometry."""
    layout = GridLayout("15", 3, 5, config.spacing_m)
    xy_e = layout.positions_m()
    pad = fit.best["margin_rel"] * config.spacing_m
    lo, hi = xy_e.min(axis=0) - pad, xy_e.max(axis=0) + pad
    xy = lo + rng.random((n, 2)) * (hi - lo)
    heading = rng.uniform(-np.pi, np.pi, n)
    params = DipoleParams()
    dz = params.electrode_depth_m - params.fish_depth_m
    a_eff = fit.best["regulariser_rel"] * config.spacing_m
    a = math.sqrt(max(a_eff**2 - dz**2, 0.0))
    gain = dipole_gain(
        xy,
        heading,
        xy_e,
        fish_depth_m=params.fish_depth_m,
        electrode_depth_m=params.electrode_depth_m,
        min_distance_m=a,
        distance_exponent=fit.best["distance_exponent"],
        reference_distance_m=params.reference_distance_m,
    )
    return np.abs(gain).max(axis=1).astype(np.float64)


def aggregate(
    trials: Sequence[TrialStats], config: AggregateConfig | None = None
) -> Calibration:
    """Pool trials, fit the field falloff and propose simulator parameters.

    Parameters
    ----------
    trials : sequence of TrialStats
    config : AggregateConfig, optional

    Returns
    -------
    Calibration
    """
    config = AggregateConfig() if config is None else config
    rng = np.random.default_rng(np.random.SeedSequence(config.seed, spawn_key=(1000,)))
    rows = _fish_table(trials)
    fish = [f for t in trials for f in t.fish]
    figures: dict[str, Any] = {"fish_rows": rows}
    params: Json = {}
    new: Json = {}

    # 1. Baseline EODf ------------------------------------------------------
    eodf = _eodf_summary(rows)
    figures["eodf"] = rows
    new["population.baseline_hz"] = {
        "value": {
            "female": {"mean": eodf["sex_f"]["mean"], "sd": eodf["sex_f"]["sd"]},
            "male": {"mean": eodf["sex_m"]["mean"], "sd": eodf["sex_m"]["sd"]},
            "pooled_quantiles": eodf["all"],
        },
        "method": "median of the slow baseline (5th percentile of 5-min windows) per fish and trial, at the recording temperature (23-26 degC); sex from trial_summary.csv",
        "n": len(rows),
        "note": "fish recur across trials (6 groups), so the n fish-trials are not independent",
    }

    # 2. Delta F -----------------------------------------------------------
    df_summary = _delta_f_summary(trials)
    figures["abs_delta_f"] = np.concatenate([t.pair.abs_delta_f_hz for t in trials])

    # 3. Drift -------------------------------------------------------------
    lags = fish[0].drift_lags_s
    pooled_gamma = np.nanmedian(np.vstack([f.drift_gamma_hz2 for f in fish]), axis=0)
    pooled_fit = fit_drift_variogram(lags, pooled_gamma, "gaussian")
    pooled_exp = fit_drift_variogram(lags, pooled_gamma, "exponential")
    sds = np.array([f.drift_gaussian.sd for f in fish])
    taus = np.array([f.drift_gaussian.timescale_s for f in fish])
    shared_lags = trials[0].pair.shared_lags_s
    shared_gamma = np.nanmedian(
        np.vstack([t.pair.shared_gamma for t in trials]), axis=0
    )
    sel = shared_lags >= 60.0
    shared_fit = fit_drift_variogram(shared_lags[sel], shared_gamma[sel], "gaussian")
    shared_sds = np.array([t.pair.shared_gaussian.sd for t in trials])
    shared_taus = np.array([t.pair.shared_gaussian.timescale_s for t in trials])
    slopes = np.array([t.pair.shared_slope_per_h for t in trials])
    q10s = np.array([f.q10 for f in fish])
    per_degc = np.array([t.pair.shared_per_degc for t in trials])
    drift = {
        "individual_pooled_gaussian": dataclasses.asdict(pooled_fit),
        "individual_pooled_exponential": dataclasses.asdict(pooled_exp),
        "individual_sd_hz": _q(sds),
        "individual_timescale_s": _q(taus),
        "individual_sd_by_role": {
            r: _q([f.drift_gaussian.sd for f in fish if f.role == r])
            for r in ("winner", "loser")
        },
        "individual_rms_log_gaussian": _q(
            [f.drift_gaussian.rms_log_residual for f in fish]
        ),
        "individual_rms_log_exponential": _q(
            [f.drift_exponential.rms_log_residual for f in fish]
        ),
        "nugget_sd_hz": _q(np.sqrt([f.drift_gaussian.nugget for f in fish])),
        "baseline_slope_hz_per_h": _q([f.slope_hz_per_h for f in fish]),
        "baseline_change_10min_hz": _q(
            np.concatenate([f.change_10min_hz for f in fish]), (0.5, 0.9, 0.99)
        ),
        "shared_pooled_gaussian": dataclasses.asdict(shared_fit),
        "shared_sd_rel": _q(shared_sds),
        "shared_timescale_s": _q(shared_taus),
        "shared_slope_percent_per_h": _q(100 * slopes),
        "corr_level": _q([t.pair.corr_level for t in trials]),
        "corr_increments_30min": _q([t.pair.corr_increments_30min for t in trials]),
        "temperature_range_c": _q([t.pair.temperature_range_c for t in trials]),
        "temperature_slope_c_per_h": _q(
            [t.pair.temperature_slope_c_per_h for t in trials]
        ),
        "shared_temp_corr": _q([t.pair.shared_temp_corr for t in trials]),
        "q10_per_fish": _q(q10s),
        "q10_from_shared": _q(np.exp(10 * per_degc)),
    }
    figures["drift"] = {
        "lags": lags,
        "gammas": [f.drift_gamma_hz2 for f in fish],
        "pooled": pooled_gamma,
        "pooled_fit": pooled_fit,
        "pooled_exp": pooled_exp,
        "shared_lags": shared_lags,
        "shared_gammas": [t.pair.shared_gamma for t in trials],
        "shared_pooled": shared_gamma,
        "shared_fit": shared_fit,
    }
    params["gridtools.sim.eod.DriftParams.sd_hz"] = _param(
        round(pooled_fit.sd, 3),
        current=_default(DriftParams, "sd_hz"),
        uncertainty={"across_fish": _q(sds)},
        method="semivariogram of each fish's despiked 1-s EODf minus its shared (relative) drift, light phase (>= 3.25 h), lags 1-1800 s, 1% trimmed; median over fish; Gaussian-kernel fit with nugget",
        n=len(fish),
    )
    params["gridtools.sim.eod.DriftParams.timescale_s"] = _param(
        round(pooled_fit.timescale_s, 1),
        current=_default(DriftParams, "timescale_s"),
        uncertainty={"across_fish": _q(taus)},
        method="as sd_hz",
        n=len(fish),
        note=(
            f"Gaussian kernel RMS log residual {pooled_fit.rms_log_residual:.2f}, "
            f"exponential {pooled_exp.rms_log_residual:.2f} (tau_exp = {pooled_exp.timescale_s:.0f} s); "
            "the real variogram rises more slowly than Delta t^2 below ~5 s (tracking smooths over 0.8-1.6 s windows) "
            "and keeps rising slightly beyond the plateau"
        ),
    )
    control_rate = max(
        _default(SharedDriftParams, "control_rate_hz"), 2.0 / shared_fit.timescale_s
    )
    params["gridtools.sim.eod.SharedDriftParams.sd_rel"] = _param(
        round(shared_fit.sd, 4),
        current=_default(SharedDriftParams, "sd_rel"),
        uncertainty={"across_trials": _q(shared_sds)},
        method="mean of the two fish's relative slow baselines per trial; semivariogram (lags 60 s - 4 h), median over trials, Gaussian-kernel fit",
        n=len(trials),
        note="real shared drift is mostly a monotonic decline with water temperature (median slope "
        f"{np.nanmedian(100 * slopes):.2f} %/h); a stationary process with this sd and timescale reproduces its increments over minutes to hours, not its sign",
    )
    params["gridtools.sim.eod.SharedDriftParams.timescale_s"] = _param(
        round(shared_fit.timescale_s, 0),
        current=_default(SharedDriftParams, "timescale_s"),
        uncertainty={"across_trials": _q(shared_taus)},
        method="as sd_rel",
        n=len(trials),
    )
    params["gridtools.sim.eod.SharedDriftParams.control_rate_hz"] = _param(
        control_rate,
        current=_default(SharedDriftParams, "control_rate_hz"),
        method="unchanged unless needed for timescale_s * control_rate_hz >= 2",
    )

    # 4. Field falloff -------------------------------------------------------
    layout_names = sorted({t.layout for t in trials})
    layouts = {name: layout_for(name, config) for name in layout_names}
    profiles_by_layout = {
        name: [
            np.concatenate([f.profiles for f in t.fish])
            for t in trials
            if t.layout == name
        ]
        for name in layout_names
    }
    snr_by_layout = {
        name: np.concatenate(
            [f.peak_snr_db for t in trials if t.layout == name for f in t.fish]
        )
        for name in layout_names
    }
    spread = float(np.median([_noise_spread(t) for t in trials]))
    draws = {
        name: draw_simulation(
            rng,
            config.n_field_samples,
            layouts[name].n_simulated,
            snr_by_layout[name],
            spread,
        )
        for name in layout_names
    }
    fit = fit_field_falloff(
        profiles_by_layout,
        layouts,
        draws,
        exponents=config.exponents,
        regularisers_rel=config.regularisers_rel,
        margins_rel=config.margins_rel,
        n_bootstrap=config.n_bootstrap,
        rng=rng,
    )
    sensitivity: Json = {}
    # Per-layout fits (is the answer consistent between grids?).
    for name in layout_names:
        i, j, k = np.unravel_index(
            int(np.argmin(fit.loss_by_layout[name])), fit.loss.shape
        )
        sensitivity[f"layout {name} alone"] = {
            "distance_exponent": float(fit.exponents[i]),
            "regulariser_rel": float(fit.regularisers_rel[j]),
            "margin_rel": float(fit.margins_rel[k]),
            "loss": float(fit.loss_by_layout[name][i, j, k]),
            "n_trials": len(profiles_by_layout[name]),
        }
    # Best exponent per fixed margin (dependence on the position prior).
    for k, m in enumerate(fit.margins_rel):
        i, j = np.unravel_index(int(np.argmin(fit.loss[:, :, k])), fit.loss.shape[:2])
        sensitivity[f"margin {m:g} fixed"] = {
            "distance_exponent": float(fit.exponents[i]),
            "regulariser_rel": float(fit.regularisers_rel[j]),
            "loss": float(fit.loss[i, j, k]),
        }
    # The physical dipole (n = 2) at its best regulariser and margin.
    i2 = int(np.argmin(np.abs(fit.exponents - 2.0)))
    j2, k2 = np.unravel_index(int(np.argmin(fit.loss[i2])), fit.loss.shape[1:])
    sensitivity["n = 2 (physical dipole)"] = {
        "regulariser_rel": float(fit.regularisers_rel[j2]),
        "margin_rel": float(fit.margins_rel[k2]),
        "loss": float(fit.loss[i2, j2, k2]),
    }
    # 2019 missing electrode at an unknown position instead of a corner.
    if "9-8" in layouts:
        alt = GridLayout("9-8 random", 3, 3, config.spacing_m, (), n_random_missing=1)
        alt_draws = draw_simulation(
            rng, config.n_field_samples, 9, snr_by_layout["9-8"], spread
        )
        alt_fit = fit_field_falloff(
            {"9-8 random": profiles_by_layout["9-8"]},
            {"9-8 random": alt},
            {"9-8 random": alt_draws},
            exponents=config.exponents,
            regularisers_rel=config.regularisers_rel,
            margins_rel=config.margins_rel,
            n_bootstrap=0,
            rng=rng,
        )
        sensitivity["2019 missing electrode at random position"] = alt_fit.best
    # Noise-floor spread switched off.
    flat_draws = {
        name: draw_simulation(
            rng,
            config.n_field_samples,
            layouts[name].n_simulated,
            snr_by_layout[name],
            0.0,
        )
        for name in layout_names
    }
    flat_fit = fit_field_falloff(
        profiles_by_layout,
        layouts,
        flat_draws,
        exponents=config.exponents,
        regularisers_rel=config.regularisers_rel,
        margins_rel=config.margins_rel,
        n_bootstrap=0,
        rng=rng,
    )
    sensitivity["equal noise floors on all electrodes"] = flat_fit.best
    # Identifiability: parameter sets within the bootstrap spread of the loss.
    boot_losses = []
    for b in fit.bootstrap:
        i = int(np.argmin(np.abs(fit.exponents - b[0])))
        j = int(np.argmin(np.abs(fit.regularisers_rel - b[1])))
        k = int(np.argmin(np.abs(fit.margins_rel - b[2])))
        boot_losses.append(fit.loss[i, j, k])
    tolerance = (
        float(np.quantile(boot_losses, 0.9) - fit.best["loss"]) if boot_losses else 0.0
    )
    near = fit.loss <= fit.best["loss"] + max(tolerance, 1e-9)
    idx = np.argwhere(near)
    sensitivity["near_optimal_set"] = {
        "loss_tolerance_db2": tolerance,
        "distance_exponent_range": [
            float(fit.exponents[idx[:, 0]].min()),
            float(fit.exponents[idx[:, 0]].max()),
        ],
        "regulariser_rel_range": [
            float(fit.regularisers_rel[idx[:, 1]].min()),
            float(fit.regularisers_rel[idx[:, 1]].max()),
        ],
        "margin_rel_range": [
            float(fit.margins_rel[idx[:, 2]].min()),
            float(fit.margins_rel[idx[:, 2]].max()),
        ],
        "n_sets": int(idx.shape[0]),
    }
    # Effective log-log slope of the gain at one spacing on the body axis:
    # d ln g / d ln r = 1 - (n + 1) r^2 / (r^2 + a^2), r = 1 spacing.
    slopes = {}
    candidates = [(fit.best["distance_exponent"], fit.best["regulariser_rel"])]
    candidates += [
        (float(fit.exponents[i]), float(fit.regularisers_rel[j])) for i, j, _ in idx
    ]
    candidates.append((2.0, sensitivity["n = 2 (physical dipole)"]["regulariser_rel"]))
    for n_c, a_c in candidates:
        slopes[f"n={n_c:g}, a_eff={a_c:g}"] = 1.0 - (n_c + 1.0) / (1.0 + a_c**2)
    sensitivity["effective_slope_at_one_spacing"] = slopes

    a_eff = fit.best["regulariser_rel"] * config.spacing_m
    dip = DipoleParams()
    dz = dip.electrode_depth_m - dip.fish_depth_m
    min_distance = math.sqrt(max(a_eff**2 - dz**2, 0.0))
    n_lo, n_hi = fit.interval("distance_exponent")
    a_lo, a_hi = fit.interval("regulariser_rel")
    m_lo, m_hi = fit.interval("margin_rel")
    params["gridtools.sim.field.DipoleParams.distance_exponent"] = _param(
        fit.best["distance_exponent"],
        current=_default(DipoleParams, "distance_exponent"),
        uncertainty={
            "bootstrap_90": [n_lo, n_hi],
            "near_optimal": sensitivity["near_optimal_set"]["distance_exponent_range"],
        },
        method="simulation-based fit of order-free sorted sign_v profiles (quantiles of ranks 2..E), random positions (uniform over grid + margin) and headings, noise floor and strongest-electrode SNR from the same recordings; grid search; bootstrap over trials",
        n=len(trials),
    )
    params["gridtools.sim.field.DipoleParams.min_distance_m"] = _param(
        round(min_distance, 3),
        current=_default(DipoleParams, "min_distance_m"),
        uncertainty={
            "effective_regulariser_rel_bootstrap_90": [a_lo, a_hi],
            "effective_regulariser_m": a_eff,
        },
        method="effective regulariser a_eff = sqrt(a^2 + dz^2) from the profile fit, in units of the spacing, times the fishgrid.cfg spacing (0.5 m); minus the default 5-cm depth offset",
        note="a and the fish-electrode depth offset are not separately identifiable; the metre value scales with the true electrode spacing, which conflicts with the tank size in the paper (see report)",
    )
    new["scene.arena_margin_rel"] = {
        "value": fit.best["margin_rel"],
        "uncertainty": {"bootstrap_90": [m_lo, m_hi]},
        "method": "position prior margin beyond the outer electrodes (spacings) that best reproduces the profiles; weakly identified",
    }

    # 5. Noise, hum, amplitudes ---------------------------------------------
    noise_summary, noise_extra = _noise_summary(trials)
    floor_v2 = 10 ** (noise_summary["floor_db_v2_hz"]["q50"] / 10)
    nyquist = 10000.0
    alpha = noise_summary["alpha"]["q50"]
    params["gridtools.sim.noise.ColoredNoiseParams.exponent"] = _param(
        0.0 if abs(alpha) < 0.15 else _round(alpha, 2),
        current=_default(ColoredNoiseParams, "exponent"),
        uncertainty={"alpha_across_electrodes": noise_summary["alpha"]},
        method="power-law fit of the band-wise median PSD floor, 100-5000 Hz, per live electrode",
        note="the floor is white within +/-0.15 in the exponent; the simulator's pink default puts too much noise below the fish band",
    )
    params["gridtools.sim.noise.ColoredNoiseParams.sd"] = _param(
        float(math.sqrt(floor_v2 * nyquist)),
        current=_default(ColoredNoiseParams, "sd"),
        uncertainty={
            "floor_db_v2_hz_across_trials": noise_summary["floor_db_v2_hz"],
            "by_year": noise_summary["floor_by_year"],
        },
        method="sqrt(N0 * fs/2) for a white floor N0 = median PSD in 400-1500 Hz (median over live electrodes and 20 x 10-s segments per trial; median over trials); raw ADC volts",
        n=len(trials),
        note="draw per scene from the across-trial range (2019 floors are higher than 2020)",
    )
    params["gridtools.sim.noise.ElectrodeFaultParams.dead_prob"] = _param(
        round(noise_summary["dead_electrodes"] / noise_summary["electrodes"], 3),
        current=_default(ElectrodeFaultParams, "dead_prob"),
        method="electrodes whose fish fundamental never exceeds the floor by 10 dB in any segment",
        n=noise_summary["electrodes"],
    )
    params["gridtools.sim.noise.ElectrodeFaultParams.noisy_prob"] = _param(
        round(
            noise_summary["noisy_electrodes"]
            / max(1, noise_summary["electrodes"] - noise_summary["dead_electrodes"]),
            3,
        ),
        current=_default(ElectrodeFaultParams, "noisy_prob"),
        method="live electrodes with a floor > 6 dB above the trial median",
        n=noise_summary["electrodes"],
    )
    if noise_summary["noisy_factor"]["n"]:
        params["gridtools.sim.noise.ElectrodeFaultParams.noisy_factor"] = _param(
            _round(noise_summary["noisy_factor"]["q50"], 2),
            current=_default(ElectrodeFaultParams, "noisy_factor"),
            uncertainty=noise_summary["noisy_factor"],
            method="amplitude ratio of noisy electrodes' floor to the trial median",
        )
    params["gridtools.sim.noise.GainMismatchParams.log_sd"] = _param(
        _default(GainMismatchParams, "log_sd"),
        current=_default(GainMismatchParams, "log_sd"),
        uncertainty={
            "floor_spread_db_upper_bound": noise_summary["electrode_spread_db"]
        },
        method="not identifiable without positions; the per-electrode floor spread (SD in dB across live electrodes) bounds gain mismatch plus electrode noise differences",
    )
    params["gridtools.sim.noise.ColoredNoiseParams.per_electrode_sd_db"] = _param(
        _round(noise_summary["electrode_spread_db"]["q50"], 2),
        current=None,
        uncertainty=noise_summary["electrode_spread_db"],
        method="SD across live electrodes of the floor (dB); pass as colored_noise(sd_scale=10**(N(0, sd)/20))",
        note="not a field of ColoredNoiseParams; realised through sd_scale",
    )
    hum_range = noise_summary["hum_50hz_max_electrode_v"]
    params["gridtools.sim.noise.HumParams.amplitude_range"] = _param(
        [hum_range["q05"], hum_range["q95"]],
        current=list(_default(HumParams, "amplitude_range")),
        uncertainty={"by_year": noise_summary["hum_50hz_by_year_v"]},
        method="50-Hz amplitude (lobe sum above floor, Welch PSD) at the electrode with the most hum, median over segments; 5-95% across trials, raw ADC volts",
        note="the 100-Hz high-pass of the amplifier is already in these numbers",
    )
    ratio = noise_summary["hum_harmonic_ratio"]
    odd = [
        ratio[str(h)]["q50"] for h in (3, 5, 7, 9) if ratio.get(str(h), {}).get("q50")
    ]
    decay = (
        float(
            np.exp(
                np.mean(np.log(np.asarray(odd) + 1e-12))
                / np.mean([2, 4, 6, 8][: len(odd)])
            )
        )
        if odd
        else float("nan")
    )
    params["gridtools.sim.noise.HumParams.harmonic_decay"] = _param(
        _round(decay, 3),
        current=_default(HumParams, "harmonic_decay"),
        uncertainty={"harmonic_to_fundamental_ratio": ratio},
        method="geometric decay matching the median ratios of the odd harmonics 3, 5, 7, 9 to the fundamental",
        note="real hum is dominated by odd harmonics (even ones are 10-20 dB weaker), which the simulator's single geometric decay does not express",
    )
    params["gridtools.sim.noise.HumParams.n_harmonics"] = _param(
        round(noise_summary["hum_harmonics_detected"]["q50"]),
        current=_default(HumParams, "n_harmonics"),
        uncertainty=noise_summary["hum_harmonics_detected"],
        method="harmonics of 50 Hz (up to 2 kHz) detected 6 dB over the local floor in >= half of the segments, on the electrode with most hum",
    )
    params["gridtools.sim.noise.HumParams.coupling_log_sd"] = _param(
        _round(noise_summary["hum_coupling_log_sd"]["q50"], 2),
        current=_default(HumParams, "coupling_log_sd"),
        uncertainty=noise_summary["hum_coupling_log_sd"],
        method="SD of ln(50-Hz amplitude) across live electrodes within a trial; median over trials",
    )
    params["gridtools.sim.noise.ClipParams.limit"] = _param(
        0.02048,
        current=_default(ClipParams, "limit"),
        uncertainty={"max_abs_v_across_trials": noise_summary["max_abs_v"]},
        method="ADC range: quantisation step 0.625 uV = 40.96 mV / 2**16 in the raw data, i.e. +/-20.48 mV; raw units are volts",
        note=f"no clipping observed (largest clipped fraction {noise_summary['clipped_fraction_max']:.1e})",
    )

    peak_amp = np.concatenate([f.peak_amplitude_v for f in fish])
    raw_amp = np.concatenate([f.raw_peak_amplitude_v for f in fish])
    per_fish_amp = np.array(
        [np.median(f.raw_peak_amplitude_v) for f in fish if f.raw_peak_amplitude_v.size]
    )
    cn0 = np.concatenate([f.cn0_db_hz for f in fish])
    gains = _gain_quantiles(fit, config, rng)
    fundamental = APTERONOTUS_LEPTORHYNCHUS.amplitudes[0]
    amp_scale = float(np.median(raw_amp) / (fundamental * np.median(gains)))
    log_sd_fish = (
        float(np.std(np.log(per_fish_amp), ddof=1))
        if per_fish_amp.size > 2
        else float("nan")
    )
    amplitudes = {
        "peak_fundamental_v_sign_v": _q(peak_amp),
        "peak_fundamental_v_raw": _q(raw_amp),
        "per_fish_median_v_raw": _q(per_fish_amp),
        "cn0_db_hz": _q(cn0),
        "snr_db_in_v1_spectrogram_4096": _q(cn0 - 10 * math.log10(1.5 * 20000 / 4096)),
        "simulated_max_gain_quantiles": _q(gains),
    }
    new["scene.eod_amplitude"] = {
        "value": amp_scale,
        "uncertainty": {"log_sd_across_fish": log_sd_fish},
        "method": "EOD amplitude A (synthesize_eod) so that 0.90 * A * gain at the strongest electrode matches the median real fundamental amplitude there, with the fitted field, 0.5-m geometry and default depths; raw ADC volts",
        "note": "draw per fish log-normally with the given log SD",
    }
    new["scene.noise_relative_to_fish"] = {
        "value": {"cn0_db_hz": amplitudes["cn0_db_hz"]},
        "method": "carrier-to-noise density of the fish fundamental at its strongest electrode, from sign_v and the measured floor; FFT-length independent",
    }
    new["noise.click_rate_per_h"] = {
        "value": noise_summary["clicks_per_h"]["q50"],
        "uncertainty": noise_summary["clicks_per_h"],
        "method": "high-band (> 4 kHz) transients 12 robust SD over the floor, merged across electrodes, in 20 x 10-s segments per trial (audian/wavetracker definition, simplified)",
        "note": "mostly periodic recording artefacts (pairs 30 ms apart; in some 2020 trials onset/offset pairs 0.1 s apart about every second, likely the 100-ms LED sync pulses), not random clicks",
    }

    # 6. Movement -----------------------------------------------------------
    movement = _movement_summary(rows, trials)
    sim_moves: Json = {}
    base = MovementParams()
    variants = {
        "default": base,
        "half speed": dataclasses.replace(base, speed_mean_m_s=base.speed_mean_m_s / 2),
        "double speed": dataclasses.replace(
            base, speed_mean_m_s=base.speed_mean_m_s * 2
        ),
        "mostly resting (rest 60 s, move 10 s)": dataclasses.replace(
            base, mean_rest_s=60.0, mean_move_s=10.0
        ),
        "resting (rest 600 s, move 10 s)": dataclasses.replace(
            base, mean_rest_s=600.0, mean_move_s=10.0
        ),
        "resting, little rotation (rest 600 s, move 10 s, rest_turn_factor 0.05)": (
            dataclasses.replace(
                base, mean_rest_s=600.0, mean_move_s=10.0, rest_turn_factor=0.05
            )
        ),
        "1.5x speed, rest 5 s, move 30 s": dataclasses.replace(
            base,
            speed_mean_m_s=base.speed_mean_m_s * 1.5,
            mean_rest_s=5.0,
            mean_move_s=30.0,
        ),
        "always moving": dataclasses.replace(base, mean_rest_s=0.0),
    }
    counts = {name: sum(t.layout == name for t in trials) for name in layout_names}
    move_layout = "15-15" if "15-15" in counts else max(counts, key=lambda k: counts[k])
    move_trials = [t for t in trials if t.layout == move_layout]
    move_snr = np.concatenate([f.peak_snr_db for t in move_trials for f in t.fish])
    dt = float(np.median([t.frame_dt_s for t in move_trials]))
    movement["simulated_layout"] = move_layout
    for label, mp in variants.items():
        stats = [
            simulated_dwell_stats(
                rng,
                layouts[move_layout],
                mp,
                distance_exponent=fit.best["distance_exponent"],
                regulariser_rel=fit.best["regulariser_rel"],
                margin_rel=fit.best["margin_rel"],
                duration_s=config.movement_duration_s,
                frame_dt_s=dt,
                median_peak_snr_db=float(np.median(move_snr)),
            )
            for _ in range(4)
        ]
        sim_moves[label] = {
            "switches_per_min_3db": float(np.mean([s.switches_per_min for s in stats])),
            "still_fraction_3db": float(np.mean([s.still_fraction for s in stats])),
            "median_dwell_s": float(
                np.median(np.concatenate([s.dwell_times_s for s in stats]))
            ),
        }
    movement["simulated"] = sim_moves
    real = movement[move_layout]
    regimes: Json = {}
    for phase in ("dark", "light"):
        target_rate = real[f"switches_per_min_3db_{phase}"]["q50"]
        target_still = real[f"still_fraction_3db_{phase}"]["q50"]

        def distance(
            label: str, rate: float = target_rate, still: float = target_still
        ) -> float:
            sim = sim_moves[label]
            return abs(
                math.log(max(sim["switches_per_min_3db"], 1e-3) / max(rate, 1e-3))
            ) + abs(sim["still_fraction_3db"] - still)

        closest = min(sim_moves, key=distance)
        regimes[phase] = {
            "closest_variant": closest,
            "real_switches_per_min": target_rate,
            "real_still_fraction": target_still,
            "simulated_switches_per_min": sim_moves[closest]["switches_per_min_3db"],
            "simulated_still_fraction": sim_moves[closest]["still_fraction_3db"],
            "params": {
                k: v
                for k, v in dataclasses.asdict(variants[closest]).items()
                if v != dataclasses.asdict(base)[k]
            },
        }
    new["movement.regimes"] = {
        "value": regimes,
        "method": f"closest of {len(sim_moves)} MovementParams variants to the real median switch rate and still fraction ({move_layout} trials), per phase; simulated with the fitted field",
        "note": "coarse: only order-free proxies, depends on the field model and electrode spacing; speeds and turning are not identified",
    }

    # 7. Chirps ---------------------------------------------------------------
    chirp_summary, chirp_extra = _chirp_summary(trials)
    new["chirps.rate_per_h"] = {
        "value": {
            "loser": chirp_summary["rate_per_h"]["loser"],
            "winner": chirp_summary["rate_per_h"]["winner"],
        },
        "method": "v1-detected chirps per fish and hour (whole trial; dark/light in the report)",
        "note": chirp_summary["bias_note"],
    }
    new["chirps.min_interval_s"] = {
        "value": chirp_summary["ici_s"]["q01"],
        "uncertainty": {"per_fish_minimum": chirp_summary["min_ici_per_fish_s"]},
        "method": "1st percentile of within-fish inter-chirp intervals after merging v1 duplicates closer than 50 ms",
    }
    new["chirps.ici_s"] = {
        "value": chirp_summary["ici_s"],
        "fraction_below": chirp_summary["ici_fraction_below"],
    }
    new["chirps.cross_fish"] = {
        "value": {
            "coincidence_0.25s": chirp_summary["coincidence_0.25s"],
            "latency_excess_0_1s": chirp_summary.get("latency_excess_0_1s"),
            "latency_excess_1_3s": chirp_summary.get("latency_excess_1_3s"),
        },
        "method": "next-chirp latency from one fish's chirp to the other's, observed vs circular-shift null (200 shifts per trial)",
    }

    summary = {
        "n_trials": len(trials),
        "trials": [t.name for t in trials],
        "layouts": {
            name: [t.name for t in trials if t.layout == name] for name in layout_names
        },
        "eodf": eodf,
        "delta_f": df_summary,
        "drift": drift,
        "field": {
            "best": fit.best,
            "bootstrap_90": {
                "distance_exponent": [n_lo, n_hi],
                "regulariser_rel": [a_lo, a_hi],
                "margin_rel": [m_lo, m_hi],
            },
            "sensitivity": sensitivity,
            "noise_offset_sd_db": spread,
            "max_minus_median_db": {
                name: _q(
                    -np.nanmedian(np.concatenate(profiles_by_layout[name]), axis=1)
                )
                for name in layout_names
            },
            "drop_to_second_db": {
                name: _q(np.concatenate(profiles_by_layout[name])[:, 1])
                for name in layout_names
            },
            "peak_snr_db": {name: _q(snr_by_layout[name]) for name in layout_names},
        },
        "noise": noise_summary,
        "amplitudes": amplitudes,
        "movement": movement,
        "chirps": chirp_summary,
    }
    figures.update(
        {
            "field": {
                "fit": fit,
                "layouts": layouts,
                "draws": draws,
                "profiles": profiles_by_layout,
            },
            "noise": noise_extra,
            "noise_spectra": {
                t.name: (t.noise.floor_band_hz, t.noise.floor_spectrum, t.noise.dead)
                for t in trials
            },
            "amplitudes": {
                "raw": raw_amp,
                "sign_v": peak_amp,
                "cn0": cn0,
                "gains": gains,
            },
            "chirps": chirp_extra,
            "movement_rows": rows,
        }
    )
    return Calibration(summary, params, new, fit, sensitivity, figures)


def _noise_spread(t: TrialStats) -> float:
    live = ~t.noise.dead
    db = 10.0 * np.log10(np.median(t.noise.floor_v2_hz[:, live], axis=0))
    return float(np.std(db - np.median(db), ddof=1))
