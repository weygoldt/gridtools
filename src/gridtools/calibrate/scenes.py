"""Measure simulated scenes with the calibration's definitions.

Closes the loop between :mod:`gridtools.sim` and the calibration: the same
estimators that produced ``calibration.json`` from the tube-competition
recordings (noise floor and hum from Welch spectra, ``sign_v`` profiles,
strongest-electrode switches, drift variograms, chirp intervals, clicks) are
run on scene directories written by :func:`gridtools.sim.scene.write_scene`,
and :func:`compare_scenes` tabulates simulated against real values.

Scenes are short (minutes) and are drawn to be harder than real (close
frequencies, high chirp rates, noisier recordings), so the comparison is a
plausibility check of the calibrated components, not a goodness-of-fit.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gridtools.calibrate.frequency import fit_drift_variogram, semivariogram
from gridtools.calibrate.profiles import (
    dwell_stats,
    relative_sorted_profiles,
    strongest_electrode,
)
from gridtools.calibrate.spectra import (
    HANN_ENBW_BINS,
    count_clicks,
    line_amplitude,
    noise_floor,
    psd,
)
from gridtools.io import load_tracks, open_recording

__all__ = ["SceneStats", "compare_scenes", "measure_scene"]

logger = logging.getLogger(__name__)

_NOISE_NFFT = 8192
_SEGMENT_S = 10.0


@dataclass
class SceneStats:
    """Measurements of one simulated scene (see :func:`measure_scene`)."""

    name: str
    n_electrodes: int
    harder: bool
    floor_db: float
    hum_50hz_v: float
    clicks_per_h: float
    baselines_hz: list[float] = field(default_factory=list)
    abs_delta_f_hz: list[float] = field(default_factory=list)
    drift_sd_hz: list[float] = field(default_factory=list)
    drift_timescale_s: list[float] = field(default_factory=list)
    max_minus_median_db: list[float] = field(default_factory=list)
    drop_to_second_db: list[float] = field(default_factory=list)
    peak_snr_db: list[float] = field(default_factory=list)
    peak_amplitude_v: list[float] = field(default_factory=list)
    switches_per_min: dict[str, list[float]] = field(default_factory=dict)
    ici_s: list[float] = field(default_factory=list)
    chirps_per_h: list[float] = field(default_factory=list)
    coincident_fraction: float = float("nan")


def measure_scene(directory: str | Path, *, max_frames: int = 2000) -> SceneStats:
    """Measure one scene directory with the calibration's definitions.

    Parameters
    ----------
    directory : path
        Scene directory.
    max_frames : int
        Profile frames used per fish at most (evenly spaced).

    Returns
    -------
    SceneStats
    """
    directory = Path(directory)
    info = json.loads((directory / "scene.json").read_text())
    recording = open_recording(directory / "recording.wav")
    fs = recording.samplerate_hz
    dead = np.asarray(info["channels"]["dead"], dtype=bool)
    live = ~dead

    # noise floor, hum and clicks on consecutive 10-s segments
    seg = round(_SEGMENT_S * fs)
    floors, hums, events = [], [], 0
    n_segments = max(1, recording.n_samples // seg)
    for k in range(n_segments):
        x = recording.read(k * seg, min((k + 1) * seg, recording.n_samples))
        freqs, power = psd(x, fs, _NOISE_NFFT)
        floors.append(noise_floor(freqs, power, (400.0, 1500.0)))
        hums.append(float(np.max(line_amplitude(freqs, power, 50.0)[live])))
        _, onsets = count_clicks(x, fs)
        events += onsets.size
    floor = np.median(np.stack(floors), axis=0)
    floor_db = 10.0 * np.log10(floor)
    stats = SceneStats(
        name=directory.name,
        n_electrodes=int(live.sum()),
        harder=bool(info["levels"].get("harder", False)),
        floor_db=float(np.median(floor_db[live])),
        hum_50hz_v=float(np.median(hums)),
        clicks_per_h=events / (n_segments * _SEGMENT_S) * 3600.0,
    )

    # tracks: EODf, delta F, drift, profiles, switches
    tracks = load_tracks(directory)
    tracks_cfg = info["config"]["tracks"]
    nfft = (
        round(fs / float(tracks_cfg["power_resolution_hz"]))
        if "power_resolution_hz" in tracks_cfg
        else int(tracks_cfg["power_nfft"])
    )
    enbw = HANN_ENBW_BINS * fs / nfft
    fish_tracks = list(tracks)
    fish_info = {int(f["fish_id"]): f for f in info["fish"]}
    for track in fish_tracks:
        stats.baselines_hz.append(float(np.median(track.freqs_hz)))
    for a in range(len(fish_tracks)):
        for b in range(a + 1, len(fish_tracks)):
            ta, tb = fish_tracks[a], fish_tracks[b]
            step = max(1, round(10.0 / float(np.median(np.diff(ta.times_s)))))
            stats.abs_delta_f_hz.extend(
                np.abs(ta.freqs_hz[::step] - tb.freqs_hz[::step]).tolist()
            )
    for track in fish_tracks:
        entry = fish_info.get(track.ident, {})  # {}: a real fish (hybrid scene)
        if not entry.get("ramps"):
            grid = np.arange(0.0, track.times_s[-1], 1.0)
            f = np.interp(grid, track.times_s, track.freqs_hz)
            lags = np.unique(np.round(np.geomspace(1, len(grid) // 2, 30)).astype(int))
            gamma, _ = semivariogram(f - np.median(f), lags)
            fit = fit_drift_variogram(lags.astype(float), gamma)
            stats.drift_sd_hz.append(float(fit.sd))
            stats.drift_timescale_s.append(float(fit.timescale_s))
        assert track.powers_db is not None
        others = [o for o in fish_tracks if o.ident != track.ident]
        usable = np.ones(track.times_s.size, dtype=bool)
        for other in others:
            usable &= np.abs(other.frequency_at(track.times_s) - track.freqs_hz) >= 5.0
        rows = np.flatnonzero(usable)
        rows = rows[:: max(1, rows.size // max_frames)]
        powers = track.powers_db[rows][:, live]
        if rows.size and powers.shape[1] >= 3:
            profiles = relative_sorted_profiles(powers)
            stats.max_minus_median_db.extend((-np.nanmedian(profiles, axis=1)).tolist())
            stats.drop_to_second_db.extend(profiles[:, 1].tolist())
            peak = np.max(powers, axis=1)
            e_max = np.argmax(powers, axis=1)
            stats.peak_snr_db.extend((peak - floor_db[live][e_max]).tolist())
            stats.peak_amplitude_v.extend(
                np.sqrt(2.0 * 10.0 ** (peak / 10.0) * enbw).tolist()
            )
        if live.sum() >= 3:
            sequence = strongest_electrode(track.powers_db[:, live], 3.0)
            dwell = dwell_stats(track.times_s, sequence)
            stats.switches_per_min.setdefault(entry.get("movement", "dark"), []).append(
                dwell.switches_per_min
            )

    # chirps
    chirps = pd.read_parquet(directory / "chirps.parquet")
    duration_h = recording.duration_s / 3600.0
    emitters = chirps["emitter_id"].to_numpy(dtype=np.int64)
    t_centre = chirps["t_center_s"].to_numpy(dtype=np.float64)
    centres = {fid: np.sort(t_centre[emitters == fid]) for fid in fish_info}
    for fid in fish_info:
        times = centres.get(fid, np.zeros(0))
        stats.chirps_per_h.append(times.size / duration_h)
        stats.ici_s.extend(np.diff(times).tolist())
    if len(chirps) and len(fish_info) > 1:
        dt = chirps["dt_nearest_other_chirp_s"].to_numpy(dtype=float)
        stats.coincident_fraction = float(np.mean(dt <= 0.25))
    return stats


def _q(values: Any, scale: float = 1.0) -> str:
    v = np.asarray(values, dtype=np.float64) * scale
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "-"
    q05, q50, q95 = np.quantile(v, [0.05, 0.5, 0.95])
    return f"{q50:.3g} ({q05:.3g}-{q95:.3g})"


def _real(summary: dict[str, Any], *keys: str, scale: float = 1.0) -> str:
    node: Any = summary
    for key in keys:
        node = node[key]
    return f"{node['q50'] * scale:.3g} ({node['q05'] * scale:.3g}-{node['q95'] * scale:.3g})"


def compare_scenes(
    scene_dirs: list[str | Path], calibration_json: str | Path
) -> pd.DataFrame:
    """Simulated vs real statistics, one row per quantity.

    Values are medians with 5-95 % ranges (pooled over fish, frames, pairs or
    intervals as in the calibration report).
    """
    calibration = json.loads(Path(calibration_json).read_text())
    summary = calibration["summary"]
    measured = [measure_scene(d) for d in scene_dirs]

    def pooled(name: str, scenes: list[SceneStats] | None = None) -> list[float]:
        return [
            v
            for s in (measured if scenes is None else scenes)
            for v in getattr(s, name)
        ]

    grids = [s for s in measured if s.n_electrodes >= 9]

    switches: dict[str, list[float]] = {}
    for s in grids:
        for regime, values in s.switches_per_min.items():
            switches.setdefault(regime, []).extend(values)
    delta_f = np.asarray(pooled("abs_delta_f_hz"))
    ici = np.asarray(pooled("ici_s"))
    real_df = summary["delta_f"]["fraction_below_pooled"]
    real_ici = summary["chirps"]["ici_fraction_below"]
    real_coincidence = summary["chirps"]["coincidence_0.25s"]["pooled"]
    realistic = [s for s in measured if not s.harder]
    harder = [s for s in measured if s.harder]
    rows = [
        (
            "baseline EODf (Hz)",
            _q(pooled("baselines_hz")),
            _real(summary, "eodf", "all"),
        ),
        (
            "|dF| (Hz)",
            _q(delta_f),
            _real(summary, "delta_f", "abs_delta_f_hz"),
        ),
        (
            "fraction |dF| < 5 / 10 / 20 Hz",
            " / ".join(f"{np.mean(delta_f < t):.3f}" for t in (5, 10, 20))
            if delta_f.size
            else "-",
            " / ".join(f"{real_df[str(t)]:.3f}" for t in (5.0, 10.0, 20.0)),
        ),
        (
            "drift SD (Hz), fish without ramps",
            _q(pooled("drift_sd_hz")),
            _real(summary, "drift", "individual_sd_hz"),
        ),
        (
            "drift time-scale (s)",
            _q(pooled("drift_timescale_s")),
            _real(summary, "drift", "individual_timescale_s"),
        ),
        (
            "noise floor (dB re 1 V2/Hz), realistic scenes",
            _q([s.floor_db for s in realistic]),
            _real(summary, "noise", "floor_db_v2_hz"),
        ),
        (
            "noise floor, harder scenes",
            _q([s.floor_db for s in harder]),
            "-",
        ),
        (
            "hum 50 Hz, strongest electrode (V)",
            _q([s.hum_50hz_v for s in measured]),
            _real(summary, "noise", "hum_50hz_max_electrode_v"),
        ),
        (
            "fundamental at strongest electrode (V, sign_v; >= 9 electrodes)",
            _q(pooled("peak_amplitude_v", grids)),
            _real(summary, "amplitudes", "peak_fundamental_v_sign_v"),
        ),
        (
            "strongest minus median electrode (dB; >= 9 electrodes)",
            _q(pooled("max_minus_median_db", grids)),
            _real(summary, "field", "max_minus_median_db", "15-15"),
        ),
        (
            "drop to second electrode (dB; >= 9 electrodes)",
            _q(pooled("drop_to_second_db", grids)),
            _real(summary, "field", "drop_to_second_db", "15-15"),
        ),
        (
            "strongest-electrode SNR (dB, sign_v bin over floor; >= 9 el., realistic)",
            _q(pooled("peak_snr_db", [s for s in grids if not s.harder])),
            _real(summary, "field", "peak_snr_db", "15-15"),
        ),
        (
            "electrode switches/min, dark regime (>= 9 electrodes)",
            _q(switches.get("dark", [])),
            _real(summary, "movement", "15-15", "switches_per_min_3db_dark"),
        ),
        (
            "electrode switches/min, light regime (>= 9 electrodes)",
            _q(switches.get("light", [])),
            _real(summary, "movement", "15-15", "switches_per_min_3db_light"),
        ),
        (
            "chirps per fish and hour",
            _q(pooled("chirps_per_h")),
            _real(summary, "chirps", "rate_per_h", "loser"),
        ),
        (
            "within-fish intervals < 0.5 / 1 / 2 s",
            " / ".join(f"{np.mean(ici < t):.2f}" for t in (0.5, 1.0, 2.0))
            if ici.size
            else "-",
            " / ".join(f"{real_ici[str(t)]:.2f}" for t in (0.5, 1.0, 2.0)),
        ),
        (
            "chirps with another fish's chirp within 0.25 s",
            f"{np.nanmean([s.coincident_fraction for s in measured]):.3f}",
            f"{real_coincidence['observed']:.4f} (v1; null {real_coincidence['null']:.3f})",
        ),
        (
            "clicks per hour",
            _q([s.clicks_per_h for s in measured]),
            _real(summary, "noise", "clicks_per_h"),
        ),
    ]
    return pd.DataFrame(rows, columns=["quantity", "simulated", "real"])
