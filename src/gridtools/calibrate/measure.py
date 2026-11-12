"""Measure one trial: frequencies, drift, profiles, noise, hum, chirps.

:func:`measure_trial` reads a trial's tracks (whole), a seeded sample of raw
segments and the v1 detections, and returns a :class:`TrialStats` with
everything :mod:`gridtools.calibrate.aggregate` needs. Only the two tracked
fish of a dyad (winner and loser from ``trial_summary.csv``) are measured.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from gridtools.calibrate.chirps import (
    CrossFishStats,
    cross_fish_stats,
    merge_duplicates,
)
from gridtools.calibrate.frequency import (
    DeltaFStats,
    DriftFit,
    delta_f_stats,
    fit_drift_variogram,
    resample_track,
    running_quantile,
    semivariogram,
)
from gridtools.calibrate.profiles import (
    dwell_stats,
    relative_sorted_profiles,
    strongest_electrode,
)
from gridtools.calibrate.spectra import (
    HANN_ENBW_BINS,
    count_clicks,
    floor_spectrum,
    hum_lines,
    line_amplitude,
    noise_floor,
    power_law_fit,
    psd,
    segment_starts,
)
from gridtools.calibrate.trials import (
    TrialMeta,
    open_raw,
    read_temperatures,
    sign_v_nfft,
)
from gridtools.io import (
    Track,
    Tracks,
    load_tracks,
    read_fishgrid_config,
    read_v1_detections,
)

__all__ = [
    "ChirpTrain",
    "FishStats",
    "MeasureConfig",
    "NoiseStats",
    "PairStats",
    "TrialStats",
    "measure_trial",
]

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class MeasureConfig:
    """Settings of :func:`measure_trial` (defaults used for calibration v1).

    Attributes
    ----------
    grid_s : float
        Resampling interval of frequency traces.
    baseline_window_s, baseline_quantile : float
        Running low quantile for the slow baseline (5th percentile of 5-min
        windows, as in Raab et al. 2021), robust against rises.
    despike_window_s : float
        Running median that removes chirp spikes from the 1-s trace.
    dark_phase_s : float
        Trials start 3 h before lights-on; the first ``dark_phase_s`` are
        the dark (active) phase.
    light_phase_start_s : float
        Start of the part used for the individual drift (rises are rarer
        in the light phase).
    drift_max_lag_s : float
        Longest lag of the individual-drift fit.
    shared_lags_s : tuple of float
        Lag range of the shared-drift fit.
    q10_reference : float
        Q10 for normalising EODf to 25 degC (1.37, Raab et al. 2021).
    n_profile_samples : int
        Frames per fish kept for the field fit.
    min_delta_f_profiles_hz : float
        Frames with a smaller frequency difference to the other fish are
        not used for profiles (their ``sign_v`` bins overlap).
    hysteresis_db : tuple of float
        Hysteresis settings of the strongest-electrode switch counts.
    n_segments, segment_s : int, float
        Raw segments sampled per trial for spectra.
    noise_nfft : int
        FFT length for noise and hum spectra.
    noise_band_hz : tuple of float
        Band of the reference noise floor (the fish band).
    floor_band_edges_hz : tuple of float
        Bands of the floor spectrum.
    slope_band_hz : tuple of float
        Band for the power-law fit of the floor.
    n_hum_harmonics : int
    chirp_merge_s : float
        Same-fish detections closer than this are merged (v1 duplicates).
    coincidence_window_s : float
    n_shuffles : int
    null_shift_s : tuple of float
        Shift range of the cross-fish null (see
        :func:`gridtools.calibrate.chirps.cross_fish_stats`).
    clip_level_v : float
        Clipping level of the ADC in raw units (volts).
    """

    grid_s: float = 1.0
    baseline_window_s: float = 301.0
    baseline_quantile: float = 0.05
    despike_window_s: float = 5.0
    dark_phase_s: float = 3 * 3600.0
    light_phase_start_s: float = 3.25 * 3600.0
    drift_max_lag_s: float = 1800.0
    shared_lags_s: tuple[float, float] = (60.0, 4 * 3600.0)
    q10_reference: float = 1.37
    n_profile_samples: int = 4000
    min_delta_f_profiles_hz: float = 5.0
    hysteresis_db: tuple[float, ...] = (0.0, 3.0)
    n_segments: int = 20
    segment_s: float = 10.0
    noise_nfft: int = 8192
    noise_band_hz: tuple[float, float] = (400.0, 1500.0)
    floor_band_edges_hz: tuple[float, ...] = tuple(
        float(v) for v in np.geomspace(20.0, 9500.0, 25)
    )
    slope_band_hz: tuple[float, float] = (100.0, 5000.0)
    n_hum_harmonics: int = 40
    chirp_merge_s: float = 0.05
    coincidence_window_s: float = 0.25
    n_shuffles: int = 200
    null_shift_s: tuple[float, float] = (5.0, 30.0)
    clip_level_v: float = 0.02048


@dataclass(frozen=True, eq=False)
class FishStats:
    """Per-fish measurements.

    Attributes
    ----------
    ident : int
    role : str
        ``"winner"`` or ``"loser"``.
    sex : str
    coverage_h : float
        Hours with a tracked frequency (1-s grid).
    baseline_hz, baseline_dark_hz, baseline_light_hz : float
        Median of the slow baseline over the trial, dark and light phase.
    baseline_25c_hz : float
        Baseline normalised to 25 degC with ``q10_reference`` (NaN without
        temperatures).
    slope_hz_per_h : float
        Linear trend of the slow baseline.
    q10 : float
        From regressing log baseline on temperature (NaN if the temperature
        range is below 0.5 degC).
    drift_lags_s, drift_gamma_hz2 : numpy.ndarray
        Semivariogram of the individual drift.
    drift_gaussian, drift_exponential : DriftFit
    change_10min_hz : numpy.ndarray
        Absolute change of the slow baseline over 10 min (one per 10 min).
    profiles : numpy.ndarray, shape (n, E_live)
        Sorted relative ``sign_v`` profiles of sampled frames, electrodes
        without fish signal excluded.
    peak_snr_db : numpy.ndarray, shape (n,)
        ``sign_v`` at the strongest electrode minus that electrode's noise
        density (both PSD, dB) for the same frames.
    cn0_db_hz : numpy.ndarray, shape (n,)
        Carrier-to-noise density at the strongest electrode.
    peak_amplitude_v : numpy.ndarray, shape (n,)
        Fundamental amplitude at the strongest electrode from ``sign_v``
        (bin-centred Hann conversion; up to 1.4 dB low between bins).
    raw_peak_amplitude_v : numpy.ndarray
        Fundamental amplitude at the strongest electrode from the raw
        segments (lobe sum, scalloping-free).
    dwell : dict of float to dict
        Strongest-electrode switch statistics per hysteresis (dB), each with
        ``switches_per_min``, ``still_fraction``, ``dwell_times_s``,
        ``switches_per_min_dark``, ``switches_per_min_light``.
    """

    ident: int
    role: str
    sex: str
    coverage_h: float
    baseline_hz: float
    baseline_dark_hz: float
    baseline_light_hz: float
    baseline_25c_hz: float
    slope_hz_per_h: float
    q10: float
    drift_lags_s: FloatArray
    drift_gamma_hz2: FloatArray
    drift_gaussian: DriftFit
    drift_exponential: DriftFit
    change_10min_hz: FloatArray
    profiles: FloatArray
    peak_snr_db: FloatArray
    cn0_db_hz: FloatArray
    peak_amplitude_v: FloatArray
    raw_peak_amplitude_v: FloatArray
    dwell: dict[float, dict[str, object]]


@dataclass(frozen=True, eq=False)
class PairStats:
    """Measurements of the dyad.

    Attributes
    ----------
    delta_f : DeltaFStats
        From the despiked 1-s traces (instantaneous, rises included).
    delta_f_baseline : DeltaFStats
        From the slow baselines.
    abs_delta_f_hz : numpy.ndarray
        Despiked :math:`|\\Delta F|` on a 10-s grid (for pooled histograms).
    shared_lags_s, shared_gamma : numpy.ndarray
        Semivariogram of the shared relative drift.
    shared_gaussian : DriftFit
        Fit (``sd`` is relative, dimensionless).
    shared_slope_per_h : float
        Linear trend of the shared relative drift (fraction per hour).
    corr_level : float
        Correlation of the two fish's relative baseline changes.
    corr_increments_30min : float
        Correlation of their 30-min increments.
    temperature_range_c : float
    temperature_slope_c_per_h : float
    shared_per_degc : float
        Regression slope of the shared relative drift on temperature.
    shared_temp_corr : float
    """

    delta_f: DeltaFStats
    delta_f_baseline: DeltaFStats
    abs_delta_f_hz: FloatArray
    shared_lags_s: FloatArray
    shared_gamma: FloatArray
    shared_gaussian: DriftFit
    shared_slope_per_h: float
    corr_level: float
    corr_increments_30min: float
    temperature_range_c: float
    temperature_slope_c_per_h: float
    shared_per_degc: float
    shared_temp_corr: float


@dataclass(frozen=True, eq=False)
class NoiseStats:
    """Noise, hum, clipping and clicks from raw segments.

    Attributes
    ----------
    segment_starts_s : numpy.ndarray
    floor_v2_hz : numpy.ndarray, shape (S, E)
        Noise density in the fish band per segment and electrode (V**2/Hz).
    floor_band_hz : numpy.ndarray, shape (B,)
    floor_spectrum : numpy.ndarray, shape (B, E)
        Median over segments of the band-wise floor.
    alpha : numpy.ndarray, shape (E,)
        Power-law exponent of the floor per electrode.
    hum_frequencies_hz : numpy.ndarray, shape (H,)
    hum_amplitude_v : numpy.ndarray, shape (S, H, E)
    hum_detected : numpy.ndarray of bool, shape (S, H, E)
    fish_power_over_floor_db : numpy.ndarray, shape (E,)
        Maximum over segments and fish of a fish's fundamental power over
        the floor, per electrode (low for electrodes without signal).
    dead : numpy.ndarray of bool, shape (E,)
    noisy : numpy.ndarray of bool, shape (E,)
    max_abs_v : float
    clipped_fraction : float
    click_counts : numpy.ndarray, shape (S, E)
    click_events : numpy.ndarray, shape (S,)
        Clicks on any electrode (merged across electrodes) per segment.
    click_times_s : numpy.ndarray
        Onset times of those clicks in the recording.
    segment_s : float
    sign_v_check_db : numpy.ndarray, shape (k, E_sign)
        ``sign_v`` minus its recomputation from raw for k frames (columns
        mapped to raw channels in order).
    sign_v_best_channel : numpy.ndarray of int, shape (E_sign,)
        Raw channel whose recomputed power matches each ``sign_v`` column
        best (median absolute difference).
    """

    segment_starts_s: FloatArray
    floor_v2_hz: FloatArray
    floor_band_hz: FloatArray
    floor_spectrum: FloatArray
    alpha: FloatArray
    hum_frequencies_hz: FloatArray
    hum_amplitude_v: FloatArray
    hum_detected: npt.NDArray[np.bool_]
    fish_power_over_floor_db: FloatArray
    dead: npt.NDArray[np.bool_]
    noisy: npt.NDArray[np.bool_]
    max_abs_v: float
    clipped_fraction: float
    click_counts: npt.NDArray[np.int64]
    click_events: npt.NDArray[np.int64]
    click_times_s: FloatArray
    segment_s: float
    sign_v_check_db: FloatArray
    sign_v_best_channel: npt.NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class ChirpTrain:
    """v1-detected chirps of one fish.

    Attributes
    ----------
    ident : int
    role : str
    times_s : numpy.ndarray
        Merged chirp times.
    n_raw : int
        Detections before merging duplicates.
    n_merged : int
    """

    ident: int
    role: str
    times_s: FloatArray
    n_raw: int
    n_merged: int


@dataclass(frozen=True, eq=False)
class TrialStats:
    """Everything measured on one trial.

    Attributes
    ----------
    name : str
    layout : str
        ``"<electrodes>-<electrodes with fish signal>"``, e.g. ``"9-8"``
        (2019: 3x3 grid, channel 9 without signal) or ``"15-15"``.
    n_rows, n_columns : int
    n_channels : int
    n_sign_columns : int
    sign_v_nfft : int
    frame_dt_s : float
    duration_s : float
    fish : list of FishStats
    pair : PairStats
    noise : NoiseStats
    chirps : list of ChirpTrain
    cross : dict of str to CrossFishStats
        ``"winner->loser"`` and ``"loser->winner"``.
    unassigned_chirps : int
    """

    name: str
    layout: str
    n_rows: int
    n_columns: int
    n_channels: int
    n_sign_columns: int
    sign_v_nfft: int
    frame_dt_s: float
    duration_s: float
    fish: list[FishStats]
    pair: PairStats
    noise: NoiseStats
    chirps: list[ChirpTrain]
    cross: dict[str, CrossFishStats]
    unassigned_chirps: int
    notes: list[str] = field(default_factory=list)


def _log_lags(max_lag: int, n: int = 40) -> npt.NDArray[np.int64]:
    return np.unique(np.round(np.geomspace(1, max_lag, n)).astype(np.int64))


def _corr(x: FloatArray, y: FloatArray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _linear_slope(t: FloatArray, y: FloatArray) -> float:
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(t[ok], y[ok], 1)[0])


def _phase_median(t: FloatArray, y: FloatArray, lo: float, hi: float) -> float:
    sel = (t >= lo) & (t < hi) & np.isfinite(y)
    return float(np.median(y[sel])) if sel.any() else float("nan")


def _layout_name(n_electrodes: int, n_live: int) -> str:
    return f"{n_electrodes}-{n_live}"


def _sign_v_check(
    tracks: Tracks,
    recording_data: npt.NDArray[np.generic],
    samplerate_hz: float,
    nfft: int,
    rng: np.random.Generator,
    n_frames: int = 6,
) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    """Recompute ``sign_v`` for random frames from the raw data."""
    assert tracks.powers_db is not None
    assigned = np.flatnonzero(np.isfinite(tracks.idents))
    rows = rng.choice(assigned, min(n_frames, assigned.size), replace=False)
    n_sign = tracks.powers_db.shape[1]
    n_channels = recording_data.shape[1]
    diffs = np.full((rows.size, n_sign), np.nan)
    all_diffs = np.full((rows.size, n_sign, n_channels), np.nan)
    window = np.hanning(nfft + 1)[:nfft]  # periodic Hann
    for i, row in enumerate(rows):
        centre = round(float(tracks.times_s[tracks.time_indices[row]]) * samplerate_hz)
        start = centre - nfft // 2
        if start < 0 or start + nfft > recording_data.shape[0]:
            continue
        x = np.asarray(recording_data[start : start + nfft], dtype=np.float64)
        x = x - x.mean(axis=0)
        spec = np.fft.rfft(x * window[:, None], axis=0)
        power = 2.0 * np.abs(spec) ** 2 / (samplerate_hz * np.sum(window**2))
        k = round(float(tracks.freqs_hz[row]) * nfft / samplerate_hz)
        db = 10.0 * np.log10(power[k])
        sign = tracks.powers_db[row]
        diffs[i] = sign - db[:n_sign]
        all_diffs[i] = sign[:, None] - db[None, :]
    best = np.argmin(np.nanmedian(np.abs(all_diffs), axis=0), axis=1).astype(np.int64)
    return diffs, best


def _fish_series(
    track: Track, duration_s: float, config: MeasureConfig
) -> tuple[FloatArray, FloatArray, FloatArray]:
    t, f = resample_track(track, 0.0, duration_s, config.grid_s)
    window = max(1, round(config.baseline_window_s / config.grid_s))
    baseline = running_quantile(f, window, config.baseline_quantile)
    despike = max(1, round(config.despike_window_s / config.grid_s))
    fast = running_quantile(f, despike, 0.5)
    return t, baseline, fast


def measure_trial(
    trial_dir: str | Path,
    meta: TrialMeta,
    rng: np.random.Generator,
    config: MeasureConfig | None = None,
) -> TrialStats:
    """Measure one trial.

    Parameters
    ----------
    trial_dir : str or Path
        Fishgrid directory with tracks, raw data and (optionally)
        ``temperatures.csv`` and ``chirpdetector_bboxes.csv``.
    meta : TrialMeta
        Winner and loser identities.
    rng : numpy.random.Generator
        Drives every random choice (segments, frame samples, shuffles).
    config : MeasureConfig, optional

    Returns
    -------
    TrialStats
    """
    config = MeasureConfig() if config is None else config
    trial_dir = Path(trial_dir)
    notes: list[str] = []
    cfg = read_fishgrid_config(trial_dir)
    tracks = load_tracks(trial_dir)
    if tracks.powers_db is None:
        msg = f"{trial_dir}: no sign_v.npy"
        raise ValueError(msg)
    recording = open_raw(trial_dir)
    fs = recording.samplerate_hz
    nfft = sign_v_nfft(tracks.times_s, fs)
    frame_dt = float(np.median(np.diff(tracks.times_s)))
    duration = float(min(tracks.times_s[-1], recording.duration_s))
    n_sign = tracks.powers_db.shape[1]
    idents = (meta.winner_id, meta.loser_id)
    temps = read_temperatures(trial_dir)

    # ---------------------------------------------------------- noise ----
    seg_len = round(config.segment_s * fs)
    starts = segment_starts(
        rng, recording.n_samples, seg_len, config.n_segments, margin_samples=seg_len
    )
    n_ch = recording.n_channels
    floors = np.empty((starts.size, n_ch))
    band_spectra = []
    hum_amp = []
    hum_det = []
    clicks = np.zeros((starts.size, n_ch), dtype=np.int64)
    click_events = np.zeros(starts.size, dtype=np.int64)
    click_times: list[FloatArray] = []
    max_abs = 0.0
    n_clipped = 0
    fish_over_floor = np.full(n_ch, -np.inf)
    raw_amps: dict[int, list[float]] = {i: [] for i in idents}
    harmonics_hz = 50.0 * np.arange(1, config.n_hum_harmonics + 1, dtype=np.float64)
    centres = np.zeros(0)
    for s, start in enumerate(starts):
        x = recording.read(int(start), int(start) + seg_len)
        max_abs = max(max_abs, float(np.max(np.abs(x))))
        n_clipped += int(np.count_nonzero(np.abs(x) >= config.clip_level_v * 0.999))
        freqs, power = psd(x, fs, config.noise_nfft)
        floors[s] = noise_floor(freqs, power, config.noise_band_hz)
        centres, spectrum = floor_spectrum(freqs, power, config.floor_band_edges_hz)
        band_spectra.append(spectrum)
        hum = hum_lines(freqs, power, n_harmonics=config.n_hum_harmonics)
        hum_amp.append(hum.amplitudes)
        hum_det.append(hum.detected)
        counts, onsets = count_clicks(x, fs)
        clicks[s] = counts
        click_events[s] = onsets.size
        click_times.append(start / fs + onsets)
        t_mid = (start + seg_len / 2) / fs
        f_now = {i: float(tracks.track(i).frequency_at(t_mid)[0]) for i in idents}
        for i in idents:
            fi = f_now[i]
            others = [f_now[j] for j in idents if j != i]
            if not math.isfinite(fi) or any(
                math.isfinite(o) and abs(o - fi) < 15.0 for o in others
            ):
                continue
            if np.min(np.abs(harmonics_hz - fi)) < 3.0:
                continue
            amp = line_amplitude(freqs, power, fi)
            raw_amps[i].append(float(amp.max()))
            df = freqs[1] - freqs[0]
            fish_power = (amp**2 / 2.0) / (HANN_ENBW_BINS * df)  # peak PSD
            ratio_db = 10.0 * np.log10(np.maximum(fish_power, 1e-300) / floors[s])
            fish_over_floor = np.maximum(fish_over_floor, ratio_db)
    floor_median = np.median(floors, axis=0)
    floor_spec = np.median(np.stack(band_spectra), axis=0)
    slope_sel = (centres >= config.slope_band_hz[0]) & (
        centres <= config.slope_band_hz[1]
    )
    alpha = np.array(
        [
            power_law_fit(centres[slope_sel], floor_spec[slope_sel, e])[0]
            for e in range(n_ch)
        ]
    )
    floor_db = 10.0 * np.log10(floor_median)
    dead = fish_over_floor < 10.0
    noisy = (
        (floor_db > np.median(floor_db[~dead]) + 6.0) & ~dead
        if (~dead).any()
        else np.zeros(n_ch, bool)
    )
    sign_check, best_channel = _sign_v_check(tracks, recording.data, fs, nfft, rng)
    if dead.any():
        notes.append(
            f"electrodes without fish signal (raw channel index): {np.flatnonzero(dead).tolist()}"
        )
    noise = NoiseStats(
        segment_starts_s=starts / fs,
        floor_v2_hz=floors,
        floor_band_hz=centres,
        floor_spectrum=floor_spec,
        alpha=alpha,
        hum_frequencies_hz=harmonics_hz,
        hum_amplitude_v=np.stack(hum_amp),
        hum_detected=np.stack(hum_det),
        fish_power_over_floor_db=fish_over_floor,
        dead=dead,
        noisy=noisy,
        max_abs_v=max_abs,
        clipped_fraction=n_clipped / (starts.size * seg_len * n_ch),
        click_counts=clicks,
        click_events=click_events,
        click_times_s=np.concatenate(click_times),
        segment_s=config.segment_s,
        sign_v_check_db=sign_check,
        sign_v_best_channel=best_channel,
    )
    # sign_v column c holds raw channel c (verified by _sign_v_check).
    floor_sign_db = floor_db[:n_sign]
    live = ~dead[:n_sign]
    layout = _layout_name(cfg.n_rows * cfg.n_columns, int(live.sum()))

    # ------------------------------------------------------ frequencies ----
    series = {i: _fish_series(tracks.track(i), duration, config) for i in idents}
    t = series[idents[0]][0]
    rel = {i: series[i][1] / np.nanmedian(series[i][1]) - 1.0 for i in idents}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        shared = np.nanmean(np.vstack([rel[i] for i in idents]), axis=0)
    temp_on_grid = (
        np.interp(t, temps[0], temps[1], left=np.nan, right=np.nan)
        if temps is not None
        else np.full(t.size, np.nan)
    )

    lags_shared = _log_lags(int(config.shared_lags_s[1] / config.grid_s))
    gamma_shared, _ = semivariogram(shared, lags_shared)
    lag_s = lags_shared.astype(np.float64) * config.grid_s
    sel = lag_s >= config.shared_lags_s[0]
    shared_fit = fit_drift_variogram(lag_s[sel], gamma_shared[sel])
    step = round(1800.0 / config.grid_s)
    inc = {i: rel[i][step:] - rel[i][:-step] for i in idents}
    temp_range = (
        float(np.nanmax(temps[1]) - np.nanmin(temps[1]))
        if temps is not None
        else float("nan")
    )
    ok_t = np.isfinite(temp_on_grid) & np.isfinite(shared)
    if temps is not None and ok_t.sum() > 10 and temp_range >= 0.5:
        per_degc = float(np.polyfit(temp_on_grid[ok_t], shared[ok_t], 1)[0])
    else:
        per_degc = float("nan")

    fast_df = series[idents[0]][2] - series[idents[1]][2]
    base_df = series[idents[0]][1] - series[idents[1]][1]
    pair = PairStats(
        delta_f=delta_f_stats(fast_df, config.grid_s),
        delta_f_baseline=delta_f_stats(base_df, config.grid_s),
        abs_delta_f_hz=np.abs(fast_df[:: max(1, round(10.0 / config.grid_s))]),
        shared_lags_s=lag_s,
        shared_gamma=gamma_shared,
        shared_gaussian=shared_fit,
        shared_slope_per_h=_linear_slope(t / 3600.0, shared),
        corr_level=_corr(rel[idents[0]], rel[idents[1]]),
        corr_increments_30min=_corr(inc[idents[0]], inc[idents[1]]),
        temperature_range_c=temp_range,
        temperature_slope_c_per_h=_linear_slope(temps[0] / 3600.0, temps[1])
        if temps is not None
        else float("nan"),
        shared_per_degc=per_degc,
        shared_temp_corr=_corr(temp_on_grid, shared),
    )

    # ------------------------------------------------------------ fish ----
    fish_stats: list[FishStats] = []
    lags_drift = _log_lags(int(config.drift_max_lag_s / config.grid_s))
    other_freq = {
        i: tracks.track(j).frequency_at
        for i, j in ((idents[0], idents[1]), (idents[1], idents[0]))
    }
    for ident in idents:
        tt, baseline, fast = series[ident]
        track = tracks.track(ident)
        coverage = float(np.isfinite(fast).sum() * config.grid_s / 3600.0)
        temp_ok = np.isfinite(temp_on_grid) & np.isfinite(baseline)
        if temp_ok.any():
            b25 = float(
                np.median(
                    baseline[temp_ok]
                    * config.q10_reference ** ((25.0 - temp_on_grid[temp_ok]) / 10.0)
                )
            )
        else:
            b25 = float("nan")
        if temps is not None and temp_ok.sum() > 10 and temp_range >= 0.5:
            beta = np.polyfit(temp_on_grid[temp_ok], np.log(baseline[temp_ok]), 1)[0]
            q10 = float(math.exp(10.0 * beta))
        else:
            q10 = float("nan")
        individual = fast - np.nanmedian(baseline) * (1.0 + shared)
        individual = np.where(tt >= config.light_phase_start_s, individual, np.nan)
        gamma, _ = semivariogram(individual, lags_drift, trim=0.01)
        lags_s = lags_drift.astype(np.float64) * config.grid_s
        ten = round(600.0 / config.grid_s)
        b10 = baseline[::ten]
        change = np.abs(np.diff(b10))
        change = change[np.isfinite(change)]

        # Profiles and SNR from sign_v.
        powers = track.powers_db
        assert powers is not None
        f_other = other_freq[ident](track.times_s)
        usable = ~(
            np.isfinite(f_other)
            & (np.abs(f_other - track.freqs_hz) < config.min_delta_f_profiles_hz)
        )
        usable &= np.all(np.isfinite(powers), axis=1)
        rows = np.flatnonzero(usable)
        pick = np.sort(
            rng.choice(rows, min(config.n_profile_samples, rows.size), replace=False)
        )
        sampled = powers[pick][:, live]
        e_max = np.flatnonzero(live)[np.argmax(sampled, axis=1)]
        peak_db = np.max(sampled, axis=1)
        snr = peak_db - floor_sign_db[e_max]
        enbw = HANN_ENBW_BINS * fs / nfft
        cn0 = snr + 10.0 * math.log10(enbw)
        peak_amp = np.sqrt(2.0 * 10.0 ** (peak_db / 10.0) * enbw)

        dwell: dict[float, dict[str, object]] = {}
        valid_rows = np.flatnonzero(np.all(np.isfinite(powers), axis=1))
        times_valid = track.times_s[valid_rows]
        for hyst in config.hysteresis_db:
            seq = strongest_electrode(powers[valid_rows][:, live], hyst)
            overall = dwell_stats(times_valid, seq, max_gap_s=2.0 + 2 * frame_dt)
            dark = times_valid < config.dark_phase_s
            dark_stats = dwell_stats(
                times_valid[dark], seq[dark], max_gap_s=2.0 + 2 * frame_dt
            )
            light_stats = dwell_stats(
                times_valid[~dark], seq[~dark], max_gap_s=2.0 + 2 * frame_dt
            )
            dwell[hyst] = {
                "switches_per_min": overall.switches_per_min,
                "still_fraction": overall.still_fraction,
                "dwell_times_s": overall.dwell_times_s,
                "switches_per_min_dark": dark_stats.switches_per_min,
                "switches_per_min_light": light_stats.switches_per_min,
                "still_fraction_dark": dark_stats.still_fraction,
                "still_fraction_light": light_stats.still_fraction,
            }

        fish_stats.append(
            FishStats(
                ident=ident,
                role=meta.role(ident),
                sex=meta.sex(ident),
                coverage_h=coverage,
                baseline_hz=float(np.nanmedian(baseline)),
                baseline_dark_hz=_phase_median(tt, baseline, 0.0, config.dark_phase_s),
                baseline_light_hz=_phase_median(
                    tt, baseline, config.dark_phase_s, np.inf
                ),
                baseline_25c_hz=b25,
                slope_hz_per_h=_linear_slope(tt / 3600.0, baseline),
                q10=q10,
                drift_lags_s=lags_s,
                drift_gamma_hz2=gamma,
                drift_gaussian=fit_drift_variogram(lags_s, gamma, "gaussian"),
                drift_exponential=fit_drift_variogram(lags_s, gamma, "exponential"),
                change_10min_hz=change,
                profiles=relative_sorted_profiles(sampled),
                peak_snr_db=snr,
                cn0_db_hz=cn0,
                peak_amplitude_v=peak_amp,
                raw_peak_amplitude_v=np.asarray(raw_amps[ident]),
                dwell=dwell,
            )
        )

    # ---------------------------------------------------------- chirps ----
    chirps: list[ChirpTrain] = []
    cross: dict[str, CrossFishStats] = {}
    unassigned = 0
    bbox = trial_dir / "chirpdetector_bboxes.csv"
    if bbox.exists():
        det = read_v1_detections(bbox)
        centre = 0.5 * (det["t_start_s"] + det["t_end_s"]).to_numpy(dtype=np.float64)
        track_id = det["track_id"].to_numpy(dtype=np.float64, na_value=np.nan)
        unassigned = int(np.count_nonzero(~np.isin(track_id, idents)))
        for ident in idents:
            times = centre[track_id == ident]
            merged, n_merged = merge_duplicates(times, config.chirp_merge_s)
            chirps.append(
                ChirpTrain(ident, meta.role(ident), merged, times.size, n_merged)
            )
        by_role = {c.role: c.times_s for c in chirps}
        for a, b in (("winner", "loser"), ("loser", "winner")):
            cross[f"{a}->{b}"] = cross_fish_stats(
                by_role[a],
                by_role[b],
                0.0,
                duration,
                rng,
                window_s=config.coincidence_window_s,
                n_shuffles=config.n_shuffles,
                min_shift_s=config.null_shift_s[0],
                max_shift_s=config.null_shift_s[1],
            )
    else:
        notes.append("no chirpdetector_bboxes.csv")

    return TrialStats(
        name=trial_dir.name,
        layout=layout,
        n_rows=cfg.n_rows,
        n_columns=cfg.n_columns,
        n_channels=recording.n_channels,
        n_sign_columns=n_sign,
        sign_v_nfft=nfft,
        frame_dt_s=frame_dt,
        duration_s=duration,
        fish=fish_stats,
        pair=pair,
        noise=noise,
        chirps=chirps,
        cross=cross,
        unassigned_chirps=unassigned,
        notes=notes,
    )
