"""Build the background library from real recordings.

:func:`build_background_library` processes every source recording
independently (one worker process per recording, each with its own random
stream derived from the library seed and the recording's name, so the result
does not depend on the order, the subset or the number of workers):

1. candidates on a grid inside the recording, screened against v1 chirp
   detections and hand labels, and one segment chosen per time stratum
   (:mod:`gridtools.background.select`);
2. fish-free candidates: windows without any assigned track detection,
   searched over the whole recording (fish are often put in after the
   recording started) and verified on the audio: no line more than
   ``tonal_peak_db`` above the floor in 200-2000 Hz outside mains
   harmonics;
3. every chosen segment is read once, measured (levels, real fish, hum,
   clicks, tonal peaks; :mod:`gridtools.background.levels`) and written
   with its configuration and clipped tracks;
4. channels whose fish signal never exceeds the floor by
   ``dead_channel_snr_db`` in any segment of the recording are marked dead.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import logging
import math
import platform
import shutil
import subprocess
import time
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import wavfile

from gridtools.background.field import FieldFolder, FieldScanConfig, scan_folder
from gridtools.background.levels import (
    LEVEL_BAND_HZ,
    MAINS_HZ,
    REFERENCE_BANDWIDTH_HZ,
    count_clicks,
    level_from_psd,
    line_amplitude,
    tonal_peaks,
    tracked_line_power,
    welch_psd,
)
from gridtools.background.library import (
    MANIFEST_COLUMNS,
    clip_tracks,
    read_manifest,
    write_manifest,
)
from gridtools.background.review import merge_review, read_review
from gridtools.background.select import (
    SelectionConfig,
    candidate_starts,
    choose_stratified,
    count_overlaps,
    expected_missed_chirps,
    frame_coverage,
    local_rate_per_s,
)
from gridtools.background.sources import SourceRecording, open_source_recording
from gridtools.io import (
    Recording,
    Tracks,
    load_tracks,
    read_fishgrid_config,
    read_labels,
    read_v1_detections,
    save_tracks,
)

__all__ = ["BackgroundConfig", "build_background_library", "segment_id"]

log = logging.getLogger(__name__)

#: Unit of fishgrid raw samples (gridtools.calibrate: quantisation step and
#: input range show raw = ADC volts; the amplifier gain is 100).
FISHGRID_UNITS = "V at the ADC (electrode potential x 100)"


@dataclass(frozen=True)
class BackgroundConfig:
    """Parameters of :func:`build_background_library`.

    Attributes
    ----------
    selection : SelectionConfig
        Segment length, chirp margin, strata (see
        :mod:`gridtools.background.select`).
    seed : int
        Library seed.
    exclude_labelled : bool
        Skip recordings with hand labels entirely (default True): they hold
        the locked real test set, and splits are by recording. If False,
        only the labelled spans (plus the margin) are excluded.
    fish_free_per_recording : int
        At most this many fish-free segments per recording (default 2).
    tonal_peak_db : float
        Lines this far above the running median of the electrode-summed PSD
        in 200-2000 Hz (outside mains harmonics) count as tonal peaks; a
        fish-free segment has none, and peaks no tracked fish explains are
        reported as untracked (default 10 dB).
    untracked_tolerance_hz : float
        A tonal peak is explained by a tracked fish if it lies within this
        distance of the fish's frequency range in the segment, or of a
        multiple of it (default 5 Hz).
    dead_channel_snr_db : float
        A channel is dead if no fish exceeds this SNR on it in any segment of
        the recording (default 10 dB).
    curated_max_ids : int
        Recordings with more track identities are flagged as fragmented
        tracks (default 6).
    stationary_fraction : float
        Tonal peaks that no tracked fish explains, at the same frequency
        (2.5 Hz) in at least this fraction of a recording's measured segments
        (and at least two), are stationary interference, not fish (default
        0.5). They are ignored by the
        fish-free check and the untracked-fish check.
    unscreened_weight : float
        Default sampling weight of segments not screened for chirps (default
        0.25; screened and fish-free segments have 1).
    level_band_hz : (float, float)
        Band of the background level (default 500-1500 Hz).
    reference_bandwidth_hz : float
        Bandwidth of levels and SNRs (default 7.32 Hz).
    field : FieldScanConfig
        Scan of field deployments for fish-free segments
        (:mod:`gridtools.background.field`).
    workers : int
        Worker processes (default 4; at most 6 on the shared machine).
    """

    selection: SelectionConfig = dataclasses.field(default_factory=SelectionConfig)
    seed: int = 0
    exclude_labelled: bool = True
    fish_free_per_recording: int = 2
    tonal_peak_db: float = 10.0
    untracked_tolerance_hz: float = 5.0
    dead_channel_snr_db: float = 10.0
    curated_max_ids: int = 6
    stationary_fraction: float = 0.5
    unscreened_weight: float = 0.25
    level_band_hz: tuple[float, float] = LEVEL_BAND_HZ
    reference_bandwidth_hz: float = REFERENCE_BANDWIDTH_HZ
    field: FieldScanConfig = dataclasses.field(default_factory=FieldScanConfig)
    workers: int = 4

    def __post_init__(self) -> None:
        if self.workers < 1:
            msg = f"workers must be >= 1, got {self.workers}"
            raise ValueError(msg)
        if self.fish_free_per_recording < 0:
            msg = "fish_free_per_recording must be >= 0"
            raise ValueError(msg)
        if not 0 <= self.unscreened_weight <= 1:
            msg = "unscreened_weight must lie in [0, 1]"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for ``library.json``."""
        return dataclasses.asdict(self)


def segment_id(source: SourceRecording, start_s: float) -> str:
    """Identifier of a segment: ``<dataset>_<recording>_<start in s>``."""
    return f"{source.dataset}_{source.name}_{round(start_s):06d}"


def _rng(seed: int, source: SourceRecording) -> np.random.Generator:
    key = zlib.crc32(source.key.encode("utf-8"))
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(key,)))


@dataclass(frozen=True)
class _Plan:
    starts_s: list[float]
    classes: list[str]
    scores: list[float]
    stats: dict[str, Any]


def _track_frames(tracks: Tracks) -> dict[int, np.ndarray]:
    """Per identity: whether it has a detection in each frame."""
    out: dict[int, np.ndarray] = {}
    assigned = np.isfinite(tracks.idents)
    for ident in tracks.ids:
        mask = np.zeros(len(tracks.times_s), dtype=bool)
        mask[tracks.time_indices[assigned & (tracks.idents == ident)]] = True
        out[int(ident)] = mask
    return out


def _plan(
    source: SourceRecording,
    duration_s: float,
    tracks: Tracks | None,
    detections: pd.DataFrame | None,
    labels: pd.DataFrame | None,
    config: BackgroundConfig,
    rng: np.random.Generator,
) -> _Plan:
    sel = config.selection
    length = sel.segment_s
    stats: dict[str, Any] = {}

    def blocked(starts: np.ndarray) -> np.ndarray:
        bad = np.zeros(starts.size, dtype=bool)
        if detections is not None and len(detections):
            bad |= (
                count_overlaps(
                    starts,
                    length,
                    detections["t_start_s"],
                    detections["t_end_s"],
                    sel.margin_s,
                )
                > 0
            )
        if labels is not None and len(labels):
            bad |= (
                count_overlaps(
                    starts, length, labels["t_start_s"], labels["t_end_s"], sel.margin_s
                )
                > 0
            )
        return bad

    frames = _track_frames(tracks) if tracks is not None else {}
    # fish segments
    starts = candidate_starts(duration_s, length, sel.step_s, sel.edge_s)
    valid = ~blocked(starts)
    score = np.zeros(starts.size)
    if detections is not None:
        centres = starts + length / 2
        rate = local_rate_per_s(
            centres,
            0.5 * (detections["t_start_s"] + detections["t_end_s"]),
            sel.rate_window_s,
        )
        score += expected_missed_chirps(rate, length, sel.v1_recall)
    if tracks is not None and frames:
        coverages = np.vstack(
            [
                frame_coverage(starts, length, tracks.times_s, mask)
                for mask in frames.values()
            ]
        )
        present = np.nan_to_num(coverages, nan=0.0) > 0
        worst = np.where(present, np.nan_to_num(coverages, nan=0.0), np.inf).min(axis=0)
        # windows without any tracked fish are fish-free candidates instead
        valid &= present.any(axis=0)
        score += np.where(np.isfinite(worst), 1.0 - worst, 0.0)
    chosen = choose_stratified(
        rng,
        starts,
        valid,
        score,
        segment_s=length,
        n_strata=sel.segments_per_recording,
        span_s=(sel.edge_s, max(sel.edge_s, duration_s - sel.edge_s - length)),
    )
    stats.update(
        n_candidates=int(starts.size),
        n_valid=int(valid.sum()),
        n_fish_segments=int(chosen.size),
    )
    plan_starts = [float(s) for s in starts[chosen]]
    plan_classes = ["fish"] * len(plan_starts)
    plan_scores = [float(s) for s in score[chosen]]

    # fish-free candidates: no assigned detection at all (whole recording)
    n_free = 0
    if tracks is not None and config.fish_free_per_recording > 0:
        free_starts = candidate_starts(duration_s, length, sel.step_s, sel.margin_s)
        any_fish = np.zeros(len(tracks.times_s), dtype=bool)
        for mask in frames.values():
            any_fish |= mask
        widened = frame_coverage(
            free_starts - sel.margin_s,
            length + 2 * sel.margin_s,
            tracks.times_s,
            any_fish,
        )
        empty = np.nan_to_num(widened, nan=0.0) == 0
        free_valid = empty & ~blocked(free_starts)
        if plan_starts:
            taken = np.asarray(plan_starts)
            free_valid &= np.all(
                np.abs(free_starts[:, None] - taken[None, :]) >= length, axis=1
            )
        n_free = int(free_valid.sum())
        if n_free:
            # a few more than wanted: some fail the audio check
            pick = choose_stratified(
                rng,
                free_starts,
                free_valid,
                np.zeros(free_starts.size),
                segment_s=length,
                n_strata=max(1, 3 * config.fish_free_per_recording),
                span_s=(0.0, duration_s),
            )
            plan_starts += [float(s) for s in free_starts[pick]]
            plan_classes += ["fish_free_candidate"] * len(pick)
            plan_scores += [0.0] * len(pick)
    stats["n_fish_free_candidates"] = n_free
    return _Plan(plan_starts, plan_classes, plan_scores, stats)


def _fish_measurements(
    x: np.ndarray,
    samplerate_hz: float,
    tracks: Tracks | None,
    noise_psd_per_electrode: np.ndarray,
    level: float,
) -> list[dict[str, Any]]:
    if tracks is None:
        return []
    fish = list(tracks)
    funcs = [(lambda t, tr=tr: tr.frequency_at(t, max_gap_s=2.0)) for tr in fish]
    out = []
    n_frames = len(tracks.times_s)
    for k, track in enumerate(fish):
        others = [f for j, f in enumerate(funcs) if j != k]
        _, power = tracked_line_power(
            x, samplerate_hz, funcs[k], noise_psd_per_electrode, others_hz=others
        )
        strongest = np.nanmax(np.where(np.isfinite(power), power, -np.inf), axis=1)
        strongest = np.where(np.isfinite(strongest), strongest, np.nan)
        usable = np.isfinite(strongest)
        fish_power = float(np.nanmedian(strongest)) if usable.any() else math.nan
        per_electrode = (
            np.nanmedian(power[usable], axis=0)
            if usable.any()
            else np.full(x.shape[1], np.nan)
        )
        out.append(
            {
                "ident": int(track.ident),
                "eodf_hz": float(np.median(track.freqs_hz)),
                "freq_min_hz": float(track.freqs_hz.min()),
                "freq_max_hz": float(track.freqs_hz.max()),
                "coverage": float(len(track.times_s) / max(1, n_frames)),
                "usable_frames": float(usable.mean()) if usable.size else 0.0,
                "power": fish_power,
                "snr_db": 10 * math.log10(fish_power / level)
                if fish_power > 0
                else math.nan,
                "power_per_electrode": per_electrode.tolist(),
            }
        )
    return out


def _untracked(
    peaks_hz: np.ndarray, fish: list[dict[str, Any]], tolerance_hz: float
) -> list[float]:
    out = []
    for f in peaks_hz:
        explained = False
        for info in fish:
            for k in (1, 2, 3, 4):
                lo = k * info["freq_min_hz"] - tolerance_hz
                hi = k * info["freq_max_hz"] + tolerance_hz
                if lo <= f <= hi:
                    explained = True
        if not explained:
            out.append(float(f))
    return out


def _near(freq_hz: float, lines_hz: Sequence[float], tolerance_hz: float) -> bool:
    return any(abs(freq_hz - f) <= tolerance_hz for f in lines_hz)


def _stationary_lines(
    peaks_per_segment: Sequence[np.ndarray], min_fraction: float, bin_hz: float = 2.5
) -> list[float]:
    """Frequencies with a tonal peak in at least ``min_fraction`` of the segments.

    Segments of a recording are spread over hours; a fish's EODf drifts by
    several Hz between them (water temperature: about -6.6 Hz/h), mains-
    locked or digital interference does not. Needs at least two segments
    with a peak at the same frequency (within ``bin_hz``).
    """
    n = len(peaks_per_segment)
    if n < 2:
        return []
    all_peaks = np.sort(
        np.concatenate(
            [np.asarray(p, dtype=float) for p in peaks_per_segment] or [np.empty(0)]
        )
    )
    lines: list[float] = []
    for f in all_peaks:
        if _near(float(f), lines, bin_hz):
            continue
        count = sum(
            bool(np.any(np.abs(np.asarray(p, dtype=float) - f) <= bin_hz))
            for p in peaks_per_segment
        )
        if count >= max(2, math.ceil(min_fraction * n)):
            lines.append(float(f))
    return lines


def _measure(
    x: np.ndarray,
    samplerate_hz: float,
    tracks: Tracks | None,
    config: BackgroundConfig,
) -> dict[str, Any]:
    freqs, psd = welch_psd(x, samplerate_hz)
    level = level_from_psd(
        freqs,
        psd,
        band_hz=config.level_band_hz,
        reference_bandwidth_hz=config.reference_bandwidth_hz,
    )
    hum = line_amplitude(freqs, psd, MAINS_HZ)
    harmonics = np.array(
        [line_amplitude(freqs, psd, k * MAINS_HZ) for k in range(1, 21)]
    )
    _, click_times = count_clicks(x, samplerate_hz)
    peaks_hz, peaks_db = tonal_peaks(freqs, psd, threshold_db=config.tonal_peak_db)
    fish = _fish_measurements(
        x, samplerate_hz, tracks, level.noise_psd_per_electrode, level.level
    )
    floors_db = 10 * np.log10(level.noise_psd_per_electrode)
    return {
        "level": level,
        "floors_db": floors_db,
        "hum_amplitude": float(hum.max()),
        "hum_per_electrode": hum.tolist(),
        "hum_harmonics_amplitude": float(np.sqrt((harmonics**2).sum(axis=0)).max()),
        "click_times_s": click_times.tolist(),
        "peaks_hz": peaks_hz,
        "peaks_db": peaks_db,
        "max_tonal_db": float(peaks_db.max()) if peaks_db.size else 0.0,
        "fish": fish,
        "untracked": _untracked(peaks_hz, fish, config.untracked_tolerance_hz),
        "max_abs": float(np.abs(x).max()),
    }


def _noisy_channels(
    floors_db: np.ndarray, dead: Sequence[int], excess_db: float = 6.0
) -> list[int]:
    """Live channels whose floor exceeds the live channels' median by ``excess_db``.

    The calibration's definition of a noisy electrode (6 dB).
    """
    floors = np.asarray(floors_db, dtype=float)
    live = np.setdiff1d(np.arange(floors.size), np.asarray(dead, dtype=int))
    if live.size == 0:
        return []
    ref = float(np.median(floors[live]))
    return [int(c) for c in live if floors[c] > ref + excess_db]


def _level_columns(m: dict[str, Any], duration_s: float) -> dict[str, Any]:
    """Manifest columns of the level, hum and click measurements."""
    level = m["level"]
    return {
        "noise_psd": level.noise_psd,
        "noise_psd_db": level.noise_psd_db,
        "noise_psd_spread_db": float(np.ptp(m["floors_db"])),
        "level": level.level,
        "level_db": level.level_db,
        "reference_bandwidth_hz": level.reference_bandwidth_hz,
        "level_band_low_hz": level.band_hz[0],
        "level_band_high_hz": level.band_hz[1],
        "tonal_fraction": level.tonal_fraction,
        "hum_amplitude": m["hum_amplitude"],
        "hum_harmonics_amplitude": m["hum_harmonics_amplitude"],
        "n_clicks": len(m["click_times_s"]),
        "click_rate_per_s": len(m["click_times_s"]) / duration_s,
        "max_abs": m["max_abs"],
    }


def _level_details(m: dict[str, Any]) -> dict[str, Any]:
    """``segment.json`` entries of the level, hum and click measurements."""
    return {
        "noise_psd_per_electrode": m["level"].noise_psd_per_electrode.tolist(),
        "hum_per_electrode": m["hum_per_electrode"],
        "click_times_s": m["click_times_s"],
        "tonal_peaks_hz": m["peaks_hz"].tolist(),
        "tonal_peaks_db": m["peaks_db"].tolist(),
    }


@dataclass(frozen=True)
class _FieldMeasure:
    """Picklable measurement callback for :func:`scan_folder`."""

    config: BackgroundConfig

    def __call__(self, x: np.ndarray, samplerate_hz: float) -> dict[str, Any]:
        m = _measure(x, samplerate_hz, None, self.config)
        floors = np.asarray(m["floors_db"], dtype=float)
        # no fish to tell dead from live: a floor 10 dB below the median is a
        # disconnected or shorted input
        dead = [int(c) for c in np.flatnonzero(floors < np.median(floors) - 10.0)]
        return {
            "columns": {
                "n_v1_detections_margin": None,
                "nearest_v1_detection_s": math.nan,
                "v1_rate_per_h": math.nan,
                "expected_missed_chirps": math.nan,
                "selection_score": 0.0,
                "track_power_nfft": None,
                "real_snr_max_db": math.nan,
                "real_level_diff_db": math.nan,
                "dead_channels": dead,
                "noisy_channels": _noisy_channels(floors, dead),
                **_level_columns(m, x.shape[0] / samplerate_hz),
            },
            "details": _level_details(m),
        }


def _process_field(
    folder: FieldFolder, out_dir: Path, config: BackgroundConfig, vetoed: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    t_start = time.perf_counter()
    rows, info = scan_folder(
        folder,
        out_dir,
        config.selection.segment_s,
        config.field,
        config.seed,
        _FieldMeasure(config),
        vetoed=vetoed,
    )
    for row in rows:
        row["accept"] = True
    info["runtime_s"] = round(time.perf_counter() - t_start, 1)
    log.info(
        "%s: %d fish-free segments in %.0f s", folder.key, len(rows), info["runtime_s"]
    )
    return rows, info


def _track_power_nfft(tracks: Tracks | None, samplerate_hz: float) -> int | None:
    """FFT length of the tracker's ``sign_v`` from its frame step.

    The production trackers used frames every 0.2 nfft (nfft = 32768: 2019,
    May/June 2020, 2022) or 0.15 nfft (16384: March 2020); verified against
    the raw data by the calibration and for 2022 here.
    """
    if tracks is None or len(tracks.times_s) < 2:
        return None
    step = float(np.median(np.diff(tracks.times_s))) * samplerate_hz
    for nfft, frac in ((32768, 0.2), (16384, 0.15)):
        if abs(step - frac * nfft) < 2:
            return nfft
    return None


def _write_wav(path: Path, data: np.ndarray, samplerate_hz: float) -> None:
    wavfile.write(path, round(samplerate_hz), np.ascontiguousarray(data, dtype="<f4"))


def _dispatch(
    source: SourceRecording | FieldFolder,
    out_dir: Path,
    config: BackgroundConfig,
    vetoed: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(source, FieldFolder):
        return _process_field(source, out_dir, config, vetoed)
    return _process(source, out_dir, config, vetoed)


def _vetoed_spans(
    source: SourceRecording, vetoed: set[str], length_s: float
) -> pd.DataFrame:
    prefix = f"{source.dataset}_{source.name}_"
    starts = [
        float(int(v[len(prefix) :]))
        for v in vetoed
        if v.startswith(prefix) and v[len(prefix) :].isdigit()
    ]
    return pd.DataFrame(
        {"t_start_s": starts, "t_end_s": [s + length_s for s in starts]}, dtype=float
    )


def _process(
    source: SourceRecording, out_dir: Path, config: BackgroundConfig, vetoed: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    t_start = time.perf_counter()
    info: dict[str, Any] = {"dataset": source.dataset, "recording": source.name}
    if source.labels_path is not None and config.exclude_labelled:
        info["skipped"] = "hand-labelled (locked test set)"
        return [], info
    rec: Recording = open_source_recording(source)
    fs = rec.samplerate_hz
    duration = rec.duration_s
    cfg = read_fishgrid_config(source.config_path) if source.config_path else None
    tracks = load_tracks(source.tracks_dir) if source.tracks_dir else None
    detections = (
        read_v1_detections(source.detections_path) if source.detections_path else None
    )
    labels = read_labels(source.labels_path) if source.labels_path else None
    # segments the author rejected in review.csv are avoided like labels
    rejected = _vetoed_spans(source, vetoed, config.selection.segment_s)
    if len(rejected):
        labels = (
            rejected
            if labels is None
            else pd.concat([labels, rejected], ignore_index=True)
        )
        info["n_vetoed"] = len(rejected)
    rng = _rng(config.seed, source)
    plan = _plan(source, duration, tracks, detections, labels, config, rng)
    info.update(plan.stats)
    info.update(
        duration_s=duration,
        n_electrodes=rec.n_channels,
        n_detections=None if detections is None else len(detections),
        n_track_ids=None if tracks is None else int(tracks.ids.size),
    )
    length = config.selection.segment_s
    n = round(length * fs)
    n_ids = 0 if tracks is None else int(tracks.ids.size)
    track_quality = (
        "none"
        if tracks is None
        else ("curated" if n_ids <= config.curated_max_ids else "fragmented")
    )
    nfft_tracks = _track_power_nfft(tracks, fs)
    det_centres = (
        None
        if detections is None
        else np.sort(0.5 * (detections["t_start_s"] + detections["t_end_s"]).to_numpy())
    )
    # pass 1: measure every planned segment (audio is not kept)
    measured: list[tuple[float, str, float, Tracks | None, dict[str, Any]]] = []
    for start_s, kind, score in zip(
        plan.starts_s, plan.classes, plan.scores, strict=True
    ):
        start = round(start_s * fs)
        clipped = (
            None
            if tracks is None
            else clip_tracks(
                tracks, start / fs, (start + n) / fs, n_electrodes=rec.n_channels
            )
        )
        m = _measure(rec.read(start, start + n), fs, clipped, config)
        measured.append((start / fs, kind, score, clipped, m))

    # lines at a fixed frequency in many segments are interference, not fish
    # (only peaks no tracked fish explains, so a stable fish's line never
    # counts as interference where it is untracked)
    stationary = _stationary_lines(
        [np.asarray(m["untracked"]) for *_, m in measured], config.stationary_fraction
    )
    info["stationary_lines_hz"] = stationary
    tol = config.untracked_tolerance_hz
    kept: list[tuple[float, str, float, Tracks | None, dict[str, Any]]] = []
    n_free_kept = 0
    for start_s, kind, score, clipped, m in measured:
        m["untracked"] = [f for f in m["untracked"] if not _near(f, stationary, tol)]
        if kind == "fish_free_candidate":
            fishy = [f for f in m["peaks_hz"] if not _near(f, stationary, tol)]
            if fishy or clipped is not None:
                continue
            if n_free_kept >= config.fish_free_per_recording:
                continue
            kind = "fish_free"
            n_free_kept += 1
        kept.append((start_s, kind, score, clipped, m))
    measured = kept
    info["n_fish_free_segments"] = n_free_kept

    # pass 2: write the accepted segments
    for start_s, _, _, clipped, _ in measured:
        start = round(start_s * fs)
        seg_dir = out_dir / "segments" / segment_id(source, start_s)
        seg_dir.mkdir(parents=True, exist_ok=True)
        _write_wav(seg_dir / "background.wav", rec.read(start, start + n), fs)
        if source.config_path is not None:
            shutil.copyfile(source.config_path, seg_dir / "fishgrid.cfg")
        if clipped is not None:
            save_tracks(clipped, seg_dir)

    # dead channels: no fish above dead_channel_snr_db on them in any segment
    best = np.full(rec.n_channels, -np.inf)
    for _, _, _, _, m in measured:
        floors = m["level"].noise_psd_per_electrode * config.reference_bandwidth_hz
        for fish in m["fish"]:
            p = np.asarray(fish["power_per_electrode"], dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                snr = 10 * np.log10(np.where(p > 0, p, np.nan) / floors)
            best = np.fmax(best, np.nan_to_num(snr, nan=-np.inf))
    dead = (
        [int(c) for c in np.flatnonzero(best < config.dead_channel_snr_db)]
        if np.isfinite(best).any()
        else []
    )
    info["dead_channels"] = dead
    noisy = (
        _noisy_channels(np.median([m["floors_db"] for *_, m in measured], axis=0), dead)
        if measured
        else []
    )
    info["noisy_channels"] = noisy

    rows = []
    geometry = rec.geometry
    for start_s, kind, score, clipped, m in measured:
        sid = segment_id(source, start_s)
        seg_dir = out_dir / "segments" / sid
        start = round(start_s * fs)
        m["level"]
        fish = m["fish"]
        fish_snr = [f["snr_db"] for f in fish]
        finite_snr = [s for s in fish_snr if math.isfinite(s)]
        stop_s = start_s + n / fs
        n_det = 0
        nearest = math.nan
        rate_h = math.nan
        expected = math.nan
        if detections is not None and det_centres is not None:
            sel = config.selection
            n_det = int(
                count_overlaps(
                    [start_s],
                    n / fs,
                    detections["t_start_s"],
                    detections["t_end_s"],
                    sel.margin_s,
                )[0]
            )
            if det_centres.size:
                inside = (det_centres >= start_s) & (det_centres <= stop_s)
                dist = np.minimum(
                    np.abs(det_centres - start_s), np.abs(det_centres - stop_s)
                )
                nearest = 0.0 if inside.any() else float(dist.min())
            rate = local_rate_per_s(
                [start_s + length / 2], det_centres, sel.rate_window_s
            )
            rate_h = float(rate[0] * 3600)
            expected = float(expected_missed_chirps(rate, length, sel.v1_recall)[0])
        screened = kind == "fish_free" or detections is not None
        complete = kind == "fish_free" or (
            clipped is not None
            and not m["untracked"]
            and all(f["coverage"] >= 0.9 for f in fish)
        )
        row = {
            "segment_id": sid,
            "dataset": source.dataset,
            "recording": source.name,
            "split_group": source.key,
            "path": str(Path("segments") / sid),
            "segment_class": kind,
            "screened": bool(screened),
            "weight": 1.0 if screened else config.unscreened_weight,
            "start_s": float(start_s),
            "stop_s": float(stop_s),
            "duration_s": float(n / fs),
            "dark_phase": bool(
                source.dark_phase_s is not None
                and start_s + length / 2 < source.dark_phase_s
            ),
            "samplerate_hz": float(fs),
            "n_samples": int(n),
            "n_electrodes": int(rec.n_channels),
            "n_rows": None if cfg is None else int(cfg.n_rows),
            "n_columns": None if cfg is None else int(cfg.n_columns),
            "row_spacing_m": None if cfg is None else float(cfg.row_spacing_m),
            "column_spacing_m": None if cfg is None else float(cfg.column_spacing_m),
            "geometry_known": bool(source.geometry_known),
            "spacing_known": bool(source.spacing_known),
            "electrode_positions_m": None
            if geometry is None
            else geometry.positions_m.tolist(),
            "units": FISHGRID_UNITS,
            "n_v1_detections_margin": n_det if detections is not None else None,
            "nearest_v1_detection_s": nearest,
            "v1_rate_per_h": rate_h,
            "expected_missed_chirps": expected,
            "selection_score": float(score),
            "tracks_available": tracks is not None,
            "track_quality": track_quality,
            "track_power_nfft": nfft_tracks,
            "n_fish": len(fish),
            "fish_ids": [f["ident"] for f in fish],
            "fish_eodf_hz": [f["eodf_hz"] for f in fish],
            "fish_coverage": [f["coverage"] for f in fish],
            "fish_power": [f["power"] for f in fish],
            "fish_snr_db": fish_snr,
            "real_snr_max_db": max(finite_snr) if finite_snr else math.nan,
            "real_level_diff_db": max(finite_snr) - min(finite_snr)
            if len(finite_snr) >= 2
            else math.nan,
            "untracked_peaks_hz": m["untracked"],
            "tracks_complete": bool(complete),
            "dead_channels": dead,
            "noisy_channels": noisy,
            **_level_columns(m, n / fs),
            "accept": sid not in vetoed,
        }
        details = {
            "segment_id": sid,
            "source": {
                "dataset": source.dataset,
                "recording": source.name,
                "path": str(source.recording_path),
                "start_sample": start,
                "n_samples": n,
                "tracks_dir": None
                if source.tracks_dir is None
                else str(source.tracks_dir),
                "detections": None
                if source.detections_path is None
                else str(source.detections_path),
            },
            **_level_details(m),
            "fish": fish,
            "dead_channels": dead,
            "noisy_channels": noisy,
        }
        (seg_dir / "segment.json").write_text(
            json.dumps(details, indent=1, default=_json_default) + "\n"
        )
        rows.append(row)
    info["n_segments"] = len(rows)
    info["runtime_s"] = round(time.perf_counter() - t_start, 1)
    log.info(
        "%s: %d segments (%d fish-free) in %.0f s",
        source.key,
        len(rows),
        n_free_kept,
        info["runtime_s"],
    )
    return rows, info


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def _git_commit() -> str | None:
    try:
        here = Path(__file__).resolve().parent
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def build_background_library(
    sources: Sequence[SourceRecording | FieldFolder],
    out_dir: str | Path,
    config: BackgroundConfig | None = None,
) -> pd.DataFrame:
    """Select, measure and write background segments; return the manifest.

    Parameters
    ----------
    sources : sequence of SourceRecording or FieldFolder
        Recordings to draw from: grid recordings with tracks and detections
        (e.g. :func:`competition_2019_sources`,
        :func:`competition_2022_sources`) and field deployments scanned for
        fish-free stretches (:mod:`gridtools.background.field`).
    out_dir : path
        Library directory; created if missing. Builds are incremental: the
        rows of the recordings in ``sources`` are replaced, rows of other
        recordings in an existing manifest are kept. Segment directories of
        the same id are overwritten.
    config : BackgroundConfig, optional

    Returns
    -------
    pandas.DataFrame
        The manifest (also written as ``manifest.parquet``/``.csv``, with
        ``recordings.csv``, ``library.json`` and the merged ``review.csv``
        next to it). Segments the author rejected in ``review.csv`` are
        avoided (another window is chosen); rows carry ``accept``.
    """
    config = BackgroundConfig() if config is None else config
    out = Path(out_dir)
    (out / "segments").mkdir(parents=True, exist_ok=True)
    keys = [s.key for s in sources]
    if len(set(keys)) != len(keys):
        msg = "source recordings must be unique (dataset/name)"
        raise ValueError(msg)
    review = read_review(out)
    vetoed = (
        set()
        if review is None
        else set(review.loc[~review["accept"], "segment_id"].astype(str))
    )
    t0 = time.perf_counter()
    results: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
    if config.workers == 1:
        results = [_dispatch(s, out, config, vetoed) for s in sources]
    else:
        with ProcessPoolExecutor(
            max_workers=min(config.workers, 6, len(sources) or 1)
        ) as pool:
            futures = [pool.submit(_dispatch, s, out, config, vetoed) for s in sources]
            results = [f.result() for f in futures]
    rows = [row for r, _ in results for row in r]
    manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    processed = set(keys)
    old_path = out / "manifest.parquet"
    if old_path.exists():
        old = read_manifest(old_path)
        keep = ~(old["dataset"] + "/" + old["recording"]).isin(list(processed))
        old = pd.DataFrame(old.loc[keep.to_numpy()])
        if "accept" not in old.columns:
            old["accept"] = True
        old = pd.DataFrame(old[list(MANIFEST_COLUMNS)])
        manifest = (
            pd.concat([old, manifest], ignore_index=True) if len(old) else manifest
        )
    manifest = manifest.sort_values(["dataset", "recording", "start_s"], kind="stable")
    manifest = manifest.reset_index(drop=True)
    write_manifest(manifest, out)
    infos = pd.DataFrame([info for _, info in results])
    rec_path = out / "recordings.csv"
    if rec_path.exists():
        previous = pd.read_csv(rec_path)
        gone = (previous["dataset"] + "/" + previous["recording"]).isin(list(processed))
        previous = pd.DataFrame(previous.loc[~gone.to_numpy()])
        infos = pd.concat([previous, infos], ignore_index=True)
    infos.to_csv(rec_path, index=False)
    merge_review(out, read_manifest(out / "manifest.parquet"))
    size = sum(p.stat().st_size for p in (out / "segments").rglob("*") if p.is_file())
    meta_path = out / "library.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    builds = meta.get("builds", [])
    builds.append(
        {
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "config": config.to_dict(),
            "sources": [
                {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in dataclasses.asdict(s).items()
                }
                for s in sources
            ],
            "n_segments_written": len(rows),
            "runtime_s": round(time.perf_counter() - t0, 1),
            "versions": {
                "gridtools": importlib.metadata.version("gridtools"),
                "gridtools_commit": _git_commit(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": importlib.metadata.version("scipy"),
            },
        }
    )
    meta = {
        "updated": builds[-1]["created"],
        "n_segments": len(manifest),
        "size_bytes": size,
        "builds": builds,
    }
    meta_path.write_text(json.dumps(meta, indent=1, default=str) + "\n")
    log.info("library %s: %d segments, %.1f GB", out, len(manifest), size / 1e9)
    return read_manifest(out / "manifest.parquet")
