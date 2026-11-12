"""Harvesting long chirps from (synthetic) grid recordings."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import hilbert

from gridtools.io import Recording
from gridtools.templates import BuildConfig
from gridtools.templates.harvest import (
    CandidateResult,
    HarvestConfig,
    contaminants,
    demodulated_frequency,
    find_candidates,
    harvest_candidate,
    mark_duplicates,
    null_weights,
    open_recording_lenient,
    pulse_pair_frequency,
    read_review,
    write_review,
)

from .conftest import (
    FS,
    SyntheticGrid,
    synthetic_tracks,
    write_grid_directory,
)


def run(grid: SyntheticGrid, config: HarvestConfig | None = None) -> CandidateResult:
    recording = Recording(Path("synthetic"), grid.samplerate_hz, grid.data)
    return harvest_candidate(
        recording,
        synthetic_tracks(grid),
        grid.box,
        config or HarvestConfig(),
        recording_name="synthetic",
    )


# ---- signal processing ------------------------------------------------------


def test_pulse_pair_recovers_constant_and_swept_frequency() -> None:
    t = np.arange(round(0.5 * FS)) / FS
    tone = np.exp(2j * np.pi * 900.0 * t)
    f = pulse_pair_frequency(tone, FS, 20.0)
    np.testing.assert_allclose(f, 900.0, atol=1e-6)
    sweep_hz = 800.0 + 400.0 * t  # 400 Hz/s
    sweep = np.exp(2j * np.pi * np.cumsum(sweep_hz) / FS)
    f = pulse_pair_frequency(sweep, FS, np.full(t.size, 20.0))
    inner = slice(200, -200)
    np.testing.assert_allclose(f[inner], sweep_hz[inner], atol=0.5)


def test_demodulation_removes_noise_bias_at_low_amplitude() -> None:
    # a weak tone in broadband noise is pulled toward the band centre by the
    # power-weighted estimate; demodulating around the estimate removes that
    rng = np.random.default_rng(1)
    n = round(0.4 * FS)
    t = np.arange(n) / FS
    tone = 0.05 * np.exp(2j * np.pi * 1200.0 * t)
    noise = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
    spectrum = np.fft.fft(noise)
    freqs = np.fft.fftfreq(n, 1 / FS)
    spectrum[(freqs < 800) | (freqs > 1300)] = 0  # band 800-1300 Hz, centre 1050
    noise = 0.02 * np.fft.ifft(spectrum) / np.sqrt(np.mean(np.abs(spectrum) ** 2) / n)
    z = tone + noise
    first = pulse_pair_frequency(z, FS, 160.0)
    refined = demodulated_frequency(z, FS, first, 160.0, (150.0, 80.0, 50.0))
    inner = slice(2000, -2000)
    assert np.median(first[inner]) < 1195.0  # biased low
    assert abs(np.median(refined[inner]) - 1200.0) < abs(
        np.median(first[inner]) - 1200.0
    )
    assert abs(np.median(refined[inner]) - 1200.0) < 3.0


def test_null_weights_cancel_an_interferer() -> None:
    emitter = np.array([1.0, -0.6, 0.35, 0.15, -0.1, 0.05])
    other = np.array([0.1, 0.3, -0.2, 0.9, 0.6, -0.4])
    w, keep = null_weights(emitter, [other])
    assert abs(w @ other) < 1e-12
    assert np.linalg.norm(w) == pytest.approx(1.0)
    assert 0.5 < keep <= 1.0
    w0, keep0 = null_weights(emitter, [])
    assert keep0 == pytest.approx(1.0)
    np.testing.assert_allclose(w0, emitter / np.linalg.norm(emitter))


def test_contaminants() -> None:
    fund, harm = contaminants(800.0, 1250.0, [700.0, 1000.0, 610.0], harmonics=True)
    assert fund == [1000.0]
    assert harm == [1220.0]  # 2 x 610
    fund, harm = contaminants(800.0, 1250.0, [610.0], harmonics=False)
    assert (fund, harm) == ([], [])


# ---- whole candidates ------------------------------------------------------


@pytest.mark.parametrize(
    ("excursion_hz", "plateau_s", "tau_s"),
    [(450.0, 0.15, 0.04), (400.0, 0.20, 0.06), (520.0, 0.25, 0.05)],
)
def test_harvest_recovers_plateau_and_decay(
    synthetic_grid: Callable[..., SyntheticGrid],
    excursion_hz: float,
    plateau_s: float,
    tau_s: float,
) -> None:
    grid = synthetic_grid(
        seed=3, excursion_hz=excursion_hz, plateau_s=plateau_s, decay_tau_s=tau_s
    )
    total_s = grid.end_2pct_s - grid.onset_s
    assert 0.3 <= total_s <= 0.5  # a long chirp
    result = run(grid)
    record, extraction = result.record, result.extraction
    assert record["status"] == "harvested", record["reason"]
    assert record["emitter_track_id"] == 1
    assert record["emitter_hz"] == pytest.approx(grid.emitter_hz, abs=1.0)
    assert extraction is not None
    snippet = extraction.snippet
    a, m = snippet.analysis, snippet.metrics
    assert a.baseline_freq == pytest.approx(grid.emitter_hz, abs=1.0)
    assert m.excursion_hz == pytest.approx(excursion_hz, rel=0.04)
    # compare the template with the true excursion inside the support
    start = round(extraction.window_start_s * FS)
    truth = grid.delta_true_hz[start : start + extraction.signal.size]
    centre = snippet.delta_f_hz.size // 2
    first = a.align_index - centre
    support = np.arange(int(np.ceil(a.support_start)), int(a.support_stop))
    error = snippet.delta_f_hz[support - first] - truth[support]
    assert np.median(np.abs(error)) < 5.0
    # the estimator window widens where the amplitude drops, so the abrupt
    # rise is smoothed over a few ms (most of the remaining error)
    assert np.quantile(np.abs(error), 0.9) < 25.0
    true_10 = np.flatnonzero(truth >= 0.1 * excursion_hz)
    assert m.duration_10_s == pytest.approx((true_10[-1] - true_10[0]) / FS, abs=0.006)
    # the support covers the whole slow decay
    assert extraction.window_start_s + a.support_stop / FS >= grid.end_2pct_s
    # exact baseline outside the support and ramps
    d, amp = snippet.delta_f_hz, snippet.amp_rel
    assert d[0] == d[-1] == 0.0
    assert amp[0] == amp[-1] == 1.0
    ramp = BuildConfig().taper_ramp_s * FS
    outside = np.ones(d.size, dtype=bool)
    outside[
        int(np.floor(a.support_start - ramp - first)) : int(
            np.ceil(a.support_stop + ramp - first)
        )
        + 1
    ] = False
    assert np.all(d[outside] == 0.0)
    assert np.all(amp[outside] == 1.0)


def test_harvest_rejects_a_fish_inside_the_span(
    synthetic_grid: Callable[..., SyntheticGrid],
) -> None:
    grid = synthetic_grid(seed=4, other_hz=1050.0)
    result = run(grid)
    assert result.record["status"] == "rejected"
    assert result.record["reason"] == "fish_in_span"
    assert result.record["emitter_track_id"] == 1


def test_harvest_rejects_a_harmonic_inside_the_span(
    synthetic_grid: Callable[..., SyntheticGrid],
) -> None:
    grid = synthetic_grid(seed=5, other_hz=560.0)  # second harmonic at 1120 Hz
    result = run(grid)
    assert result.record["status"] == "rejected"
    assert result.record["reason"] == "harmonic_in_span"
    keep = run(grid, HarvestConfig(reject_harmonics=False))
    assert keep.record["status"] == "harvested"


def test_harvest_finds_the_emitter_without_a_hint(
    synthetic_grid: Callable[..., SyntheticGrid],
) -> None:
    grid = synthetic_grid(seed=6)
    box: dict[str, object] = dict(grid.box)
    box["track_id"] = pd.NA  # v1 often has no track
    recording = Recording(Path("synthetic"), grid.samplerate_hz, grid.data)
    result = harvest_candidate(recording, synthetic_tracks(grid), box, HarvestConfig())
    assert result.record["status"] == "harvested"
    assert result.record["emitter_track_id"] == 1
    assert result.record["hint_agrees"] is None


# ---- bookkeeping -------------------------------------------------------------


def test_find_candidates_thresholds() -> None:
    detections = pd.DataFrame(
        {
            "t_start_s": [0.0, 1.0, 2.0, 3.0],
            "t_end_s": [0.3, 1.2, 2.26, 3.5],
            "f_low_hz": [800.0, 800.0, 800.0, 800.0],
            "f_high_hz": [1200.0, 1200.0, 1200.0, 900.0],
            "score": [0.9] * 4,
            "emitter_hz": [800.0] * 4,
            "track_id": pd.array([1, 1, 1, 1], dtype="Int64"),
        }
    )
    candidates = find_candidates(detections, HarvestConfig())
    assert candidates["box_index"].tolist() == [0, 2]


def test_mark_duplicates_keeps_the_best_box() -> None:
    records = pd.DataFrame(
        {
            "status": ["harvested"] * 4,
            "recording": ["a", "a", "a", "b"],
            "emitter_track_id": [1, 1, 1, 1],
            "onset_10_abs_s": [10.0, 10.02, 11.0, 10.0],
            "box_score": [0.5, 0.9, 0.7, 0.4],
            "box_index": [0, 1, 2, 0],
        }
    )
    duplicate = mark_duplicates(records, 0.05)
    assert duplicate.tolist() == [True, False, False, False]


def test_review_sheet_keeps_vetoes(tmp_path: Path) -> None:
    meta = pd.DataFrame(
        {
            "harvest_id": [0, 1],
            "harvest_key": ["rec#1", "rec#2"],
            "recording": ["rec", "rec"],
            "emitter_track_id": [3, 3],
            "individual": ["g1f1", "g1f1"],
            "onset_10_abs_s": [1.0, 2.0],
            "duration_10_s": [0.31, 0.42],
            "excursion_hz": [400.0, 410.0],
            "baseline_hz": [800.0, 801.0],
            "step_flag": [False, True],
        }
    )
    path = tmp_path / "review.csv"
    sheet = write_review(meta, path)
    assert sheet["accept"].tolist() == [True, True]
    assert sheet["flags"].tolist() == ["", "step_flag"]
    edited = pd.read_csv(path, dtype={"note": str})
    edited.loc[1, "accept"] = False
    edited.loc[1, "note"] = "plateau broken"
    edited.to_csv(path, index=False)
    sheet = write_review(meta, path)  # a re-run keeps the author's decisions
    assert sheet["accept"].tolist() == [True, False]
    review = read_review(path)
    assert review["accept"].tolist() == [True, False]
    assert review.loc[1, "note"] == "plateau broken"
    edited["accept"] = edited["accept"].astype(object)
    edited.loc[0, "accept"] = "maybe"
    edited.to_csv(path, index=False)
    with pytest.raises(ValueError, match="accept"):
        read_review(path)


def test_open_recording_drops_a_partial_frame(
    tmp_path: Path, synthetic_grid: Callable[..., SyntheticGrid]
) -> None:
    grid = synthetic_grid(seed=0, duration_s=0.2)
    path = write_grid_directory(tmp_path / "rec", grid)
    with open(path / "traces-grid1.raw", "ab") as f:
        f.write(b"\x00" * 12)  # three of six floats of a last frame
    recording, dropped = open_recording_lenient(path)
    assert dropped == 12
    assert recording.n_samples == grid.data.shape[0]
    np.testing.assert_array_equal(recording.read(0, 10), grid.data[:10])


def test_analytic_helpers_accept_hilbert_output() -> None:
    x = np.sin(2 * np.pi * 850.0 * np.arange(4000) / FS)
    f = pulse_pair_frequency(np.asarray(hilbert(x)), FS, 20.0)
    assert np.median(f) == pytest.approx(850.0, abs=0.1)


def test_harvest_survives_lost_tracks_and_fragments(
    synthetic_grid: Callable[..., SyntheticGrid],
) -> None:
    # the tracker lost the emitter near the chirp (track 1 only 40-60 s away)
    # and a fragment (track 3) sits on the emitter's frequency
    from gridtools.io import Tracks

    grid = synthetic_grid(seed=8)
    far = np.arange(40.0, 60.0, 0.1)
    near = np.arange(0.0, 3.0, 0.1)
    times = np.concatenate([near, far])
    k_near, k_far = near.size, far.size
    tracks = Tracks(
        times_s=times,
        freqs_hz=np.concatenate(
            [
                np.full(k_far, grid.emitter_hz),
                np.full(k_near, grid.other_hz),
                np.full(k_near, grid.emitter_hz + 1.0),
            ]
        ),
        idents=np.concatenate(
            [np.ones(k_far), np.full(k_near, 2.0), np.full(k_near, 3.0)]
        ),
        time_indices=np.concatenate(
            [np.arange(k_near, k_near + k_far), np.arange(k_near), np.arange(k_near)]
        ),
    )
    recording = Recording(Path("synthetic"), grid.samplerate_hz, grid.data)
    result = harvest_candidate(recording, tracks, grid.box, HarvestConfig())
    assert result.record["status"] == "harvested", result.record["reason"]
    assert result.record["emitter_track_id"] == 1  # the hint wins over the fragment
    assert result.record["n_suspects"] == 2
