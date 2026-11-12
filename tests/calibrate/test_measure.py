"""End-to-end measurement and aggregation of a synthetic trial."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gridtools.calibrate.aggregate import AggregateConfig, aggregate, layout_for
from gridtools.calibrate.measure import MeasureConfig, TrialStats, measure_trial
from gridtools.calibrate.report import write_calibration_json, write_report
from gridtools.calibrate.run import measure_trials, trial_rng
from gridtools.calibrate.trials import read_trial_summary, resolve_trial_dir

from .conftest import AMPLITUDES, FISH_HZ, SyntheticTrial

SMALL = MeasureConfig(
    baseline_window_s=5.0,
    despike_window_s=3.0,
    dark_phase_s=20.0,
    light_phase_start_s=10.0,
    drift_max_lag_s=10.0,
    shared_lags_s=(2.0, 10.0),
    n_profile_samples=100,
    n_segments=2,
    segment_s=5.0,
    n_shuffles=5,
)


@pytest.fixture(scope="module")
def stats(synthetic_trial: SyntheticTrial) -> TrialStats:
    return measure_trial(
        synthetic_trial.path, synthetic_trial.meta, np.random.default_rng(0), SMALL
    )


def test_layout_and_dead_channel(stats: TrialStats) -> None:
    assert stats.layout == "9-8"
    assert stats.sign_v_nfft == 16384
    assert stats.noise.dead.tolist() == [False] * 8 + [True]
    assert not stats.noise.noisy.any()


def test_sign_v_matches_raw_channels(stats: TrialStats) -> None:
    np.testing.assert_allclose(
        np.nanmedian(stats.noise.sign_v_check_db, axis=0), 0.0, atol=1e-6
    )
    assert stats.noise.sign_v_best_channel.tolist() == list(range(8))


def test_noise_floor_and_amplitudes(stats: TrialStats) -> None:
    floor = np.median(stats.noise.floor_v2_hz[:, :8])
    assert floor == pytest.approx(1e-10 / 10_000, rel=0.1)
    np.testing.assert_allclose(stats.noise.alpha[:8], 0.0, atol=0.2)
    winner, loser = stats.fish
    assert winner.raw_peak_amplitude_v == pytest.approx(AMPLITUDES[0].max(), rel=0.03)
    assert loser.raw_peak_amplitude_v == pytest.approx(AMPLITUDES[1].max(), rel=0.03)
    # sign_v-based amplitude (bin-centred conversion) loses up to 1.4 dB.
    assert np.median(winner.peak_amplitude_v) <= AMPLITUDES[0].max() * 1.01
    assert np.median(winner.peak_amplitude_v) >= AMPLITUDES[0].max() * 10 ** (-1.5 / 20)
    expected_cn0 = 10 * np.log10((AMPLITUDES[0].max() ** 2 / 2) / floor)
    assert np.median(winner.cn0_db_hz) == pytest.approx(expected_cn0, abs=1.5)


def test_frequencies_and_delta_f(stats: TrialStats) -> None:
    winner, loser = stats.fish
    assert winner.baseline_hz == pytest.approx(FISH_HZ[0], abs=0.1)
    assert loser.baseline_hz == pytest.approx(FISH_HZ[1], abs=0.1)
    assert (winner.role, winner.sex, loser.role, loser.sex) == (
        "winner",
        "f",
        "loser",
        "m",
    )
    assert stats.pair.delta_f.quantiles_hz[0.5] == pytest.approx(60.0, abs=0.1)
    assert stats.pair.delta_f.n_crossings == 0
    assert stats.pair.delta_f.fraction_below[50.0] == 0.0
    assert not winner.drift_gaussian.sd > 0.05  # NaN for a perfectly constant trace


def test_profiles_are_order_free_and_constant(stats: TrialStats) -> None:
    winner = stats.fish[0]
    assert winner.profiles.shape[1] == 8
    expected = np.sort(20 * np.log10(AMPLITUDES[0][:8] / AMPLITUDES[0].max()))[::-1]
    np.testing.assert_allclose(np.median(winner.profiles, axis=0), expected, atol=0.3)
    # The strongest electrode never changes for a fish that does not move.
    assert winner.dwell[3.0]["switches_per_min"] == 0.0


def test_chirps(stats: TrialStats, synthetic_trial: SyntheticTrial) -> None:
    winner, loser = stats.chirps
    assert (winner.n_raw, winner.n_merged) == (4, 1)
    np.testing.assert_allclose(winner.times_s, [5.0, 12.0, 30.0], atol=1e-9)
    assert loser.times_s.size == 2
    assert stats.unassigned_chirps == 1
    cross = stats.cross["winner->loser"]
    # The winner chirp at 5.0 s has the loser chirp at 5.35 s, outside +/-0.25 s.
    assert cross.coincidence == 0.0
    lat = cross.latency_counts
    assert lat[3] == 1  # 0.3-0.4 s


def test_measure_trials_is_deterministic_and_cached(
    synthetic_trial: SyntheticTrial, tmp_path: Path
) -> None:
    root = synthetic_trial.path.parent
    (root / "trial_summary.csv").write_text(
        ",recording,win_ID,lose_ID,sex_win,sex_lose,size_win,size_lose\n"
        f"0,{synthetic_trial.path.name},1.0,2.0,f,m,12.0,11.0\n",
        encoding="utf-8",
    )
    assert read_trial_summary(root)[synthetic_trial.path.name].winner_id == 1
    assert resolve_trial_dir(root, "2019-01-01") == synthetic_trial.path
    first = measure_trials(
        root, ["2019-01-01"], seed=3, config=SMALL, cache_dir=tmp_path
    )
    assert (tmp_path / f"{synthetic_trial.path.name}.pkl").exists()
    again = measure_trials(root, ["2019-01-01"], seed=3, config=SMALL)
    np.testing.assert_array_equal(first[0].fish[0].profiles, again[0].fish[0].profiles)
    a = trial_rng(3, 0).random(3)
    b = trial_rng(3, 0).random(3)
    np.testing.assert_array_equal(a, b)


def test_layout_for_names() -> None:
    config = AggregateConfig()
    assert layout_for("9-8", config).missing == (8,)
    assert layout_for("15-14", config).n_random_missing == 1
    with pytest.raises(ValueError, match="unknown layout"):
        layout_for("4-4", config)


def test_aggregate_smoke(stats: TrialStats, tmp_path: Path) -> None:
    config = AggregateConfig(
        exponents=(2.0, 3.0),
        regularisers_rel=(0.1, 0.3),
        margins_rel=(0.5,),
        n_field_samples=500,
        n_bootstrap=3,
        movement_duration_s=60.0,
    )
    calibration = aggregate([stats, stats], config)
    params = calibration.parameters
    assert params["gridtools.sim.noise.ClipParams.limit"]["value"] == pytest.approx(
        0.02048
    )
    assert params["gridtools.sim.noise.ColoredNoiseParams.exponent"]["value"] == 0.0
    sd = params["gridtools.sim.noise.ColoredNoiseParams.sd"]["value"]
    assert sd == pytest.approx(1e-5, rel=0.1)
    assert params["gridtools.sim.field.DipoleParams.distance_exponent"]["value"] in (
        2.0,
        3.0,
    )
    assert calibration.summary["n_trials"] == 2
    path = tmp_path / "calibration.json"
    write_calibration_json(calibration, path, provenance={"test": True})
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert "gridtools.sim.eod.DriftParams.sd_hz" in loaded["parameters"]
    assert loaded["provenance"]["test"] is True
    report = write_report(calibration, tmp_path, {"test": True})
    text = report.read_text(encoding="utf-8")
    assert "## Proposed defaults vs current defaults" in text
    assert "`noise.ClipParams.limit`" in text
    assert (tmp_path / "figures" / "field_loss.png").exists()
