"""Smoke tests on the tube competition recordings (skipped if absent)."""

import numpy as np
import pytest

from gridtools.io import load_tracks, open_recording, read_labels, read_v1_detections

from ..conftest import DATA_ROOT

pytestmark = pytest.mark.data


@pytest.mark.parametrize(
    ("trial", "n_channels"), [("2019-12-09-10_00", 9), ("2020-03-16-10_00", 15)]
)
def test_open_trial(trial: str, n_channels: int) -> None:
    directory = DATA_ROOT / trial
    rec = open_recording(directory)
    assert rec.n_channels == n_channels
    assert rec.samplerate_hz == 20_000.0
    assert 5.5 * 3600 < rec.duration_s < 6.5 * 3600
    tail = rec.read(rec.n_samples - 100, rec.n_samples)
    assert tail.shape == (100, n_channels)
    tracks = load_tracks(directory)
    assert len(tracks.ids) >= 2
    assert tracks.times_s[-1] <= rec.duration_s + 1.0


def test_trial_ending_in_a_partial_frame_opens() -> None:
    # 2020-03-19 is one of the 33 analysed trials; its raw file ends mid-frame.
    rec = open_recording(DATA_ROOT / "2020-03-19-10_00")
    assert rec.n_channels == 15
    assert rec.read(rec.n_samples - 10, rec.n_samples).shape == (10, 15)


def test_sign_v_may_have_fewer_electrodes_than_the_recording() -> None:
    # All 2019 trials: raw channel 9 has no fish signal; sign_v holds channels 1-8.
    tracks = load_tracks(DATA_ROOT / "2019-12-09-10_00")
    assert tracks.n_electrodes == 8


def test_hand_labels_and_v1_detections_overlap() -> None:
    directory = DATA_ROOT / "2019-12-09-10_00"
    labels = read_labels(directory / "traces-grid1-editable-labels.csv")
    chirps = labels[labels["category"] == "chirp"]
    assert len(chirps) == 204
    detections = read_v1_detections(directory)
    centres = np.asarray(detections["t_start_s"] + detections["t_end_s"]) / 2
    label_centres = np.asarray(chirps["t_start_s"] + chirps["t_end_s"]) / 2
    nearest = np.min(np.abs(label_centres[:, None] - centres[None, :]), axis=1)
    assert np.mean(nearest < 0.1) > 0.5
