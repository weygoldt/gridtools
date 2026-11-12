from pathlib import Path

import numpy as np
import pytest

from gridtools.io import Tracks, load_tracks, save_tracks


def make_tracks() -> Tracks:
    # Fish 1: frames 0-4 at 500 Hz rising by 1 Hz per frame, with a gap at frame 2.
    # Fish 2: frames 0-1 only. One unassigned detection.
    times = np.arange(6, dtype=np.float64) * 0.5
    freqs = np.array([500, 501, 503, 504, 700, 700, 900], dtype=np.float64)
    idents = np.array([1, 1, 1, 1, 2, 2, np.nan], dtype=np.float64)
    indices = np.array([0, 1, 3, 4, 0, 1, 5], dtype=np.int64)
    powers = np.arange(14, dtype=np.float64).reshape(7, 2)
    return Tracks(times, freqs, idents, indices, powers)


def test_ids_and_tracks() -> None:
    tracks = make_tracks()
    np.testing.assert_array_equal(tracks.ids, [1, 2])
    fish = tracks.track(1)
    np.testing.assert_allclose(fish.times_s, [0.0, 0.5, 1.5, 2.0])
    np.testing.assert_allclose(fish.freqs_hz, [500, 501, 503, 504])
    assert (fish.start_s, fish.stop_s) == (0.0, 2.0)
    assert [t.ident for t in tracks] == [1, 2]
    with pytest.raises(KeyError):
        tracks.track(3)


def test_frequency_is_undefined_outside_track_and_in_long_gaps() -> None:
    fish = make_tracks().track(1)
    f = fish.frequency_at([-0.1, 0.0, 0.25, 1.0, 1.5, 2.0, 2.1], max_gap_s=0.6)
    np.testing.assert_allclose(f, [np.nan, 500, 500.5, np.nan, 503, 504, np.nan])
    # the 1 s gap is bridged when allowed
    assert fish.frequency_at(1.0, max_gap_s=1.0)[0] == pytest.approx(502.0)


def test_frequencies_at_lists_only_live_tracks() -> None:
    tracks = make_tracks()
    assert tracks.frequencies_at(0.25) == pytest.approx({1: 500.5, 2: 700.0})
    assert tracks.frequencies_at(1.75) == pytest.approx({1: 503.5})


def test_powers_at() -> None:
    fish = make_tracks().track(2)
    np.testing.assert_allclose(fish.powers_at([0.25, 3.0]), [[9, 10], [np.nan, np.nan]])


def test_duplicate_detections_in_one_frame_are_averaged() -> None:
    tracks = Tracks(
        np.arange(2.0),
        np.array([500, 502, 510.0]),
        np.array([1, 1, 1.0]),
        np.array([0, 0, 1]),
    )
    np.testing.assert_allclose(tracks.track(1).freqs_hz, [501, 510])


def test_validation() -> None:
    with pytest.raises(ValueError, match="outside times"):
        Tracks(np.arange(2.0), np.array([500.0]), np.array([1.0]), np.array([2]))
    with pytest.raises(ValueError, match="finite frequencies"):
        Tracks(np.arange(2.0), np.array([np.nan]), np.array([1.0]), np.array([0]))
    with pytest.raises(ValueError, match="entries"):
        Tracks(
            np.arange(2.0), np.array([500.0, 501]), np.array([1.0]), np.array([0, 1])
        )
    with pytest.raises(ValueError, match="read-only"):
        make_tracks().freqs_hz[0] = 1.0


def test_save_load_roundtrip(tmp_path: Path) -> None:
    tracks = make_tracks()
    save_tracks(tracks, tmp_path)
    loaded = load_tracks(tmp_path)
    for name in ("times_s", "freqs_hz", "idents", "time_indices", "powers_db"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(tracks, name))


def test_fixture_tracks(fixture_dir: Path) -> None:
    tracks = load_tracks(fixture_dir)
    np.testing.assert_array_equal(tracks.ids, [2, 3])
    assert tracks.n_electrodes == 15
    for track in tracks:
        assert np.all(np.diff(track.times_s) > 0)
        assert 600 < np.median(track.freqs_hz) < 900
