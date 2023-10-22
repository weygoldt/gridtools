"""
Test the datasets module.
"""

import pathlib

import numpy as np
import pytest

from gridtools.datasets import ChirpData, Dataset, RawData, WavetrackerData
from gridtools.exceptions import GridDataMismatch

path = pathlib.Path(__file__).resolve().parent.parent / "data"


def test_wavetracker_data_init():
    """
    Tests the initialization of the WavetrackerData class on real data.
    """
    data = WavetrackerData(path)
    assert data.path == path


def test_wavetracker_data_load_data(data_dir):
    """
    Tests the loading of data into the WavetrackerData class.
    Checks that the data that is loaded is the data that is expected.
    """

    # Create and save test data
    np.save(data_dir / "fund_v.npy", np.array([1, 2, 3]))
    np.save(data_dir / "sign_v.npy", np.array([4, 5, 6]))
    np.save(data_dir / "ident_v.npy", np.array([7, 8, 9]))
    np.save(data_dir / "idx_v.npy", np.array([1, 2, 3]))
    np.save(data_dir / "times.npy", np.array([0.1, 0.2, 0.3]))

    data = WavetrackerData(data_dir)
    assert np.array_equal(data.freqs, np.array([1, 2, 3]))
    assert np.array_equal(data.powers, np.array([4, 5, 6]))
    assert np.array_equal(data.idents, np.array([7, 8, 9]))
    assert np.array_equal(data.indices, np.array([1, 2, 3]))
    assert np.array_equal(data.times, np.array([0.1, 0.2, 0.3]))


# def test_raw_data_init_with_raw():
#     """
#     Test the initialization of the RawData class with a `traces-grid1.raw` file.
#     """

#     test_path.mkdir()

#     traces_raw_data = np.array([[1, 2, 3], [4, 5, 6]])
#     traces_raw_data.tofile(test_path / "traces-grid1.raw")

#     data = RawData(test_path)
#     assert data.path == test_path
#     assert np.array_equal(data.raw, traces_raw_data)
#     assert data.samplerate == 20000
#     assert data.channels == traces_raw_data.shape[1]

#     shutil.rmtree(test_path)


def test_raw_data_init_with_npy(data_dir):
    """
    Tests the initialization of the RawData class with a `raw.npy` file.
    """

    # Create and save test data
    test_data = np.array([[1, 2, 3], [4, 5, 6]])
    np.save(data_dir / "raw.npy", test_data)

    data = RawData(data_dir)
    assert data.path == data_dir
    assert np.array_equal(data.raw, test_data)
    assert data.samplerate == 20000
    assert data.channels == np.shape(test_data)[1]


def test_raw_data_init_with_empty_raw(data_dir):
    """
    Tests the initialization of the RawData class with an empty `raw.npy` file.
    """

    # Create and save an empty test data
    empty_raw_data = np.array([]).reshape(0, 3)
    np.save(data_dir / "raw.npy", empty_raw_data)

    with pytest.raises(GridDataMismatch):
        RawData(data_dir)


def test_raw_data_init_with_missing_file(data_dir):
    """
    Tests the initialization of the RawData class with a missing `raw.npy` file.
    """
    with pytest.raises(FileNotFoundError):
        RawData(data_dir)


def test_valid_detector(chirp_data):
    """
    Tests that the detector is valid.
    """
    assert chirp_data.detector == "gp"


def test_invalid_detector():
    """
    Tests that an invalid detector raises an AssertionError.
    """
    with pytest.raises(AssertionError):
        invalid_detector = "invalid"
        ChirpData(pathlib.Path("nonexistent_path"), invalid_detector)


def test_nonexistent_path():
    """
    Tests that a nonexistent path raises a FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        non_existent_path = pathlib.Path("nonexistent_path")
        ChirpData(non_existent_path, "gp")


def test_get_fish(chirp_data):
    """
    Tests that the get_fish method returns the correct times for a given fish.
    """
    fish_1_times = chirp_data.get_fish(1)
    assert np.array_equal(fish_1_times, np.array([1.0, 3.0]))


def test_data_mismatch(tmp_path):
    """
    Tests that a GridDataMismatch is raised when the times and idents do not
    match.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    detector = "gp"
    invalid_idents = np.array([1, 2, 3, 4, 5])
    np.save(
        data_dir / f"chirp_times_{detector}.npy", np.array([1.0, 2.0, 3.0, 4.0])
    )
    np.save(data_dir / f"chirp_ids_{detector}.npy", invalid_idents)

    with pytest.raises(GridDataMismatch, match="Times and idents do not match"):
        ChirpData(data_dir, detector)


def test_dataset_init():
    """
    Tests the initialization of the Dataset class.
    """
    track = WavetrackerData(path)
    dataset = Dataset(path, track)
    assert dataset.path == path
    assert dataset.track == track


def test_dataset_init_with_optional_args():
    """
    Tests the initialization of the Dataset class with optional arguments.
    """
    track = WavetrackerData(path)
    rec = RawData(path)
    chirp = ChirpData(path, "gt")
    dataset = Dataset(path, track, rec, chirp)
    assert dataset.path == path
    assert dataset.track == track
    assert dataset.rec == rec
    assert dataset.chirp == chirp


def test_dataset_init_with_wrong_types():
    """
    Tests that the Dataset class raises an AssertionError when initialized with
    arguments of the wrong type.
    """
    with pytest.raises(AssertionError):
        Dataset("wrong_type", "wrong_type")


def test_dataset_check_type_with_wrong_types():
    """
    Tests that the _check_type method of the Dataset class raises an AssertionError
    when called with arguments of the wrong type.
    """
    wrong_type = "wrong_type"
    with pytest.raises(AssertionError):
        Dataset(path, wrong_type)


def test_dataset_repr():
    """
    Tests the __repr__ method of the Dataset class.
    """
    track = WavetrackerData(path)
    dataset = Dataset(path, track)
    assert repr(dataset) == f"Dataset({track}, None, None)"


def test_dataset_str():
    """
    Tests the __str__ method of the Dataset class.
    """
    track = WavetrackerData(path)
    dataset = Dataset(path, track)
    assert str(dataset) == f"Dataset({track}, None, None)"


def test_dataset_check_type_with_optional_args():
    """
    Tests that the _check_type method of the Dataset class does not raise an
    AssertionError when called with optional arguments of the correct type.
    """
    track = WavetrackerData(path)
    rec = RawData(path)
    chirp = ChirpData(path, "gt")
    Dataset(path, track, rec, chirp)


def test_dataset_check_type_with_missing_optional_args():
    """
    Tests that the _check_type method of the Dataset class does not raise an
    AssertionError when called with missing optional arguments.
    """
    track = WavetrackerData(path)
    Dataset(path, track)


def test_dataset_check_type_with_wrong_optional_args():
    """
    Tests that the _check_type method of the Dataset class raises an AssertionError
    when called with optional arguments of the wrong type.
    """
    track = WavetrackerData(path)
    rec = "wrong_type"
    chirp = "wrong_type"
    with pytest.raises(AssertionError):
        Dataset(path, track, rec, chirp)
