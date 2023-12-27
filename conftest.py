"""
This module provides dataset generation functions for the testing suite
of the gridtools package.
"""
import shutil

import numpy as np
import pytest

from gridtools.datasets import ChirpData


# Define a pytest fixture for creating a temporary data directory
@pytest.fixture(name="data_dir")
def fixture_data_dir(tmp_path):
    """
    Creates a temporary data directory for testing. Handles cleanup after
    each test. This is a pytest fixture, so it is automatically passed to
    each test function that uses it.
    """
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    yield data_dir  # Provide the data directory to the test functions
    shutil.rmtree(data_dir)  # Cleanup after each test


@pytest.fixture(name="chirp_data")
def fixture_chirp_data(tmp_path):
    """
    Creates a temporary chirp dataset for testing. Handles cleanup after
    data is created. This is a pytest fixture, so it is automatically passed to
    each test function that uses it.
    """
    # Create temporary data for testing
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    detector = "gp"
    example_times = np.array([1.0, 2.0, 3.0, 4.0])
    example_idents = np.array([1, 2, 1, 3])
    np.save(data_dir / f"chirp_times_{detector}.npy", example_times)
    np.save(data_dir / f"chirp_ids_{detector}.npy", example_idents)

    return ChirpData(data_dir, detector)
