#!/usr/bin/env python3

"""
Wavetracker dataset classes using composition. The main class is the `Dataset`
class, which can be used to load data from the wavetracker, the raw data and
the chirp data. The `WavetrackerData` class loads the wavetracker data, the
`RawData` class loads the raw data and the `ChirpData` class loads the chirp
data. The `Dataset` class is a composition of the other three classes. This way,
the user can choose which data to load and which not to load. It is also easily
extensible to other data types, e.g. rises or behaviour data.
"""

import pathlib
from dataclasses import dataclass
from typing import Optional

from numpy import isnan, load, ndarray, unique
from rich import print as rprint
from thunderfish.dataloader import DataLoader

from .exceptions import GridDataMismatch


class WavetrackerData:
    """
    Loads the `.npy` files produced by the `wavetracker` into an easy to use
    class format. Provides methods to extract single or specific fish. Checks
    data set health (demension mismatches, etc.) before returning the object.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist.")

        self._load_data()
        self._check_health()

    def get_fish(self, fish: int) -> ndarray:
        """
        Function to extract a single fish or a subset of individuals from the
        dataset.

        Parameters
        ----------
        fish : Union[int, list, ndarray]
            The IDs of the fish to extract.

        Returns
        -------
        ndarray
            Time, freq and powers for the fish(es) specified.

        Raises
        ------
        ValueError
            When the fish ID is not in the dataset.
        """
        times = self.times[self.indices[self.idents == fish]]
        freqs = self.freqs[self.idents == fish]
        powers = self.powers[self.idents == fish]
        return times, freqs, powers

    def _load_data(self) -> None:
        self.freqs = load(self.path / "fund_v.npy")
        self.powers = load(self.path / "sign_v.npy")
        self.idents = load(self.path / "ident_v.npy")
        self.indices = load(self.path / "idx_v.npy")
        self.ids = unique(self.idents[~isnan(self.idents)]).astype(int)
        self.times = load(self.path / "times.npy")

    def _check_health(self) -> None:
        if (
            self.freqs.shape[0]
            != self.powers.shape[0]
            != self.idents.shape[0]
            != self.indices.shape[0]
        ):
            rprint(f"{self.freqs.shape[0]=}")
            rprint(f"{self.powers.shape[0]=}")
            rprint(f"{self.idents.shape[0]=}")
            rprint(f"{self.indices.shape[0]=}")
            raise GridDataMismatch("Data shapes do not match!")

        if self.times.shape[0] < unique(self.indices).shape[0]:
            rprint(f"{self.times.shape[0]=}")
            rprint(f"{unique(self.indices).shape[0]=}")
            raise GridDataMismatch(
                "Number of times is less than number of unique indices"
            )

    def __repr__(self) -> str:
        return f"WavetrackerData({self.path})"

    def __str__(self) -> str:
        return f"WavetrackerData({self.path})"


class RawData:
    """
    Loads the raw dataset (real: `traces-grid1.raw`, simulated: `raw.npy`)
    into an easy to use class format.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist.")

        self._load_data()
        self._check_health()

    def _load_data(self) -> None:
        if pathlib.Path(self.path / "raw.npy").exists():
            self.raw = load(self.path / "raw.npy")
            self.samplerate = 20000
            self.channels = self.raw.shape[1]
        elif pathlib.Path(self.path / "traces-grid1.raw").exists():
            self.raw = DataLoader(str(self.path) + "/traces-grid1.raw", 60)
            self.samplerate = self.raw.samplerate
            self.channels = self.raw.shape[1]
        else:
            raise FileNotFoundError(f"Could not find raw data in {self.path}.")

    def _check_health(self) -> None:
        if self.raw.shape[0] == 0:
            raise GridDataMismatch("Raw data has zero length.")

    def __repr__(self) -> str:
        return f"RawData({self.path})"

    def __str__(self) -> str:
        return f"RawData({self.path})"


class ChirpData:
    """
    Loads the chirp times and chirp ids into an easy to use class format.
    The chirp times are the times at which the chirps were detected, and the
    chirp ids are the fish that were detected at that time.
    Files must have the format `chirp_times_{detector}.npy` and
    `chirp_ids_{detector}.npy` where `detector` is one of `gp` (Grosspraktikum),
    `gt` (ground truth) or `cnn` (cnn-chirpdetector).
    """

    def __init__(self, path: pathlib.Path, detector: str) -> None:
        assert detector in [
            "gp",
            "gt",
            "cnn",
        ], "Detector must be one of 'gp', 'gt' or 'cnn'"
        self.path = path
        self.detector = detector
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist.")

        self._load_data()
        self._check_health()

    def get_fish(self, fish: int) -> ndarray:
        """
        Returns the chirp times for a single fish.

        Parameters
        ----------
        fish : int
            The ID of the fish to extract.

        Returns
        -------
        ndarray
            The chirp times for the fish specified.
        """
        times = self.times[self.idents == fish]
        return times

    def _load_data(self) -> None:
        self.times = load(self.path / f"chirp_times_{self.detector}.npy")
        self.idents = load(self.path / f"chirp_ids_{self.detector}.npy")

    def _check_health(self) -> None:
        if self.times.shape[0] != self.idents.shape[0]:
            raise GridDataMismatch(
                f"Times and idents do not match: {self.times.shape[0]}, {self.idents.shape[0]}"
            )

    def __repr__(self) -> str:
        return f"ChirpData({self.path})"

    def __str__(self) -> str:
        return f"ChirpData({self.path})"


@dataclass
class Dataset:
    """
    The main dataset class to load data extracted from electrode grid recordings
    of wave-type weakly electric fish. Every dataset must at least get a path
    to a wavetracker dataset. Optionally, a raw dataset and/or a chirp dataset
    can be provided. The raw dataset can be used to extract e.g. the chirp times
    from the raw data.
    """

    path: pathlib.Path
    track: WavetrackerData
    rec: Optional[RawData] = None
    chirp: Optional[ChirpData] = None

    def __post_init__(self) -> None:
        self._check_type()

    def _check_type(self) -> None:
        assert isinstance(
            self.track, WavetrackerData
        ), "track must be a WavetrackerData object."
        if self.rec is not None:
            assert isinstance(
                self.rec, RawData
            ), "raw must be a RawData object."
        if self.chirp is not None:
            assert isinstance(
                self.chirp, ChirpData
            ), "chirp must be a ChirpData object."

    def __repr__(self) -> str:
        return f"Dataset({self.track}, {self.rec}, {self.chirp})"

    def __str__(self) -> str:
        return f"Dataset({self.track}, {self.rec}, {self.chirp})"
