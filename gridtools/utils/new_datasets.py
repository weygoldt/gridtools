#!/usr/bin/env python3

"""
Grid dataset classes using composition instead of inheritance.
"""

import pathlib
from dataclasses import dataclass
from typing import Optional

from numpy import array, isnan, load, ndarray, save, unique
from rich import print
from thunderfish.dataloader import DataLoader

from ..exceptions.exceptions import GridDataMismatch


class WavetrackerData:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist.")

        self._load_data()
        self._check_health()

    def get_fish(self, fish: int) -> ndarray:
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
            raise GridDataMismatch(
                f"Data shapes do not match: {self.freqs.shape[0]}, {self.powers.shape[0]}, {self.idents.shape[0]}, {self.indices.shape[0]}"
            )

        if self.times.shape[0] < unique(self.indices).shape[0]:
            raise GridDataMismatch(
                f"Number of times ({self.times.shape[0]}) is less than number of unique indices ({unique(self.indices).shape[0]})"
            )

    def __repr__(self) -> str:
        return f"WavetrackerData({self.path})"

    def __str__(self) -> str:
        return f"WavetrackerData({self.path})"


class RawData:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist.")

        self._load_data()
        self._check_health()

    def _load_data(self) -> None:
        if pathlib.Path(self.path / "raw.npy").exists():
            self.raw = load(self.path / "raw.npy")
            self.samplerate = 20000  # TODO: get from metadata
            self.channels = self.raw.shape[1]
        elif pathlib.Path(self.path / "traces-grid1.raw").exists():
            self.raw = DataLoader(self.path, 60)
            self.samplerate = self.raw.samplerate
            self.channels = self.raw.shape[1]
        else:
            raise FileNotFoundError(f"Could not find raw data in {self.path}.")

    def _check_health(self) -> None:
        if self.raw.shape[0] == 0:
            raise GridDataMismatch(f"Raw data has zero length.")

    def __repr__(self) -> str:
        return f"RawData({self.path})"

    def __str__(self) -> str:
        return f"RawData({self.path})"


class ChirpData:
    def __init__(self, path: pathlib.Path, detector: str) -> None:
        assert detector in [
            "gp",
            "gt",
            "cnn",
            "yolo",
        ], "Detector must be one of 'gp', 'gt', 'cnn', or 'yolo'."
        self.path = path
        self.detector = detector
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist.")

        self._load_data()
        self._check_health()

    def get_fish(self, fish: int) -> ndarray:
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
