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

import numpy as np
from numpy import isnan, load, ndarray, unique
from pydantic import BaseModel
from rich import print as rprint
from thunderfish.dataloader import DataLoader

from .exceptions import GridDataMismatch


def save(dataset, output_path: pathlib.Path) -> None:
    """Save a Dataset object to disk

    Parameters
    ----------
    dataset : Dataset
        Dataset to save to file.
    output_path : pathlib.Path
        Path where to save the dataset.

    Raises
    ------
    FileExistsError
        When there already is a dataset.
    """
    output_dir = output_path / dataset.path.name

    if output_dir.exists():
        raise FileExistsError(f"Output directory {output_dir} already exists.")

    output_dir.mkdir(parents=True)

    np.save(output_dir / "fund_v.npy", dataset.track.freqs)
    np.save(output_dir / "sign_v.npy", dataset.track.powers)
    np.save(output_dir / "ident_v.npy", dataset.track.idents)
    np.save(output_dir / "idx_v.npy", dataset.track.indices)
    np.save(output_dir / "times.npy", dataset.track.times)

    np.save(output_dir / "raw.npy", dataset.rec.raw)

    if dataset.chirp is not None:
        np.save(output_dir / "chirp_times_gt.npy", dataset.chirp.times)
        np.save(output_dir / "chirp_ids_gt.npy", dataset.chirp.idents)
    if dataset.rise is not None:
        np.save(output_dir / "rise_times_gt.npy", dataset.rise.times)
        np.save(output_dir / "rise_ids_gt.npy", dataset.rise.idents)


def subset(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    start: float,
    stop: float,
):
    """Creates and saves a subset of a dataset.

    Parameters
    ----------
    input_path : pathlib.Path
        Where the original dataset is
    output_path : pathlib.Path
        Where the subset should go
    start : float
        Where to start the subset in seconds
    stop : float
        Where to stop the subset in seconds

    Raises
    ------
    GridDataMismatch
        When the start and stop times are not in the dataset.
    """
    assert start < stop, "Start time must be smaller than stop time."

    wt = WavetrackerData(input_path)
    raw = RawData(input_path)

    assert start > wt.times[0], "Start time must be larger than the beginning."
    assert stop < wt.times[-1], "Stop time must be smaller than the end."

    # check if there are chirps and use ground truth (gt) if available
    if len(list(input_path.glob("chirp_times_*"))) > 0:
        if (input_path / "chirp_times_gt.npy").exists():
            chirps = ChirpData(input_path, "gt")
        else:
            chirps = ChirpData(input_path, "cnn")
    else:
        chirps = None

    # check for rise times and use ground truth if available
    if len(list(input_path.glob("rise_times_*"))) > 0:
        if (input_path / "rise_times_gt.npy").exists():
            rises = RiseData(input_path, "gt")
        else:
            rises = RiseData(input_path, "pd")
    else:
        rises = None

    # construct dataset object
    ds = Dataset(input_path, wt, raw, chirps, rises)

    # estimate the start and stop as indices to get the raw data
    start_idx = int(start * ds.rec.samplerate)
    stop_idx = int(stop * ds.rec.samplerate)
    raw = ds.rec.raw[start_idx:stop_idx, :]

    tracks = []
    powers = []
    indices = []
    idents = []

    for track_id in np.unique(ds.track.idents[~np.isnan(ds.track.idents)]):
        track = ds.track.freqs[ds.track.idents == track_id]
        power = ds.track.powers[ds.track.idents == track_id]
        time = ds.track.times[ds.track.indices[ds.track.idents == track_id]]
        index = ds.track.indices[ds.track.idents == track_id]

        track = track[(time >= start) & (time <= stop)]
        power = power[(time >= start) & (time <= stop)]
        index = index[(time >= start) & (time <= stop)]
        ident = np.repeat(track_id, len(track))

        tracks.append(track)
        powers.append(power)
        indices.append(index)
        idents.append(ident)

    tracks = np.concatenate(tracks)
    powers = np.concatenate(powers)
    indices = np.concatenate(indices)
    idents = np.concatenate(idents)
    time = ds.track.times[ds.track.times >= start & ds.track.times <= stop]

    if len(indices) == 0:
        raise GridDataMismatch("No data in the specified time range.")
    else:
        indices -= indices[0]

    # reconstruct dataset
    wt.freqs = tracks
    wt.powers = powers
    wt.idents = idents
    wt.indices = indices
    wt.ids = np.unique(idents)
    wt.times = time
    raw.raw = raw

    # extract chirps
    if chirps is not None:
        chirp_ids = chirps.idents[
            (chirps.times >= start) & (chirps.times <= stop)
        ]
        chirp_times = chirps.times[
            (chirps.times >= start) & (chirps.times <= stop)
        ]
        chirps.times = chirp_times
        chirps.idents = chirp_ids
    if rises is not None:
        rise_ids = rises.idents[(rises.times >= start) & (rises.times <= stop)]
        rise_times = rises.times[(rises.times >= start) & (rises.times <= stop)]
        rises.times = rise_times
        rises.idents = rise_ids

    # rebuild dataset
    subset_ds = Dataset(output_path, wt, raw, chirps, rises)

    save(subset_ds, output_path)


class WavetrackerData(BaseModel):
    """
    Loads the `.npy` files produced by the `wavetracker` into an easy to use
    class format. Provides methods to extract single or specific fish. Checks
    data set health (demension mismatches, etc.) before returning the object.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist.")

        self.freqs = load(self.path / "fund_v.npy")
        self.powers = load(self.path / "sign_v.npy")
        self.idents = load(self.path / "ident_v.npy")
        self.indices = load(self.path / "idx_v.npy")
        self.ids = unique(self.idents[~isnan(self.idents)]).astype(int)
        self.times = load(self.path / "times.npy")
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


class RawData(BaseModel):
    """
    Loads the raw dataset (real: `traces-grid1.raw`, simulated: `raw.npy`)
    into an easy to use class format.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist.")

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
        self._check_health()

    def _check_health(self) -> None:
        if self.raw.shape[0] == 0:
            raise GridDataMismatch("Raw data has zero length.")

    def __repr__(self) -> str:
        return f"RawData({self.path})"

    def __str__(self) -> str:
        return f"RawData({self.path})"


class ChirpData(BaseModel):
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

        self.times = load(self.path / f"chirp_times_{self.detector}.npy")
        self.idents = load(self.path / f"chirp_ids_{self.detector}.npy")
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

    def _check_health(self) -> None:
        if self.times.shape[0] != self.idents.shape[0]:
            raise GridDataMismatch(
                f"Times and idents do not match: {self.times.shape[0]}, {self.idents.shape[0]}"
            )

    def __repr__(self) -> str:
        return f"ChirpData({self.path})"

    def __str__(self) -> str:
        return f"ChirpData({self.path})"


class RiseData(BaseModel):
    """
    Loads rise times and rise identities into an easy to use class format.
    The rise times are the times at which the rises were detected, and the
    rise ids are the fish that were detected at that time. Files must have the
    format `rise_times_{detector}.npy` and `rise_ids_{detector}.npy` where
    `detector` is either `pd` (peakdetection) or `gt` (ground truth).
    """

    def __init__(self, path: pathlib.Path, detector: str) -> None:
        assert detector in [
            "pd",
            "gt",
        ], "Detector must be one of 'pd' or 'gt'"
        self.path = path
        self.detector = detector
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist.")

        self.times = load(self.path / f"rise_times_{self.detector}.npy")
        self.idents = load(self.path / f"rise_ids_{self.detector}.npy")
        self._check_health()

    def get_fish(self, fish: int) -> ndarray:
        """
        Returns the rise times for a single fish.

        Parameters
        ----------
        fish : int
            The ID of the fish to extract.

        Returns
        -------
        ndarray
            The rise times for the fish specified.
        """
        times = self.times[self.idents == fish]
        return times

    def _check_health(self) -> None:
        if self.times.shape[0] != self.idents.shape[0]:
            raise GridDataMismatch(
                f"Times and idents do not match: {self.times.shape[0]}, {self.idents.shape[0]}"
            )

    def __repr__(self) -> str:
        return f"RiseData({self.path})"

    def __str__(self) -> str:
        return f"RiseData({self.path})"


class Dataset(BaseModel):
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
    rise: Optional[RiseData] = None

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
