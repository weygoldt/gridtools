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
from typing import Optional, Union

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)
from thunderfish.dataloader import DataLoader

from .exceptions import GridDataMismatch

# def save(dataset, output_path: pathlib.Path) -> None:
#     """Save a Dataset object to disk

#     Parameters
#     ----------
#     dataset : Dataset
#         Dataset to save to file.
#     output_path : pathlib.Path
#         Path where to save the dataset.

#     Raises
#     ------
#     FileExistsError
#         When there already is a dataset.
#     """
#     output_dir = output_path / dataset.path.name

#     if output_dir.exists():
#         raise FileExistsError(f"Output directory {output_dir} already exists.")

#     output_dir.mkdir(parents=True)

#     np.save(output_dir / "fund_v.npy", dataset.track.freqs)
#     np.save(output_dir / "sign_v.npy", dataset.track.powers)
#     np.save(output_dir / "ident_v.npy", dataset.track.idents)
#     np.save(output_dir / "idx_v.npy", dataset.track.indices)
#     np.save(output_dir / "times.npy", dataset.track.times)

#     np.save(output_dir / "raw.npy", dataset.rec.raw)

#     if dataset.chirp is not None:
#         np.save(output_dir / "chirp_times_gt.npy", dataset.chirp.times)
#         np.save(output_dir / "chirp_ids_gt.npy", dataset.chirp.idents)
#     if dataset.rise is not None:
#         np.save(output_dir / "rise_times_gt.npy", dataset.rise.times)
#         np.save(output_dir / "rise_ids_gt.npy", dataset.rise.idents)


# def subset(
#     input_path: pathlib.Path,
#     output_path: pathlib.Path,
#     start: float,
#     stop: float,
# ):
#     """Creates and saves a subset of a dataset.

#     Parameters
#     ----------
#     input_path : pathlib.Path
#         Where the original dataset is
#     output_path : pathlib.Path
#         Where the subset should go
#     start : float
#         Where to start the subset in seconds
#     stop : float
#         Where to stop the subset in seconds

#     Raises
#     ------
#     GridDataMismatch
#         When the start and stop times are not in the dataset.
#     """
#     assert start < stop, "Start time must be smaller than stop time."

#     wt = WavetrackerData(input_path)
#     raw = RawData(input_path)

#     assert start > wt.times[0], "Start time must be larger than the beginning."
#     assert stop < wt.times[-1], "Stop time must be smaller than the end."

#     # check if there are chirps and use ground truth (gt) if available
#     if len(list(input_path.glob("chirp_times_*"))) > 0:
#         if (input_path / "chirp_times_gt.npy").exists():
#             chirps = ChirpData(input_path, "gt")
#         else:
#             chirps = ChirpData(input_path, "cnn")
#     else:
#         chirps = None

#     # check for rise times and use ground truth if available
#     if len(list(input_path.glob("rise_times_*"))) > 0:
#         if (input_path / "rise_times_gt.npy").exists():
#             rises = RiseData(input_path, "gt")
#         else:
#             rises = RiseData(input_path, "pd")
#     else:
#         rises = None

#     # construct dataset object
#     ds = Dataset(input_path, wt, raw, chirps, rises)

#     # estimate the start and stop as indices to get the raw data
#     start_idx = int(start * ds.rec.samplerate)
#     stop_idx = int(stop * ds.rec.samplerate)
#     raw = ds.rec.raw[start_idx:stop_idx, :]

#     tracks = []
#     powers = []
#     indices = []
#     idents = []

#     for track_id in np.unique(ds.track.idents[~np.isnan(ds.track.idents)]):
#         track = ds.track.freqs[ds.track.idents == track_id]
#         power = ds.track.powers[ds.track.idents == track_id]
#         time = ds.track.times[ds.track.indices[ds.track.idents == track_id]]
#         index = ds.track.indices[ds.track.idents == track_id]

#         track = track[(time >= start) & (time <= stop)]
#         power = power[(time >= start) & (time <= stop)]
#         index = index[(time >= start) & (time <= stop)]
#         ident = np.repeat(track_id, len(track))

#         tracks.append(track)
#         powers.append(power)
#         indices.append(index)
#         idents.append(ident)

#     tracks = np.concatenate(tracks)
#     powers = np.concatenate(powers)
#     indices = np.concatenate(indices)
#     idents = np.concatenate(idents)
#     time = ds.track.times[ds.track.times >= start & ds.track.times <= stop]

#     if len(indices) == 0:
#         raise GridDataMismatch("No data in the specified time range.")
#     else:
#         indices -= indices[0]

#     # reconstruct dataset
#     wt.freqs = tracks
#     wt.powers = powers
#     wt.idents = idents
#     wt.indices = indices
#     wt.ids = np.unique(idents)
#     wt.times = time
#     raw.raw = raw

#     # extract chirps
#     if chirps is not None:
#         chirp_ids = chirps.idents[
#             (chirps.times >= start) & (chirps.times <= stop)
#         ]
#         chirp_times = chirps.times[
#             (chirps.times >= start) & (chirps.times <= stop)
#         ]
#         chirps.times = chirp_times
#         chirps.idents = chirp_ids
#     if rises is not None:
#         rise_ids = rises.idents[(rises.times >= start) & (rises.times <= stop)]
#         rise_times = rises.times[(rises.times >= start) & (rises.times <= stop)]
#         rises.times = rise_times
#         rises.idents = rise_ids

#     # rebuild dataset
#     subset_ds = Dataset(output_path, wt, raw, chirps, rises)

#     save(subset_ds, output_path)


def load_wavetracker(path: pathlib.Path) -> "WavetrackerData":
    """
    Dataloader for the arrays produced by the wavetracker and
    other data extracted from them, such as position estimates.
    """

    files = list(path.glob("*"))
    if not any(["fund_v.npy" in str(f) for f in files]):
        raise FileNotFoundError("No wavetracker dataset found!")

    return WavetrackerData(
        freqs=np.load(path / "fund_v.npy"),
        powers=np.load(path / "sign_v.npy"),
        idents=np.load(path / "ident_v.npy"),
        indices=np.load(path / "idx_v.npy"),
        times=np.load(path / "times.npy"),
        ids=np.unique(np.load(path / "ident_v.npy")),
        xpos=np.load(path / "xpos.npy") if "xpos.npy" in files else None,
        ypos=np.load(path / "ypos.npy") if "ypos.npy" in files else None,
    )


def load_raw(path: pathlib.Path) -> "RawData":
    """
    Dataloader for a raw dataset, such as the "traces-grid1.raw" files.
    Uses the thunderfish dataloader to easily access large binary files.
    """

    files = list(path.glob("*"))
    if not any("raw.npy" in str(f) for f in files):
        raise FileNotFoundError("No raw dataset found!")
    return RawData(
        samplerate=DataLoader(path / "raw.npy").samplerate,
        raw=DataLoader(path / "raw.npy"),
    )


def load_com(path: pathlib.Path) -> "CommunicationData":
    """
    Dataloader for communication data, such as chirps and rises, if they
    are available.
    """
    files = list(path.glob("*"))
    if any("chirp_times_" in str(f) for f in files):
        if any("chirp_times_gt.npy" in str(f) for f in files):
            det = "gt"
        elif any("chirp_times_cnn.npy" in str(f) for f in files):
            det = "cnn"
        else:
            raise FileNotFoundError(
                "No chirp dataset with correct detector handle found!"
            )
        chps = ChirpData(
            times=np.load(path / f"chirp_times_{det}.npy"),
            idents=np.load(path / f"chirp_ids_{det}.npy").astype(int),
            detector=det,
        )
    else:
        chps = None

    if any("rise_times_" in str(f) for f in files):
        if any("rise_times_gt.npy" in str(f) for f in files):
            det = "gt"
        elif any("rise_times_pd.npy" in str(f) for f in files):
            det = "pd"
        else:
            raise FileNotFoundError(
                "No rise dataset with correct detector handle found!"
            )
        ris = RiseData(
            times=np.load(path / f"rise_times_{det}.npy"),
            idents=np.load(path / f"rise_ids_{det}.npy").astype(int),
            detector=det,
        )
    else:
        ris = None

    return CommunicationData(chirp=chps, rise=ris)


def load(path: pathlib.Path, raw: bool = False) -> "Dataset":
    """
    Meta-dataloader that loads all the data from a dataset and builds a
    Dataset object, in which raw data, wavetracker data and communication
    data are stored.
    """
    ds = Dataset(
        path=path,
        rec=load_raw(path) if raw else None,
        track=load_wavetracker(path),
        com=load_com(path),
    )
    return ds


class WavetrackerData(BaseModel):
    """
    Contains data extracted by the wavetracker.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    freqs: np.ndarray[float]
    powers: np.ndarray[float]
    idents: np.ndarray[float]
    indices: np.ndarray[int]
    ids: np.ndarray[int]
    times: np.ndarray[float]
    xpos: Optional[np.ndarray[float]] = None
    ypos: Optional[np.ndarray[float]] = None

    @field_validator("freqs", "powers", "idents", "indices", "ids", "times")
    @classmethod
    def check_numpy_array(cls, v):
        """
        Check if all the arrays are numpy arrays.
        """
        if not isinstance(v, np.ndarray):
            raise ValidationError("Value must be a numpy.ndarray")
        return v

    @field_validator("xpos", "ypos")
    @classmethod
    def check_numpy_array_pos(cls, v):
        """
        Check if xpos and ypos are numpy arrays when they are not none.
        """
        if v is not None and not isinstance(v, np.ndarray):
            raise ValidationError("Value must be a numpy.ndarray")
        return v

    # @field_validator("times")
    # @classmethod
    # def check_times_indices(cls, v, values):
    #     """
    #     Checks that the indices in the indices array cannot go out of bounds
    #     of the times array.
    #     """
    #     if v.shape[0] < len(set(values["indices"])):
    #         raise GridDataMismatch(
    #             "Number of times is smaller than number of unique indices!"
    #         )
    #     return v

    @field_validator("times")
    @classmethod
    def check_times_sorted(cls, v):
        """
        Checks that the times are monotonically increasing.
        """
        if not np.all(np.diff(v) > 0):
            raise GridDataMismatch(
                "Wavetracker times are not monotonically increasing!"
            )
        return v

    @model_validator(mode="after")
    def check_wavetracker_data(self) -> "WavetrackerData":
        """
        Check if the wavetracker data is of equal length.
        """
        lengths = [
            len(x) for x in [self.freqs, self.powers, self.idents, self.indices]
        ]
        if len(set(lengths)) > 1:
            raise GridDataMismatch("Wavetracker data is not of equal length!")
        return self


class RawData(BaseModel):
    """
    Contains raw data from the electrode grid recordings.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    samplerate: int
    raw: Union[np.ndarray, DataLoader]

    @field_validator("raw")
    @classmethod
    def check_raw(cls, v):
        """
        Checks if the raw data is a numpy array or a DataLoader object.
        """
        if not isinstance(v, (np.ndarray, DataLoader)):
            raise ValidationError(
                "Raw data must be a numpy array or a DataLoader."
            )
        return v

    @field_validator("samplerate")
    @classmethod
    def check_samplerate(cls, v):
        """
        Checks if the samplerate is a positive integer.
        """
        if not isinstance(v, int):
            raise ValidationError("Samplerate must be an integer.")
        if v <= 0:
            raise ValidationError("Samplerate must be a positive integer.")
        return v


class ChirpData(BaseModel):
    """
    Contains data about chirps produced by fish in the dataset.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    times: np.ndarray
    idents: np.ndarray
    detector: str

    @field_validator("times", "idents")
    @classmethod
    def check_numpy_array(cls, v):
        """
        Check if times and idents are numpy arrays.
        """
        if not isinstance(v, np.ndarray):
            raise ValidationError("Value must be a numpy.ndarray")
        return v

    @field_validator("detector")
    @classmethod
    def check_detector(cls, v):
        """
        Check if detector is either 'gt' or 'cnn' and a string.
        """
        if not isinstance(v, str):
            raise ValidationError("Detector must be a string.")
        if v not in ["gt", "cnn"]:
            raise ValidationError("Detector must be 'gt' or 'cnn'.")
        return v


class RiseData(BaseModel):
    """
    Contains data of rises produced by fish in the dataset.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    times: np.ndarray
    idents: np.ndarray
    detector: str

    @field_validator("times", "idents")
    @classmethod
    def check_numpy_array(cls, v):
        """
        Check if times and idents are numpy arrays.
        """
        if not isinstance(v, np.ndarray):
            raise ValidationError("Value must be a numpy.ndarray")
        return v

    @field_validator("detector")
    @classmethod
    def check_detector(cls, v):
        """
        Check if detector is either 'gt' or 'pd' and a string.
        """
        if not isinstance(v, str):
            raise ValidationError("Detector must be a string.")
        if v not in ["gt", "pd"]:
            raise ValidationError("Detector must be 'gt' or 'pd'.")
        return v


class CommunicationData(BaseModel):
    """
    Contains data for communication signals produced by fish in the dataset.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chirp: Optional[ChirpData] = None
    rise: Optional[RiseData] = None

    @field_validator("chirp")
    @classmethod
    def typecheck_chirp(cls, v):
        """
        Check if chirp data is a ChirpData object if it is not none.
        """
        if v is not None and not isinstance(v, ChirpData):
            raise ValidationError("Chirp data must be a ChirpData object.")
        return v

    @field_validator("rise")
    @classmethod
    def typecheck_rise(cls, v):
        """
        Check if rise data is a RiseData object if it is not none.
        """
        if v is not None and not isinstance(v, RiseData):
            raise ValidationError("Rise data must be a RiseData object.")
        return v

    @model_validator(mode="after")
    def check_communication_data(self):
        """Check if chirp or rise data is provided. Class should
        not be instantiated when no data is provided."""

        if self.chirp is None and self.rise is None:
            raise ValidationError(
                "Communication data must contain either chirp or rise data."
            )
        return self


class Dataset(BaseModel):
    """
    The main dataset class to load data extracted from electrode grid recordings
    of wave-type weakly electric fish. Every dataset must at least get a path
    to a wavetracker dataset. Optionally, a raw dataset and/or a chirp dataset
    can be provided. The raw dataset can be used to extract e.g. the chirp times
    from the raw data.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: pathlib.Path
    rec: Optional[RawData] = None
    track: WavetrackerData
    com: Optional[CommunicationData] = None

    @field_validator("path")
    @classmethod
    def check_path(cls, v):
        """
        Check if path is a pathlib.Path object.
        """
        if not isinstance(v, pathlib.Path):
            raise ValueError("Path must be a pathlib.Path object.")
        return v

    @field_validator("rec")
    @classmethod
    def check_rec(cls, v):
        """
        Check if raw data is a RawData object or none.
        """
        if v is not None and not isinstance(v, RawData):
            raise ValueError("Raw data must be a RawData object.")
        return v

    @field_validator("track")
    @classmethod
    def check_track(cls, v):
        """
        Check if wavetracker data is a WavetrackerData object.
        """
        if v is not None and not isinstance(v, WavetrackerData):
            raise ValueError(
                "Wavetracker data must be a WavetrackerData object."
            )
        return v

    @field_validator("com")
    @classmethod
    def check_com(cls, v):
        """
        Check if communication data is a CommunicationData object or none.
        """
        if v is not None and not isinstance(v, CommunicationData):
            raise ValueError(
                "Communication data must be a CommunicationData object."
            )
        return v
