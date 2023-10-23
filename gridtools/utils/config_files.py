#!/usr/bin/env python3

"""
A module that keeps common tools to handle files.
"""

import pathlib
from typing import Union

import numpy as np
import yaml
from rich import print as rprint
from thunderfish.dataloader import DataLoader


def todict(obj, classkey=None):
    """Recursively convert an object into a dictionary.

    Parameters
    ----------
    obj : _object_
        Some object to convert into a dictionary.
    classkey : str, optional
        The key to that should be converted. If None,
        converts everything in the object. By default None

    Returns
    -------
    dict
        The converted dictionary.
    """
    if isinstance(obj, dict):
        data = {}
        for k, v in obj.items():
            data[k] = todict(v, classkey)
        return data
    elif hasattr(obj, "_ast"):
        return todict(obj._ast())
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        return [todict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict(
            [
                (key, todict(value, classkey))
                for key, value in obj.__dict__.items()
                if not callable(value) and not key.startswith("_")
            ]
        )
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj


class Config:
    """
    Class to recursively load a YAML file and access its contents using
    dot notation.

    Parameters
    ----------
    config_file : str
        The path to the YAML file to load.

    Attributes
    ----------
    <key> : Any
        The value associated with the specified key in the loaded YAML file. If the value is
        a dictionary, it will be recursively converted to another `Config` object and accessible
        as a subclass attribute.
    """

    def __init__(self, config_file: Union[str, dict]) -> None:
        """
        Load the YAML file and convert its keys to class attributes.
        """

        if isinstance(config_file, dict):
            config_dict = config_file
        else:
            with open(config_file, "r") as f:
                config_dict = yaml.safe_load(f)
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        """
        Return a string representation of the `Config` object.
        """
        return f"Config({vars(self)})"

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the `Config` object.
        """
        return str(vars(self))

    def pprint(self) -> None:
        """
        Pretty print the `Config` object.
        """
        rprint(todict(self))


# class WaveTrackerDataset:
#     def __init__(self, datapath: pathlib.Path) -> None:
#         if not datapath.is_dir():
#             raise NotADirectoryError(f"{datapath} is not a directory")

#         self.path = datapath
#         self.date = datapath.name

#         if pathlib.Path(datapath / "raw.npy").exists():
#             self.raw = np.load(datapath / "raw.npy", allow_pickle=True)
#             self.samplerate = 20000.0
#         elif pathlib.Path(datapath / "traces-grid1.raw").exists():
#             self.raw = DataLoader(
#                 datapath / "traces-grid1.raw", 60.0, 0, channel=-1
#             )
#             self.samplerate = self.raw.samplerate
#         else:
#             raise FileNotFoundError(
#                 f"Could not find raw data file in {datapath}"
#             )

#         self.n_electrodes = self.raw.shape[1]

#         self.track_times = np.load(datapath / "times.npy", allow_pickle=True)
#         self.track_freqs = np.load(datapath / "fund_v.npy", allow_pickle=True)
#         self.track_indices = np.load(datapath / "idx_v.npy", allow_pickle=True)
#         self.track_idents = np.load(datapath / "ident_v.npy", allow_pickle=True)
#         self.track_powers = np.load(datapath / "sign_v.npy", allow_pickle=True)
#         self.ids = np.unique(self.track_idents[~np.isnan(self.track_idents)])

#     def save(self):
#         np.save(self.path / "times.npy", self.track_times)
#         np.save(self.path / "fund_v.npy", self.track_freqs)
#         np.save(self.path / "idx_v.npy", self.track_indices)
#         np.save(self.path / "ident_v.npy", self.track_idents)
#         np.save(self.path / "sign_v.npy", self.track_powers)
#         np.save(self.path / "raw.npy", self.raw)

#     def __repr__(self) -> str:
#         return f"WaveTrackerDataset({self.file})"

#     def __str__(self) -> str:
#         return f"WaveTrackerDataset({self.file})"


# class SimulatedDataset(WaveTrackerDataset):
#     def __init__(self, datapath: pathlib.Path) -> None:
#         super().__init__(datapath)

#         self.chirp_times = np.load(datapath / "chirp_times_gt.npy")
#         self.chirp_ids = np.load(datapath / "chirp_ids_gt.npy")
#         self.rise_times = np.load(datapath / "rise_times_gt.npy")
#         self.rise_ids = np.load(datapath / "rise_ids_gt.npy")

#     def __repr__(self) -> str:
#         return f"SimulatedDataset({self.file})"

#     def __str__(self) -> str:
#         return f"SimulatedDataset({self.file})"
