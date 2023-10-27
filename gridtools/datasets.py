#!/usr/bin/env python3

"""
Classes and functions to load, work with and save data associated with 
electrode grid recordings of wave-type weakly electric fish.

The main class is the Dataset class, which is able to load all data extracted
by the `wavetracker` as well as raw data, communication signals, etc.

The main functionalities include the following:
- gridtools.datasets.load: Load a dataset from a given path.
- gridtools.datasets.save: Save a dataset to disk.
- gridtools.datasets.subset: Make a subset of a dataset.

The Dataset class is a composition of the WavetrackerData, GridData and
CommunicationData classes. This way, the user can choose which data to load
and which not to load. This design might seem complicated at first, but it
allows for a lot of flexibility and extensibility. For example, if in the
future, better detectors are available, or other kinds of data is extracted,
they can easily be added to the Dataset class, without breaking the existing
code.
"""

import argparse
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
from rich.pretty import pprint as rpprint

from thunderfish.dataloader import DataLoader
from thunderfish.datawriter import write_data

from .exceptions import GridDataMismatch


def load_wavetracker(path: Union[pathlib.Path, str]) -> "WavetrackerData":
    """
    Load data produced by the wavetracker and other data extracted from them,
    such as position estimates.

    Parameters
    ----------
    path : pathlib.Path
        Path to the directory containing the data files.

    Returns
    -------
    WavetrackerData
        An instance of the WavetrackerData class containing the loaded data.

    Raises
    ------
    FileNotFoundError
        If no wavetracker dataset is found in the specified directory.

    Examples
    --------
    >>> import pathlib
    >>> from gridtools.datasets import load_wavetracker
    >>> wt = load_wavetracker(pathlib.Path("path/to/wavetracker"))
    """

    if isinstance(path, str):
        path = pathlib.Path(path)

    files = list(path.glob("*"))

    if not any("fund_v.npy" in str(f) for f in files):
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


def load_grid(path: Union[pathlib.Path, str]) -> "GridData":
    """
    Load a raw dataset from a given path.

    Parameters
    ----------
    path : pathlib.Path
        The path to the directory containing the raw dataset.

    Returns
    -------
    GridData
        An object containing the loaded raw dataset.

    Raises
    ------
    FileNotFoundError
        If no raw dataset is found in the given directory.

    Notes
    -----
    This function uses the thunderfish dataloader to easily access large binary files.
    The function checks if the directory contains a "traces*" file.
    The function is tested using .raw and .wav files.
    If neither file is found, a FileNotFoundError is raised.
    If a "traces-grid1.raw" file is found, it is loaded using the thunderfish DataLoader.
    The function returns a GridData object containing the loaded raw dataset.
    Instead of directly passing the DataLoader object, I wrap it in this class
    to later be able to add metadata such as electrode positions, etc. to the
    recording class.

    Examples
    --------
    >>> import pathlib
    >>> from gridtools.datasets load_grid
    >>> rec = load_grid(pathlib.Path("path/to/raw"))
    """

    if isinstance(path, str):
        path = pathlib.Path(path)

    files = list(path.glob("traces*"))

    if len(files) == 0:
        raise FileNotFoundError("No raw dataset found!")
    if len(files) > 1:
        raise FileNotFoundError("More than one raw dataset found! Check path.")

    file = files[0]
    rec = DataLoader(str(path / file.name))

    return GridData(rec=rec, samplerate=rec.samplerate)


def load_com(path: Union[pathlib.Path, str]) -> "CommunicationData":
    """
    Load communication data from a given path. Loads chirps if available, loads
    rises if available, or loads both if available. If no communication data is
    available, returns None.

    Data on disk must follow the following naming convention:
    - {signal type}_times_{detector handle}.npy
    - {signal type}_ids_{detector handle}.npy

    For example, chirp data is loaded from the following files:
    - chirp_times_gt.npy
    - chirp_ids_gt.npy
    - chirp_times_cnn.npy
    - chirp_ids_cnn.npy

    If no detector handle is found, the data is not loaded. If in the future,
    better detectors are available, they need to be added to this function.

    Parameters
    ----------
    path : pathlib.Path
        The path to the directory containing the communication data.

    Returns
    -------
    CommunicationData
        An object containing the loaded communication data.

    Raises
    ------
    FileNotFoundError
        If no chirp or rise dataset with the correct detector handle is found.

    Examples
    --------
    >>> import pathlib
    >>> from gridtools.datasets import load_com
    >>> com = load_com(pathlib.Path("path/to/communication"))
    >>> # get rise times and ids
    >>> com.rise.times
    >>> com.rise.idents
    """

    if isinstance(path, str):
        path = pathlib.Path(path)

    files = list(path.glob("*"))

    # Load chirp data if available
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

    # Load rise data if available
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


def load(path: Union[pathlib.Path, str], grid: bool = False) -> "Dataset":
    """
    Load all data from a dataset and build a Dataset object. A dataset object
    contains at least all data produces by the wavetracker and optionally, also
    the raw data, communication signals or position estimates.

    A dataset is not just a dataclass but a data model: Upon instantiation, the
    data is checked for consistency and errors are raised if the data is
    inconsistent.

    the Dataset model is a composition of the WavetrackerData, GridData and
    CommunicationData models. This way, the user can choose which data to load
    and which not to load. It is also easily extensible to other data types,
    e.g. rises or behaviour data, if this is needed in the future.

    Parameters
    ----------
    path : pathlib.Path
        The path to the dataset.
    raw : bool, optional
        Whether to load the raw data or not. Default is False.

    Returns
    -------
    Dataset
        A Dataset object containing the raw data, wavetracker data, and communication data.

    Example
    -------
    >>> from gridtools.datasets import load
    >>> ds = load(pathlib.Path("path/to/dataset"))
    >>> # You can easily access wavetracker data using the dot notation
    >>> ds.track.freqs
    >>> # You can also access the raw data
    >>> ds.rec.raw
    >>> # Or the communication data
    >>> ds.com.chirp.times
    >>> ds.com.chirp.idents
    >>> # Or the position estimates
    >>> ds.track.xpos
    """
    if isinstance(path, str):
        path = pathlib.Path(path)

    ds = Dataset(
        path=path,
        grid=load_grid(path) if grid else None,
        track=load_wavetracker(path),
        com=load_com(path),
    )
    return ds


def subset_wavetracker(
    wt: "WavetrackerData", start_time: float, stop_time: float
) -> "WavetrackerData":
    """
    Extracts a subset of a WavetrackerData object between start_time and stop_time.

    Parameters
    ----------
    wt : WavetrackerData
        The WavetrackerData object to extract a subset from.
    start_time : float
        The start time of the subset.
    stop_time : float
        The stop time of the subset.

    Returns
    -------
    WavetrackerData
        A new WavetrackerData object containing the subset of data between start_time and stop_time.

    Raises
    ------
    GridDataMismatch
        If there is no data in the specified time range.

    Example
    -------
    >>> import pathlib
    >>> from gridtools.datasets import load_wavetracker, subset_wavetracker
    >>> wt = load_wavetracker(pathlib.Path("path/to/wavetracker"))
    >>> wt_sub = subset_wavetracker(wt, 0.5, 1.5)
    """

    freqs = []
    powers = []
    indices = []
    idents = []
    if wt.xpos is not None:
        xpos = []
        ypos = []

    for track_id in np.unique(wt.idents[~np.isnan(wt.idents)]):
        freq = wt.freqs[wt.idents == track_id]
        power = wt.powers[wt.idents == track_id]
        time = wt.times[wt.indices[wt.idents == track_id]]
        index = wt.indices[wt.idents == track_id]

        freq = freq[(time >= start_time) & (time <= stop_time)]
        power = power[(time >= start_time) & (time <= stop_time)]
        index = index[(time >= start_time) & (time <= stop_time)]
        ident = np.repeat(track_id, len(freq))

        if wt.xpos is not None:
            x = wt.xpos[wt.idents == track_id]
            y = wt.ypos[wt.idents == track_id]
            x = xpos[(time >= start_time) & (time <= stop_time)]
            y = ypos[(time >= start_time) & (time <= stop_time)]
            xpos.append(x)
            ypos.append(y)

        freqs.append(freq)
        powers.append(power)
        indices.append(index)
        idents.append(ident)

    tracks = np.concatenate(freqs)
    powers = np.concatenate(powers)
    indices = np.concatenate(indices)
    idents = np.concatenate(idents)
    if wt.xpos is not None:
        xpos = np.concatenate(xpos)
        ypos = np.concatenate(ypos)
    time = wt.times[(wt.times >= start_time) & (wt.times <= stop_time)]

    if len(indices) == 0:
        raise GridDataMismatch("No data in the specified time range.")
    indices -= indices[0]

    # rebuild wavetracker object
    wt_sub = WavetrackerData(
        freqs=tracks,
        powers=powers,
        idents=idents,
        indices=indices,
        ids=np.unique(idents),
        times=time,
        xpos=xpos if wt.xpos is not None else None,
        ypos=ypos if wt.ypos is not None else None,
    )
    return wt_sub


def subset_grid(
    rec: "GridData", start_time: float, stop_time: float
) -> "GridData":
    """
     Returns a subset of a raw dataset.

     Parameters
     ----------
     rec :GridData
         The raw dataset to subset.
     start_time : float
         The start time of the subset in seconds.
     stop_time : float
         The stop time of the subset in seconds.

     Returns
     -------
    GridData
         The subset of the raw dataset.

     Examples
     --------
     >>> import pathlib
     >>> from gridtools.datasets import load_grid,subset_grid
     >>> rec = load_grid(pathlib.Path("path/to/raw")
     >>> subset = subset_grid(rec, 0.1, 0.5)
    """

    assert start_time < stop_time, "Start time must be smaller than stop time."
    assert start_time > 0, "Start time must be larger than 0."
    assert (
        stop_time < rec.rec.shape[0] / rec.samplerate
    ), "Stop time must be smaller than the end."

    start_idx = int(start_time * rec.samplerate)
    stop_idx = int(stop_time * rec.samplerate)

    assert start_idx < rec.rec.shape[0], "Start index out of bounds."
    assert stop_idx < rec.rec.shape[0], "Stop index out of bounds."

    raw = rec.rec[start_idx:stop_idx, :]
    return GridData(rec=raw, samplerate=rec.samplerate)


def subset_com(
    com: "CommunicationData", start_time: float, stop_time: float
) -> "CommunicationData":
    """
    Makes a subset of a communication dataset.

    Parameters
    ----------
    com : CommunicationData
        The communication dataset to subset.
    start_time : float
        The start time of the subset.
    stop_time : float
        The stop time of the subset.

    Returns
    -------
    CommunicationData
        The subset of the communication dataset.

    Examples
    --------
    >>> import pathlib
    >>> from gridtools.datasets import load_com, subset_com
    >>> com = load_com(pathlib.Path("path/to/communication"))
    >>> subset = subset_com(com, 0.1, 0.5)
    """

    if hasattr(com, "chirp"):
        ci = com.chirp.idents[
            (com.chirp.times >= start_time) & (com.chirp.times <= stop_time)
        ]
        ct = com.chirp.times[
            (com.chirp.times >= start_time) & (com.chirp.times <= stop_time)
        ]
    if hasattr(com, "rise"):
        ri = com.rise.idents[
            (com.rise.times >= start_time) & (com.rise.times <= stop_time)
        ]
        rt = com.rise.times[
            (com.rise.times >= start_time) & (com.rise.times <= stop_time)
        ]

    chirp = (
        ChirpData(times=ct, idents=ci, detector=com.chirp.detector)
        if hasattr(com, "chirp")
        else None
    )
    rise = (
        RiseData(times=rt, idents=ri, detector=com.rise.detector)
        if hasattr(com, "rise")
        else None
    )
    if not hasattr(com, "chirp") and not hasattr(com, "rise"):
        return None
    return CommunicationData(chirp=chirp, rise=rise)


def subset(data: "Dataset", start_time: float, stop_time: float) -> "Dataset":
    """
    Makes a subset of a full dataset.

    Parameters
    ----------
    data : Dataset
        The full dataset to be subsetted.
    start_time : float
        The start time of the subset, in seconds.
    stop_time : float
        The stop time of the subset, in seconds.

    Returns
    -------
    Dataset
        The subsetted dataset.

    Raises
    ------
    AssertionError
        If the start time is greater than or equal to the stop time, or if the
        start time is less than the beginning of the full dataset, or if the
        stop time is greater than the end of the full dataset.

    Notes
    -----
    This function subsets the wave tracker, recorder, and communicator data in
    the full dataset, if they exist. If the recorder or communicator data do
    not exist, the corresponding subsetted data will be None.

    Examples
    --------
    >>> import pathlib
    >>> from gridtools.datasets import load, subset, save
    >>> ds = load(pathlib.Path("path/to/dataset"))
    >>> subset = subset(ds, 0.1, 0.5)
    >>> save(subset, pathlib.Path("path/to/save"))
    """

    assert start_time < stop_time, "Start time must be smaller than stop time."
    assert (
        start_time >= data.track.times[0]
    ), "Start time must be larger than the beginning."
    assert (
        stop_time <= data.track.times[-1]
    ), "Stop time must be smaller than the end."

    wt_sub = subset_wavetracker(data.track, start_time, stop_time)
    raw_sub = (
        subset_grid(data.grid, start_time, stop_time) if data.grid else None
    )
    com_sub = subset_com(data.com, start_time, stop_time) if data.com else None

    return Dataset(path=data.path, grid=raw_sub, track=wt_sub, com=com_sub)


def save_wavetracker(
    wt: "WavetrackerData", output_path: Union[pathlib.Path, str]
) -> None:
    """
    Save WavetrackerData object to disk.

    Parameters
    ----------
    wt : WavetrackerData
        WavetrackerData object to save.
    output_path : pathlib.Path
        Path to save the object to.

    Returns
    -------
    None

    Notes
    -----
    This function saves the following attributes of the WavetrackerData object to disk:
    - freqs: numpy.ndarray
        Frequencies of each fish over time.
    - powers: numpy.ndarray
        Powers for each frequency.
    - idents: numpy.ndarray
        Identifiers of the fish.
    - indices: numpy.ndarray
        Indices to connect fish ID and freq/power/positions to the time axis.
    - times: numpy.ndarray
        Time axis.
    - xpos: numpy.ndarray, optional
        X-coordinates estimated for each fish, if available.
    - ypos: numpy.ndarray, optional
        Y-coordinates estimated for each fish, if available.

    Examples
    --------
    >>> import pathlib
    >>> from gridtools.datasets import load_wavetracker
    >>> wt = load_wavetracker(pathlib.Path("path/to/wavetracker"))
    >>> save_wavetracker(wt, pathlib.Path("path/to/save"))
    """

    if isinstance(output_path, str):
        output_path = pathlib.Path(output_path)

    np.save(output_path / "fund_v.npy", wt.freqs)
    np.save(output_path / "sign_v.npy", wt.powers)
    np.save(output_path / "ident_v.npy", wt.idents)
    np.save(output_path / "idx_v.npy", wt.indices)
    np.save(output_path / "times.npy", wt.times)
    if wt.xpos is not None:
        np.save(output_path / "xpos.npy", wt.xpos)
        np.save(output_path / "ypos.npy", wt.ypos)


def save_grid(rec: "GridData", output_path: Union[pathlib.Path, str]) -> None:
    """
    Save raw data to a WAV file using `thunderfish.datawriter`.

    Parameters
    ----------
    rec :GridData
        The raw data to be saved.
    output_path : pathlib.Path
        The path to save the file to.

    Example
    -------
    >>> import pathlib
    >>> from gridtools.datasets import load_grid,save_grid
    >>> rec = load_grid(pathlib.Path("path/to/raw"))
    >>> save_grid(rec, pathlib.Path("path/to/save"))
    """

    if isinstance(output_path, str):
        output_path = pathlib.Path(output_path)

    write_data(str(output_path / "traces_grid1.wav"), rec.rec, rec.samplerate)


def save_com(
    com: "CommunicationData", output_path: Union[pathlib.Path, str]
) -> None:
    """
    Save communication data to disk.

    Parameters
    ----------
    com : CommunicationData
        The communication data to save.
    output_path : pathlib.Path
        The path to the directory where the data should be saved.

    Returns
    -------
    None

    Example
    -------
    >>> import pathlib
    >>> from gridtools.datasets import load_com, save_com
    >>> com = load_com(pathlib.Path("path/to/communication"))
    >>> save_com(com, pathlib.Path("path/to/save"))
    """

    if com.chirp is not None:
        np.save(output_path / "chirp_times_gt.npy", com.chirp.times)
        np.save(output_path / "chirp_ids_gt.npy", com.chirp.idents)


def save(dataset: "Dataset", output_path: Union[pathlib.Path, str]) -> None:
    """
    Save a Dataset object to disk. This function saves the wavetracker data,
    the raw data and the communication data to disk, depending on what is
    available in the Dataset object.

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

    Example
    -------
    >>> import pathlib
    >>> from gridtools.datasets import load, save
    >>> ds = load(pathlib.Path("path/to/dataset"))
    >>> save(ds, pathlib.Path("path/to/save"))
    """

    if isinstance(output_path, str):
        output_path = pathlib.Path(output_path)

    output_dir = output_path / dataset.path.name

    if output_dir.exists():
        raise FileExistsError(f"Output directory {output_dir} already exists.")

    output_dir.mkdir(parents=True)

    save_wavetracker(dataset.track, output_dir)

    if dataset.grid is not None:
        save_grid(dataset.grid, output_dir)

    if dataset.com is not None:
        save_com(dataset.com, output_dir)


def subset_cli():
    """
    Subset a dataset to a given time range. Parameters are passed via the
    command line.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the dataset to be subsetted.
    output_path : pathlib.Path
        Path to the directory where the subsetted dataset should be saved.
    start_time : float
        Start time of the subset in seconds.
    stop_time : float
        Stop time of the subset in seconds.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Subset a dataset to a given time range."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=pathlib.Path,
        help="Path to the dataset to be subsetted.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        help="Path to the directory where the subsetted dataset should be saved.",
    )
    parser.add_argument(
        "--start",
        "-s",
        type=float,
        help="Start time of the subset in seconds.",
    )
    parser.add_argument(
        "--end",
        "-e",
        type=float,
        help="Stop time of the subset in seconds.",
    )
    args = parser.parse_args()

    ds = load(args.input, grid=True)
    ds_sub = subset(ds, args.start, args.end)
    save(ds_sub, args.output)


def pprint(obj: BaseModel) -> None:
    """
    Pretty-print the attributes of the object.
    """

    def collect_vars(obj):
        if isinstance(obj, BaseModel):
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith("_") and value is not None:
                    if isinstance(value, BaseModel):
                        result[key] = collect_vars(value)
                    else:
                        result[key] = type(value).__name__
            return result
        else:
            return None

    return rpprint(collect_vars(obj), expand_all=True)


class WavetrackerData(BaseModel):
    """
    Contains data extracted by the wavetracker.
    All check functions are automatically run when the object is instantiated.

    Parameters
    ----------
    freqs : np.ndarray[float]
        Array of frequencies.
    powers : np.ndarray[float]
        Array of powers.
    idents : np.ndarray[float]
        Array of idents.
    indices : np.ndarray[int]
        Array of indices.
    ids : np.ndarray[int]
        Array of ids.
    times : np.ndarray[float]
        Array of times.
    xpos : Optional[np.ndarray[float]], optional
        Array of x positions, by default None.
    ypos : Optional[np.ndarray[float]], optional
        Array of y positions, by default None.

    Methods
    -------
    check_numpy_array(v)
        Check if all the arrays are numpy arrays.
    check_numpy_array_pos(v)
        Check if xpos and ypos are numpy arrays when they are not none.
    check_times_sorted(v)
        Checks that the times are monotonically increasing.
    check_times_indices()
        Checks that the indices in the indices array cannot go out of bounds
        of the times array.
    check_wavetracker_data()
        Check if the wavetracker data is of equal length.
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

        Parameters
        ----------
        v : np.ndarray
            Array to be checked.

        Returns
        -------
        np.ndarray
            The input array.

        Raises
        ------
        ValidationError
            If the input is not a numpy array.
        """
        if not isinstance(v, np.ndarray):
            raise ValidationError("Value must be a numpy.ndarray")
        return v

    @field_validator("xpos", "ypos")
    @classmethod
    def check_numpy_array_pos(cls, v):
        """
        Check if xpos and ypos are numpy arrays when they are not none.

        Parameters
        ----------
        v : np.ndarray
            Array to be checked.

        Returns
        -------
        np.ndarray
            The input array.

        Raises
        ------
        ValidationError
            If the input is not a numpy array.
        """
        if v is not None and not isinstance(v, np.ndarray):
            raise ValidationError("Value must be a numpy.ndarray")
        return v

    @field_validator("times")
    @classmethod
    def check_times_sorted(cls, v):
        """
        Checks that the times are monotonically increasing.

        Parameters
        ----------
        v : np.ndarray
            Array of times.

        Returns
        -------
        np.ndarray
            The input array.

        Raises
        ------
        GridDataMismatch
            If the times are not monotonically increasing.
        """
        if not np.all(np.diff(v) > 0):
            raise GridDataMismatch(
                "Wavetracker times are not monotonically increasing!"
            )
        return v

    @model_validator(mode="after")
    def check_times_indices(self):
        """
        Checks that the indices in the indices array cannot go out of bounds
        of the times array.

        Returns
        -------
        WavetrackerData
            The current instance of the WavetrackerData class.

        Raises
        ------
        GridDataMismatch
            If the number of times is smaller than the number of unique indices.
        """
        if self.times.shape[0] < len(set(self.indices)):
            raise GridDataMismatch(
                "Number of times is smaller than number of unique indices!"
            )
        return self

    @model_validator(mode="after")
    def check_wavetracker_data(self) -> "WavetrackerData":
        """
        Check if the wavetracker data is of correct length.

        Returns
        -------
        WavetrackerData
            The current instance of the WavetrackerData class.

        Raises
        ------
        GridDataMismatch
            If the wavetracker data is not of equal length.
        """
        lengths = [
            np.shape(x)[0]
            for x in [self.freqs, self.powers, self.idents, self.indices]
        ]
        if len(set(lengths)) > 1:
            raise GridDataMismatch("Wavetracker data is not of equal length!")
        return self

    def pprint(self) -> None:
        """
        Pretty-print the attributes of the object.
        """
        pprint(self)


class GridData(BaseModel):
    """
    Contains raw data from the electrode grid recordings.

    Parameters
    ----------
    raw : Union[np.ndarray, DataLoader]
        The raw data from the electrode grid recordings.

    Raises
    ------
    ValidationError
        If the raw data is not a numpy array or a DataLoader object.

    Attributes
    ----------
    model_config : ConfigDict
        A dictionary containing the configuration for the model.
    raw : Union[np.ndarray, DataLoader]
        The raw data from the electrode grid recordings.

    Examples
    --------
    >>> import pathlib
    >>> from gridtools.datasets load_grid
    >>> rec = load_grid(pathlib.Path("path/to/raw"))
    >>> rec.rec
    >>> rec.samplerate
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rec: Union[np.ndarray, DataLoader]
    samplerate: int

    @field_validator("rec")
    @classmethod
    def check_raw(
        cls, v: Union[np.ndarray, DataLoader]
    ) -> Union[np.ndarray, DataLoader]:
        """
        Check if the raw data is a numpy array or a DataLoader object.

        Parameters
        ----------
        v : Union[np.ndarray, DataLoader]
            The raw data to be validated.

        Returns
        -------
        Union[np.ndarray, DataLoader]
            The validated raw data.

        Raises
        ------
        ValidationError
            If the raw data is not a numpy array or a DataLoader object.
        """
        if not isinstance(v, (np.ndarray, DataLoader)):
            raise ValidationError(
                "Raw data must be a numpy array or a DataLoader."
            )
        return v

    def pprint(self) -> None:
        """
        Pretty-print the attributes of the object.
        """
        pprint(self)


class ChirpData(BaseModel):
    """
    Contains data about chirps produced by fish in the dataset.

    Parameters
    ----------
    times : np.ndarray
        A numpy array containing the times at which chirps were detected.
    idents : np.ndarray
        A numpy array containing the identities of the fish that produced the chirps.
    detector : str
        The type of detector used to detect the chirps. Must be either 'gt' or 'cnn'.
        More detector types will probably follow.

    Methods
    -------
    check_numpy_array(v)
        Check if times and idents are numpy arrays.
    check_detector(v)
        Check if detector is either 'gt' or 'cnn' and a string.

    Attributes
    ----------
    times : np.ndarray
        A numpy array containing the times at which chirps were detected.
    idents : np.ndarray
        A numpy array containing the identities of the fish that produced the chirps.
    detector : str
        The type of detector used to detect the chirps. Must be either 'gt' or 'cnn'.
        More detector types will probably follow.
    model_config : ConfigDict
        A dictionary containing the configuration for the model.

    Raises
    ------
    ValidationError
        If the input values do not meet the specified requirements.

    Examples
    --------
    >>> times = np.array([0.1, 0.2, 0.3])
    >>> idents = np.array([1, 2, 3])
    >>> detector = 'gt'
    >>> chirps = ChirpData(times=times, idents=idents, detector=detector)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    times: np.ndarray
    idents: np.ndarray
    detector: str
    params: Optional[np.ndarray]

    @field_validator("times", "idents")
    @classmethod
    def check_numpy_array(cls, v):
        """
        Check if times and idents are numpy arrays.

        Parameters
        ----------
        v : Any
            The value to be checked.

        Returns
        -------
        np.ndarray
            The input value if it is a numpy array.

        Raises
        ------
        ValidationError
            If the input value is not a numpy array.
        """
        if not isinstance(v, np.ndarray):
            raise ValidationError("Value must be a numpy.ndarray")
        return v

    @field_validator("detector")
    @classmethod
    def check_detector(cls, v):
        """
        Check if detector is either 'gt' or 'cnn' and a string.

        Parameters
        ----------
        v : Any
            The value to be checked.

        Returns
        -------
        str
            The input value if it is a string and either 'gt' or 'cnn'.

        Raises
        ------
        ValidationError
            If the input value is not a string or is not 'gt' or 'cnn'.
        """
        if not isinstance(v, str):
            raise ValidationError("Detector must be a string.")
        if v not in ["gt", "cnn"]:
            raise ValidationError("Detector must be 'gt' or 'cnn'.")
        return v

    def pprint(self) -> None:
        """
        Pretty-print the attributes of the object.
        """
        pprint(self)


class RiseData(BaseModel):
    """
    Contains data of rises produced by fish in the dataset.

    Attributes
    ----------
    times : numpy.ndarray
        An array of times.
    idents : numpy.ndarray
        An array of identifiers.
    detector : str
        The detector used to produce the data. Must be either 'gt' (ground
        truth) or 'pd' (peak detection).

    Examples
    --------
    >>> times = np.array([0.1, 0.2, 0.3])
    >>> idents = np.array([1, 2, 3])
    >>> detector = 'gt'
    >>> data = RiseData(times=times, idents=idents, detector=detector)
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

        Parameters
        ----------
        v : numpy.ndarray
            The value to be validated.

        Returns
        -------
        numpy.ndarray
            The validated value.

        Raises
        ------
        ValidationError
            If the value is not a numpy array.
        """
        if not isinstance(v, np.ndarray):
            raise ValidationError("Value must be a numpy.ndarray")
        return v

    @field_validator("detector")
    @classmethod
    def check_detector(cls, v):
        """
        Check if detector is either 'gt' or 'pd' and a string.

        Parameters
        ----------
        v : str
            The value to be validated.

        Returns
        -------
        str
            The validated value.

        Raises
        ------
        ValidationError
            If the value is not a string or is not 'gt' or 'pd'.
        """
        if not isinstance(v, str):
            raise ValidationError("Detector must be a string.")
        if v not in ["gt", "pd"]:
            raise ValidationError("Detector must be 'gt' or 'pd'.")
        return v

    def pprint(self) -> None:
        """
        Pretty-print the attributes of the object.
        """
        pprint(self)


class CommunicationData(BaseModel):
    """
    Contains data for communication signals produced by fish in the dataset.
    If a variable is set to None, it is removed.

    Parameters
    ----------
    chirp : ChirpData, optional
        Data for the chirp signal produced by the fish.
    rise : RiseData, optional
        Data for the rise signal produced by the fish.

    Methods
    -------
    typecheck_chirp(v)
        Check if chirp data is a ChirpData object if it is not none.
    typecheck_rise(v)
        Check if rise data is a RiseData object if it is not none.
    check_communication_data()
        Check if chirp or rise data is provided. Class should not be instantiated when no data is provided.

    Raises
    ------
    ValidationError
        If chirp data is not a ChirpData object or if rise data is not a RiseData object, or if neither chirp nor rise data is provided.

    Returns
    -------
    CommunicationData
        An instance of the CommunicationData class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chirp: Optional[ChirpData] = None
    rise: Optional[RiseData] = None

    @field_validator("chirp")
    @classmethod
    def typecheck_chirp(cls, v):
        """
        Check if chirp data is a ChirpData object if it is not none.

        Parameters
        ----------
        v : Any
            The value to check.

        Raises
        ------
        ValidationError
            If chirp data is not a ChirpData object.

        Returns
        -------
        Any
            The value that was passed in.
        """
        if v is not None and not isinstance(v, ChirpData):
            raise ValidationError("Chirp data must be a ChirpData object.")
        return v

    @field_validator("rise")
    @classmethod
    def typecheck_rise(cls, v):
        """
        Check if rise data is a RiseData object if it is not none.

        Parameters
        ----------
        v : Any
            The value to check.

        Raises
        ------
        ValidationError
            If rise data is not a RiseData object.

        Returns
        -------
        Any
            The value that was passed in.
        """
        if v is not None and not isinstance(v, RiseData):
            raise ValidationError("Rise data must be a RiseData object.")
        return v

    @model_validator(mode="after")
    def check_communication_data(self):
        """
        Check if chirp or rise data is provided. Class should not be instantiated when no data is provided.

        Raises
        ------
        ValidationError
            If neither chirp nor rise data is provided.

        Returns
        -------
        CommunicationData
            An instance of the CommunicationData class.
        """
        if self.chirp is None and self.rise is None:
            raise ValidationError(
                "Communication data must contain either chirp or rise data."
            )
        return self

    @model_validator(mode="after")
    def delete_if_none(self):
        """
        Deletes attributes from the model if their values are None.

        Returns:
        --------
        self : object
            The modified instance of the object.
        """
        for key, value in self.model_dump().items():
            if value is None:
                delattr(self, key)
        return self

    def pprint(self) -> None:
        """
        Pretty-print the attributes of the object.
        """
        pprint(self)


class Dataset(BaseModel):
    """
    The main dataset class to load data extracted from electrode grid recordings
    of wave-type weakly electric fish.

    Parameters
    ----------
    path : pathlib.Path
        The path to the wavetracker dataset.
    grid : Optional[GridData], optional
        The raw data, by default None.
    track : WavetrackerData
        The wavetracker data.
    com : Optional[CommunicationData], optional
        The communication data, by default None.

    Notes
    -----
    Every dataset must at least get a path to a wavetracker dataset. Optionally,
    a raw dataset and/or a chirp dataset can be provided. The raw dataset can
    be used to extract e.g. the chirp times from the raw data. Best instantiated
    with the `load` function as demonstrated in the examples. If a variable is
    set to None, it is removed.

    Examples
    --------
    >>> import pathlib
    >>> from gridtools.datasets import load
    >>> ds = load(pathlib.Path("path/to/dataset"))
    >>> # look at the chirps
    >>> ds.com.chirp.times
    >>> # or do something with the wavetracker data
    >>> ds.track.freqs
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: pathlib.Path
    grid: Optional[GridData]
    track: WavetrackerData
    com: Optional[CommunicationData]

    @field_validator("path")
    @classmethod
    def check_path(cls, v):
        """
        Check if path is a pathlib.Path object.

        Parameters
        ----------
        v : Any
            The value to be validated.

        Returns
        -------
        pathlib.Path
            The validated path.

        Raises
        ------
        ValidationError
            If the value is not a pathlib.Path object.
        """
        if not isinstance(v, pathlib.Path):
            raise ValidationError("Path must be a pathlib.Path object.")
        return v

    @field_validator("grid")
    @classmethod
    def check_rec(cls, v):
        """
        Check if raw data is a GridData object or none.

        Parameters
        ----------
        v : Any
            The value to be validated.

        Returns
        -------
        GridData or None
            The validated raw data.

        Raises
        ------
        ValidationError
            If the value is not a GridData object.
        """
        if v is not None and not isinstance(v, GridData):
            raise ValidationError("Raw data must be a GridData object.")
        return v

    @field_validator("track")
    @classmethod
    def check_track(cls, v):
        """
        Check if wavetracker data is a WavetrackerData object.

        Parameters
        ----------
        v : Any
            The value to be validated.

        Returns
        -------
        WavetrackerData
            The validated wavetracker data.

        Raises
        ------
        ValidationError
            If the value is not a WavetrackerData object.
        """
        if v is not None and not isinstance(v, WavetrackerData):
            raise ValidationError(
                "Wavetracker data must be a WavetrackerData object."
            )
        return v

    @field_validator("com")
    @classmethod
    def check_com(cls, v):
        """
        Check if communication data is a CommunicationData object or none.

        Parameters
        ----------
        v : Any
            The value to be validated.

        Returns
        -------
        CommunicationData or None
            The validated communication data.

        Raises
        ------
        ValidationError
            If the value is not a CommunicationData object.
        """
        if v is not None and not isinstance(v, CommunicationData):
            raise ValidationError(
                "Communication data must be a CommunicationData object."
            )
        return v

    @model_validator(mode="after")
    def delete_if_none(self):
        """
        Deletes attributes from the model if their values are None.

        Returns:
        --------
        self : object
            The modified instance of the object.
        """
        for key, value in self.model_dump().items():
            if value is None:
                delattr(self, key)
        return self

    def pprint(self) -> None:
        """
        Pretty-print the attributes of the object.
        """
        pprint(self)
