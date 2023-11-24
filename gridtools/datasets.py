"""# Datasets.

Classes and functions to load, work with and save data associated with
electrode grid recordings of wave-type weakly electric fish.

The main functionalities include the following:
- gridtools.datasets.load: Load a dataset from a given path.
- gridtools.datasets.save: Save a dataset to disk.
- gridtools.datasets.subset: Make a subset of a dataset.

The main class is the `Dataset` class, which is able to load all data extracted
by the `wavetracker` as well as raw data, communication signals, etc.

## Architecture and design principles

The architecture of the `datasets` module follows these design principles:
- **Composition over inheritance**: The `Dataset` class is a composition of
different subclasses, making it easily extensible to other data types in the
future.
- **Data models**: The `Dataset` class is not just a dataclass but a data
model: Upon instantiation, the data is checked for consistency and errors are
raised if the data is inconsistent.

The Dataset class is a composition of:
- `GridData`: The raw recording from the electrode grid.
- `WavetrackerData`: Tracking arrays produced by - and derived from - the
[`wavetracker`](https://github.com/tillraab/wavetracker.git).
- `CommunicationData`: Chirp and rise times, identifiers and optionally,
extracted parameters such as height, width, etc.

## Usage

Loading a dataset is as easy as calling the `load` function with the path to
the dataset as an argument. The function returns a `Dataset` object containing
the loaded data.
```python
from gridtools.datasets import load
ds = load(pathlib.Path("path/to/dataset"))
```

To also load the raw data, set the `grid` argument to `True`.
```python
from gridtools.datasets import load
ds = load(pathlib.Path("path/to/dataset"), grid=True)
```

To create a subset of a dataset, use the `subset` function. The function takes
the dataset to subset, the start and stop time of the subset, and the mode
("time" or "index") as arguments. The function returns a new `Dataset` object
containing the subsetted data.
```python
from gridtools.datasets import load, subset
ds = load(pathlib.Path("path/to/dataset"))
subset = subset(ds, 0.1, 0.5)
```

To save this subset to disk, use the `save` function. The function takes the
subsetted dataset and the path to the directory where the dataset should be
saved as arguments.
```python
from gridtools.datasets import load, subset, save
ds = load(pathlib.Path("path/to/dataset"))
subset = subset(ds, 0.1, 0.5)
save(subset, pathlib.Path("path/to/save"))
```

If you are just interested in some part of the dataset, such as the
`wavetracker` arrays, you can simply use the subclass specific methods instead:
```python
from gridtools.datasets import (
    load_wavetracker, subset_wavetracker, save_wavetracker
)
wt = load_wavetracker(pathlib.Path("path/to/wavetracker"))
subset = subset_wavetracker(wt, 0.1, 0.5)
save_wavetracker(subset, pathlib.Path("path/to/save"))
```
"""

import pathlib
from typing import Optional, Union, Any, Type

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

# The order determines the priority of the detectors.
chirp_detectors = ["gt", "rcnn", "cnn", "None"]
rise_detectors = ["gt", "rcnn", "pd", "None"]


def load_wavetracker(path: Union[pathlib.Path, str]) -> "WavetrackerData":
    """Load wavetracker files.

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
    ```python
    >>> import pathlib
    >>> from gridtools.datasets import load_wavetracker
    >>> wt = load_wavetracker(pathlib.Path("path/to/wavetracker"))
    ```
    """
    if isinstance(path, str):
        path = pathlib.Path(path)

    files = list(path.glob("*"))

    if not any("fund_v.npy" in str(f) for f in files):
        msg = "No wavetracker dataset found in the provided directory!"
        raise FileNotFoundError(msg)

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
    """Load a raw dataset from a given path.

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
    This function uses the thunderfish dataloader to easily access large binary
    files. The function checks if the directory contains a "traces*" file.
    The function is tested using .raw and .wav files.
    If neither file is found, a FileNotFoundError is raised.
    If a "traces-grid1.raw" file is found, it is loaded using the thunderfish
    DataLoader.The function returns a GridData object containing the loaded raw
    dataset. Instead of directly passing the DataLoader object, I wrap it in
    this class to later be able to add metadata such as electrode positions,
    etc. to the recording class.

    Examples
    --------
    ```python
    >>> import pathlib
    >>> from gridtools.datasets load_grid
    >>> rec = load_grid(pathlib.Path("path/to/raw"))
    ```
    """
    if isinstance(path, str):
        path = pathlib.Path(path)

    files = list(path.glob("traces*"))

    if len(files) == 0:
        msg = "No raw dataset found in the provided directory!"
        raise FileNotFoundError(msg)
    if len(files) > 1:
        msg = "More than one raw dataset found! Check path."
        raise FileNotFoundError(msg)

    file = files[0]
    rec = DataLoader(str(path / file.name))

    return GridData(rec=rec, samplerate=rec.samplerate)


def load_chirps(path: pathlib.Path) -> "ChirpData":
    """Load the chirp data from a given path.

    Parameters
    ----------
    - `path`: `pathlib.Path`
        The path to the directory containing the chirp data.

    Returns
    -------
    - `ChirpData`
        An object containing the loaded chirp data.

    Raises
    ------
    - `FileNotFoundError`
        If no chirp dataset is found in the given directory.
    """
    files = list(path.glob("*"))

    # Load chirp data if available: Check if chirp data is available
    # for any detector
    det = None
    exist = False
    chirp_times = np.array([])
    chirp_ids = np.array([])
    params = np.array([])

    for detector in chirp_detectors:
        if any(f"chirp_times_{detector}.npy" in str(f) for f in files):
            det = detector
            exist = True
            chirp_times = np.load(path / f"chirp_times_{det}.npy")
            chirp_ids = np.load(path / f"chirp_ids_{det}.npy").astype(int)
            if any(f"chirp_params_{det}.npy" in str(f) for f in files):
                params = np.load(path / f"chirp_params_{det}.npy")
            break

    return ChirpData(
        times=chirp_times,
        idents=chirp_ids,
        params=params,
        detector=str(det),
        exist=exist,
    )


def load_rises(path: pathlib.Path) -> "RiseData":
    """Load the rise data from a given path.

    Parameters
    ----------
    - `path`: `pathlib.Path`
        The path to the directory containing the rise data.

    Returns
    -------
    - `RiseData`
        An object containing the loaded rise data.

    Raises
    ------
    - `FileNotFoundError`
        If no rise dataset is found in the given directory.
    """
    files = list(path.glob("*"))

    # Load rise data if available: Check if rise data is available
    # for any detector
    det = None
    exist = False
    rise_times = np.array([])
    rise_ids = np.array([])
    params = np.array([])

    for detector in rise_detectors:
        if any(f"rise_times_{detector}.npy" in str(f) for f in files):
            det = detector
            exist = True
            rise_times = np.load(path / f"rise_times_{det}.npy")
            rise_ids = np.load(path / f"rise_ids_{det}.npy").astype(int)
            if any(f"rise_params_{det}.npy" in str(f) for f in files):
                params = np.load(path / f"rise_params_{det}.npy")
            break

    return RiseData(
        times=rise_times,
        idents=rise_ids,
        params=params,
        detector=str(det),
        exist=exist,
    )


def load_com(path: pathlib.Path) -> "CommunicationData":
    """Load communication data from a given path.

    Loads chirps if available, loads
    rises if available, or loads both if available.

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
    ```python
    >>> import pathlib
    >>> from gridtools.datasets import load_com
    >>> com = load_com(pathlib.Path("path/to/communication"))
    >>> # get rise times and ids
    >>> com.rise.times
    >>> com.rise.idents
    ```
    """
    chirp = load_chirps(path)
    rise = load_rises(path)
    return CommunicationData(chirp=chirp, rise=rise)


def load(path: pathlib.Path) -> "Dataset":
    """
    Load all data from a dataset and build a Dataset object.

    Parameters
    ----------
    path : pathlib.Path
        The path to the dataset.

    Returns
    -------
    Dataset
        A Dataset object containing the raw data, wavetracker data, and
        communication data.

    Examples
    --------
    ```python
    >>> import pathlib
    >>> from gridtools.datasets import load
    >>> ds = load(pathlib.Path("path/to/dataset"))
    >>> ds.track.freqs
    >>> ds.rec.raw
    >>> ds.com.chirp.times
    >>> ds.com.chirp.idents
    >>> ds.track.xpos
    ```
    """
    return Dataset(
        path=path,
        grid=load_grid(path),
        track=load_wavetracker(path),
        com=load_com(path),
    )


def subset_wavetracker(
    wt: "WavetrackerData",
    start: Union[float, int],
    stop: Union[float, int],
    mode: str = "time",
    samplerate: int = 20000,
) -> "WavetrackerData":
    """
    Extracts a subset of a WavetrackerData object between start_time and stop_time.

    Parameters
    ----------
    wt : WavetrackerData
        The WavetrackerData object to extract a subset from.
    start : float
        The start time or index of the subset.
    stop : float
        The stop time or index of the subset.
    mode: str
        Whether to use time or index method.
    samplerate: int
        Samplerate to use for conversion between time and index.

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
    ```python
    >>> import pathlib
    >>> from gridtools.datasets import load_wavetracker, subset_wavetracker
    >>> wt = load_wavetracker(pathlib.Path("path/to/wavetracker"))
    >>> wt_sub = subset_wavetracker(wt, 0.5, 1.5)
    ```
    """

    assert mode in ["time", "index"], "Mode must be either 'time' or 'index'."

    if mode == "index":
        start_time = start / samplerate
        stop_time = stop / samplerate
    else:
        start_time = start
        stop_time = stop

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
    time -= start_time

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
    rec: "GridData",
    start: float,
    stop: float,
    mode: str = "time",
    samplerate: int = 20000,
) -> "GridData":
    """
    Returns a subset of a raw dataset.

    Parameters
    ----------
    rec :GridData
        The raw dataset to subset.
    start : float
        The start time / index of the subset.
    stop: float
        The stop time / index of the subset.
    mode: str
        Whether to use time or index method.
    samplerate: int
        Samplerate to use for conversion between time and index.

    Returns
    -------
    GridData
         The subset of the raw dataset.

    Examples
    --------
    ```python
    >>> import pathlib
    >>> from gridtools.datasets import load_grid,subset_grid
    >>> rec = load_grid(pathlib.Path("path/to/raw")
    >>> subset = subset_grid(rec, 0.1, 0.5)
    ```
    """

    assert mode in ["time", "index"], "Mode must be either 'time' or 'index'."

    if mode == "index":
        start_time = start / samplerate
        stop_time = stop / samplerate
    else:
        start_time = start
        stop_time = stop

    assert start_time < stop_time, "Start time must be smaller than stop time."
    assert start_time >= 0, "Start time must be larger or equal to 0."
    assert (
        stop_time <= rec.rec.shape[0] / rec.samplerate
    ), "Stop time must be smaller than the end."

    start_idx = int(start_time * rec.samplerate)
    stop_idx = int(stop_time * rec.samplerate)

    assert start_idx < rec.rec.shape[0], "Start index out of bounds."
    assert stop_idx <= rec.rec.shape[0], "Stop index out of bounds."

    raw = rec.rec[start_idx:stop_idx, :]
    return GridData(rec=raw, samplerate=rec.samplerate)


def subset_com(
    com: "CommunicationData",
    start: float,
    stop: float,
    mode: str = "time",
    samplerate: int = 20000,
) -> "CommunicationData":
    """
    Makes a subset of a communication dataset.

    Parameters
    ----------
    com : CommunicationData
        The communication dataset to subset.
    start : float
        The start time / index of the subset.
    stop: float
        The stop time / index of the subset.
    mode: str
        Whether to use time or index method.
    samplerate: int
        Samplerate to use for conversion between time and index.

    Returns
    -------
    CommunicationData
        The subset of the communication dataset.

    Examples
    --------
    ```python
    >>> import pathlib
    >>> from gridtools.datasets import load_com, subset_com
    >>> com = load_com(pathlib.Path("path/to/communication"))
    >>> subset = subset_com(com, 0.1, 0.5)
    ```
    """

    assert mode in ["time", "index"], "Mode must be either 'time' or 'index'."

    if mode == "index":
        start_time = start / samplerate
        stop_time = stop / samplerate
    else:
        start_time = start
        stop_time = stop

    if hasattr(com, "chirp"):
        ci = com.chirp.idents[
            (com.chirp.times >= start_time) & (com.chirp.times <= stop_time)
        ]
        ct = com.chirp.times[
            (com.chirp.times >= start_time) & (com.chirp.times <= stop_time)
        ]
        if hasattr(com.chirp, "params"):
            cp = com.chirp.params[
                (com.chirp.times >= start_time) & (com.chirp.times <= stop_time)
            ]
        else:
            cp = None
        chirp = (
            ChirpData(
                times=ct,
                idents=ci,
                params=cp,
                detector=com.chirp.detector,
                exist=com.chirp.exist,
            )
            if hasattr(com, "chirp")
            else None
        )
    else:
        chirp = None
    if hasattr(com, "rise"):
        ri = com.rise.idents[
            (com.rise.times >= start_time) & (com.rise.times <= stop_time)
        ]
        rt = com.rise.times[
            (com.rise.times >= start_time) & (com.rise.times <= stop_time)
        ]
        if hasattr(com.rise, "params"):
            rp = com.rise.params[
                (com.rise.times >= start_time) & (com.rise.times <= stop_time)
            ]
        else:
            rp = None
        rise = (
            RiseData(
                times=rt,
                idents=ri,
                params=rp,
                detector=com.rise.detector,
                exist=com.rise.exist,
            )
            if hasattr(com, "rise")
            else None
        )
    else:
        rise = None

    if not hasattr(com, "chirp") and not hasattr(com, "rise"):
        return None
    return CommunicationData(chirp=chirp, rise=rise)


def subset(
    data: "Dataset", start: float, stop: float, mode: str = "time"
) -> "Dataset":
    """
    Makes a subset of a full dataset.

    Parameters
    ----------
    data : Dataset
        The full dataset to be subsetted.
    start : float
        The start time / index of the subset, in seconds.
    stop : float
        The stop time / index of the subset, in seconds.
    mode : str = "time" or "index"
        Whether to use time or index method.

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
    ```python
    >>> import pathlib
    >>> from gridtools.datasets import load, subset, save
    >>> ds = load(pathlib.Path("path/to/dataset"))
    >>> subset = subset(ds, 0.1, 0.5)
    >>> save(subset, pathlib.Path("path/to/save"))
    ```
    """

    assert mode in ["time", "index"], "Mode must be either 'time' or 'index'."
    samplerate = data.grid.samplerate

    if mode == "index":
        start_time = start / samplerate
        stop_time = stop / samplerate
    else:
        start_time = start
        stop_time = stop

    assert start_time < stop_time, "Start time must be smaller than stop time."

    wt_sub = subset_wavetracker(
        data.track, start_time, stop_time, samplerate=samplerate
    )
    raw_sub = (
        subset_grid(data.grid, start_time, stop_time, samplerate=samplerate)
        if data.grid
        else None
    )
    com_sub = (
        subset_com(data.com, start_time, stop_time, samplerate=samplerate)
        if hasattr(data, "com")
        else None
    )

    new_path = (
        data.path.parent
        / f"subset_{data.path.name}_t0_{start_time}_t1_{stop_time}"
    )

    return Dataset(path=new_path, grid=raw_sub, track=wt_sub, com=com_sub)


def save_wavetracker(
    wt: "WavetrackerData", output_path: Union[pathlib.Path, str]
) -> None:
    """
    Save WavetrackerData object to disk as numpy files, like the original
    wavetracker data.

    Parameters
    ----------
    wt : WavetrackerData
        WavetrackerData object to save.
    output_path : pathlib.Path
        Path to save the object to.

    Returns
    -------
    None

    Examples
    --------
    ```python
    >>> import pathlib
    >>> from gridtools.datasets import load_wavetracker
    >>> wt = load_wavetracker(pathlib.Path("path/to/wavetracker"))
    >>> save_wavetracker(wt, pathlib.Path("path/to/save"))
    ```
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
    ```python
    >>> import pathlib
    >>> from gridtools.datasets import load_grid,save_grid
    >>> rec = load_grid(pathlib.Path("path/to/raw"))
    >>> save_grid(rec, pathlib.Path("path/to/save"))
    ```
    """

    if isinstance(output_path, str):
        output_path = pathlib.Path(output_path)

    write_data(
        str(output_path / "traces_grid1.wav"),
        rec.rec,
        rec.samplerate,
        verbose=2,
    )


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

    if hasattr(com, "chirp"):
        np.save(
            output_path / f"chirp_times_{com.chirp.detector}.npy",
            com.chirp.times,
        )
        np.save(
            output_path / f"chirp_ids_{com.chirp.detector}.npy",
            com.chirp.idents,
        )
        if hasattr(com.chirp, "params"):
            np.save(
                output_path / f"chirp_params_{com.chirp.detector}.npy",
                com.chirp.params,
            )

    if hasattr(com, "rise"):
        np.save(
            output_path / f"rise_times_{com.rise.detector}.npy", com.rise.times
        )
        np.save(
            output_path / f"rise_ids_{com.rise.detector}.npy", com.rise.idents
        )
        if hasattr(com.rise, "params"):
            np.save(
                output_path / f"rise_params_{com.rise.detector}.npy",
                com.rise.params,
            )


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


def _pprint(obj: BaseModel) -> None:
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
    """Contains data extracted by the wavetracker.

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

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

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
    def _check_numpy_array(cls, v):
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
    def _check_numpy_array_pos(cls, v):
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
    def _check_times_sorted(cls, v):
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
    def _check_times_indices(self):
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
    def _check_wavetracker_data(self) -> "WavetrackerData":
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
        _pprint(self)


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

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    rec: Union[np.ndarray, DataLoader]
    samplerate: int

    @field_validator("rec")
    @classmethod
    def _check_raw(
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
        _pprint(self)


class ChirpData(BaseModel):
    """Contains data about chirps produced by fish in the dataset.

    Parameters
    ----------
    times : np.ndarray
        A numpy array containing the times at which chirps were detected.
    idents : np.ndarray
        A numpy array containing the identities of the fish that produced
        the chirps.
    detector : str
        The type of detector used to detect the chirps. Must be either 'gt'
        or 'cnn'. More detector types will probably follow.

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
        A numpy array containing the identities of the fish that produced the
        chirps.
    detector : str
        The type of detector used to detect the chirps. Must be either
        'gt' or 'cnn'. More detector types will probably follow.
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

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    times: np.ndarray
    idents: np.ndarray
    detector: str
    params: np.ndarray
    exist: bool

    @field_validator("times", "idents", "params")
    @classmethod
    def _check_numpy_array(cls: Type["ChirpData"], v: np.ndarray) -> np.ndarray:
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
            msg = "Value must be a numpy.ndarray"
            raise ValidationError(msg)
        return v

    @field_validator("detector")
    @classmethod
    def _check_detector(cls: Type["ChirpData"], v: str) -> str:
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
            msg = "Detector must be a string."
            raise ValidationError(msg)
        if v not in chirp_detectors:
            msg = f"Detector must be in {chirp_detectors}."
            raise ValidationError(msg)
        return v

    def pprint(self: "ChirpData") -> None:
        """Pretty-print the attributes of the object."""
        _pprint(self)


class RiseData(BaseModel):
    """Contains data of rises produced by fish in the dataset.

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

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    times: np.ndarray
    idents: np.ndarray
    params: np.ndarray
    detector: str
    exist: bool

    @field_validator("times", "idents", "params")
    @classmethod
    def _check_numpy_array(cls: Type["RiseData"], v: np.ndarray) -> np.ndarray:
        """Check if times and idents are numpy arrays.

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
            msg = "Value must be a numpy.ndarray"
            raise ValidationError(msg)
        return v

    @field_validator("detector")
    @classmethod
    def _check_detector(cls: Type["RiseData"], v: str) -> str:
        """Check if detector is either 'gt' or 'pd' and a string.

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
            msg = "Detector must be a string."
            raise ValidationError(msg)
        if v not in rise_detectors:
            msg = f"Detector must be in {rise_detectors}."
            raise ValidationError(msg)
        return v

    def pprint(self: "RiseData") -> None:
        """Pretty-print the attributes of the object."""
        _pprint(self)


class CommunicationData(BaseModel):
    """Contains data for communication signals.

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
        Check if chirp or rise data is provided. Class should not be
        instantiated when no data is provided.

    Raises
    ------
    ValidationError
        If chirp data is not a ChirpData object or if rise data is not a
        RiseData object, or if neither chirp nor rise data is provided.

    Returns
    -------
    CommunicationData
        An instance of the CommunicationData class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chirp: ChirpData
    rise: RiseData

    @field_validator("chirp")
    @classmethod
    def _typecheck_chirp(
        cls: Type["CommunicationData"], v: Type["ChirpData"]
    ) -> Type["ChirpData"]:
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
    def _typecheck_rise(cls, v):
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
    def _check_communication_data(self):
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
    def _delete_if_none(self):
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
        _pprint(self)


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

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    path: pathlib.Path
    grid: Optional[GridData]
    track: WavetrackerData
    com: Optional[CommunicationData]

    @field_validator("path")
    @classmethod
    def _check_path(cls, v):
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
    def _check_rec(cls, v):
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
    def _check_track(cls, v):
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
    def _check_com(cls, v):
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
    def _delete_if_none(self):
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
        _pprint(self)


def subset_cli(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    start_time: float,
    end_time: float,
) -> None:
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
    end_time : float
        Stop time of the subset in seconds.

    Returns
    -------
    None
    """
    ds = load(input_path, grid=True)
    ds_sub = subset(ds, start_time, end_time)
    print(f"Saving dataset to {output_path}")
    save(ds_sub, output_path)
