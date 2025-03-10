"""Data loader functions for different types of electrode grid datasets."""

import pathlib

from IPython import embed
import numpy as np
from thunderlab.dataloader import DataLoader

from gridtools.datasets.models import (
    ChirpData,
    CommunicationData,
    Dataset,
    GridData,
    RiseData,
    WavetrackerData,
)

# The order determines the priority of the detectors.
chirp_detectors = ["gt", "rcnn", "cnn", "None"]
rise_detectors = ["gt", "rcnn", "pd", "None"]


def load_wavetracker(path: pathlib.Path) -> WavetrackerData:
    """Load wavetracker files.

    Load data produced by the wavetracker and other data extracted from them,
    such as position estimates.

    Parameters
    ----------
    - `path` : `pathlib.Path`
        Path to the directory containing the data files.

    Returns
    -------
    - `WavetrackerData`
        An instance of the WavetrackerData class containing the loaded data.

    Raises
    ------
    - `FileNotFoundError`
        If no wavetracker dataset is found in the specified directory.

    Examples
    --------
    ```python
    import pathlib
    from gridtools.datasets import load_wavetracker
    wt = load_wavetracker(pathlib.Path("path/to/wavetracker"))
    wt.pprint()
    ```
    """
    files = list(path.glob("*"))
    has_positions = True

    if not any("fund_v.npy" in str(f) for f in files):
        msg = f"No wavetracker dataset found in {path}"
        raise FileNotFoundError(msg)

    if not any("xpos.npy" in str(f) for f in files):
        has_positions = False
    else:
        xpos = np.load(path / "xpos.npy")
        freqs = np.load(path / "fund_v.npy")
        if len(xpos) == 0 and len(freqs) > 0:
            has_positions = False

    if has_positions:
        xpos = np.load(path / "xpos.npy")
        ypos = np.load(path / "ypos.npy")
    else:
        xpos = np.ndarray([])
        ypos = np.ndarray([])

    return WavetrackerData(
        freqs=np.load(path / "fund_v.npy"),
        powers=np.load(path / "sign_v.npy"),
        idents=np.load(path / "ident_v.npy"),
        indices=np.load(path / "idx_v.npy"),
        times=np.load(path / "times.npy"),
        ids=np.unique(np.load(path / "ident_v.npy")),
        has_positions=has_positions,
        xpos=xpos,
        ypos=ypos,
        path=path,
    )


def load_grid(path: pathlib.Path) -> GridData:
    """Load a raw dataset from a given path.

    Parameters
    ----------
    - `path` : `pathlib.Path`
        The path to the directory containing the raw dataset.

    Returns
    -------
    - `GridData`
        An object containing the loaded raw dataset.

    Raises
    ------
    - `FileNotFoundError`
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
    import pathlib
    from gridtools.datasets load_grid
    rec = load_grid(pathlib.Path("path/to/raw"))
    rec.pprint()
    ```
    """
    # check if dir exists
    if not path.exists():
        msg = f"Directory {path} does not exist."
        raise FileNotFoundError(msg)

    rec_path = path / "recordings"
    wav_files = sorted(list(path.glob("*wav")))
    raw_files = sorted(list(path.glob("*raw")))
    rec_wav_files = sorted(list(rec_path.glob("*wav")))

    has_files = False
    files = []
    if len(wav_files) > 0:
        files = wav_files
        has_files = True
    if len(raw_files) > 0:
        files = raw_files
        has_files = True
    if len(rec_wav_files) > 0:
        files = rec_wav_files
        has_files = True

    if not has_files:
        msg = f"No raw dataset found in {path}"
        raise FileNotFoundError(msg)

    # Check if single file or multi file dataset
    load_path = [str(file) for file in files]
    if len(files) > 1:
        msg = (
            f"Warning: More than one raw dataset found in {path}"
            "Assuming multi-file .wav dataset."
        )
    else:
        load_path = str(files[0])

    print(f"Load path: {load_path}")
    print(f"Files: {files}")

    rec = DataLoader(load_path)
    shape = rec.shape
    samplerate = rec.rate

    if not isinstance(samplerate, float):
        msg = "DataLoader samplerate must be a float."
        raise TypeError(msg)
    if not isinstance(shape, tuple):
        msg = (
            "DataLoader shape must be a tuple."
            f"Error loading raw dataset in {path}"
        )
        raise TypeError(msg)
    if not isinstance(shape[0], int):
        msg = (
            "DataLoader shape must have at least one dimension."
            f"Error loading raw dataset in {path}."
        )
        raise TypeError(msg)

    return GridData(
        rec=rec,
        samplerate=samplerate,
        shape=shape,
        path=path,
    )


def load_chirps(path: pathlib.Path) -> ChirpData:
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

    det = None
    are_detected = False
    have_params = False
    chirp_times = np.array([])
    chirp_ids = np.array([])
    params = np.array([])

    for detector in chirp_detectors:
        print(detector)
        if any(f"chirp_params_{detector}.npy" in str(f) for f in files):
            params = np.load(path / f"chirp_params_{detector}.npy")
            have_params = True
        if any(f"chirp_times_{detector}.npy" in str(f) for f in files):
            det = detector
            are_detected = True
            chirp_times = np.load(path / f"chirp_times_{detector}.npy")
            chirp_ids = np.load(path / f"chirp_ids_{detector}.npy").astype(int)
            break

    return ChirpData(
        times=chirp_times,
        idents=chirp_ids,
        params=params,
        detector=str(det),
        are_detected=are_detected,
        have_params=have_params,
        path=path,
    )


def load_rises(path: pathlib.Path) -> RiseData:
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

    det = None
    are_detected = False
    have_params = False
    rise_times = np.array([])
    rise_ids = np.array([])
    params = np.array([])

    for detector in rise_detectors:
        if any(f"rise_params_{det}.npy" in str(f) for f in files):
            params = np.load(path / f"rise_params_{det}.npy")
            have_params = True
        if any(f"rise_times_{detector}.npy" in str(f) for f in files):
            det = detector
            are_detected = True
            rise_times = np.load(path / f"rise_times_{det}.npy")
            rise_ids = np.load(path / f"rise_ids_{det}.npy").astype(int)
            break

    return RiseData(
        times=rise_times,
        idents=rise_ids,
        params=params,
        detector=str(det),
        are_detected=are_detected,
        have_params=have_params,
        path=path,
    )


def load_com(path: pathlib.Path) -> CommunicationData:
    """Load communication data from a given path.

    Loads chirps if available, loads
    rises if available, or loads both if available.
    Communication signals are point events, i.e. they have a time stamp and an
    identifier.

    Data on disk must follow the following naming convention:
    - {signal type}_times_{detector handle}.npy
    - {signal type}_ids_{detector handle}.npy

    For example, chirp data is loaded from the following files
    for the ground truths (gt) or the cnn detector (cnn):
    Which detectors are available and in which order they are checked is
    determined by the chirp_detectors and rise_detectors lists in this file.
    - chirp_times_gt.npy
    - chirp_ids_gt.npy
    - chirp_times_cnn.npy
    - chirp_ids_cnn.npy
    - ...

    If no detector handle is found, the data is not loaded. If in the future,
    better detectors are available, they need to be added to this function.

    Parameters
    ----------
    - `path` : `pathlib.Path`
        The path to the directory containing the communication data.

    Returns
    -------
    - `CommunicationData`
        An object containing the loaded communication data.

    Raises
    ------
    - `FileNotFoundError`
        If no chirp or rise dataset with the correct detector handle is found.

    Examples
    --------
    ```python
    import pathlib
    from gridtools.datasets import load_com
    com = load_com(pathlib.Path("path/to/communication"))
    # get rise times and ids
    com.rise.times
    com.rise.idents
    ```
    """
    chirp = load_chirps(path)
    rise = load_rises(path)
    are_detected = False
    if chirp.are_detected or rise.are_detected:
        are_detected = True
    return CommunicationData(chirp=chirp, rise=rise, are_detected=are_detected)


def find_derived_path(raw_path: pathlib.Path) -> pathlib.Path:
    """
    Finds the derived data path based on the raw data path.

    Parameters
    ----------
    - `raw_path` : `pathlib.Path`
        The path to the raw data.

    Returns
    -------
    - `pathlib.Path`
        The path to the derived data if found, otherwise returns the raw path.
    """
    parent = raw_path.parent.parent
    print(parent)
    derived_candidates = [
        "derived_data",
        "intermediate",
        "interim",
        "derived",
        "intermediate_data",
        "interim_data",
    ]

    found_derived_path = False
    derived_path = raw_path
    for candidate in derived_candidates:
        derived_path = parent / candidate / raw_path.parts[-1]
        if derived_path.exists():
            found_derived_path = True
            break

    found_wavetracker_files = False
    files = list(derived_path.glob("*.npy"))
    if any("fund_v.npy" in str(f) for f in files):
        found_wavetracker_files = True
    else:
        derived_path = derived_path / "recordings"

    files = list(derived_path.glob("*.npy"))
    if any("fund_v.npy" in str(f) for f in files):
        found_wavetracker_files = True

    if found_derived_path and found_wavetracker_files:
        return derived_path
    msg = "Could not find wavetracker (derived) data!"
    raise FileNotFoundError(msg)


def load(path: pathlib.Path, search_intermediate: bool = False) -> Dataset:
    """
    Load all data from a dataset and build a Dataset object.

    Parameters
    ----------
    - `path` : `pathlib.Path`
        The path to the dataset.

    - `seach_intermediate` : `bool`
        If True, search for intermediate data.

    Returns
    -------
    - `Dataset`
        A Dataset object containing the raw data, wavetracker data, and
        communication data.

    Examples
    --------
    ```python
    import pathlib
    from gridtools.datasets import load
    ds = load(pathlib.Path("path/to/dataset"))
    ds.track.freqs
    ds.rec.raw
    ds.com.chirp.times
    ds.com.chirp.idents
    ds.track.xpos
    ds.pprint()
    ```
    """
    derived_path = path
    print(path.parts[-2])
    if search_intermediate:
        raw_keywords = ["raw", "raw_data"]
        if any(keyword in path.parts[-2].lower() for keyword in raw_keywords):
            print("searching for intermediate data")
            derived_path = find_derived_path(path)

    print(derived_path)

    return Dataset(
        path=path,
        grid=load_grid(path),
        track=load_wavetracker(derived_path),
        com=load_com(derived_path),
    )
