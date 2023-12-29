"""Functions for making subsets of grid datasets."""

from typing import Union
import pathlib

import numpy as np

from gridtools.datasets.models import (
    ChirpData,
    CommunicationData,
    Dataset,
    GridData,
    RiseData,
    WavetrackerData,
)
from gridtools.datasets.loaders import load
from gridtools.datasets.savers import save

def subset_wavetracker(
    wt: WavetrackerData,
    start: Union[float, int],
    stop: Union[float, int],
    mode: str = "time",
    samplerate: float = 20000.0,
) -> WavetrackerData:
    """Extract a subset of a WavetrackerData object.

    Parameters
    ----------
    - `wt` : `WavetrackerData`
        The WavetrackerData object to extract a subset from.
    - `start` : `float`
        The start time or index of the subset.
    - `stop` : `float`
        The stop time or index of the subset.
    - `mode` : `str`
        Whether to use time or index method.
    - `samplerate` : `int`
        Samplerate to use for conversion between time and index.

    Returns
    -------
    - `WavetrackerData`
        A new WavetrackerData object containing the subset of data between
        start_time and stop_time.

    Raises
    ------
    - `GridDataMismatch`
        If there is no data in the specified time range.

    Example
    -------
    ```python
    import pathlib
    from gridtools.datasets import load_wavetracker, subset_wavetracker
    wt = load_wavetracker(pathlib.Path("path/to/wavetracker"))
    wt_sub = subset_wavetracker(wt, 0.5, 1.5)
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

        if wt.has_positions:
            x = wt.xpos[wt.idents == track_id]
            y = wt.ypos[wt.idents == track_id]
            x = x[(time >= start_time) & (time <= stop_time)]
            y = y[(time >= start_time) & (time <= stop_time)]
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
    time = wt.times[(wt.times >= start_time) & (wt.times <= stop_time)]
    time -= start_time

    if wt.has_positions:
        xpos = np.concatenate(xpos)
        ypos = np.concatenate(ypos)
    else:
        xpos = np.ndarray([])
        ypos = np.ndarray([])

    # reset index array so that it fits on the now shorter time array
    if len(indices) > 0:
        indices -= indices[0]

    # rebuild wavetracker object
    return WavetrackerData(
        freqs=tracks,
        powers=powers,
        idents=idents,
        indices=indices,
        ids=np.unique(idents),
        times=time,
        xpos=xpos,
        ypos=ypos,
        has_positions=wt.has_positions,
    )


def subset_grid(
    rec: "GridData",
    start: float,
    stop: float,
    mode: str = "time",
    samplerate: float = 20000.0,
) -> "GridData":
    """Return a subset of a raw dataset.

    Parameters
    ----------
    - `rec` :`GridData`
        The raw dataset to subset.
    - `start` : `float`
        The start time / index of the subset.
    - `stop` : `float`
        The stop time / index of the subset.
    - `mode` : `str`
        Whether to use time or index method.
    - `samplerate` : `int`
        Samplerate to use for conversion between time and index.

    Returns
    -------
    - `GridData`
         The subset of the raw dataset.

    Examples
    --------
    ```python
    import pathlib
    from gridtools.datasets import load_grid,subset_grid
    rec = load_grid(pathlib.Path("path/to/raw")
    subset = subset_grid(rec, 0.1, 0.5)
    ```
    """
    assert mode in ["time", "index"], "Mode must be either 'time' or 'index'."

    # make the boundaries for the subset
    if mode == "index":
        start_time = start / samplerate
        stop_time = stop / samplerate
    else:
        start_time = start
        stop_time = stop

    # check if content of the thunderfish dataloader is correct
    if not isinstance(rec.samplerate, float):
        msg = "Samplerate must be a float."
        raise TypeError(msg)
    rec_shape = rec.shape
    if not isinstance(rec_shape, tuple):
        msg = "Raw data must have a shape."
        raise TypeError(msg)
    if not isinstance(rec_shape[0], int):
        msg = "Raw data must have at least one dimension."
        raise TypeError(msg)

    # check that boundaries make sense given the data
    assert start_time < stop_time, "Start time must be smaller than stop time."
    assert start_time >= 0, "Start time must be larger or equal to 0."
    assert (
        stop_time <= rec_shape[0] / rec.samplerate
    ), "Stop time must be smaller than the end."
    start_idx = int(start_time * rec.samplerate)
    stop_idx = int(stop_time * rec.samplerate)
    assert start_idx < rec_shape[0], "Start index out of bounds."
    assert stop_idx <= rec_shape[0], "Stop index out of bounds."

    raw = rec.rec[start_idx:stop_idx, :]
    return GridData(rec=raw, samplerate=rec.samplerate, shape=rec.shape)


def subset_com(
    com: CommunicationData,
    start: float,
    stop: float,
    mode: str = "time",
    samplerate: float = 20000.0,
) -> CommunicationData:
    """Make a subset of a communication dataset.

    Parameters
    ----------
    - `com` : `CommunicationData`
        The communication dataset to subset.
    - `start` : `float`
        The start time / index of the subset.
    - `stop` : `float`
        The stop time / index of the subset.
    - `mode` : `str`
        Whether to use time or index method.
    - `samplerate` : `int`
        Samplerate to use for conversion between time and index.

    Returns
    -------
    - `CommunicationData`
        The subset of the communication dataset.

    Examples
    --------
    ```python
    import pathlib
    from gridtools.datasets import load_com, subset_com
    com = load_com(pathlib.Path("path/to/communication"))
    subset = subset_com(com, 0.1, 0.5)
    ```
    """
    assert mode in ["time", "index"], "Mode must be either 'time' or 'index'."

    if mode == "index":
        start_time = start / samplerate
        stop_time = stop / samplerate
    else:
        start_time = start
        stop_time = stop

    com_are_detected = False
    if com.chirp.are_detected:
        com_are_detected = True
        ci = com.chirp.idents[
            (com.chirp.times >= start_time) & (com.chirp.times <= stop_time)
        ]
        ct = com.chirp.times[
            (com.chirp.times >= start_time) & (com.chirp.times <= stop_time)
        ]
        cp = np.array([])
        if com.chirp.have_params:
            cp = com.chirp.params[
                (com.chirp.times >= start_time)
                & (com.chirp.times <= stop_time)
            ]
        chirp = ChirpData(
            times=ct,
            idents=ci,
            params=cp,
            detector=com.chirp.detector,
            are_detected=com.chirp.are_detected,
            have_params=com.chirp.have_params,
        )
    else:
        chirp = ChirpData(
            times=np.array([]),
            idents=np.array([]),
            params=np.array([]),
            detector="None",
            are_detected=False,
            have_params=False,
        )

    if com.rise.are_detected:
        com_are_detected = True
        ri = com.rise.idents[
            (com.rise.times >= start_time) & (com.rise.times <= stop_time)
        ]
        rt = com.rise.times[
            (com.rise.times >= start_time) & (com.rise.times <= stop_time)
        ]
        rp = np.array([])
        if com.rise.have_params:
            rp = com.rise.params[
                (com.rise.times >= start_time) & (com.rise.times <= stop_time)
            ]
        rise = RiseData(
            times=rt,
            idents=ri,
            params=rp,
            detector=com.rise.detector,
            are_detected=com.rise.are_detected,
            have_params=com.rise.have_params,
        )
    else:
        rise = RiseData(
            times=np.array([]),
            idents=np.array([]),
            params=np.array([]),
            detector="None",
            are_detected=False,
            have_params=False,
        )

    return CommunicationData(
        chirp=chirp, rise=rise, are_detected=com_are_detected
    )


def subset(
    data: Dataset, start: float, stop: float, mode: str = "time"
) -> Dataset:
    """Make a subset of a full dataset.

    Parameters
    ----------
    - `data` : `Dataset`
        The full dataset to be subsetted.
    - `start` : `float`
        The start time / index of the subset, in seconds.
    - `stop` : `float`
        The stop time / index of the subset, in seconds.
    - `mode` : `str = "time" or "index"`
        Whether to use time or index method.

    Returns
    -------
    - `Dataset`
        The subsetted dataset.

    Raises
    ------
    - `AssertionError`
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
    import pathlib
    from gridtools.datasets import load, subset, save
    ds = load(pathlib.Path("path/to/dataset"))
    subset = subset(ds, 0.1, 0.5)
    save(subset, pathlib.Path("path/to/save"))
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
    raw_sub = subset_grid(
        data.grid, start_time, stop_time, samplerate=samplerate
    )
    com_sub = subset_com(
        data.com, start_time, stop_time, samplerate=samplerate
    )

    new_path = (
        data.path.parent
        / f"subset_{data.path.name}_t0_{start_time}_t1_{stop_time}"
    )

    return Dataset(path=new_path, grid=raw_sub, track=wt_sub, com=com_sub)


def subset_cli(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    start_time: float,
    end_time: float,
) -> None:
    """Subset a dataset to a given time range.

    Parameters are passed via the command line.

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
    ds = load(input_path)
    ds_sub = subset(ds, start_time, end_time)
    print(f"Saving dataset to {output_path}")
    save(ds_sub, output_path)
