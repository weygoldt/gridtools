"""Functions for saving grid datasets in various formats."""

import pathlib

import numpy as np
from thunderfish.datawriter import write_data

from gridtools.datasets.models import (
    CommunicationData,
    Dataset,
    GridData,
    WavetrackerData,
)


def save_wavetracker(wt: WavetrackerData, output_path: pathlib.Path) -> None:
    """Save WavetrackerData object to disk as numpy files.

    ...like the original wavetracker data.

    Parameters
    ----------
    - `wt` : `WavetrackerData`
        WavetrackerData object to save.
    - `output_path` : `pathlib.Path`
        Path to save the object to.

    Returns
    -------
    - `None`

    Examples
    --------
    ```python
    import pathlib
    from gridtools.datasets import load_wavetracker
    wt = load_wavetracker(pathlib.Path("path/to/wavetracker"))
    wt.pprint()
    save_wavetracker(wt, pathlib.Path("path/to/save"))
    ```
    """
    if not output_path.exists():
        msg = (
            f"❗ Output directory {output_path} does not exist."
            "Please provide an existing directory."
        )
        raise FileNotFoundError(msg)

    np.save(output_path / "fund_v.npy", wt.freqs)
    np.save(output_path / "sign_v.npy", wt.powers)
    np.save(output_path / "ident_v.npy", wt.idents)
    np.save(output_path / "idx_v.npy", wt.indices)
    np.save(output_path / "times.npy", wt.times)
    if wt.has_positions:
        np.save(output_path / "xpos.npy", wt.xpos)
        np.save(output_path / "ypos.npy", wt.ypos)


def save_grid(rec: GridData, output_path: pathlib.Path) -> None:
    """Save raw data to a WAV file using `thunderfish.datawriter`.

    Parameters
    ----------
    - `rec` : `GridData`
        The raw data to be saved.
    - `output_path` : `pathlib.Path`
        The path to save the file to.

    Example
    -------
    ```python
    import pathlib
    from gridtools.datasets import load_grid,save_grid
    rec = load_grid(pathlib.Path("path/to/raw"))
    save_grid(rec, pathlib.Path("path/to/save"))
    ```
    """
    if not output_path.exists():
        msg = (
            f"❗ Output directory {output_path} does not exist."
            "Please provide an existing directory."
        )
        raise FileNotFoundError(msg)

    write_data(
        str(output_path / "traces_grid1.wav"),
        rec.rec,
        rec.samplerate,
        verbose=2,
    )


def save_com(com: CommunicationData, output_path: pathlib.Path) -> None:
    """Save communication data to disk.

    Parameters
    ----------
    - `com` : `CommunicationData`
        The communication data to save.
    - `output_path` : `pathlib.Path`
        The path to the directory where the data should be saved.

    Returns
    -------
    - `None`

    Example
    -------
    import pathlib
    from gridtools.datasets import load_com, save_com
    com = load_com(pathlib.Path("path/to/communication"))
    save_com(com, pathlib.Path("path/to/save"))
    """
    if not output_path.exists():
        msg = (
            f"❗ Output directory {output_path} does not exist."
            "Please provide an existing directory."
        )
        raise FileNotFoundError(msg)

    if com.chirp.are_detected:
        np.save(
            output_path / f"chirp_times_{com.chirp.detector}.npy",
            com.chirp.times,
        )
        np.save(
            output_path / f"chirp_ids_{com.chirp.detector}.npy",
            com.chirp.idents,
        )
    if com.chirp.have_params and com.chirp.are_detected:
        np.save(
            output_path / f"chirp_params_{com.chirp.detector}.npy",
            com.chirp.params,
        )

    if com.rise.are_detected:
        np.save(
            output_path / f"rise_times_{com.rise.detector}.npy", com.rise.times
        )
        np.save(
            output_path / f"rise_ids_{com.rise.detector}.npy", com.rise.idents
        )
    if com.rise.have_params and com.rise.are_detected:
        np.save(
            output_path / f"rise_params_{com.rise.detector}.npy",
            com.rise.params,
        )


def save(dataset: Dataset, output_path: pathlib.Path) -> None:
    """Save a Dataset object to disk.

    This function saves the wavetracker data,
    the raw data and the communication data to disk, depending on what is
    available in the Dataset object.

    Parameters
    ----------
    - `dataset` : `Dataset`
        Dataset to save to file.
    - `output_path` : `pathlib.Path`
        Path where to save the dataset.

    Raises
    ------
    - `FileExistsError`
        When there already is a dataset.

    Example
    -------
    ```python
    import pathlib
    from gridtools.datasets import load, save
    ds = load(pathlib.Path("path/to/dataset"))
    save(ds, pathlib.Path("path/to/save"))
    ```
    """
    output_dir = output_path / dataset.path.name

    if output_dir.exists():
        msg = (
            f"🚫 Output directory {output_dir} already exists."
            "Aborting to prevent overwriting existing data."
            "Please delete the directory or choose a different output path."
        )
        raise FileExistsError(msg)

    output_dir.mkdir(parents=True)

    save_wavetracker(dataset.track, output_dir)
    save_grid(dataset.grid, output_dir)
    if dataset.com.are_detected:
        save_com(dataset.com, output_dir)
