#!/usr/bin/env python3

"""
A module that keeps common tools to handle the two kinds of config 
files used in this project: simulation config files and preprocessing
config files.
"""

import pathlib
import shutil

import toml
from pydantic import BaseModel, ConfigDict


def copy_config(path: str, configfile: str) -> None:
    """
    Copy the default config file from the package root directory into a
    specified path. Depenting on the configfile str, the preprocessing
    or simulation config file is copied.
    """

    assert isinstance(path, str), "The path argument must be a string."
    assert isinstance(
        configfile, str
    ), "The configfile argument must be a string."
    assert configfile in [
        "preprocessing",
        "simulations",
    ], "The configfile argument must be either 'preprocessing' or 'simulation'."

    if configfile == "preprocessing":
        configfile = configfile + ".toml"
    if configfile == "simulations":
        configfile = configfile + ".toml"

    origin = pathlib.Path(__file__).parent.parent / configfile
    if not origin.exists():
        raise FileNotFoundError(
            f"Could not find the default config file for {configfile}. "
            f"Please make sure that the file '{configfile}.toml' exists in "
            "the package root directory."
        )

    destination = pathlib.Path(path)

    if destination.is_dir():
        shutil.copy(origin, destination / f"gridtools_{configfile}")

    elif destination.is_file():
        raise FileExistsError(
            "The specified path already exists and is a file. "
            "Please specify a directory or a non-existing path."
        )

    elif not destination.exists():
        raise FileNotFoundError("Please specify an existing directory.")


class SimulationConfigMeta(BaseModel):
    """
    Load the meta chapter in the simulation config file.
    """

    ngrids: int


class SimulationConfigGrid(BaseModel):
    """
    Load the grid chapter in the simulation config file.
    """

    samplerate: int
    wavetracker_samplerate: int
    duration: float
    origin: tuple
    shape: tuple
    spacing: float
    style: str
    boundaries: tuple
    downsample_lowpass: float


class SimulationConfigFish(BaseModel):
    """
    Simulation parameters for the fish.
    """

    nfish: tuple
    eodfrange: tuple
    noise_std: float
    eodfnoise_std: float
    eodfnoise_band: tuple


class SimulationConfigChirps(BaseModel):
    """
    Load chirp config for the simulation.
    """

    min_chirp_dt: float
    max_chirp_freq: float
    max_chirp_contrast: float
    chirpnoise_std: float
    chirpnoise_band: tuple
    chirp_params_path: str


class SimulationConfig(BaseModel):
    """
    The main config object for the simulation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    path: str
    meta: SimulationConfigMeta
    grid: SimulationConfigGrid
    fish: SimulationConfigFish
    chirps: SimulationConfigChirps


class PreprocessingConfig(BaseModel):
    """
    The main config object for preprocessing.
    """

    pass


def load_sim_config(config_file: str) -> SimulationConfig:
    """
    Load the simulation config file.

    Parameters
    ----------
    config_file : str
        The path to the YAML file to load.

    Returns
    -------
    SimulationConfig
        The simulation config.
    """
    config_dict = toml.load(config_file)
    meta = SimulationConfigMeta(**config_dict["meta"])
    grid = SimulationConfigGrid(**config_dict["grid"])
    fish = SimulationConfigFish(**config_dict["fish"])
    chirps = SimulationConfigChirps(**config_dict["chirps"])
    config = SimulationConfig(
        path=config_file, meta=meta, grid=grid, fish=fish, chirps=chirps
    )
    return config


def load_prepro_config(config_file: str) -> PreprocessingConfig:
    """
    Load the preprocessing config file.

    Parameters
    ----------
    config_file : str
        The path to the YAML file to load.

    Returns
    -------
    PreprocessingConfig
        The preprocessing config.
    """
    config_dict = toml.load(config_file)
    config = PreprocessingConfig(**config_dict)
