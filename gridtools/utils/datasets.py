#!/usr/bin/env python3

"""
Classes to load wavetracker-generated dataset. 
"""

import datetime
import pathlib

import numpy as np
from thunderfish import dataloader
from thunderfish.dataloader import DataLoader

from ..exceptions.exceptions import GridDataMismatch, GridDataMissing
from ..utils.logger import make_logger

logger = make_logger(__name__)


class WTData:
    """
    Base class for loading wavetracker-generated dataset.
    """

    def __init__(self, path: pathlib.Path):
        self.path = path

        logger.info("Loading dataset from %s", self.path)
        self._load_dataset()
        self._check_health()

    @property
    def start_time(self):
        year, month, day, time = self.path.name("-")
        try:
            hour, minute = time.split("_")
        except ValueError:
            hour, minute = time.split(":")

        start_time = datetime.datetime(
            int(year), int(month), int(day), int(hour), int(minute)
        )
        return start_time

    @property
    def stop_time(self):
        return self.start_time + datetime.timedelta(seconds=self.track_times[-1])

    def _load_dataset(self):
        """
        Load dataset.
        """
        self.track_times = np.load(self.path / "times.npy")
        self.track_freqs = np.load(self.path / "fund_v.npy")
        self.track_powers = np.load(self.path / "sign_v.npy")
        self.track_idents = np.load(self.path / "ident_v.npy")
        self.track_indices = np.load(self.path / "idx_v.npy")
        self.ids = np.unique(self.track_idents[~np.isnan(self.track_idents)])
        self.grid_rate = dataloader.fishgrid_samplerate(self.path)
        self.grid_spacing = dataloader.fishgrid_spacings(self.path)
        self.grid_shape = dataloader.fishgrid_grids(self.path)

        logger.info("Dataset loaded from %s", self.path)

    def _check_health(self):
        """
        Check dataset health.
        """
        time_length = self.track_times.shape[0]
        freq_length = self.track_freqs.shape[0]
        power_length = self.track_powers.shape[0]
        index_length = self.track_indices.shape[0]
        unique_index_length = np.unique(self.track_indices).shape[0]
        ident_length = self.track_idents.shape[0]
        power_electrode_count = self.track_powers.shape[1]
        grid_electrode_count = np.prod(self.grid_shape)

        if time_length < unique_index_length:
            raise GridDataMismatch("Index vector is longer than time vector!.")
        else:
            logger.info("Passed check: Index vector is not longer than time vector.")

        if time_length < np.max(self.track_indices):
            raise GridDataMismatch(
                "Index vector contains indices larger than time vector!"
            )
        else:
            logger.info(
                "Passed check: Index vector does not contain indices larger than time vector."
            )

        if index_length == freq_length == power_length == ident_length:
            logger.info("Passed check: All tracked vectors have the same length.")
        else:
            raise GridDataMismatch("Not all tracked vectors have the same length!")

        if power_electrode_count == grid_electrode_count:
            logger.info(
                "Passed check: Power vector has the number of electrodes as the grid."
            )
        else:
            raise GridDataMismatch(
                "Power vector has not the same number of electrodes as the grid!"
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(path={self.path})"

    def __str__(self):
        return f"{self.__class__.__name__}(path={self.path})"


class WTRaw(WTData):
    """
    Class to load wavetracker-generated dataset with raw data.
    """

    def __init__(self, path: pathlib.Path):
        super().__init__(path)

    def _load_dataset(self):
        raw_path = self.path / "traces-grid1.raw"
        if not raw_path.exists():
            raw_path = self.path / "raw.npy"
        if not raw_path.exists():
            raise GridDataMissing(
                f"Raw data missing from {self.path}. Please run wavetracker first."
            )
        self.raw = DataLoader(raw_path, buffersize=60, channel=-1)
        return super()._load_dataset()

    def _check_health(self):
        if self.raw.shape[1] != self.track_powers.shape[1]:
            raise GridDataMismatch(
                "Raw data has not the same number of electrodes as power vector!"
            )
        else:
            logger.info("Passed check: Raw data has the same length as power vector.")
        return super()._check_health()

    def __repr__(self):
        return super().__repr__() + f"\nRaw data: {self.raw.shape}"

    def __str__(self):
        return super().__str__() + f"\nRaw data: {self.raw.shape}"


class WTPreprocessing(WTData):
    """
    Class to load wavetracker-generated dataset and initialize empty
    class attributes to fill in during preprocessing.
    """

    def __init__(self, path: pathlib.Path):
        super().__init__(path)
        self._init_preprocessing()

    def _init_preprocessing(self):
        self.temperature = None
        self.light = None
        self.q10 = None
        self.sex = None
        self.x = None
        self.y = None

    def __repr__(self):
        return super().__repr__() + f"\nPreprocessing data: {self.temperature.shape}"

    def __str__(self):
        return super().__str__() + f"\nPreprocessing data: {self.temperature.shape}"


class WTProcessed(WTData):
    """
    Class to load wavetracker-generated dataset with processed data.
    """

    def __init__(self, path: pathlib.Path):
        super().__init__(path)

    def _load_dataset(self):
        self.temperature = np.load(self.path / "temperature.npy")
        self.light = np.load(self.path / "light.npy")
        self.q10 = np.load(self.path / "q10.npy")
        self.sex = np.load(self.path / "sex.npy")
        self.x = np.load(self.path / "x.npy")
        self.y = np.load(self.path / "y.npy")
        return super()._load_dataset()

    def _check_health(self):
        if self.temperature.shape[0] != self.light.shape[0]:
            raise GridDataMismatch(
                "Temperature vector has not the same length as light vector!"
            )
        else:
            logger.info(
                "Passed check: Temperature vector has the same length as light vector."
            )
        if self.ids.shape[0] == self.q10.shape[0] == self.sex.shape[0]:
            logger.info("Passed check: All metadata vectors have the same length.")
        else:
            raise GridDataMismatch("Not all metadata vectors have the same length!")
        if self.x.shape[0] == self.y.shape[0] == self.track_freqs:
            logger.info(
                "Passed check: All positions have the same length as tracked vectors."
            )
        else:
            raise GridDataMismatch(
                "Not all positions have the same length as tracked vectors!"
            )
        return super()._check_health()

    def __repr__(self):
        return super().__repr__() + f"\nProcessed data: {self.temperature.shape}"

    def __str__(self):
        return super().__str__() + f"\nProcessed data: {self.temperature.shape}"
