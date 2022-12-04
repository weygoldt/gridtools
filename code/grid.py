import datetime
import logging
import os

import numpy as np
from makeLogger import makeLogger
from tqdm import tqdm

logger = makeLogger(__name__)


class CoustomError(Exception):
    def __init__(self, value: str, message: str) -> None:
        self.value = value
        self.message = message
        super().__init__(message)


class RawGrid:
    def __init__(self, datapath: str, finespec: bool) -> None:

        # print a message
        logger.info("Initialising grid instance ...")

        # initialize directory variables
        self._datapath = datapath
        self._dataroot = os.path.split(self._datapath[:-1])

        # to load or not to load the fine spectrogram
        self._finespec = finespec

        # initialize list to track functions applied to dataset
        self.type = ["raw"]

        # try to load wavetracker output files
        try:
            self.times = np.load(datapath + "times.npy", allow_pickle=True)
            self.meta = np.load(datapath + "meta.npy", allow_pickle=True)
            self.idx_v = np.load(datapath + "idx_v.npy", allow_pickle=True)
            self.spec = np.load(datapath + "spec.npy", allow_pickle=True)
            self.fund_v = np.load(datapath + "fund_v.npy", allow_pickle=True)
            self.sign_v = np.load(datapath + "sign_v.npy", allow_pickle=True)
            self.ident_v = np.load(datapath + "ident_v.npy", allow_pickle=True)
            self.ids = np.unique(self.ident_v[~np.isnan(self.ident_v)])
        except FileNotFoundError as error:
            logger.error(error)

        # try to load hobologger data
        try:
            self.temp = np.load(datapath + "temp.npy", allow_pickle=True)
            self.light = np.load(datapath + "light.npy", allow_pickle=True)
        except FileNotFoundError as error:
            logger.warn(error)

        # try to load fine spectrogram if specified
        if self._finespec == True:
            try:
                self.fill_freqs = np.load(
                    datapath + "fill_freqs.npy", allow_pickle=True)
                self.fill_times = np.load(
                    datapath + "fill_times.npy", allow_pickle=True)
                self.fill_spec_shape = np.load(
                    datapath + "fill_spec_shape.npy", allow_pickle=True
                )
                self.fill_spec = np.memmap(
                    datapath + "fill_spec.npy",
                    dtype="float",
                    mode="r",
                    shape=(self.fill_spec_shape[0], self.fill_spec_shape[1]),
                    order="F",
                )
            except FileNotFoundError as error:
                logger.error(error)

        # try to load position estimates

    @property
    def rec_datetime(self):

        # get the folder we are in
        folder = self._datapath[:-1]

        # split the folder name into year, month, day, etc.
        try:
            rec_year, rec_month, rec_day, rec_time = os.path.split(
                os.path.split(folder)[-1]
            )[-1].split("-")
        except ValueError:
            logger.error(
                "The directory name does not match the datetime naming pattern!")
            return None

        # make them integers
        rec_year = int(rec_year)
        rec_month = int(rec_month)
        rec_day = int(rec_day)

        # try to split time stamp
        try:
            rec_time = [
                int(rec_time.split("_")[0]),
                int(rec_time.split("_")[1]),
                0,
            ]
        except ValueError:
            try:
                rec_time = [
                    int(rec_time.split(":")[0]),
                    int(rec_time.split(":")[1]),
                    0,
                ]
            except ValueError:
                logging.error(
                    "Time string of folder name does not contain '_' or ':' !")

            # combine all to datetime object
        rec_datetime = datetime.datetime(
            year=rec_year,
            month=rec_month,
            day=rec_day,
            hour=rec_time[0],
            minute=rec_time[1],
            second=rec_time[2],
        )

        return rec_datetime


if __name__ == "__main__":
    datapath = "../data/"
    g = Grid(datapath, False)
