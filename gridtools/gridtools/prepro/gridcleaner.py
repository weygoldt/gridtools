import datetime
import logging
import os

import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter
from thunderfish import dataloader, powerspectrum
from tqdm import tqdm

from ..exceptions import GridDataMismatch, GridDataMissing
from ..logger import makeLogger
from ..toolbox.datahandling import (estimateMode, findClosest, findOnTime,
                                    normQ10)
from ..toolbox.spatial import velocity2d

logger = makeLogger(__name__)


class GridCleaner:
    def __init__(self, datapath: str) -> None:

        # print a message
        logger.info("Initialising grid instance ...")

        # initialize directory variables
        self._datapath = datapath
        self._dataroot = os.path.split(self._datapath[:-1])

        # initialize list to track functions applied to dataset
        self.type = ["wavetracker output", ]

        # initialize class data that is not loaded yet
        self.temp = None
        self.light = None
        self.sex = None
        self.xpos = None
        self.ypos = None

        # try to load wavetracker output files
        try:
            # the wavetracker output
            self.times = np.load(datapath + "times.npy", allow_pickle=True)
            self.idx_v = np.load(datapath + "idx_v.npy", allow_pickle=True)
            self.fund_v = np.load(datapath + "fund_v.npy", allow_pickle=True)
            self.sign_v = np.load(datapath + "sign_v.npy", allow_pickle=True)
            self.ident_v = np.load(datapath + "ident_v.npy", allow_pickle=True)
            self.ids = np.unique(self.ident_v[~np.isnan(self.ident_v)])

            # some grid metadata
            self.grid_rate = dataloader.fishgrid_samplerate(self._datapath)
            self.grid_spacings = dataloader.fishgrid_spacings(self._datapath)
            self.grid_grid = dataloader.fishgrid_grids(self._datapath)

        except FileNotFoundError as error:
            logger.error(error)

        # check if files are have no errors
        try:
            # get data for checks
            time_len = len(self.times)
            idx_len = len(self.idx_v)
            uidx_len = len(np.unique(self.idx_v))
            fund_len = len(self.fund_v)
            sign_len = np.shape(self.sign_v[:, 0])[0]
            ident_len = len(self.ident_v)
            sign_elno = np.shape(self.sign_v[0, :])[0]
            grid_elno = np.prod(self.grid_grid)

            # check idx and time match in len
            if time_len < uidx_len:
                raise GridDataMismatch(
                    f"Index vector is longer than time vector! Time vector lenght: {time_len} Unique index vector length: {uidx_len}!")
            else:
                logger.debug('Passed time-index length check!')

            # check idx and time match in indexing value
            if time_len < np.max(self.idx_v):
                raise GridDataMismatch(
                    f"Too many indices for the time vector! Time vector lenght: {time_len} Index out of bounds: {np.max(self.idx_v)}!")
            else:
                logger.debug('Passed time-index bounds check!')

            # check if idx len fits to fund, sign, ident
            if idx_len == fund_len == sign_len == ident_len:
                logger.debug('Passed data array-length check!')
            else:
                raise GridDataMismatch(
                    "Lengths of idx, fund, sign and ident are not the same!")

            # check if electrode number matches in sign_v
            if sign_elno == grid_elno:
                logger.debug("Passed electrode number check!")
            else:
                raise GridDataMismatch(
                    'Number of electrodes in sign_v does not match grid metadata!')

            logger.info('Grid initialized succesfully!')

        except GridDataMismatch as error:
            logging.error(str(error))
            raise error

    @ property
    def starttime(self) -> datetime.datetime:

        logger.debug("Computing starttime ...")

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
        starttime = datetime.datetime(
            year=rec_year,
            month=rec_month,
            day=rec_day,
            hour=int(rec_time[0]),
            minute=int(rec_time[1]),
            second=int(rec_time[2]),
        )

        return starttime

    @property
    def stoptime(self) -> datetime.date:
        logger.debug("Computing stoptime ...")
        stoptime = list(
            map(lambda x: self.starttime +
                datetime.timedelta(seconds=x), self.times)
        )[-1]

        return stoptime

    def purgeNans(self) -> None:

        logger.debug("Removing all NaNs ...")

        self.idx_v = np.delete(self.idx_v, np.isnan(self.ident_v))
        self.fund_v = np.delete(self.fund_v, np.isnan(self.ident_v))
        self.sign_v = np.delete(self.sign_v, np.isnan(self.ident_v), axis=0)
        self.ident_v = np.delete(self.ident_v, np.isnan(self.ident_v))
        self.ids = np.delete(self.ids, np.isnan(self.ids))

    def purgeShort(self, thresh: float) -> None:

        logger.debug("Removing short tracks ...")

        counter = 0
        for track_id in tqdm(self.ids):
            times = self.times[self.idx_v[self.ident_v == track_id]]
            dur = times.max() - times.min()

            if dur < thresh:
                self.idx_v = np.delete(self.idx_v, self.ident_v == track_id)
                self.fund_v = np.delete(self.fund_v, self.ident_v == track_id)

                try:
                    self.sign_v = np.delete(
                        self.sign_v, self.ident_v == track_id, axis=0
                    )
                except GridDataMismatch as error:
                    logger.error(
                        "The dimensions of sign_v do not match. Either sign_v is not recomputed (>> fillPowers) or the data is corrupted!")
                    raise error

                self.ident_v = np.delete(
                    self.ident_v, self.ident_v == track_id)
                self.ids = np.delete(self.ids, self.ids == track_id)
                counter += 1

        logger.info("Removed %i short frequency tracks.", counter)

    def purgeBad(self, thresh: float) -> None:

        logger.debug("Removing poorly tracked traces ...")

        counter = 1
        for track_id in tqdm(self.ids):

            # get min and max sampled time
            tmin = np.min(self.times[self.idx_v[self.ident_v == track_id]])
            tmax = np.max(self.times[self.idx_v[self.ident_v == track_id]])

            # get index for those on time
            start = findOnTime(self.times, tmin)
            stop = findOnTime(self.times, tmax)

            # get true times for ID including non-sampled
            tru_times = self.times[start:stop]

            # get sampled times
            sam_times = self.times[self.idx_v[self.ident_v == track_id]]

            # compute sampling coverage as proportion of times vs times sampled
            perf = len(sam_times)/len(tru_times)

            # delete data for IDs that fall below threshold
            if perf < thresh:
                self.idx_v = np.delete(self.idx_v, self.ident_v == track_id)
                self.fund_v = np.delete(self.fund_v, self.ident_v == track_id)
                self.sign_v = np.delete(
                    self.sign_v, self.ident_v == track_id, axis=0)
                self.ident_v = np.delete(
                    self.ident_v, self.ident_v == track_id)
                self.ids = np.delete(self.ids, self.ids == track_id)
                counter += 1

        logger.info("Removed %i poorly tracked tracks", counter)

    def fillPowers(self, filename: str = 'traces-grid1.raw') -> None:

        def recomputePowers() -> None:

            # parameters to recompute powerspectrum from raw grid file
            samplingrate = self.grid_rate
            path = self._datapath + filename
            raw = dataloader.open_data(path, -1, 60.0, 10.0)
            nfft = powerspectrum.nfft(samplingrate, freq_resolution=1)

            # update powers in signature vector
            for track_id in tqdm(self.ids):

                # get id where signature vector has nans
                id_powers = self.sign_v[:, 0][self.ident_v == track_id]
                indices = np.arange(len(id_powers))
                idxs = indices[np.isnan(id_powers)]

                # get times for missing powers
                times = self.times[self.idx_v[self.ident_v == track_id]][idxs]

                # convert times to to indices for raw data by multiplying by original sampling rate
                iois = times * samplingrate

                # loop over raw sample points and compute power spectrum
                for ioi, idx in zip(iois, idxs):

                    # get frequency for id at the time point of ioi
                    freq = self.fund_v[self.ident_v == track_id][idx]

                    for channel in np.arange(np.shape(raw)[1]):

                        # calculate power spectral density for channel at roi
                        freqs, powers = powerspectrum.psd(
                            raw[int(ioi - nfft / 2): int(ioi + nfft / 2), channel],
                            ratetime=samplingrate,
                            overlap_frac=0.9,
                            freq_resolution=1,
                        )

                        # select power for frequency that matches fundamental frequency of track id most closely
                        power_sel = powers[findClosest(freqs, freq)]
                        freq_sel = freqs[findClosest(freqs, freq)]

                        # log transform power
                        power_sel_log = powerspectrum.decibel(power_sel)

                        # insert computed power into loaded signature vector
                        insert_idx = np.arange(len(self.sign_v))[self.ident_v == track_id][
                            idx
                        ]

                        self.sign_v[insert_idx, channel] = power_sel_log

            logger.debug(
                "Finished computing the new powers, starting checks ...")

        def savePowers() -> None:

            # create paths to files
            backup = self._datapath + "sign_v_backup.npy"
            current = self._datapath + "sign_v.npy"

            # Check if we already backed up the old signature vector
            if os.path.isfile(backup):
                logger.info(
                    "Backup signature vector found!")

            # if not, back it up!
            else:
                try:
                    # rename old sign vector to backup
                    os.rename(current, backup)
                    logger.warning("Backup sign_v.npy created succesfully!")

                except OSError as error:
                    logger.error(
                        "Failed creating a backup sign_v.npy! Aborting ...")
                    raise error

            # check if we now have a backup and save new one to file if true.
            if os.path.isfile(backup):
                np.save(current, self.sign_v)  # save sign_v to file
                logger.info(
                    "Backup signature vector exists, saving newly computed signature vector to disk!")

                # load newly saved signature vector back into namespace to continue computation.
                try:
                    self.sign_v = np.load(current, allow_pickle=True)
                    logging.info("New sign_v loaded into namespace")

                except FileNotFoundError as error:
                    logging.error(
                        "Error loading newly generated sign_v into namespace!")
                    raise error
            else:
                logger.error("Backup signature vector not found! Aborting ...")
                raise FileNotFoundError

        logger.debug('Updating power matrix ...')

        recomputePowers()

        savePowers()

    def triangPositionsSTUB(self) -> None:

        # check if current signature vector is usable
        if len(self.ident_v) != len(self.sign_v):
            msg = "The dimensions of sign_v do not match. Either sign_v is not recomputed (>> fillPowers) or the data is corrupted!"
            logger.error(msg)
            raise GridDataMismatch(msg)

    def interpolateAll(self) -> None:

        logger.debug("Interpolating all data ...")

        # check positions are already computed
        if self.xpos == None:
            msg = "The current class instance has no position estimations! Compute positions before interpolation!"
            logger.error(msg)
            raise GridDataMissing(msg)

        # init lists for data collection
        collect_ident = []
        collect_idx = []
        collect_sign = []
        collect_fund = []
        collect_xpos = []
        collect_ypos = []

        for track_id in tqdm(self.ids):

            # get min and max time for ID
            tmin = np.min(self.times[self.idx_v[self.ident_v == track_id]])
            tmax = np.max(self.times[self.idx_v[self.ident_v == track_id]])

            # get time index for tmin and tmax
            start = findOnTime(self.times, tmin)
            stop = findOnTime(self.times, tmax)

            # get true times including non-sampled
            sam_times = self.times[start:stop]
            tru_times = self.times[self.idx_v[self.ident_v == track_id]]

            # get sampled data
            powers = self.sign_v[self.ident_v == track_id]
            fund = self.fund_v[self.ident_v == track_id]
            xpos = self.xpos[self.ident_v == track_id]
            ypos = self.ypos[self.ident_v == track_id]

            # interpolate signature matrix
            num_el = np.shape(powers)[1]
            new_length = len(tru_times)
            powers_interp = np.zeros(shape=(new_length, num_el))
            for el in range(num_el):
                p = powers[:, el]
                p_interp = np.interp(tru_times, sam_times, p)
                powers_interp[:, el] = p_interp

            # interpolate 1d arrays
            fund_interp = np.interp(tru_times, sam_times, fund)
            xpos_interp = np.interp(tru_times, sam_times, xpos)
            ypos_interp = np.interp(tru_times, sam_times, ypos)

            # build new index vector that includes the generated datapoints
            idx_v = np.arange(start, start+len(tru_times))

            # build new ident_v that includes the generated datapoints
            ident_v = np.ones(len(idx_v), dtype=int) * int(track_id)

            # append it all to lists
            collect_ident.append(ident_v)
            collect_idx.append(idx_v)
            collect_fund.append(fund_interp)
            collect_xpos.append(xpos_interp)
            collect_ypos.append(ypos_interp)
            collect_sign.append(powers_interp)

        # overwrite old entries
        self.ident_v = np.asarray(np.ravel(collect_ident), dtype=int)
        self.idx_v = np.asarray(np.ravel(collect_idx), dtype=int)
        self.fund_v = np.ravel(collect_fund)
        self.xpos = np.ravel(collect_xpos)
        self.ypos = np.ravel(collect_ypos)
        self.sign_v = np.concatenate(collect_sign, axis=0)

        logger.info("Interpolation finished.")

    def smoothPositions(self, params) -> None:

        logger.debug("Smoothing position estimates ...")

        # retrieve preprocessing parameters from config file
        veloc_thresh = params["vthresh"]
        median_window = params["median_window"]
        smth_window = params["smoothing_window"]
        polyorder = params["smoothing_polyorder"]

        # iterate through all ids
        for track_id in tqdm(self.ids):

            # get data
            times = self.times[self.idx_v[self.ident_v == track_id]]
            xpos = self.xpos[self.ident_v == track_id]
            ypos = self.ypos[self.ident_v == track_id]

            # compute velocity
            veloc = velocity2d(times, xpos, ypos)

            # get index for datapoints where fish was NOT unrealistically fast
            index = np.arange(len(veloc))
            valid_data = index[veloc < veloc_thresh]

            # intepolate too fast datapoints
            xpos_interp = np.interp(times, times[valid_data], xpos[valid_data])
            ypos_interp = np.interp(times, times[valid_data], ypos[valid_data])

            # median filter to remove small scale outliers
            xpos_medfilt = medfilt(xpos_interp, kernel_size=median_window)
            ypos_medfilt = medfilt(ypos_interp, kernel_size=median_window)

            # savitzky-golay filter
            xpos_savgol = savgol_filter(xpos_medfilt, smth_window, polyorder)
            ypos_savgol = savgol_filter(ypos_medfilt, smth_window, polyorder)

            # overwrite class instance data
            self.xpos[self.ident_v == track_id] = xpos_savgol
            self.ypos[self.ident_v == track_id] = ypos_savgol

        logger.info("Smoothed positions!")

    def loadLogger(self, filename: str = 'hobologger.csv') -> None:

        logger.debug("Loading hobologger file ...")

        # import PROCESSED (upsampled & interpolated) hobologger file and format dates
        hobo = pd.read_csv(f"{self._dataroot}/{filename}")
        hobo["date"] = pd.to_datetime(hobo["date"])
        hobo = hobo.set_index("date")

        # convert datetime objects to str for dataframe indexing
        startdt = self.starttime.strftime("%Y-%m-%d %H:%M:%S")
        stopdt = self.stoptime.strftime("%Y-%m-%d %H:%M:%S")

        # index dataframe using starttime and stopttime datetime objects
        start = hobo.index.get_indexer([startdt], method="nearest")[0]
        stop = hobo.index.get_indexer([stopdt], method="nearest")[0]

        # grab data window from logger dataframe
        hobo = hobo.reset_index()
        temp = hobo["temp_filt"][start:stop]
        light = hobo["lux_filt"][start:stop]
        time = np.arange(len(hobo["date"][start:stop]))

        # check if logger data was available for this recording
        if len(temp) < len(self.times)/2:
            msg = "Hobologger data does not cover this recording!"
            logger.error(msg)
            raise GridDataMismatch(msg)

        # interpolate data to match sampling of frequency and positions
        self.temp = np.interp(self.times, time, temp)
        self.light = np.interp(self.times, time, light)
        logger.info("Loaded hobologger successfully!")

    def sexFish(self, upper="m", thresh=750, normtemp=25, q10=1.6) -> None:

        def sexing(upper: str, thresh: float, mode: float) -> str:

            if upper == "m":
                sex = "m" if mode > thresh else "f"
            if upper == "f":
                sex = "f" if mode > thresh else "m"

            return sex

        logger.debug("Estimating fish sex ...")

        # check if instance has temp data
        if self.temp == None:
            msg = "This dataset has no temperature data! Aborting ..."
            logger.error(msg)
            raise GridDataMissing(msg)

        # iterate through all "individuals"
        self.sex = []
        for track_id in self.ids:

            # normalize by q10 value
            tmin = self.times[self.idx_v[self.ident_v == track_id]][0]
            tmax = self.times[self.idx_v[self.ident_v == track_id]][-1]

            index = np.arange(len(self.times))
            start, stop = (
                index[self.times == tmin][0],
                index[self.times == tmax][0] + 1,
            )

            temp = self.temp[start:stop]
            data = self.fund_v[self.ident_v == track_id]

            # normalize by Q10
            normed = normQ10(data, temp, normtemp, q10)

            # estimate mode
            mode = estimateMode(normed)

            # decide sex
            sex = sexing(upper, thresh, mode)

            self.sex.append(sex)

    def integrityCheck(self) -> bool:

        logger.debug("Starting integrity check ...")

        try:
            # check overall array lengths
            len_ident_v = len(self.ident_v)
            len_idx_v = len(self.idx_v)
            len_fund_v = len(self.fund_v)
            len_xpos = len(self.xpos)
            len_ypos = len(self.ypos)
            len_sign_v = len(self.sign_v)
            len_time = len(self.times)

            # check data array dimensions
            same_dim = [len_ident_v, len_idx_v,
                        len_fund_v, len_xpos, len_ypos, len_sign_v]
            if len(np.unique(same_dim)) != 1:
                msg = "Mismatch in main data arrays!"
                logger.error(msg)
                raise GridDataMismatch(msg)

            # check index to time match
            if len_time < np.max(self.idx_v):
                msg = "Too many indices for time array!"
                logger.error(msg)
                raise GridDataMismatch(msg)

            # check sex to id match
            if len(self.sex) != len(self.ids):
                msg = "Sex array does not match id array!"
                logger.error(msg)
                raise GridDataMismatch(msg)

            # check for every tracked fish
            for track_id in self.ids:
                time = self.times[self.idx_v[self.ident_v == track_id]]
                fund = self.fund_v[self.ident_v == track_id]
                power = self.sign_v[self.ident_v == track_id]
                xpos = self.xpos[self.ident_v == track_id]
                ypos = self.xpos[self.ident_v == track_id]

                passed = True
                if len(time) != len(fund):
                    passed = False
                if len(time) != len(power):
                    passed = False
                if len(time) != len(xpos):
                    passed = False
                if len(time) != len(ypos):
                    passed = False

                if passed is False:
                    msg = f"Data mismatch in fish ID {track_id}"
                    logger.error(msg)
                    raise GridDataMismatch(msg)

            logger.info("Integrity check passed!")
            return True

        except GridDataMismatch as error:
            msg = "Integrity check failed!"
            logger.error(msg)
            raise error

    def saveDataSTUB(self) -> None:
        pass
