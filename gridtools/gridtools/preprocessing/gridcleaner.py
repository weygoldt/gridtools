import datetime
import glob
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from IPython import embed
from scipy.signal import medfilt, savgol_filter
from thunderfish import dataloader, powerspectrum
from tqdm import tqdm

from ..exceptions import BadOutputDir, GridDataMismatch, GridDataMissing
from ..logger import makeLogger
from ..utils.datahandling import (
    estimateMode,
    findClosest,
    findOnTime,
    normQ10,
    removeOutliers,
)
from ..utils.spatial import velocity2d

logger = makeLogger(__name__)


class GridCleaner:
    """this is a docstring"""

    def __init__(self, datapath: str) -> None:

        # print a message
        logger.info("Initialising grid instance ...")

        # initialize directory variables
        self._datapath = datapath
        self._dataroot = os.path.split(self._datapath[:-1])[0]

        # initialize list to track functions applied to dataset
        self.type = [
            "raw wavetracker output",
        ]

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
            raise error

        # check if files are have no errors
        try:
            # get data for checks
            time_len = len(self.times)
            idx_len = len(self.idx_v)
            uidx_len = len(np.unique(self.idx_v))
            fund_len = len(self.fund_v)
            sign_len = len(self.sign_v)
            ident_len = len(self.ident_v)
            sign_elno = np.shape(self.sign_v)[0]
            grid_elno = np.prod(self.grid_grid)

            # check idx and time match in len
            """
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
            """

        except GridDataMismatch as error:
            logging.error(str(error))
            raise error

    @property
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
                "The directory name does not match the datetime naming pattern!"
            )
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
                logger.error("Time string of folder name does not contain '_' or ':' !")

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
            map(lambda x: self.starttime + datetime.timedelta(seconds=x), self.times)
        )[-1]

        return stoptime

    def purgeUnassigned(self) -> None:

        logger.info("Removing all unassigned frequencies (NaNs in ident_v) ...")

        self.idx_v = np.delete(self.idx_v, np.isnan(self.ident_v))
        self.fund_v = np.delete(self.fund_v, np.isnan(self.ident_v))
        self.sign_v = np.delete(self.sign_v, np.isnan(self.ident_v), axis=0)
        self.ident_v = np.delete(self.ident_v, np.isnan(self.ident_v))
        self.ids = np.delete(self.ids, np.isnan(self.ids))

        self.type.extend("purged unassigned")

    def purgeShort(self, thresh: float) -> None:

        logger.info("Removing short tracks ...")

        counter = 0
        index_ids = np.arange(len(self.ids))
        index_ids_del = []
        index_ident = np.arange(len(self.ident_v))
        index_ident_del = []

        for track_id in tqdm(self.ids, desc="Purging short   "):

            times = self.times[self.idx_v[self.ident_v == track_id]]
            dur = times.max() - times.min()

            if dur < thresh:
                index_ids_del.extend(index_ids[self.ids == track_id])
                index_ident_del.extend(index_ident[self.ident_v == track_id])
                #                self.idx_v = self.idx_v[self.ident_v != track_id]
                #                self.fund_v = self.fund_v[self.ident_v != track_id]
                #
                #                try:
                #                    self.sign_v = self.sign_v[self.ident_v != track_id, :]
                #                except GridDataMismatch as error:
                #                    logger.error(
                #                        "The dimensions of sign_v do not match. Either sign_v is not recomputed (>> fillPowers) or the data is corrupted!")
                #                    raise error
                #
                #                self.ident_v = self.ident_v[self.ident_v != track_id]
                #                self.ids = self.ids[self.ids != track_id]
                counter += 1

        # make a mask from the delete indices
        mask_ident = np.ones(len(self.ident_v), bool)
        mask_ident[index_ident_del] = 0
        mask_ids = np.ones(len(self.ids), bool)
        mask_ids[index_ids_del] = 0

        # take only data that is not masked
        self.idx_v = self.idx_v[mask_ident]
        self.fund_v = self.fund_v[mask_ident]
        self.sign_v = self.sign_v[mask_ident, :]
        self.ident_v = self.ident_v[mask_ident]
        self.ids = self.ids[mask_ids]

        logger.info("Removed %i short frequency tracks.", counter)
        self.type.extend("purged short")

    def purgeBad(self, thresh: float) -> None:

        logger.info("Removing poorly tracked traces ...")

        counter = 1
        for track_id in tqdm(self.ids, desc="Purging bad     "):

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
            perf = len(sam_times) / len(tru_times)

            # delete data for IDs that fall below threshold
            if perf < thresh:
                self.idx_v = np.delete(self.idx_v, self.ident_v == track_id)
                self.fund_v = np.delete(self.fund_v, self.ident_v == track_id)
                self.sign_v = np.delete(self.sign_v, self.ident_v == track_id, axis=0)
                self.ident_v = np.delete(self.ident_v, self.ident_v == track_id)
                self.ids = np.delete(self.ids, self.ids == track_id)
                counter += 1

        logger.info("Removed %i poorly tracked tracks", counter)
        self.type.extend("purged bad")

    def fillPowers(self, filename: str = "traces-grid1.raw") -> None:
        def recomputePowers() -> None:

            # parameters to recompute powerspectrum from raw grid file
            samplingrate = self.grid_rate
            path = self._datapath + filename
            raw = dataloader.open_data(path, -1, 60.0, 10.0)
            nfft = powerspectrum.nfft(samplingrate, freq_resolution=1)

            # update powers in signature vector
            for track_id in tqdm(self.ids, desc="Filling powers  "):

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
                            raw[int(ioi - nfft / 2) : int(ioi + nfft / 2), channel],
                            ratetime=samplingrate,
                            overlap_frac=0.9,
                            freq_resolution=1,
                            window="hann",
                        )

                        # select power for frequency that matches fundamental frequency of track id most closely
                        power_sel = powers[findClosest(freqs, freq)]
                        freq_sel = freqs[findClosest(freqs, freq)]

                        # log transform power
                        power_sel_log = powerspectrum.decibel(power_sel)

                        # insert computed power into loaded signature vector
                        insert_idx = np.arange(len(self.sign_v))[
                            self.ident_v == track_id
                        ][idx]

                        self.sign_v[insert_idx, channel] = power_sel_log

            logger.debug("Finished computing the new powers, starting checks ...")

        def savePowers() -> None:

            # create paths to files
            backup = self._datapath + "sign_v_backup.npy"
            current = self._datapath + "sign_v.npy"

            # Check if we already backed up the old signature vector
            if os.path.isfile(backup):
                logger.info("Backup signature vector found!")

            # if not, back it up!
            else:
                try:
                    # rename old sign vector to backup
                    os.rename(current, backup)
                    logger.warning("Backup sign_v.npy created succesfully!")

                except OSError as error:
                    logger.error("Failed creating a backup sign_v.npy! Aborting ...")
                    raise error

            # check if we now have a backup and save new one to file if true.
            if os.path.isfile(backup):
                np.save(current, self.sign_v)  # save sign_v to file
                logger.info(
                    "Backup signature vector exists, saving newly computed signature vector to disk!"
                )

                # load newly saved signature vector back into namespace to continue computation.
                try:
                    self.sign_v = np.load(current, allow_pickle=True)
                    logger.info("New sign_v loaded into namespace")

                except FileNotFoundError as error:
                    logger.error("Error loading newly generated sign_v into namespace!")
                    raise error
            else:
                logger.error("Backup signature vector not found! Aborting ...")
                raise FileNotFoundError

        logger.info("Updating power matrix ...")

        try:
            recomputePowers()
        except Exception as error:
            logger.error("Exception during power recomputation!")
            raise error

        savePowers()
        self.type.extend("powers recalculated")

    def triangPositions(self, electrode_number: int) -> None:

        logger.info("Starting position triangulation ...")

        # check if current signature vector is usable
        if len(self.ident_v) != len(self.sign_v):
            msg = "The dimensions of sign_v do not match. Either sign_v is not recomputed (>> fillPowers) or the data is corrupted!"
            logger.error(msg)
            raise GridDataMismatch(msg)

        # create grid coordinates
        xdist = self.grid_spacings[0][0]
        ydist = self.grid_spacings[0][1]
        dims = self.grid_grid[0]

        # build distance constructors in x and y dimension
        x_constr = np.arange(0, xdist * (dims[0]), xdist)
        y_vals = np.arange(0, ydist * (dims[1]), ydist)

        # build grid of distances
        gridx = []
        gridy = []
        for x, y in zip(x_constr, y_vals):
            y_constr = np.ones(dims[1]) * y
            gridx.extend(x_constr)
            gridy.extend(y_constr)

        # initialize empty arrays for data collection
        x_pos = np.zeros(np.shape(self.ident_v))
        y_pos = np.zeros(np.shape(self.ident_v))

        # also collect ident_v for positions for ordering them like the class data later
        ident_v_tmp = np.full(np.shape(self.ident_v), np.nan)

        index = 0  # to index in two nested for loops

        # interpolate for every fish
        for track_id in tqdm(np.unique(self.ids), desc="Triangulating   "):

            # get times
            times = self.times[self.idx_v[self.ident_v == track_id]]

            # get powers across all electrodes for this frequency
            powers = self.sign_v[self.ident_v == track_id, :]

            # upsample powers
            newtime = np.arange(times.min(), times.max(), 0.1)
            print(np.shape(powers))
            print(np.shape(self.sign_v))

            # test if interpolated powers make positions legg gritty
            # newpowers = np.asarray(
            #     [
            #         np.interp(newtime, times, powers[:, i])
            #         for i in range(len(powers[0, :]))
            #     ]
            # )

            # iterate through every single point in time for this fish
            for idx in range(len(powers[:, 0])):
                # for idx in range(len(newpowers[:, 0])):

                # extract momentary powers
                mom_powers = powers[idx, :]
                # mom_powers = newpowers[:, idx]

                # calculate max n powers
                ind = np.argpartition(mom_powers, -electrode_number)[-electrode_number:]
                max_powers = mom_powers[ind]

                # get respective coordinates on grid distance matrix
                x_maxs = np.array(gridx)[ind]
                y_maxs = np.array(gridy)[ind]

                # compute weighted mean
                x_wm = sum(x_maxs * max_powers) / sum(max_powers)
                y_wm = sum(y_maxs * max_powers) / sum(max_powers)

                # add to empty arrays
                x_pos[index] = x_wm
                y_pos[index] = y_wm
                ident_v_tmp[index] = track_id
                index += 1

            del powers
            del mom_powers

        # make empty class data arrays
        self.xpos = np.zeros(np.shape(self.ident_v))
        self.ypos = np.zeros(np.shape(self.ident_v))

        # self.xpos = x_pos
        # self.ypos = y_pos

        # append to class data in same order
        for track_id in tqdm(self.ids, desc="Reordering      "):
            self.xpos[self.ident_v == int(track_id)] = x_pos[
                ident_v_tmp == int(track_id)
            ]
            self.ypos[self.ident_v == int(track_id)] = y_pos[
                ident_v_tmp == int(track_id)
            ]

        self.type.extend("positions estimated")

    def interpolateAll(self) -> None:

        logger.info("Interpolating all data ...")

        # check positions are already computed
        if len(self.xpos) == 1:
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

        for track_id in tqdm(self.ids, desc="Interpolating   "):

            # get min and max time for ID
            tmin = np.min(self.times[self.idx_v[self.ident_v == track_id]])
            tmax = np.max(self.times[self.idx_v[self.ident_v == track_id]])

            # get time index for tmin and tmax
            start = findOnTime(self.times, tmin)
            stop = findOnTime(self.times, tmax)

            # get true times including non-sampled
            sam_times = self.times[start:stop]  # all sampling points
            tru_times = self.times[
                self.idx_v[self.ident_v == track_id]
            ]  # the points where data is available

            # get sampled data
            powers = self.sign_v[self.ident_v == track_id]
            fund = self.fund_v[self.ident_v == track_id]
            xpos = self.xpos[self.ident_v == track_id]
            ypos = self.ypos[self.ident_v == track_id]

            # interpolate signature matrix
            num_el = np.shape(powers)[1]
            new_length = len(sam_times)
            powers_interp = np.zeros(shape=(new_length, num_el))

            for el in range(num_el):
                p = powers[:, el]
                p_interp = np.interp(sam_times, tru_times, p)
                powers_interp[:, el] = p_interp

            # interpolate 1d arrays
            fund_interp = np.interp(sam_times, tru_times, fund)
            xpos_interp = np.interp(sam_times, tru_times, xpos)
            ypos_interp = np.interp(sam_times, tru_times, ypos)

            # build new index vector that includes the generated datapoints
            idx_v = np.arange(start, start + len(sam_times))

            # build new ident_v that includes the generated datapoints
            ident_v = np.ones(len(idx_v), dtype=int) * int(track_id)

            # append it all to lists
            collect_ident.extend(ident_v)
            collect_idx.extend(idx_v)
            collect_fund.extend(fund_interp)
            collect_xpos.extend(xpos_interp)
            collect_ypos.extend(ypos_interp)
            collect_sign.extend(powers_interp)

        # overwrite old entries
        self.ident_v = np.asarray(np.ravel(collect_ident), dtype=int)
        self.idx_v = np.asarray(np.ravel(collect_idx), dtype=int)
        self.fund_v = np.ravel(collect_fund)
        self.xpos = np.ravel(collect_xpos)
        self.ypos = np.ravel(collect_ypos)
        self.sign_v = np.concatenate(collect_sign, axis=0)

        logger.info("Interpolation finished.")
        self.type.extend("interpolated")

    def smoothPositions(self, params) -> None:

        logger.info("Smoothing position estimates ...")

        # retrieve preprocessing parameters from config file
        veloc_thresh = params["vthresh"]
        median_window = params["median_window"]
        smth_window = params["smoothing_window"]
        polyorder = params["smoothing_polyorder"]

        # check if median window is odd
        if median_window % 2 == 0:
            median_window += 1
            logger.warning(
                "Median filter kernel width is even! Changing to {median_window}. Consider changing the value in the config file!"
            )

        # iterate through all ids
        for track_id in tqdm(self.ids, desc="Position cleanup"):

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
        self.type.extend("position estimates smoothed")

    def loadLogger(self, filename: str = "hobologger.csv") -> None:

        logger.info("Loading hobologger file ...")

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

        # interpolate data to match sampling of frequency and positions
        self.temp = np.interp(self.times, time, temp)
        self.light = np.interp(self.times, time, light)

        logger.info("Loaded hobologger successfully!")
        self.type.extend("temperature and light data loaded")

    def sexFish(self, upper="m", thresh=750, normtemp=25, q10=1.6) -> None:
        def sexing(upper: str, thresh: float, mode: float) -> str:

            if upper == "m":
                sex = "m" if mode > thresh else "f"
            if upper == "f":
                sex = "f" if mode > thresh else "m"

            return sex

        logger.info("Estimating fish sex ...")

        # check if instance has temp data
        if len(self.temp) == 1:
            msg = "This dataset has no temperature data! Aborting ..."
            logger.error(msg)
            raise GridDataMissing(msg)

        # iterate through all "individuals"
        self.sex = []
        for track_id in tqdm(self.ids, desc="Estimate sex    "):

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

            # remove outliers
            cleaned = removeOutliers(normed, bar=1.5, fillnan=False)

            # estimate mode
            mode = estimateMode(cleaned)

            # decide sex
            sex = sexing(upper, thresh, mode)

            self.sex.append(sex)

        self.type.extend("sex estimated")

    def integrityCheck(self) -> bool:

        logger.info("Starting integrity check ...")

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
            same_dim = [
                len_ident_v,
                len_idx_v,
                len_fund_v,
                len_xpos,
                len_ypos,
                len_sign_v,
            ]
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
            self.type.extend("integrity checked")
            return True

        except GridDataMismatch as error:
            msg = "Integrity check failed!"
            logger.error(msg)
            raise error

    def saveData(
        self, outputpath: str, overwritable: bool = False, check: bool = True
    ) -> None:

        logger.info("Saving data ...")

        def save(self, outputpath: str) -> None:

            np.save(outputpath + "/times.npy", self.times)
            np.save(outputpath + "/idx.npy", self.idx_v)
            np.save(outputpath + "/fund.npy", self.fund_v)
            np.save(outputpath + "/sign.npy", self.sign_v)
            np.save(outputpath + "/ident.npy", self.ident_v)
            np.save(outputpath + "/xpos.npy", self.xpos)
            np.save(outputpath + "/ypos.npy", self.ypos)
            np.save(outputpath + "/temp.npy", self.temp)
            np.save(outputpath + "/light.npy", self.light)
            np.save(outputpath + "/sex.npy", self.sex)

            logger.info("Transferring spectrograms to output directory ...")

            shutil.copy(
                f"{self._datapath}/fill_freqs.npy", f"{outputpath}/fill_freqs.npy"
            )

            shutil.copy(
                f"{self._datapath}/fill_times.npy", f"{outputpath}/fill_times.npy"
            )
            shutil.copy(
                f"{self._datapath}/fill_spec.npy", f"{outputpath}/fill_spec.npy"
            )
            shutil.copy(
                f"{self._datapath}/fill_spec_shape.npy",
                f"{outputpath}/fill_spec_shape.npy",
            )
            shutil.copy(f"{self._datapath}/spec.npy", f"{outputpath}/spec.npy")

        if not overwritable:
            if len(glob.glob(outputpath + "*.raw")) > 0:
                msg = "The output path contains a raw file! Do not overwrite exisiting data! Run 'saveData' in overwrite mode if desired."
                logger.error(msg)
                raise BadOutputDir(msg)

            if Path(outputpath) is Path(self._datapath):
                msg = "Outputpath and datapath are the same! Run 'saveData' in overwrite mode if desired."
                logger.error(msg)
                raise BadOutputDir(msg)

        if check:
            logger.info("Running integrity check before saving to disk ...")
            save(self, outputpath)
        else:
            logger.info("Saving data without checks ...")
            save(self, outputpath)

        logger.info("Data saved!")
        self.type.extend("saved")
