import datetime
import glob
import logging
import os
from pathlib import Path

import nixio as nio
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter1d
from thunderfish import dataloader, powerspectrum
from tqdm import tqdm

from ..exceptions import BadOutputDir, GridDataMismatch, GridDataMissing
from ..logger import makeLogger
from ..utils.datahandling import (
    estimateMode,
    findClosest,
    findOnTime,
    lowpass_filter,
    normQ10,
    removeOutliers,
)
from ..utils.spatial import velocity2d

logger = makeLogger(__name__)


class GridCleaner:
    """Loads wavetracker output files and provides many methods for preprocessing
    (e.g. interpolation, position estimation, etc. and subsequent saving."""

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
        self.temperature = [None]
        self.light = [None]
        self.q10 = [None]
        self.sex = [None]
        self.xpositions = [None]
        self.ypositions = [None]

        # try to load wavetracker output files
        try:
            # the wavetracker output
            self.times = np.load(datapath + "times.npy", allow_pickle=True)
            self.indices = np.load(datapath + "idx_v.npy", allow_pickle=True)
            self.frequencies = np.load(
                datapath + "fund_v.npy", allow_pickle=True
            )
            self.powers = np.load(datapath + "sign_v.npy", allow_pickle=True)
            self.identities = np.load(
                datapath + "ident_v.npy", allow_pickle=True
            )
            self.ids = np.unique(self.identities[~np.isnan(self.identities)])

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
            idx_len = len(self.indices)
            uidx_len = len(np.unique(self.indices))
            fund_len = len(self.frequencies)
            sign_len = len(self.powers[:, 0])
            ident_len = len(self.identities)
            sign_elno = len(self.powers[0, :])
            grid_elno = np.prod(self.grid_grid)

            # check idx and time match in len

            if time_len < uidx_len:
                raise GridDataMismatch(
                    f"Index vector is longer than time vector! Time vector lenght: {time_len} Unique index vector length: {uidx_len}!"
                )
            else:
                logger.debug("Passed time-index length check!")

            # check idx and time match in indexing value
            if time_len < np.max(self.indices):
                raise GridDataMismatch(
                    f"Too many indices for the time vector! Time vector lenght: {time_len} Index out of bounds: {np.max(self.indices)}!"
                )
            else:
                logger.debug("Passed time-index bounds check!")

            # check if idx len fits to fund, sign, ident
            if idx_len == fund_len == sign_len == ident_len:
                logger.debug("Passed data array-length check!")
            else:
                raise GridDataMismatch(
                    "Lengths of idx, fund, sign and ident are not the same!"
                )

            # check if electrode number matches in sign_v
            if sign_elno == grid_elno:
                logger.debug("Passed electrode number check!")
            else:
                raise GridDataMismatch(
                    "Number of electrodes in sign_v does not match grid metadata!"
                )

            logger.info("Grid initialized succesfully!")

        except GridDataMismatch as error:
            logging.error(str(error))
            raise error

    @property
    def starttime(self) -> datetime.datetime:
        """
        starttime returns the start time of the recording.

        Returns
        -------
        datetime.datetime
            start time of recording
        """
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
                logger.error(
                    "Time string of folder name does not contain '_' or ':' !"
                )

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
        """
        stoptime returns the stop time of the recording

        Returns
        -------
        datetime.date
            stop time of recording
        """
        logger.debug("Computing stoptime ...")
        stoptime = list(
            map(
                lambda x: self.starttime + datetime.timedelta(seconds=x),
                self.times,
            )
        )[-1]

        return stoptime

    def purge_unassigned(self) -> None:
        """
        purge_unassigned removes all unassigned frequencies from the dataset.
        """

        logger.info("Removing all unassigned frequencies (NaNs in ident_v) ...")

        self.indices = np.delete(self.indices, np.isnan(self.identities))
        self.frequencies = np.delete(
            self.frequencies, np.isnan(self.identities)
        )
        self.powers = np.delete(self.powers, np.isnan(self.identities), axis=0)
        self.identities = np.delete(self.identities, np.isnan(self.identities))
        self.ids = np.delete(self.ids, np.isnan(self.ids))

        self.type.extend("purged unassigned")

    def purge_short(self, thresh: float) -> None:
        """
        purge_short removes tracks below a given duration threshold from the dataset.

        Parameters
        ----------
        thresh : float
            duration threshold in seconds.
        """

        logger.info("Removing short tracks ...")

        counter = 0
        index_ids = np.arange(len(self.ids))
        index_ids_del = []
        index_ident = np.arange(len(self.identities))
        index_ident_del = []

        for track_id in tqdm(self.ids, desc="Purging short   "):
            times = self.times[self.indices[self.identities == track_id]]
            dur = times.max() - times.min()

            if dur < thresh:
                index_ids_del.extend(index_ids[self.ids == track_id])
                index_ident_del.extend(index_ident[self.identities == track_id])
                counter += 1

        # make a mask from the delete indices
        mask_ident = np.ones(len(self.identities), bool)
        mask_ident[index_ident_del] = 0
        mask_ids = np.ones(len(self.ids), bool)
        mask_ids[index_ids_del] = 0

        # take only data that is not masked
        self.indices = self.indices[mask_ident]
        self.frequencies = self.frequencies[mask_ident]
        self.powers = self.powers[mask_ident, :]
        self.identities = self.identities[mask_ident]
        self.ids = self.ids[mask_ids]

        logger.info("Removed %i short frequency tracks.", counter)
        self.type.extend("purged short")

    def purge_bad(self, thresh: float) -> None:
        """
        purge_bad removes poorly tracked tracks from the dataset.

        Parameters
        ----------
        thresh : float
            Percentage tracked threshold.
        """

        logger.info("Removing poorly tracked traces ...")

        counter = 1
        for track_id in tqdm(self.ids, desc="Purging bad     "):
            # get min and max sampled time
            tmin = np.min(self.times[self.indices[self.identities == track_id]])
            tmax = np.max(self.times[self.indices[self.identities == track_id]])

            # get index for those on time
            start = findOnTime(self.times, tmin)
            stop = findOnTime(self.times, tmax)

            # get true times for ID including non-sampled
            tru_times = self.times[start:stop]

            # get sampled times
            sam_times = self.times[self.indices[self.identities == track_id]]

            # compute sampling coverage as proportion of times vs times sampled
            perf = len(sam_times) / len(tru_times)

            # delete data for IDs that fall below threshold
            if perf < thresh:
                self.indices = np.delete(
                    self.indices, self.identities == track_id
                )
                self.frequencies = np.delete(
                    self.frequencies, self.identities == track_id
                )
                self.powers = np.delete(
                    self.powers, self.identities == track_id, axis=0
                )
                self.identities = np.delete(
                    self.identities, self.identities == track_id
                )
                self.ids = np.delete(self.ids, self.ids == track_id)
                counter += 1

        logger.info("Removed %i poorly tracked tracks", counter)
        self.type.extend("purged bad")

    def recompute_powers(self, filename: str = "traces-grid1.raw") -> None:
        """
        recompute_powers recomputes missing powers in the power matrix that result from manually
        tracking frequencies using the wavetracker.

        Parameters
        ----------
        filename : str, optional
            Filename of raw grid file, by default "traces-grid1.raw"
        """

        def fill_powers() -> None:
            # parameters to recompute powerspectrum from raw grid file
            samplingrate = self.grid_rate
            path = self._datapath + filename
            raw = dataloader.open_data(path, -1, 60.0, 10.0)
            nfft = powerspectrum.nfft(samplingrate, freq_resolution=1)

            # update powers in signature vector
            for track_id in tqdm(self.ids, desc="Filling powers  "):
                # get id where signature vector has nans
                id_powers = self.powers[:, 0][self.identities == track_id]
                indices = np.arange(len(id_powers))
                idxs = indices[np.isnan(id_powers)]

                # get times for missing powers
                times = self.times[self.indices[self.identities == track_id]][
                    idxs
                ]

                # convert times to to indices for raw data by multiplying by original sampling rate
                iois = times * samplingrate

                # loop over raw sample points and compute power spectrum
                for ioi, idx in zip(iois, idxs):
                    # get frequency for id at the time point of ioi
                    freq = self.frequencies[self.identities == track_id][idx]

                    for channel in np.arange(np.shape(raw)[1]):
                        # calculate power spectral density for channel at roi
                        freqs, powers = powerspectrum.psd(
                            raw[
                                int(ioi - nfft / 2) : int(ioi + nfft / 2),
                                channel,
                            ],
                            ratetime=samplingrate,
                            overlap_frac=0.9,
                            freq_resolution=1,
                            window="hann",
                        )

                        # select power for frequency that matches fundamental frequency of track id most closely
                        power_sel = powers[findClosest(freqs, freq)]

                        # log transform power
                        power_sel_log = powerspectrum.decibel(power_sel)

                        # insert computed power into loaded signature vector
                        insert_idx = np.arange(len(self.powers))[
                            self.identities == track_id
                        ][idx]

                        self.powers[insert_idx, channel] = power_sel_log

            logger.debug(
                "Finished computing the new powers, starting checks ..."
            )

        def save_powers() -> None:
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
                    logger.error(
                        "Failed creating a backup sign_v.npy! Aborting ..."
                    )
                    raise error

            # check if we now have a backup and save new one to file if true.
            if os.path.isfile(backup):
                np.save(current, self.powers)  # save sign_v to file
                logger.info(
                    "Backup signature vector exists, saving newly computed signature vector to disk!"
                )

                # load newly saved signature vector back into namespace to continue computation.
                try:
                    self.powers = np.load(current, allow_pickle=True)
                    logger.info("New sign_v loaded into namespace")

                except FileNotFoundError as error:
                    logger.error(
                        "Error loading newly generated sign_v into namespace!"
                    )
                    raise error
            else:
                logger.error("Backup signature vector not found! Aborting ...")
                raise FileNotFoundError

        logger.info("Updating power matrix ...")

        try:
            fill_powers()
        except Exception as error:
            logger.error("Exception during power recomputation!")
            raise error

        save_powers()
        self.type.extend("powers recalculated")

    def triangulate_positions(self, electrode_number: int) -> None:
        """
        triangulate_positions triangulates positions from the power matrix.

        Parameters
        ----------
        electrode_number : int
            Number of electrodes to use.

        Raises
        ------
        GridDataMismatch
            If power matrix and identity array mismatch.
        """

        logger.info("Starting position triangulation ...")

        # check if current signature vector is usable
        if len(self.identities) != len(self.powers):
            msg = "The dimensions of sign_v do not match. Either sign_v is not recomputed (>> recompute_powers) or the data is corrupted!"
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
        for y in y_vals:
            y_constr = np.ones(dims[1]) * y
            gridx.extend(x_constr)
            gridy.extend(y_constr)

        # initialize empty arrays for data collection
        x_pos = np.zeros(np.shape(self.identities))
        y_pos = np.zeros(np.shape(self.identities))

        # also collect ident_v for positions for ordering them like the class data later
        ident_v_tmp = np.full(np.shape(self.identities), np.nan)

        index = 0  # to index in two nested for loops

        # interpolate for every fish
        for track_id in tqdm(np.unique(self.ids), desc="Triangulating   "):
            # get powers across all electrodes for this frequency
            powers = self.powers[self.identities == track_id, :]

            # iterate through every single point in time for this fish
            for idx in range(len(powers[:, 0])):
                # extract momentary powers
                mom_powers = powers[idx, :]

                # calculate max n powers
                # ind = np.argpartition(mom_powers, -electrode_number)[-electrode_number:]
                ind = np.argsort(mom_powers)[-electrode_number:]
                max_powers = mom_powers[ind]

                # get respective coordinates on grid distance matrix
                x_maxs = np.array(gridx)[ind]
                y_maxs = np.array(gridy)[ind]

                # compute weighted mean
                x_wm = np.sum(x_maxs * max_powers) / np.sum(max_powers)
                y_wm = np.sum(y_maxs * max_powers) / np.sum(max_powers)

                # add to empty arrays
                x_pos[index] = x_wm
                y_pos[index] = y_wm

                ident_v_tmp[index] = track_id
                index += 1

            del powers
            del mom_powers

        # make empty class data arrays
        self.xpositions = np.zeros(np.shape(self.identities))
        self.ypositions = np.zeros(np.shape(self.identities))

        # append to class data in same order
        for track_id in tqdm(self.ids, desc="Reordering      "):
            self.xpositions[self.identities == int(track_id)] = x_pos[
                ident_v_tmp == int(track_id)
            ]
            self.ypositions[self.identities == int(track_id)] = y_pos[
                ident_v_tmp == int(track_id)
            ]

        self.type.extend("positions estimated")

    def interpolate_all(self) -> None:
        """
        interpolate_all interpolates all data arrays in the class instance.

        Raises
        ------
        GridDataMissing
            Is raised if data is missing.
        """

        logger.info("Interpolating all data ...")

        # check positions are already computed
        if len(self.xpositions) == 1:
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
            tmin = np.min(self.times[self.indices[self.identities == track_id]])
            tmax = np.max(self.times[self.indices[self.identities == track_id]])

            # get time index for tmin and tmax
            start = findOnTime(self.times, tmin)
            stop = findOnTime(self.times, tmax)

            # get true times including non-sampled
            sam_times = self.times[start:stop]  # all sampling points
            tru_times = self.times[
                self.indices[self.identities == track_id]
            ]  # the points where data is available

            # get sampled data
            powers = self.powers[self.identities == track_id]
            fund = self.frequencies[self.identities == track_id]
            xpos = self.xpositions[self.identities == track_id]
            ypos = self.ypositions[self.identities == track_id]

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
        self.identities = np.asarray(np.ravel(collect_ident), dtype=int)
        self.indices = np.asarray(np.ravel(collect_idx), dtype=int)
        self.frequencies = np.ravel(collect_fund)
        self.xpositions = np.ravel(collect_xpos)
        self.ypositions = np.ravel(collect_ypos)
        self.powers = np.concatenate(collect_sign, axis=0)

        logger.info("Interpolation finished.")
        self.type.extend("interpolated")

    def smooth_positions(self, params) -> None:
        """
        smooth_positions uses a combination of velocity thresholding, median filtering
        and Savitzky Golay smoothing to smooth position estimates.

        Parameters
        ----------
        params : dict
            Smoothing parameters
        """

        logger.info("Smoothing position estimates ...")

        # retrieve preprocessing parameters from config file
        veloc_thresh = params["vthresh"]
        lowpass_cutoff = params["lowpass_cutoff"]

        # iterate through all ids
        for track_id in tqdm(self.ids, desc="Position cleanup"):
            # get data
            times = self.times[self.indices[self.identities == track_id]]
            xpos = self.xpositions[self.identities == track_id]
            ypos = self.ypositions[self.identities == track_id]

            # compute velocity
            veloc = velocity2d(times, xpos, ypos)

            # get index for datapoints where fish was NOT unrealistically fast
            index = np.arange(len(veloc))
            valid_data = index[veloc < veloc_thresh]

            # intepolate too fast datapoints
            xpos_interp = np.interp(times, times[valid_data], xpos[valid_data])
            ypos_interp = np.interp(times, times[valid_data], ypos[valid_data])

            # lowpass filter position estimates
            samplingrate = times[1] - times[0]
            xpos_lowpass = lowpass_filter(
                xpos_interp, samplingrate, lowpass_cutoff
            )
            ypos_lowpass = lowpass_filter(
                ypos_interp, samplingrate, lowpass_cutoff
            )

            # overwrite class instance data
            self.xpositions[self.identities == track_id] = xpos_lowpass
            self.ypositions[self.identities == track_id] = ypos_lowpass

        logger.info("Smoothed positions!")
        self.type.extend("position estimates smoothed")

    def load_logger(self, filename: str = "hobologger.csv") -> None:
        """
        load_logger loads temperature and light from the data logger file placed in the dataroot directory.

        Parameters
        ----------
        filename : str, optional
            Name of the logger file, by default "hobologger.csv"
        """

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
        self.temperature = np.interp(self.times, time, temp)
        self.light = np.interp(self.times, time, light)

        logger.info("Loaded hobologger successfully!")
        self.type.extend("temperature and light data loaded")

    def compute_q10(self) -> None:
        """
        compute_q10 calculates the q10 (temperature coefficient) for each invidual fish
        using the temperature information from the logger.
        """

        logger.info("Computing individual Q10 values ...")

        def temperature_coefficient(tmax, tmin, fmax, fmin):
            return np.round(np.mean((fmax / fmin) ** (10 / (tmax - tmin))), 4)

        # check if instance has temp data
        if len(self.temperature) == 1:
            msg = "This dataset has no temperature data! Aborting ..."
            logger.error(msg)
            raise GridDataMissing(msg)

        q10 = []
        for track_id in self.ids:
            # load data
            frequency = self.frequencies[self.identities == track_id]
            temperature = self.temperature[
                self.indices[self.identities == track_id]
            ]

            # get index for min and max of temperature
            indices = np.arange(len(temperature))
            imin = indices[temperature == temperature.min()]
            imax = indices[temperature == temperature.max()]

            # filter signal to remove peaks
            f_minima = minimum_filter1d(frequency, 801)
            f_lowpass = lowpass_filter(f_minima, 3, 0.0005, 2)

            q10.append(
                temperature_coefficient(
                    temperature[imax],
                    temperature[imin],
                    f_lowpass[imax],
                    f_lowpass[imin],
                )
            )
        self.q10 = np.asarray(q10)

    def sex_fish(self, upper="m", thresh=750, normtemp=25) -> None:
        """
        sex_fish estimates fish sex based on fundamental frequency, temperature and Q10 value.

        Parameters
        ----------
        upper : str, optional
            sex of fish with higher frequencies, by default "m"
        thresh : int, optional
            frequency threshold, by default 750
        normtemp : int, optional
            temperature to normalize to, by default 25
        q10 : float, optional
            Q10 value of species from literature, by default 1.6
        """

        def sexing(upper: str, thresh: float, mode: float) -> str:
            sex = None
            if upper == "m":
                sex = "m" if mode > thresh else "f"
            if upper == "f":
                sex = "f" if mode > thresh else "m"

            return sex

        logger.info("Estimating fish sex ...")

        # check if instance has temp data
        if len(self.temperature) == 1:
            msg = "This dataset has no temperature data! Aborting ..."
            logger.error(msg)
            raise GridDataMissing(msg)

        # check if instance has q10 data
        if len(self.q10) == 1:
            msg = "Instance has no Q10 values! Estimate them before sexing."
            logger.error(msg)
            raise GridDataMissing(msg)

        # iterate through all "individuals"
        self.sex = []
        for track_id in tqdm(self.ids, desc="Estimate sex    "):
            # normalize by q10 value
            tmin = self.times[self.indices[self.identities == track_id]][0]
            tmax = self.times[self.indices[self.identities == track_id]][-1]

            index = np.arange(len(self.times))
            start, stop = (
                index[self.times == tmin][0],
                index[self.times == tmax][0] + 1,
            )

            temp = self.temperature[start:stop]
            data = self.frequencies[self.identities == track_id]

            # get q10
            q10 = self.q10[self.ids == track_id]

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

    def integrity_check(self) -> bool:
        """
        integrity_check checks if the data arrays in the dataset match in dimensions.

        Returns
        -------
        bool
            True if check passed, else False

        Raises
        ------
        GridDataMismatch
            Indicates error in data arrays.
        """

        logger.info("Starting integrity check ...")

        try:
            # check overall array lengths
            len_ident_v = len(self.identities)
            len_idx_v = len(self.indices)
            len_fund_v = len(self.frequencies)
            len_xpos = len(self.xpositions)
            len_ypos = len(self.ypositions)
            len_sign_v = len(self.powers)
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
            if len_time < np.max(self.indices):
                msg = "Too many indices for time array!"
                logger.error(msg)
                raise GridDataMismatch(msg)

            # check sex to id match
            if len(self.sex) != len(self.ids):
                msg = "Sex array does not match id array!"
                logger.error(msg)
                raise GridDataMismatch(msg)

            # check Q10 to id match
            if len(self.q10) != len(self.ids):
                msg = "Q10 array does not match id array!"
                logger.error(msg)
                raise GridDataMismatch(msg)

            # check for every tracked fish
            for track_id in self.ids:
                time = self.times[self.indices[self.identities == track_id]]
                fund = self.frequencies[self.identities == track_id]
                power = self.powers[self.identities == track_id]
                xpos = self.xpositions[self.identities == track_id]
                ypos = self.xpositions[self.identities == track_id]

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

    def save_data(
        self,
        outputpath: str,
        filename: str,
        overwritable: bool = False,
        check: bool = True,
    ) -> None:
        """
        save_data saves class instance data to disk.

        Parameters
        ----------
        outputpath : str
            Path to the output directory
        overwritable : bool, optional
            Whether to allow overwriting original data, by default False
        check : bool, optional
            Whether to enable integrity_check before saving, by default True

        Raises
        ------
        BadOutputDir
            If the output dir is the same as the input or contains a raw file.
        """

        logger.info("Saving data ...")

        def save(self, outputpath: str) -> None:
            # check if specified output file already exists
            target_path = f"{outputpath}/{filename}.nix"
            files = os.listdir(outputpath)
            nixfiles = [file for file in files if f"{filename}.nix" in file]

            # create block for the current recording
            blockname = str(self.starttime)

            # open file in overwrite mode in both cases but supply logging messages
            if len(nixfiles) != 0:
                logger.warning(
                    "The file you specified already exists. Overwriting ..."
                )
                file = nio.File.open(target_path, nio.FileMode.ReadWrite)
            else:
                logger.info("Creating new {}.nix file ...".format(filename))
                file = nio.File.open(target_path, nio.FileMode.Overwrite)

            if blockname in file.blocks:
                del file.blocks[blockname]

            # delete block if it exists
            block = file.create_block(blockname, f"Recording at {blockname}")

            # write time array
            block.create_data_array(
                name="times",
                array_type="nix.sampled",
                data=self.times,
                label="shared time array",
                unit="s",
            )

            # write data array in the current block
            block.create_data_array(
                name="frequencies",
                array_type="nix.sampled",
                data=self.frequencies,
                label="fundamental frequencies",
                unit="Hz",
            )

            # write identity array
            block.create_data_array(
                name="identities",
                array_type="nix.sampled",
                data=self.identities,
                label="frequency identities",
                unit=None,
            )

            # write index to access time
            block.create_data_array(
                name="indices",
                array_type="nix.sampled",
                data=self.indices,
                label="index on time",
                unit=None,
            )

            # write powers
            block.create_data_array(
                name="powers",
                array_type="nix.sampled",
                data=self.powers,
                label="powers for frequencies",
                unit="dB",
            )

            # write positions
            block.create_data_array(
                name="xpositions",
                array_type="nix.sampled",
                data=self.xpositions,
                label="x coordinates for ids",
                unit="cm",
            )

            block.create_data_array(
                name="ypositions",
                array_type="nix.sampled",
                data=self.ypositions,
                label="y coordinates for ids",
                unit="cm",
            )

            # write temperature
            block.create_data_array(
                name="temperature",
                array_type="nix.sampled",
                data=self.temperature,
                label="water temperature",
                unit="°C",
            )

            # write light
            block.create_data_array(
                name="light",
                array_type="nix.sampled",
                data=self.light,
                label="ambient luminance",
                unit="lx",
            )

            # add ids
            block.create_data_array(
                name="ids",
                array_type="nix.sampled",
                data=self.ids,
                label="unique identities",
                unit=None,
            )

            # write estimated sex
            block.create_data_array(
                name="sex",
                array_type="nix.sampled",
                dtype=nio.DataType.String,
                data=self.sex,
                label="sex of each identity",
                unit=None,
            )

            # write estimated Q10 values
            block.create_data_array(
                name="q10",
                array_type="nix.sampled",
                data=self.q10,
                label="Q10 of each identity",
                unit=None,
            )

            logger.info("Loading and saving spectrograms ...")

            # load spectrograms
            self.coarse_spectrogram = np.load(
                self._datapath + "spec.npy", allow_pickle=True
            )
            self.fine_spectrogram_frequencies = np.load(
                self._datapath + "fill_freqs.npy", allow_pickle=True
            )
            self.fine_spectrogram_times = np.load(
                self._datapath + "fill_times.npy", allow_pickle=True
            )
            self.fine_spectrogram_shape = np.load(
                self._datapath + "fill_spec_shape.npy", allow_pickle=True
            )
            self.fine_spectrogram = np.memmap(
                self._datapath + "fill_spec.npy",
                dtype="float",
                mode="r",
                shape=(
                    self.fine_spectrogram_shape[0],
                    self.fine_spectrogram_shape[1],
                ),
                order="F",
            )

            # write spectrograms
            # I think this does not work for some reason
            # need to look into how I handle the memmap stuff

            print("coarse spectrogram")
            block.create_data_array(
                name="coarse spectrogram",
                array_type="nix.sampled",
                data=self.coarse_spectrogram,
                label="coarse grid spectrogram",
                unit=None,
            )

            """
            print("fine spectrogram")
            block.create_data_array(
                name="fine spectrogram",
                array_type="nix.sampled",
                data=self.fine_spectrogram,
                label="fine grid spectrogram",
                unit=None,
            )
            """

            print("fine spectrogram shape")
            block.create_data_array(
                name="fine spectrogram shape",
                array_type="nix.sampled",
                data=self.fine_spectrogram_shape,
                label="shape of fine spectrogram",
                unit=None,
            )

            print("fine spectrogram times")
            block.create_data_array(
                name="fine spectrogram times",
                array_type="nix.sampled",
                data=self.fine_spectrogram_times,
                label="times for fine spectrogram",
                unit="s",
            )

            print("fine spectrogram frequencies")
            block.create_data_array(
                name="fine spectrogram frequencies",
                array_type="nix.sampled",
                data=self.fine_spectrogram_frequencies,
                label="frequencies for fine spectrogram",
                unit="Hz",
            )

            file.flush()
            file.close()

        if not overwritable:
            if len(glob.glob(outputpath + "*.raw")) > 0:
                msg = "The output path contains a raw file! Do not overwrite exisiting data! Run 'save_data' in overwrite mode if desired."
                logger.error(msg)
                raise BadOutputDir(msg)

            if Path(outputpath) is Path(self._datapath):
                msg = "Outputpath and datapath are the same! Run 'save_data' in overwrite mode if desired."
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
