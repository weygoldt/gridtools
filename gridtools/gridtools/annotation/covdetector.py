import datetime
import os
import string
from inspect import FrameInfo

import gridtools as gt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import thunderfish as tf
import yaml
from gridtools import utils as fs
from matplotlib.colorbar import Colorbar
from scipy.signal import find_peaks
from tabulate import tabulate
from tqdm import tqdm

from plotstyle import PlotStyle
from termcolors import TermColor as tc


class SWXCov:
    def __init__(self, data_1, data_2, time, bin_width, maxlag, step, verbose=False):

        # intialize private variables
        self.__verbose = verbose
        self.__tqdm_disable = True if self.__verbose is False else False

        # initialize variables
        self.covs, self.times, self.lags = self.sliding_window_xcov(
            data_1, data_2, time, bin_width, maxlag, step
        )
        self.maxcovs, self.maxlags = self.maxima()
        self.mincovs, self.minlags = self.minima()

    def xcov(self, data_1, data_2, maxlag):
        """
        Cross covariance with lags.
        `x` and `y` must be one-dimensional numpy arrays with the same length.
        The return value has length 2*maxlag + 1.
        """

        # reduce maxlag size when data arrays are shorter than lags
        if maxlag >= len(data_1):
            raise Exception(
                tc.err("[ WARNING ]")
                + " Decrease xcorr maxlags or use longer data arrays! Data arrays are shorter than provided lags!"
            )
            return None

        # pad y vector according to maxlags
        data_2_pad = np.pad(data_2.conj(), 2 * maxlag, mode="constant")

        # create all shifted y arrays (should be maxlags + 1 number of arrays)
        T = np.lib.stride_tricks.as_strided(
            data_2_pad[2 * maxlag :],
            shape=(2 * maxlag + 1, len(data_2) + 2 * maxlag),
            strides=(-data_2_pad.strides[0], data_2_pad.strides[0]),
        )

        # pad x vector according to maxlags
        data_1_pad = np.pad(data_1, maxlag, mode="constant")

        # calculate covariance
        covs = T.dot(data_1_pad) / data_1_pad.size - (
            T.mean(axis=1) * data_1_pad.mean()
        )

        # make lags vector to return
        lags = np.arange(-maxlag, maxlag + 1, dtype=int)

        return covs, lags

    def sliding_window_xcov(
        self,
        data_1,
        data_2,
        time,
        bin_width,
        maxlag,
        step=1,
    ):

        cov_radius = int((bin_width - 1) / 2)  # area left and right of center
        covs_times = []  # times at covs
        covs_m = []  # times at covs# covariance matrix

        if self.__verbose:
            print(
                f"{tc.GREEN}{tc.BOLD}[ SwxCov.sliding_window_xcov ]{tc.END} Computing rolling window cross covariance ..."
            )

        for idx in tqdm(
            range(cov_radius + 1, len(time) - cov_radius + 1, step),
            disable=self.__tqdm_disable,
        ):
            bin = np.arange(idx - cov_radius - 1, idx + cov_radius)
            tp = time[idx]  # timepoint

            covs, lags = self.xcov(data_1[bin], data_2[bin], maxlag)

            covs_times.append(tp)
            covs_m.append(np.array(covs))
            covs_lags = lags

        covs_t = np.asanyarray(covs_times)
        covs = np.asarray(covs_m)
        covs = covs.transpose()

        return covs, covs_t, covs_lags

    def maxima(self):
        maxcovs = np.zeros(len(self.covs[0, :]), dtype=np.float_)  # maximum covariances
        maxlags = np.zeros(len(self.covs[0, :]), dtype=np.int_)  # lags at max cov

        for index in range(len(maxcovs)):

            # get max covariances at time point
            maxcovs[index] = np.max(self.covs[:, index])

            # in rare cases there are two peaks, take first one if happens
            maxlags[index] = self.lags[self.covs[:, index] == maxcovs[index]][0]

        return maxcovs, maxlags

    def minima(self):
        mincovs = np.zeros(len(self.covs[0, :]))  # maximum covariances
        minlags = np.zeros(len(self.covs[0, :]))  # lags at max cov

        for index in range(len(mincovs)):
            mincovs[index] = np.min(self.covs[:, index])
            minlags[index] = self.lags[self.covs[:, index] == mincovs[index]][0]

        return mincovs, minlags


class CovDetector:
    def __init__(self, datapath, config, ids="all", verbose=False):

        # private variables
        self.__verbose = verbose  # set verbosity of functions and classes
        self.__tqdm_disable = True if self.__verbose is False else True
        self.__plotoutput = {}  # where gui saves its data while running
        self.__split_event = False  # control parameter to split events
        self.__dry_run = config["dry_run"]
        self.__eventcounter = 0

        # bp filter parameters
        self.rate_bp = config["track_rate"]
        self.hcutoffs_bp = config["hcutoffs"]
        self.lcutoffs_bp = config["lcutoffs"]

        # dyad params
        self.duration_thresh = config["dyad_duration_threshold"]

        # swxcov parameters
        bin_width = config["swxcov_bin_duration"] * config["track_rate"]
        self.bin_width = bin_width if fs.is_odd(bin_width) else (bin_width - 1)
        self.radius = (self.bin_width - 1) / 2
        self.step_cov = config["step"]
        self.maxlag = config["swxcov_lags"] * config["track_rate"]

        # peak detection parameters
        self.peakprom_h = config["swxcov_peakprom_h"]
        self.peakprom_l = config["swxcov_peakprom_l"]

        # initialize empty output arrays
        self.rec_out = []  # recording
        self.id1_out = []  # interacting id 1
        self.id2_out = []  # interacting id 2
        self.initiator_out = []  # id of initiator
        self.start_out = []  # event start timestamp
        self.stop_out = []  # event stop timestamp

        # prepare data aquisition
        self.datapath = datapath
        self.recording = os.path.split(self.datapath[:-1])[-1]

        # get data
        self.grid = gt.GridTracks(self.datapath, finespec=True)  # raw grid
        self.grid_p = gt.GridTracks(self.datapath, finespec=False)  # normalized
        self.grid_h = gt.GridTracks(self.datapath, finespec=False)  # filtered higher
        self.grid_l = gt.GridTracks(self.datapath, finespec=False)  # filtered lower

        # normalize p grid for sexing
        self.grid_p.q10_norm()

        # filter p grid for plotting in GUI
        self.grid_p.freq_bandpass(
            rate=3, flow=0.00004, fhigh=0.4, order=2, eod_shift=False
        )

        # apply bandpass filters for event detection
        self.grid_h.freq_bandpass(
            rate=self.rate_bp,
            flow=self.hcutoffs_bp[0],
            fhigh=self.hcutoffs_bp[1],
            order=1,
            eod_shift=False,
        )
        self.grid_l.freq_bandpass(
            rate=self.rate_bp,
            flow=self.lcutoffs_bp[0],
            fhigh=self.lcutoffs_bp[1],
            order=1,
            eod_shift=False,
        )

        # extract ids if specified
        if ids == "all":
            self.ids = self.grid.ids
        elif isinstance(ids, list):
            self.ids = ids
            self.grid.extract_ids(self.ids)
            self.grid_p.extract_ids(self.ids)
            self.grid_h.extract_ids(self.ids)
            self.grid_l.extract_ids(self.ids)
        else:
            print(
                tc.err("[ CovDetector.__init__ ]")
                + " please specify valid ids ('all' | list)!"
            )

        # make unique id pairs
        self.id_dyads = fs.unique_combinations1d(self.ids)

    def run_detection(self):
        def detector(self, dyad_h, dyad_l):

            # compute sliding window cross covariances
            covs_h = SWXCov(
                data_1=dyad_h.fund_id1,
                data_2=dyad_h.fund_id2,
                time=dyad_h.times,
                bin_width=self.bin_width,
                maxlag=self.maxlag,
                step=self.step_cov,
                verbose=self.__verbose,
            )
            covs_l = SWXCov(
                data_1=dyad_l.fund_id1,
                data_2=dyad_l.fund_id2,
                time=dyad_l.times,
                bin_width=self.bin_width,
                maxlag=self.maxlag,
                step=self.step_cov,
                verbose=self.__verbose,
            )

            # get shared times
            times = covs_h.times

            # make index vector for time vector
            int_times = np.arange(len(times))

            # find peaks in maximum covs
            peaks_h, _ = find_peaks(covs_h.maxcovs, prominence=self.peakprom_h)
            peaks_l, _ = find_peaks(covs_l.maxcovs, prominence=self.peakprom_l)

            # find simultaneously cooccurring peaks
            event_indices, event_peaks_l, event_peaks_h = fs.combine_cooccurences(
                peaks_l, peaks_h, int_times, self.radius, merge=False, crop=False
            )

            # convert indices of cov time vector to timestamps
            events = []
            peaks_h = []
            peaks_l = []
            for event_idxs, peaks_l_idxs, peaks_h_idxs in zip(
                event_indices, event_peaks_l, event_peaks_h
            ):

                # convert indices on shared dyad time array to timestamps
                start, stop = np.min(times[event_idxs]), np.max(times[event_idxs])
                event = [start, stop]
                peak_h = times[peaks_h_idxs]
                peak_l = times[peaks_l_idxs]

                events.append(event)
                peaks_h.append(peak_h)
                peaks_l.append(peak_l)

            return covs_h, covs_l, events, peaks_h, peaks_l

        for id_dyad in tqdm(self.id_dyads, disable=self.__tqdm_disable):

            # initialize dyads for loaded datasets
            dyad = gt.Dyad(
                self.grid, id_dyad, verbose=False
            )  # raw data for spectrogram
            dyad_p = gt.Dyad(self.grid_p, id_dyad, verbose=False)  # normalized for GUI
            dyad_h = gt.Dyad(
                self.grid_h, id_dyad, verbose=False
            )  # higher bp for detect.
            dyad_l = gt.Dyad(
                self.grid_l, id_dyad, verbose=False
            )  # lower bp for detect.

            # current ids to class namespace
            id1 = int(dyad.id1)
            id2 = int(dyad.id2)

            # check if dyad overlaps, exit function if not
            if dyad.overlap is False:
                print(
                    tc.warn("[ CovDetector.detect_events ]")
                    + f" Skipping dyad, {id1}, {id2} do not overlap."
                )
                continue

            # compute overlap duration, check if long enough
            duration = np.max(dyad.times) - np.min(dyad.times)
            if duration < self.duration_thresh:
                print(
                    tc.warn("[ CovDetector.detect_events ]")
                    + f" Skipping since overlapping duration of {id1}, {id2} not long enough!"
                )
                continue

            print(tc.succ("[ CovDetector.detect_events ]") + f" {id1}, {id2} loaded.")

            # run detector
            covs_h, covs_l, events, peaks_h, peaks_l = detector(self, dyad_h, dyad_l)

            # sort events in gui
            for event, peak_h, peak_l in zip(events, peaks_h, peaks_l):

                print(event)

                self.__split_event = True  # to start while loop for single event

                # while repeats for same event if gui input enables split
                while self.__split_event:
                    self.gui_plot(dyad_p, event)

                    if self.__plotoutput["event"] == 1:
                        self.backend_plot(
                            dyad, dyad_h, dyad_l, covs_h, covs_l, peak_h, peak_l
                        )

    def gui_plot(self, dyad_p, event):
        def freq_lim(track1, track2, start, stop, padding):
            """Helper function to get frequency y axis limits from two fundamental frequency tracks.

            Args:
                track1 (array): First track
                track2 (array): Second track
                start (int): Index for first value to be plotted
                stop (int): Index for second value to be plotted
                padding (int): Padding for the upper and lower limit

            Returns:
                lower (float): lower limit
                upper (float): upper limit

            """
            allfunds_tmp = (
                np.concatenate(
                    [
                        track1[start:stop],
                        track2[start:stop],
                    ]
                )
                .ravel()
                .tolist()
            )
            lower = np.min(allfunds_tmp) - padding
            upper = np.max(allfunds_tmp) + padding
            return lower, upper

        s = PlotStyle()

        fig, ax = plt.subplots(figsize=(297 * s.mm, 180 * s.mm))

        ax.set_ylabel("norm. frequency [Hz]")
        ax.set_xlabel("time [s]")

        ax.plot(dyad_p.times, dyad_p.fund_id1, label="id1", **s.id1)
        ax.plot(dyad_p.times, dyad_p.fund_id2, label="id2", **s.id2)

        # plot previously detected events
        if len(self.start_out) > 0:
            for start_ts, stop_ts in zip(self.start_out, self.stop_out):
                ax.axvspan(start_ts, stop_ts, color="red", alpha=0.2)

        ylims = fs.get_ylims(
            dyad_p.fund_id1,
            dyad_p.fund_id2,
            dyad_p.times,
            event[0],
            event[1],
            padding=0.2,
        )

        # set limits to event
        ax.set_ylim(ylims[0], ylims[1])
        ax.set_xlim(event[0], event[1])

        # add legend
        leg = ax.legend(
            loc="upper right",
            ncol=2,
            frameon=False,
        )
        leg.get_texts()[0].set_text(f"ID 1 [{int(dyad_p.id1)}]")
        leg.get_texts()[1].set_text(f"ID 2 [{int(dyad_p.id2)}]")

        # where plot gui saves input
        self.__plotoutput = {}

        # hint for help
        print('Enter "?" for help!')

        # initiate mpl connect
        fig.canvas.mpl_connect(
            "key_press_event",
            lambda event: self.on_key(event, dyad_p.id1, dyad_p.id2, ax),
        )

        plt.show()

        # convert plot lims to timestamps
        start_idx = fs.find_closest(dyad_p.times, self.__plotoutput["lims"][0])
        stop_idx = fs.find_closest(dyad_p.times, self.__plotoutput["lims"][1])
        start = dyad_p.times[start_idx]
        stop = dyad_p.times[stop_idx]

        # append gui data to output lists
        if self.__plotoutput["event"] == 1:
            self.__eventcounter += 1
            self.rec_out.append(self.recording)
            self.id1_out.append(dyad_p.id1)
            self.id2_out.append(dyad_p.id2)
            self.initiator_out.append(self.__plotoutput["init_id"])
            self.start_out.append(start)
            self.stop_out.append(stop)

        # control control variable to split or not to split event
        self.__split_event = self.__plotoutput["split"]

    def on_key(self, event, id1, id2, ax):
        if event.key == "?":
            print("--------------------------------------------")
            print("Welcome to the CovDetector!")
            print("Mark event on- and offset by using the zoom function and then:")
            print(
                "Enter "
                + tc.succ("[ 1/2 ]")
                + " to accept plotted region as event (1) or not (2"
            )
            print(
                "THEN enter "
                + tc.succ("[ 1/2/3 ]")
                + " to set the event initiaor to ID1 (1), ID2 (2) or NAN (3)."
            )
            print(
                "THEN enter "
                + tc.succ("[ a ]")
                + " to accept the plotted x limits as event start and stop."
            )
            print(
                "OPTIONALLY enter"
                + tc.succ("[ d ]")
                + " to split the plotted region in multiple events and plot it again to mark another event."
            )
            print(
                "THEN enter "
                + tc.succ("[ r/w ]")
                + " to reset variables (r) or save input (w) and close."
            )
            print("--------------------------------------------")

        elif len(self.__plotoutput) == 0:
            if event.key == "1":
                self.__plotoutput["event"] = 1
                print(f'Set event to {self.__plotoutput["event"]}')
            elif event.key == "2":
                self.__plotoutput["event"] = 0
                print(f'Set event to {self.__plotoutput["event"]}')
            elif event.key == "r":
                print("variables reset!")
                self.__plotoutput = {}
            elif event.key == "w":
                print(f"Finish input before saving. Current input: {self.__plotoutput}")
            else:
                print("Invalid input, try again!")

        elif len(self.__plotoutput) == 1:
            if event.key == "1":
                self.__plotoutput["init_id"] = int(id1)
                print(f'Set initiator to {self.__plotoutput["init_id"]}')
            elif event.key == "2":
                self.__plotoutput["init_id"] = int(id2)
                print(f'Set initiator to {self.__plotoutput["init_id"]}')
            elif event.key == "3":
                self.__plotoutput["init_id"] = np.nan
                print(f'Set initiator to {self.__plotoutput["init_id"]}')
            elif event.key == "r":
                print("variables reset!")
                self.__plotoutput = {}
            elif event.key == "w":
                print(f"Finish input before saving. Current input: {self.__plotoutput}")
            else:
                print("Invalid input, try again!")

        elif len(self.__plotoutput) == 2:
            if event.key == "a":
                self.__plotoutput["lims"] = ax.get_xlim()
                print(f'Set limits to {self.__plotoutput["lims"]}')
            elif event.key == "r":
                print("variables reset!")
                self.__plotoutput = {}

        elif len(self.__plotoutput) == 3:
            if event.key == "d":
                self.__plotoutput["split"] = True
            if event.key == "w":
                self.__plotoutput["split"] = False
                plt.close()
            elif event.key == "r":
                print("variables reset!")
                self.__plotoutput = {}

        elif len(self.__plotoutput) == 4:
            if event.key == "w":
                plt.close()
            elif event.key == "r":
                print("variables reset!")
                self.__plotoutput = {}

    def backend_plot(self, dyad, dyad_h, dyad_l, covs_h, covs_l, peak_h, peak_l):
        def get_xlims(t, tstart, tstop, padding=0.4):
            dt = tstop - tstart
            xlims = [0, 0]

            if tstart - dt * padding > np.min(t):
                xlims[0] = tstart - dt * padding
            else:
                xlims[0] = np.min(t)

            if tstop + dt * padding < np.max(t):
                xlims[1] = tstop + dt * padding
            else:
                xlims[1] = np.max(t)

            # find where x lims exists on time vector
            start_idx = fs.find_closest(t, xlims[0])
            stop_idx = fs.find_closest(t, xlims[1])

            xlims[0] = t[start_idx]
            xlims[1] = t[stop_idx]

            return xlims

        def clock_time(xlims, rec_datetime, times, axis):
            xlim = xlims
            dx = np.diff(xlim)[0]

            label_idx0 = 0
            if dx <= 20:
                res = 1
            elif dx > 20 and dx <= 120:
                res = 10
            elif dx > 120 and dx <= 600:
                res = 300
            elif dx > 600 and dx <= 1200:
                res = 300
            elif dx > 1200 and dx <= 3600:
                res = 600  # 10 min
            elif dx > 3600 and dx <= 7200:
                res = 1800  # 30 min
            else:
                res = 3600  # 60 min

            if dx > 600:
                if rec_datetime.minute % int(res / 60) != 0:
                    dmin = int(res / 60) - rec_datetime.minute % int(res / 60)
                    label_idx0 = dmin * 60
            if dx > 1200:
                if rec_datetime.minute % int(res / 60) != 0:
                    dmin = int(res / 60) - rec_datetime.minute % int(res / 60)
                    label_idx0 = dmin * 60

            xtick = np.arange(label_idx0, times[-1], res)
            datetime_xlabels = list(
                map(lambda x: rec_datetime + datetime.timedelta(seconds=x), xtick)
            )

            if dx > 120:
                xlabels = list(
                    map(
                        lambda x: ("%2s:%2s" % (str(x.hour), str(x.minute))).replace(
                            " ", "0"
                        ),
                        datetime_xlabels,
                    )
                )
                rotation = 0
            else:
                xlabels = list(
                    map(
                        lambda x: (
                            "%2s:%2s:%2s" % (str(x.hour), str(x.minute), str(x.second))
                        ).replace(" ", "0"),
                        datetime_xlabels,
                    )
                )
                rotation = 45

            # create mask
            mask = np.arange(len(xtick))[(xtick > xlim[0]) & (xtick < xlim[1])]
            axis.set_xticks(xtick[mask])
            axis.set_xticklabels(np.array(xlabels)[mask], rotation=rotation)
            axis.set_xlim(xlim)

        def plot_spec(grid, dyad, axis, style, xlims):

            # get y limits for spectrogram
            ylims = fs.get_ylims(
                dyad.fund_id1, dyad.fund_id2, dyad.times, xlims[0], xlims[1]
            )

            # get axes limits
            f0, f1 = ylims[0], ylims[1]
            t0, t1 = xlims[0], xlims[1]

            # make mask for spectrogram
            f_mask = np.arange(len(grid.fill_freqs))[
                (grid.fill_freqs >= f0 + 5) & (grid.fill_freqs <= f1 + 5)
            ]
            t_mask = np.arange(len(grid.fill_times))[
                (grid.fill_times >= t0) & (grid.fill_times <= t1)
            ]

            # plot spectrogram
            axis.imshow(
                tf.powerspectrum.decibel(
                    grid.fill_spec[f_mask[0] : f_mask[-1], t_mask[0] : t_mask[-1]][::-1]
                ),
                extent=[t0, t1, f0, f1],
                vmin=-100,
                vmax=-50,
                **style.fine_spec,
            )

            # plot tracks to spectrogram
            axis.plot(dyad.times, dyad.fund_id1, **style.id1, label="id1")
            axis.plot(dyad.times, dyad.fund_id2, **style.id2, label="id2")

            axis.set_ylabel("EOD $f$  [Hz]")  # set ylabel
            axis.set_ylim(f0, f1)

            # # init legend
            # leg = axis.legend(
            #     bbox_to_anchor=(1.03, 1.16, 0, 0),
            #     # bbox_to_anchor=(1.03, 1.14, 0, 0),
            #     loc="upper right",
            #     ncol=2,
            #     frameon=False,
            # )
            # leg.get_texts()[0].set_text(f"ID 1 [{int(dyad.id1)}]")
            # leg.get_texts()[1].set_text(f"ID 2 [{int(dyad.id2)}]")

        def plot_tracks(dyad, axis, style, xlims):

            # get y limits for spectrogram
            ylims = fs.get_ylims(
                dyad.fund_id1, dyad.fund_id2, dyad.times, xlims[0], xlims[1]
            )

            # plot tracks
            axis.plot(dyad.times, dyad.fund_id1, **style.id1)
            axis.plot(dyad.times, dyad.fund_id2, **style.id2)

            axis.set_ylim(ylims[0], ylims[1])

        def plot_covs(covs, xlims, maxlag, axis, cbar_axis, style):

            indices = np.arange(len(covs.times), dtype=int)
            start, stop = (
                indices[covs.times == xlims[0]][0],
                indices[covs.times == xlims[1]][0],
            )
            hm = axis.imshow(
                covs.covs[:, start:stop],
                extent=[
                    covs.times[start],
                    covs.times[stop] + (covs.times[1] - covs.times[0]),
                    -maxlag / self.rate_bp,
                    maxlag / self.rate_bp,
                ],
                origin="lower",
                **style.cov_heatmap,
            )

            # add a center line
            axis.plot(covs.times, np.zeros(len(covs.times)), **style.center_line)

            # plot maxlags
            axis.plot(covs.times, covs.maxlags / self.rate_bp, **style.lags)

            # add colorbar
            if cbar_axis is not None:
                Colorbar(
                    ax=cbar_axis,
                    mappable=hm,
                    orientation="vertical",
                    ticklocation="right",
                )

            # tick management
            axis.set_yticks([-maxlag / self.rate_bp, 0, maxlag / self.rate_bp])
            axis.set_xlim(xlims[0], xlims[1])

        def plot_maxcovs(covs_h, covs_l, peaks_h, peaks_l, xlims, axis, style):

            # get maxcovs in xlims for normalization
            times = covs_h.times
            indices = np.arange(len(covs_h.maxcovs))
            start, stop = (
                indices[times == xlims[0]][0],
                indices[times == xlims[1]][0],
            )
            mcnorm_h = covs_h.maxcovs[start:stop]
            mcnorm_l = covs_l.maxcovs[start:stop]

            # normalize maxcovs in xlim ranges
            maxcovs_h_norm = covs_h.maxcovs / np.max(mcnorm_h)
            maxcovs_l_norm = covs_l.maxcovs / np.max(mcnorm_l)

            # plot normalized maxcovs
            axis.plot(times, maxcovs_h_norm, **style.maxcovs_h)
            axis.plot(times, maxcovs_l_norm, **style.maxcovs_l)

            # convert peak timestamps to indices
            peaks_h_idxs = [indices[times == ts][0] for ts in peaks_h]
            peaks_l_idxs = [indices[times == ts][0] for ts in peaks_l]

            # plot peaks
            axis.scatter(
                times[peaks_h_idxs],
                maxcovs_h_norm[peaks_h_idxs],
                **style.maxcovs_h_peaks,
            )
            axis.scatter(
                times[peaks_l_idxs],
                maxcovs_l_norm[peaks_l_idxs],
                **style.maxcovs_l_peaks,
            )

            axis.set_ylim(-0.1, 1.2)

            axis.legend(
                loc="best",
                frameon=False,
                facecolor="white",
                edgecolor="none",
                framealpha=0.6,
            )
            axis.set_ylabel("norm. cov.")

        # skip if no events collected yet
        if len(self.start_out) == 0:
            return None

        # get plot stye
        s = PlotStyle()

        # Grid setup
        hr = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        wr = [0.6, 0.6]  # make [1, 0.02] to add cbar
        ny = len(hr)
        nx = len(wr)

        # init fig
        fig = plt.figure(figsize=(180 * s.mm, 150 * s.mm), constrained_layout=False)

        # init gridspec
        gs = gridspec.GridSpec(ny, nx, figure=fig, height_ratios=hr, width_ratios=wr)
        gs.update(left=0.1, right=0.99, bottom=0.09, top=0.93, wspace=0.4, hspace=0.8)

        # make axes
        ax_spec = fig.add_subplot(gs[0:6, 0])

        ax_ch = fig.add_subplot(gs[7:10, 0], sharex=ax_spec)
        ax_cl = fig.add_subplot(gs[10:13, 0], sharex=ax_ch)
        ax_help2 = fig.add_subplot(gs[7:13, 0], frameon=False)

        ax_bph = fig.add_subplot(gs[0:3, 1], sharex=ax_spec)
        ax_bpl = fig.add_subplot(gs[3:6, 1], sharex=ax_bph)
        ax_help1 = fig.add_subplot(gs[0:6, 1], frameon=False)

        ax_mcs = fig.add_subplot(gs[7:13, 1], sharex=ax_cl)

        ax_help1.set_ylabel("filt. EOD $f$  [Hz]")  # y label
        ax_help2.set_ylabel("xcov. lags [s]")  # y label

        s.hide_helper_xax(ax_help1)
        s.hide_helper_xax(ax_help2)

        # disable x labels for all except last
        plt.setp(ax_spec.get_xticklabels(), visible=False)
        plt.setp(ax_bph.get_xticklabels(), visible=False)
        plt.setp(ax_bpl.get_xticklabels(), visible=False)
        plt.setp(ax_ch.get_xticklabels(), visible=False)
        # plt.setp(ax_cl.get_xticklabels(), visible=False)

        # make titles
        ax_spec.set_title("Tracked EOD$f$", loc="center")
        ax_bph.set_title("Bandpass-filtered tracks", loc="center")
        ax_ch.set_title("Sliding-window cross-cov.", loc="center")
        ax_mcs.set_title("Co-occuring cov. maxima", loc="center")

        ax_spec.text(
            -0.26, 1.1, "A", transform=ax_spec.transAxes, size=16, weight="bold"
        )
        ax_bph.text(
            -0.28, 1.22, "B", transform=ax_bph.transAxes, size=16, weight="bold"
        )
        ax_ch.text(-0.26, 1.22, "C", transform=ax_ch.transAxes, size=16, weight="bold")
        ax_mcs.text(
            -0.265, 1.1, "D", transform=ax_mcs.transAxes, size=16, weight="bold"
        )

        # get x limits for all subplots
        tstart = self.start_out[-1]
        tstop = self.stop_out[-1]
        xlims = get_xlims(covs_h.times, tstart, tstop)

        # plot spectrogram
        plot_spec(self.grid, dyad, ax_spec, s, xlims)
        clock_time(xlims, self.grid.datetimes[0], dyad.times, ax_spec)

        # plot filtered tracks
        plot_tracks(dyad_h, ax_bph, s, xlims)
        clock_time(xlims, self.grid.datetimes[0], dyad.times, ax_bph)
        plot_tracks(dyad_l, ax_bpl, s, xlims)
        clock_time(xlims, self.grid.datetimes[0], dyad.times, ax_bpl)

        # plot swxcovs
        plot_covs(covs_h, xlims, self.maxlag, ax_ch, None, s)
        clock_time(xlims, self.grid.datetimes[0], dyad.times, ax_ch)
        plot_covs(covs_l, xlims, self.maxlag, ax_cl, None, s)
        clock_time(xlims, self.grid.datetimes[0], dyad.times, ax_cl)

        # plot covariance maxima
        plot_maxcovs(covs_h, covs_l, peak_h, peak_l, xlims, ax_mcs, s)
        clock_time(xlims, self.grid.datetimes[0], covs_h.times, ax_mcs)

        # labels
        # ax_ch_cbar.set_ylabel("covariance", labelpad=10)  # cbar label
        # ax_ch_cbar.yaxis.set_label_coords(4.2, -0.2)  # offset to share betweeen axes

        ax_mcs.set_xlabel("time [hh:mm]")
        ax_cl.set_xlabel("time [hh:mm]")

        # save fig
        fig.align_labels()

        plt.show()
        fig.savefig(
            self.datapath
            + f"CovDetector_backend_ids_{dyad.id1}_{dyad.id2}_index_{self.__eventcounter-1}.pdf"
        )
        plt.close()

    def save(self):
        def interface():
            def promt():
                var = input("Save [s] or quit [q]? ")
                return var

            var = promt()
            save = False
            if var not in "sq":
                print("Invalid input, try again!")
                var = promt()
            elif var in "q":
                print(f"Ok bye!")
                quit()
            elif var in "s":
                save = True
                print(f"Saving ...")

            return save

        if self.__dry_run:
            print("You where running in a dry run! No data saved!")
        else:
            # make dict of output arrays
            table_out = {
                "rec": self.rec_out,
                "id1": self.id1_out,
                "id2": self.id2_out,
                "init": self.initiator_out,
                "start": self.start_out,
                "stop": self.stop_out,
            }

            # export pandas dataframe to csv
            save = interface()
            try:
                if save:
                    df = pd.DataFrame(table_out)
                    df.to_csv(
                        self.datapath + "events.csv", encoding="utf-8", index=False
                    )
                    print(
                        "\n"
                        + tc.rainb("Yay! Data saved succesfully! Here's a preview:")
                    )
                    print(
                        tabulate(df, headers="keys", tablefmt="psql", showindex=False)
                    )
            except ValueError:
                print(
                    tc.err(
                        "Error making dataframe. Collected data arrays are not the same lengths."
                    )
                )


def main():
    def save_progress(tmp):
        with open("covdetector_tmp.yml", "w") as file:
            yaml.dump(tmp, file)

    def interface():
        def promt():
            var = input("Pause detection [b] or continue to next recording [c]? ")
            return var

        var = promt()
        cont = True
        if var not in "bc":
            print("Invalid input, try again!")
            var = promt()
        elif var in "b":
            print(f"Ok bye!")
            cont = False
        elif var in "c":
            print(f"Here you go!")

        return cont

    # specify ids of interest
    ids = [26788, 26789]

    # open progress tracking file
    try:
        tmp = fs.load_ymlconf("covdetector_tmp.yml")
    except FileNotFoundError:
        print("Creating new temporary file.")
        tmp = {"processed": []}

    # open config file
    config = fs.load_ymlconf("covdetector_conf.yml")

    # make list of recordings
    recs = gt.ListRecordings(path=config["data"], exclude=config["exclude"])

    # if recordings are selected in conf, only use those
    if len(config["include_only"]) > 0:
        recs.recordings = config["include_only"]

    # check what is already processed
    processed_recs = tmp["processed"]
    unprocessed_recs = list(set(recs.recordings).difference(set(processed_recs)))

    if len(unprocessed_recs) == 0:
        print("Nothing to to, all recordings processed!")
    else:
        # iterate trough unprocessed recs list
        for recording in unprocessed_recs:

            # construct path to recording
            datapath = recs.dataroot + recording + "/"

            # event detection
            events = CovDetector(datapath, config, ids, verbose=True)
            events.run_detection()
            events.save()

            # save what was already processed
            tmp["processed"].append(recording)

            # terminal promt to pause or continue
            cont = interface()
            if cont is False:
                save_progress(tmp)
                break

        # save progress if no break nessecary
        save_progress(tmp)


if __name__ == "__main__":
    main()
