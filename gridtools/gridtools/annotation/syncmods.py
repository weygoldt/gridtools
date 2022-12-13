import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import gridspec
from matplotlib.colorbar import colorbar
from scipy.signal import find_peaks
from tqdm import tqdm

from ..logger import makeLogger
from ..plotstyle import PlotStyle
from ..utils.datahandling import crossCov
from ..utils.filehandling import ConfLoader

logger = makeLogger(__name__)

class SlidingWindowCrossCov:
    
    def __init__(self, data1: np.ndarray, data2: np.ndarray, times: np.ndarray, binw: int, maxlag: int, step: int) -> None:

        self.covs, self.times, self.lags = self.slidingWindowXC(data1, data2, times, binw, maxlag, step)

        self.maxcovs, self.maxlags = self.maxima(self.covs, self.lags)
        self.mincovs, self.minlags = self.minima(self.covs, self.lags)

    @staticmethod
    def slidingWindowXC(data1: np.ndarray, data2: np.ndarray, times:np.ndarray, binw: int, maxlag: int, step: int):

        # check if cov radius is odd
        if binw %2 != 0:
            msg=f"Covariance bin width must be odd but is {binw}!"
            logger.error(msg)
            raise ValueError(msg)

        # the "radius" of the bin (i.e. left and right of center of the bin)
        cov_radius = int((binw - 1) / 2)

        # empty arrays for data
        covs_times = []
        covs_m = []
        covs_lags = []

        # iderator over time array with indices
        iterator = np.arange(cov_radius+1, len(times)-cov_radius+1, step)
        for idx in tqdm(iterator):

            covbin = np.arange(idx - cov_radius -1, idx + cov_radius)
            tp = times[idx]
            covs, lags = crossCov(data1[covbin], data2[covbin], maxlag)
            covs_times.append(tp)
            covs_m.append(np.array(covs))
            covs_lags.append(lags)

        covs_times = np.asarray(covs_times)
        covs_m = np.asarray(covs_m).transpose()
        covs_lags = np.unique(np.ravel(covs_lags))

        return covs_m, covs_times, covs_lags

    @staticmethod
    def maxima(covs: np.ndarray, lags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        maxcovs = np.zeros(len(covs[0, :]), dtype=np.float_)  # maximum covariances
        maxlags = np.zeros(len(covs[0, :]), dtype=np.int_)  # lags at max cov

        for index in range(len(maxcovs)):
            # get max covariances at time point
            # in rare cases there are two peaks, take first one if happens
            maxcovs[index] = np.max(covs[:, index])
            maxlags[index] = lags[covs[:, index] == maxcovs[index]][0]

        return  maxcovs, maxlags

    @staticmethod
    def minima(covs: np.ndarray, lags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mincovs = np.zeros(len(covs[0, :]))  # maximum covariances
        minlags = np.zeros(len(covs[0, :]))  # lags at max cov

        for index in range(len(mincovs)):
            mincovs[index] = np.min(covs[:, index])
            minlags[index] = lags[covs[:, index] == mincovs[index]][0]

        return mincovs, minlags


class CovDetector:
    def __init__(self, datapath: str, config: ConfLoader, ids: str) -> None:

        # private variables
        self.__plotoutput = {}
        self.__split_event = False
        self.__dry_run = conf.dryrun
        self.__eventcounter = 0

        # bandpass filter parameters
        self.rate_bp = conf.samplingrate
        self.hcutoffs_bp = conf.hcutoffs
        self.lcutoffs_bp = conf.lcutoffs

        # dyad params
        self.duration_thresh = conf.dyad_dur_thresh

        # sliding window crosscov params
        bin_width = conf.cov_bin_dur * conf.samplingrate
        if bin_width % 2 != 0:
            msg = f"Bin width must be odd! The supplied bin duration {conf.cov_bin_dur} translates to a bin width of {bin_width}. Bin width will be decreased by one."
            logger.warning(msg)
            bin_width -= 1

        self.binw = bin_width
        self.radius = (self.binw - 1) / 2
        self.step_cov = conf.cov_step
        self.maxlag = conf.cov_lags * conf.samplingrate

        # peak detection parameters
        self.peakprom_h = conf.peakprom_h
        self.peakprom_l = conf.peakprom_l

        # initialize empty output arrays
        self.rec_out = []
        self.id1_out = []
        self.id2_out = []
        self.initiator_out = []
        self.start_out = []
        self.stop_out = []

