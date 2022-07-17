import datetime
import os
from itertools import combinations, pairwise
from math import atan2, copysign, degrees
from operator import itemgetter
from string import Formatter

import matplotlib.pyplot as plt
import numpy as np
import yaml
from IPython import embed
from scipy import ndimage
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.spatial import cKDTree
from sklearn.metrics import auc
from sklearn.neighbors import KernelDensity

from .termcolors import tc

verbose = True  # to set the functions to verbose


"""Cool plot functions"""


def get_ylims(y1, y2, t, tstart, tstop, padding=0.2):

    # make indices
    indices = np.arange(len(t))
    start, stop = indices[t == tstart][0], indices[t == tstop][0]

    # make vectors in plot range
    y1_plt, y2_plt = y1[start:stop], y2[start:stop]

    # concatenate
    y_plt = np.concatenate([y1_plt, y2_plt]).ravel()

    # get total amplitude
    ampl = np.max(y_plt) - np.min(y_plt)

    # make limits
    lower = np.min(y_plt) - ampl * padding
    upper = np.max(y_plt) + ampl * padding

    ylims = [lower, upper]

    return ylims


def kde1d(y, bandwidth, xlims="auto", resolution=500, kernel="gaussian"):

    if xlims == "auto":
        x = np.linspace(np.min(y), np.max(y), resolution)
    else:
        try:
            x = np.linspace(xlims[0], xlims[1], resolution)
        except ValueError:
            print("Invalid argument for 'xlims'. Must be a list/array or 'auto'.")
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(y[:, None])
    log_dens = kde.score_samples(x[:, None])

    print(
        "computed AreaUnderCurve (AUC) of KDE using sklearn.metrics.auc: {}".format(
            auc(x, np.exp(log_dens))
        )
    )
    return x, np.exp(log_dens)


def kde1d_mode(y, bandwidth, resolution):
    kde = kde1d(y, bandwidth, xlims="auto", resolution=resolution)
    mode = kde[0][kde[1] == np.max(kde[1])][0]
    return mode


def kde2d(
    x, y, bandwidth, xbins=100j, ybins=100j, dims=None, kernel="gaussian", **kwargs
):
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    if dims is None:
        xx, yy = np.mgrid[x.min() : x.max() : xbins, y.min() : y.max() : ybins]
    else:
        xx, yy = np.mgrid[dims[0] : dims[1] : xbins, dims[2] : dims[3] : ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train = np.vstack([y, x]).T

    kde_skl = KernelDensity(kernel=kernel, bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)


def kde2d_bwest(x, y, ax, bw_method="silverman"):
    """Computes and plots a 2d KDE with bandwidth approximation using silverman or scott method"""
    xy = np.vstack([x, y])

    d = xy.shape[0]
    n = xy.shape[1]

    if bw_method == "silverman":
        bw = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))  # silverman

    if bw_method == "scott":
        bw = n ** (-1.0 / (d + 4))  # scott

    print("bw: {}".format(bw))

    kde = KernelDensity(
        bandwidth=bw, metric="euclidean", kernel="gaussian", algorithm="ball_tree"
    )
    kde.fit(xy.T)

    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    Z = np.reshape(np.exp(kde.score_samples(positions.T)), X.shape)

    ax.contourf(np.rot90(Z), cmap=cmocean.cm.haline, extent=[xmin, xmax, ymin, ymax])


def gaussianhist2d(x, y, extent, sigma, bins=1000):
    """Makes a gaussian smoothed 2d histogram from scatterplot data"""

    heatmap, _, _ = np.histogram2d(x, y, bins=bins, range=extent)
    heatmap = ndimage.gaussian_filter(heatmap, sigma=sigma)
    return np.rot90(heatmap)


def knn_2d_dens(xv, yv, extent, resolution, n_neighbours, dim=2):
    """Uses a cKDTree algorithm to compute the k nearest neighbours and plot them as a 2D density image."""

    def data_coord2view_coord(p, resolution, pmin, pmax):
        dp = pmax - pmin
        dv = (p - pmin) / dp * resolution
        return dv

    xv = data_coord2view_coord(xv, resolution, extent.flatten()[0], extent.flatten()[1])
    yv = data_coord2view_coord(yv, resolution, extent.flatten()[2], extent.flatten()[3])

    # Create the tree
    tree = cKDTree(np.array([xv, yv]).T)

    # Find the closest nnmax-1 neighbors (first entry is the point itself)
    grid = np.mgrid[0:resolution, 0:resolution].T.reshape(resolution**2, dim)
    dists = tree.query(grid, n_neighbours)

    # Inverse of the sum of distances to each grid point.
    inv_sum_dists = 1.0 / dists[0].sum(1)

    # Reshape
    im = inv_sum_dists.reshape(resolution, resolution)
    return im


def knn_2d_dens_slow(xs, ys, reso, n_neighbours):
    """A slow implementation of a knn 2d density plot without using special packages."""

    def data_coord2view_coord(p, vlen, pmin, pmax):
        dp = pmax - pmin
        dv = (p - pmin) / dp * vlen
        return dv

    im = np.zeros([reso, reso])
    extent = [np.min(xs), np.max(xs), np.min(ys), np.max(ys)]

    xv = data_coord2view_coord(xs, reso, extent[0], extent[1])
    yv = data_coord2view_coord(ys, reso, extent[2], extent[3])
    for x in range(reso):
        for y in range(reso):
            xp = xv - x
            yp = yv - y

            d = np.sqrt(xp**2 + yp**2)

            im[y][x] = 1 / np.sum(
                d[np.argpartition(d.ravel(), n_neighbours)[:n_neighbours]]
            )

    return im, extent


def lims(track1, track2):
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
                track1,
                track2,
            ]
        )
        .ravel()
        .tolist()
    )
    lower = np.min(allfunds_tmp)
    upper = np.max(allfunds_tmp)
    return lower, upper


"""Array/list utilities"""


def nanpad(array, position="center", padlen=1):
    nans = np.full(padlen, np.nan)
    if position == "center":
        array = np.concatenate([nans, array, nans])
    if position == "left":
        array = np.concatenate([nans, array])
    if position == "right":
        array = np.concatenate([array, nans])
    return array


def unique_combinations1d(array):
    """Combine elements in a 1d array uniquely."""
    return [i for i in combinations(array, 2)]


def unique_combinations2d(list1, list2):
    """Combine elments of two lists uniquely."""
    return [(x, y) for x in list1 for y in list2]


def find_closest(array, target):
    """Takes an array and a target and returns an index for a value of the array that matches the target most closely.

    Could also work with multiple targets and may not work for unsorted arrays, i.e. where a value occurs multiple times. Primarily used for time vectors.

    Parameters
    ----------
    array : array, required
        The array to search in.
    target : float, required
        The number that needs to be found in the array.

    Returns
    ----------
    idx : array,
        Index for the array where the closes value to target is.
    """
    idx = array.searchsorted(target)
    idx = np.clip(idx, 1, len(array) - 1)
    left = array[idx - 1]
    right = array[idx]
    idx -= target - left < right - target
    return idx


def points2ranges(timestamps, time, radius):
    """Genereates time ranges of a specified window size from point events.
    Window is sized by (radius * 2)+1.

    Args:
        timestamps (1d array of ints): Timestamps of point events.
        time (1d array of ints): Time in int, use index vector for time.
        radius (int): Radius determining window size.

    Returns:
        list of arrays: List of numpy arrays of ranges. The middle element of a range is the point event.
    """
    ranges = []
    for timestamp in timestamps:
        index = np.arange(len(time))[timestamp]

        # dynamically adjust radius for events at the beginning of time
        if index < radius:
            start = 0
        else:
            start = index - radius

        # dynmaically adjust radius for events at the end of time
        if (index + radius) > len(time):
            stop = np.arange(len(time))[-1]
        else:
            stop = index + radius

        # make area with start and stop, append to lists
        area = np.arange(start, stop + 1, dtype=int)
        ranges.append(area)

    return ranges


def combine_cooccurences(peaks1, peaks2, time, radius, merge=True, crop=False):
    """Finds ranges where some peaks2 fall within a given range of another list of peaks1. Useful to find cooccuring events on two seperate filter scales of the same data. I use it to detect peaks of local covariances on two time scales (i.e. where fast modulations and slow modulations between two datasets are similar).

    (1) Makes ranges around supplied peaks of sizes (2*radius)+1.
    (2) Selects all highpass peaks in ranges of lowpass peaks.
    -- if merge is True:
        (3) Merges ranges of selected highpass peaks with the lowpass peak range.
    -- if merge is False:
        (3) Only uses range of low pass peaks in which high pass peaks fall within
    (4) Checks if resulting ranges overlap.
    (5) Combines overlapping ranges and returns resulting ranges.
    -- if crop is True: Crops half of the specified radius from the onset and offset of the events.

    Args:
        peaks2 (int array): Peak indices of highpass filtered data.
        peaks1 (int array): Peak indices of lowpass filtered data.
        time (int array): Integer time, e.g. index vector for time.
        radius (int): Range radius around peak.
        merge (bool): Whether to merge or not to merge ranges of slow ts and fast ts peaks.
        crop (bool): Whether to crop or not to crop some of the onset and offset of the event. Currently set to crop 3/4 of the radius from start and stop.
    """

    # combine large with small ranges where small peaks are in large ranges
    # i.e. where covarying small time-scale modulations cooccur with large time
    # scale covariations.

    # make ranges around peaks
    ranges_large = np.array(points2ranges(peaks1, time, radius), dtype=object)
    ranges_small = np.array(points2ranges(peaks2, time, radius), dtype=object)

    ranges_new = []  # save new ranges here

    if merge is True:
        for ranl in ranges_large:  # iterate over large-ts ranges
            range_new = np.array([ranl], dtype=int)  # make an array of large range
            for peak in peaks2:  # iterate over small-ts peaks
                if peak in ranl:  # check if peak is in range
                    rans = ranges_small[peaks2 == peak][
                        0
                    ]  # if true get range of peak and
                    range_new = np.append(range_new, rans)  # append to new ranges

            if len(range_new) > len(
                ranl
            ):  # only save new range when new range is large than large-ts range
                ran_sorted = np.sort(np.unique(range_new))

                ranges_new.append(ran_sorted.tolist())
    elif merge is False:
        for ranl in ranges_large:  # iterate over large-ts ranges
            range_new = np.array([ranl], dtype=int)  # make an array of large range
            for peak in peaks2:  # iterate over small-ts peaks
                if peak in ranl:  # check if peak is in range
                    ranges_new.append(ranl.tolist())

    # sort list of arrays with ranges by first value of each range
    ranges_new_sort = sorted(ranges_new, key=itemgetter(0))
    cooccurrences = list_magic(ranges_new_sort)

    # get peaks where cooccurrences are
    peakbool_small = np.zeros(len(time), dtype=bool)
    peakbool_small[peaks2] = True
    peakbool_large = np.zeros(len(time), dtype=bool)
    peakbool_large[peaks1] = True

    # empty list to collect peak positions
    coocc_peaks_small = []
    coocc_peaks_large = []

    for coocc_range in cooccurrences:

        # make boolean for range
        rangebool = np.zeros(len(time), dtype=bool)
        rangebool[coocc_range[0] : coocc_range[-1]] = True

        # get peak indices
        coocc_peaks_small.append(np.where(rangebool & peakbool_small)[0].tolist())
        coocc_peaks_large.append(np.where(rangebool & peakbool_large)[0].tolist())

    if crop is True:
        clipped_cooccurrences = []
        for event in cooccurrences:
            event_clipped = event[
                int(0 + (radius - (radius // 4))) : int(
                    (len(event) - 1) - (radius - (radius // 4))
                )
            ]
            if len(event_clipped) > 0:
                clipped_cooccurrences.append(event_clipped)

        if len(clipped_cooccurrences) > 0:
            cooccurrences = clipped_cooccurrences

    return cooccurrences, coocc_peaks_large, coocc_peaks_small


def axmax_2d(image, lags, dim):
    """Returns maxima across one dimension of a 2d-array (e.g. an image).
    If given the 'x' argument to the dimesion, len(y) values are returned,
    each with the maximum of all values across the x axis for the respective index
    of the y axis point. In other words, the maximum values off datapoints across
    the x-axis. For time series data, where x is the time axis, the y argument results
    in the maxima across all y values.

    Args:
        image (2d array): 2d array, e.g. image, rolling window crosscorrelation, etc.
        lags (1d array): values for the axis of interest, e.g. time, crosscorrelation lags, etc.

    Returns:
        maxs (1d array): maxima across given dimension
        lags_maxs (1d array): values of the supplied lags values for the maxima
    """

    if dim == "y":
        maxs = np.zeros(len(image[0, :]))
    elif dim == "x":
        maxs = np.zeros(len(image[:, 0]))
    else:
        raise Exception(
            "Please specify a correct dimesion when computing maxima across one dimension of a 2d array!"
        )

    lags_maxs = np.zeros(len(image[0, :]))

    for index in range(len(maxs)):
        maxs[index] = np.max(image[:, index])
        lags_maxs[index] = lags[image[:, index] == maxs[index]]

    return maxs, lags_maxs


def array2ranges(array):
    """Finds continuous sequences of integers in array and returns ranges
    from start to stop of continuous sequences.

    Args:
        array (1d array of ints): Array that contains continuous ranges.

    Returns:
        List of touples: Containing start and stop value.
    """
    array = sorted(set(array))
    gaps = [[s, e] for s, e in zip(array, array[1:]) if s + 1 < e]
    edges = iter(array[:1] + sum(gaps, []) + array[-1:])

    return list(zip(edges, edges))


def clean_ranges(ranges):
    """Takes a sorted list of sorted lists that contain ranges and returns a list of ranges with no overlap. I AM NOT SURE IF THIS ACTUALLY WORKS!

    Args:
        ranges (int list): Sorted list of sorted lists with continuous integers.
    """
    cleaned_ranges = []
    if len(ranges) > 1:
        for a, b in pairwise(ranges):
            if len(set(a).intersection(set(b))) > 0:
                ab = sorted(list(set(a).union(set(b))))
                cleaned_ranges.append(ab)
            else:
                if b == ranges[-1]:
                    cleaned_ranges.append(b)
                    if len(set(a).intersection(set(ranges[-3]))) > 0:
                        print("???")
                else:
                    cleaned_ranges.append(a)
    else:
        cleaned_ranges = ranges

    return cleaned_ranges


def list_magic(input):
    """This is adapted from stackoverflow question https://stackoverflow.com/questions/55380743/finding-overlapping-lists-in-a-list-of-lists"""

    def Find(id, P):
        if P[id] < 0:
            return id
        P[id] = Find(P[id], P)
        return P[id]

    def Union(id1, id2, p):
        id1 = Find(id1, P)
        id2 = Find(id2, P)
        if id1 != id2:
            P[id2] = id1

    P = {}

    for sublist in input:
        for item in sublist:
            P[item] = -1

    for sublist in input:
        for i in range(1, len(sublist)):
            Union(sublist[i - 1], sublist[i], P)

    ans = {}
    for sublist in input:
        for item in sublist:
            if Find(item, P) not in ans:
                ans[Find(item, P)] = []
            ans[Find(item, P)].append(item)

    ans = [set(x) for x in ans.values()]

    lists = []
    for sets in ans:
        lists.append(sorted(list(sets)))

    return lists


"""Small tools"""


def load_ymlconf(path):
    """Small tool to open yaml configuration files."""
    with open(path) as file:
        try:
            conf = yaml.safe_load(file)
            return conf

        except yaml.YAMLError as e:
            print(e)


def is_odd(number):
    """Checks if an integer is odd."""
    if number % 2 == 0:
        odd = False
    else:
        odd = True

    return odd


def get_sign(x):
    """Takes int or float input, returns +1 if positive, -1 if negative and 0 if 0."""
    if x == 0:
        sign = 0
    else:
        sign = copysign(1, x)
    return int(sign)


def get_midpoint(x1, x2):
    """Estimate closest integer midpoint between two index values."""
    return int(round(x1 + x2) / 2)


def strfdelta(tdelta, fmt="{D:02}d {H:02}h {M:02}m {S:02}s", inputtype="timedelta"):
    """Convert a datetime.timedelta object or a regular number to a custom-
    formatted string, just like the stftime() method does for datetime.datetime
    objects.

    The fmt argument allows custom formatting to be specified.  Fields can
    include seconds, minutes, hours, days, and weeks.  Each field is optional.

    Some examples:
        '{D:02}d {H:02}h {M:02}m {S:02}s' --> '05d 08h 04m 02s' (default)
        '{W}w {D}d {H}:{M:02}:{S:02}'     --> '4w 5d 8:04:02'
        '{D:2}d {H:2}:{M:02}:{S:02}'      --> ' 5d  8:04:02'
        '{H}h {S}s'                       --> '72h 800s'

    The inputtype argument allows tdelta to be a regular number instead of the
    default, which is a datetime.timedelta object.  Valid inputtype strings:
        's', 'seconds',
        'm', 'minutes',
        'h', 'hours',
        'd', 'days',
        'w', 'weeks'
    """

    # Convert tdelta to integer seconds.
    if inputtype == "timedelta":
        remainder = int(tdelta.total_seconds())
    elif inputtype in ["s", "seconds"]:
        remainder = int(tdelta)
    elif inputtype in ["m", "minutes"]:
        remainder = int(tdelta) * 60
    elif inputtype in ["h", "hours"]:
        remainder = int(tdelta) * 3600
    elif inputtype in ["d", "days"]:
        remainder = int(tdelta) * 86400
    elif inputtype in ["w", "weeks"]:
        remainder = int(tdelta) * 604800

    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ("W", "D", "H", "M", "S")
    constants = {"W": 604800, "D": 86400, "H": 3600, "M": 60, "S": 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)


def dir2datetime(folder):
    rec_year, rec_month, rec_day, rec_time = os.path.split(os.path.split(folder)[-1])[
        -1
    ].split("-")
    rec_year = int(rec_year)
    rec_month = int(rec_month)
    rec_day = int(rec_day)
    try:
        rec_time = [int(rec_time.split("_")[0]), int(rec_time.split("_")[1]), 0]
    except:
        rec_time = [int(rec_time.split(":")[0]), int(rec_time.split(":")[1]), 0]

    rec_datetime = datetime.datetime(
        year=rec_year,
        month=rec_month,
        day=rec_day,
        hour=rec_time[0],
        minute=rec_time[1],
        second=rec_time[2],
    )

    return rec_datetime


def simple_outputdir(path):
    if os.path.isdir(path) == False:
        os.mkdir(path)
        print("new output directory created")
    else:
        print("using existing output directory")

    return path


"""Signal processing"""


def lowpass_filter(data, rate, cutoff, order=2):
    sos = butter(order, cutoff, btype="low", fs=rate, output="sos")
    y = sosfiltfilt(sos, data)
    return y


"""2D position processing functions"""


def velocity2d(t, x, y):
    """
    Compute the velocity of an object in 2D space.

    Parameters
    ----------
    t : array-like
        time axis for coordinates
    x : array-like
        x coordinates of object
    y : array-like
        y coordinates of object

    Returns
    -------
    velocity : numpy array
        velocity of object with same dimension as input (padded with nans).
    """

    # delta t
    dt = np.array([x - x0 for x0, x in zip(t, t[2:])])

    # delta d x and y
    dx = np.array([(x2 - x1) + (x1 - x0) for x0, x1, x2 in zip(x, x[1:], x[2:])])
    dy = np.array([(x2 - x1) + (x1 - x0) for x0, x1, x2 in zip(y, y[1:], y[2:])])

    # delta d tot. (pythagoras)
    dd = np.sqrt(dx**2 + dy**2)

    # velocity & correcsponding time
    v = dd / dt

    # pad to len of original time array with nans
    v = nanpad(v, position="center", padlen=1)

    return v


def aim_index(c1, c2):
    """
    And index to quanify the aim of the trajectory of one fish towards the position of another.

    A value between 0 and 1 indicating whether fish 1 swims into the opposite direciton
    of fish 2 (aim index = 0) or into the the direction where fish 2 is located (aim index = 1).

    Parameters
    ----------
    c1 : array-like
        x and y coordinates of fish 1
    c2 : array-like
        x and y coordinates of fish 2

    Returns
    -------
    aims : numpy array
        aim index of fish 1 towards fish 2
    """

    # fish 1 current position coordinates
    x1 = c1[0][:-1]
    y1 = c1[1][:-1]

    # fish 1 trajectory coordinates
    x1t = c1[0][1:]
    y1t = c1[1][1:]

    # fish 2 coordinates
    x2 = c2[0][:-1]
    y2 = c2[1][:-1]

    aims = []
    relangles = []

    for i in range(len(x1)):

        # compute trajectory angle
        adj = x1t[i] - x1[i]  # adjacent side of triangle
        opp = y1t[i] - y1[i]  # opposite side of triangle
        a = atan2(opp, adj)  # angle between them in radians

        # compute trajectory angle
        adj = x2[i] - x1[i]  # adjacent side of triangle
        opp = y2[i] - y1[i]  # opposite side of triangle
        b = atan2(opp, adj)  # angle between them in radians

        # first transform two radians to keep them positive
        a = 2 * np.pi + a if a < 0 else a
        b = 2 * np.pi + b if b < 0 else b

        # get the absolute of the relative angle
        # (because its easier and for now we dont care about left or right)
        r = np.abs(b - a)

        # transform r to degrees
        rd = degrees(r)

        # norm right side of unit circle to 0
        if rd <= 180:
            aim = 1 - rd / 180

        # norm left side of unit circle to 0
        elif rd > 180:
            aim = 1 - (180 - (rd - 180)) / 180

        aims.append(aim)
        relangles.append(rd)

    aims = nanpad(aims, "right", 1)
    relangles = nanpad(relangles, "right", 1)
    aims = np.asarray(aims)
    relangles = np.asarray(relangles)

    return aims, relangles


def veloc(times, dist):
    """
    Compute velocity as distance over time.

    Parameters
    ----------
    times : array-like
        array with time stamps, e.g. in seconds
    dist : array-like
        array with distances

    Returns
    -------
    velocity: numpy array
        velocities at time points
    """

    # make times
    dt = np.array([x - x0 for x0, x in zip(times, times[2:])])

    # compute distances
    dx = np.array(
        [(x2 - x1) + (x1 - x0) for x0, x1, x2 in zip(dist, dist[1:], dist[2:])]
    )

    # compute velocity, i.e. distance over time
    v = dx / dt

    # add nans to make same dimension as input
    v = nanpad(v, position="center", padlen=1)

    return v


def find_interactions(dyad, start, stop, maxd, peakprom, plot=False):
    """
    Uses fish trajectory, velocity and proximity to find interactions between two individuals.

    This is experimental and must be rigorously tested before being used.
    This function calculates the product of:

        - aim index:
            The "directedness" of movement of a fish. Becomes 1 if fish
            1 swims directly towards fish 2 and 0 if it swims into the
            opposite direction.

        - velocity:
            Simply the velocity of a fish at a given point in time.

        - proximity index:
            Becomes 1 if fish have a distance of 0 cm and decreases with increasing
            distance to 0. If maxd is smaller than the maximum possible distance of
            two fish on the grid, this implements a threshold. I.e. all instances where
            fish distances surpass maxd become 0, the rest are scaled between 0 and 1.

    This means that the resulting function increases, as velocities increase,
    heading precision towards the conspecific increases and distance decreases.

    By incorporating both velocity and heading direction, interactions can be
    computed for each individual seperately, instead of just thresholding the distance.

    If the correct parameters, e.g. from competition experiments are supplied,
    this could be used to detect attack events in the grid recording. I set the
    peak prominence lower to detect "social interactions" in general.


    Parameters
    ----------
    dyad : dyad object
        A class instance of the gridtools.Dyad class.
    start : float
        Start time stamp on the time axis.
    stop : float
        Stop time stamp on the time axis.
    maxd : float or int
        Maximum distance for distance index.
    peakprom : float or int
        Peak prominence of the peak detection.
    plot : bool, optional
        To plot or not to plot the parameters during calculation, by default False.

    Returns
    -------
    peaks1, peaks2 : tuple(array, array)
        The peaks, i.e. interactions for each individual.
    """

    # make arrays for aim index
    c1 = [dyad.xpos_smth_id1[start:stop], dyad.ypos_smth_id1[start:stop]]
    c2 = [dyad.xpos_smth_id2[start:stop], dyad.ypos_smth_id2[start:stop]]
    t = dyad.times[start:stop]

    # extract aim index
    aims1, relangles1 = aim_index(c1, c2)
    aims2, relangles2 = aim_index(c2, c1)

    # extract distance index (0 at max dist, 1 at min dist)
    dist_index = []
    for dpos in dyad.dpos[start:stop]:
        if dpos < maxd:
            dist_index.append(1 - dpos / maxd)
        else:
            dist_index.append(0)

    # extract velocities
    v1 = velocity2d(t, c1[0], c1[1])
    v2 = velocity2d(t, c2[0], c2[1])

    # compute relative velocities
    vr = veloc(dyad.times[start:stop], dyad.dpos[start:stop])

    interact_index1 = v1 * aims1 * dist_index
    interact_index2 = v2 * aims2 * dist_index

    # find peaks, i.e. attacks
    peaks1 = find_peaks(interact_index1, prominence=peakprom)[0]
    peaks2 = find_peaks(interact_index2, prominence=peakprom)[0]

    if plot:
        fig, ax = plt.subplots(6, 1, figsize=(6, 12), sharex=True)

        ax[0].set_title("Fundamental frequencies")
        ax[0].plot(dyad.times[start:stop], dyad.fund_id1[start:stop])
        ax[0].plot(dyad.times[start:stop], dyad.fund_id2[start:stop])

        ax[1].set_title("Velocities of both fish")
        ax[1].plot(t, v1)
        ax[1].plot(t, v2)

        ax[2].set_title("Aim index of both fish, should have peaks where attack is")
        ax[2].plot(t, aims1)
        ax[2].plot(t, aims2)

        ax[3].set_title("Distance index")
        ax[3].plot(t, dist_index)

        ax[4].set_title("Attack index, i.e. v * aim_index * dist_index")
        ax[4].plot(t, interact_index1)
        ax[4].plot(t, interact_index2)

        peakbool1 = np.full(len(interact_index1), False, dtype=bool)
        peakbool2 = np.full(len(interact_index2), False, dtype=bool)
        peakbool1[peaks1] = True
        peakbool2[peaks2] = True

        ax[5].set_title("Attack index masked for approach phases only")
        ax[5].plot(t, interact_index1)
        ax[5].plot(t, interact_index2)
        ax[5].plot(t[peakbool1], interact_index1[peakbool1], ".")
        ax[5].plot(t[peakbool2], interact_index2[peakbool2], ".")

    return peaks1, peaks2
