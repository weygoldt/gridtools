
import numpy as np
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

from ..exceptions import NotOnTimeError
from ..logger import makeLogger

logger = makeLogger(__name__)


def findClosest(array: np.ndarray, target: float) -> int:
    """
    find_closest returns an index for a value on an array that has the smallest difference to the target value. 

    Parameters
    ----------
    array : 1darray
        The array, e.g. time, ...
    target : float
        Value to be searched on the array

    Returns
    -------
    _type_
        _description_
    """

    idx = array.searchsorted(target)
    idx = np.clip(idx, 1, len(array) - 1)
    left = array[idx - 1]
    right = array[idx]
    idx -= target - left < right - target

    return idx


def findOnTime(array: np.ndarray, target: float, limit: bool = True) -> int:
    """Takes a time array and a target (e.g. timestamp) and returns an index for a value of the array that matches the target most closely.

    The time array must (a) contain unique values and (b) must be sorted from smallest to largest. If limit is True, the function checks for each target, if the difference between the target and the closest time on the time array is not larger than half of the distance between two time points at that place. When the distance exceed half the delta t, an error is returned. This also means that the time array must not nessecarily have a constant delta t.

    Parameters
    ----------
    array : array, required
        The array to search in, must be sorted.
    target : float, required
        The number that needs to be found in the array.
    limit : bool, default True
        To limit or not to limit the difference between target and array value.
    verbose : bool, default True
        To print or not to print warnings.

    Returns
    ----------
    idx : array,
        Index for the array where the closes value to target is.
    """

    # find the closest value
    idx = findClosest(array, target)

    # compute dt at this point
    found = array[idx]
    dt_target = target - found

    try:
        if target <= array[0]:
            dt_sampled = array[idx+1]-array[idx]

            if abs(array[idx]-target) > dt_sampled/2:
                if limit:
                    idx = np.nan
                    raise NotOnTimeError(
                        "The data point (target) is not on the time array!")

                logger.warning(
                    "The data point (target) is not on the time array!")

        if target > array[0] and target < array[-1]:
            if dt_target >= 0:
                dt_sampled = array[idx+1]-array[idx]
            else:
                dt_sampled = array[idx]-array[idx-1]

            if abs(array[idx]-target) > dt_sampled/2:
                if limit:
                    idx = np.nan
                    raise NotOnTimeError(
                        "The data point (target) is not on the time array!")

                logger.warning(
                    "The data point (target) is not on the time array!")

        if target >= array[-1]:
            dt_sampled = array[idx] - array[idx-1]

            if abs(array[idx]-target) > dt_sampled/2:
                if limit:
                    idx = np.nan
                    raise NotOnTimeError(
                        "The data point (target) is not on the time array!")

                logger.warning(
                    "The data point (target) is not on the time array!")

    except NotOnTimeError as error:
        logger.error("Supplied time stamp could not be found on time array!")
        raise error

    else:
        return idx


def normQ10(data, temp, normtemp, q10) -> np.ndarray:
    """
    normQ10 normalizes physiological time-series data to its Q10 value (temperature coefficient).

    Parameters
    ----------
    data : np.ndarray
        The data to be transformed
    temp : np.ndarray
        The temperature recorded along with the data
    normtemp : float
        The temperature to normalize to
    q10 : float
        The temperature coefficient of the system

    Returns
    -------
    np.ndarray[float, ...]
        The normalized data
    """
    return data + (data * (q10 % 1 * ((normtemp - temp)/10)))


def estimateMode(array: np.ndarray, cut_down: bool = False, bw_method: str = 'scott') -> float:
    """
    Estimates the "mode" of continuous data using a probability density function 
    estimated by a gaussian kernel convolution.

    The cut-down option enables mode estimation on a subset of the data, 
    making it faster. This method is highly data specific! For e.g. electrophysiological
    recordings, it might be better to constrain the "search-window" between the minimum
    and the median.

    Parameters
    ----------
    array : np.ndarray
        The data
    cut_down : bool, optional
        Enable to only use all data above the mean, by default False.
    bw_method : str, optional
        bandwidth estimation method, by default 'scott'

    Returns
    -------
    float
        The mode estimate
    """

    # make kernel density estimate
    def kde(array, cut_down=True, bw_method='scott'):

        if cut_down:
            bins, counts = np.unique(array, return_counts=True)
            f_mean = counts.mean()
            f_above_mean = bins[counts > f_mean]
            bounds = [f_above_mean.min(), f_above_mean.max()]
            array = array[np.bitwise_and(bounds[0] < array, array < bounds[1])]

        return gaussian_kde(array, bw_method=bw_method)

    # compute kde
    kernel = kde(array, cut_down=cut_down, bw_method=bw_method)

    # estimate pdf
    height = kernel.pdf(array)

    # make bounding box for mode search
    x0 = array[np.argmax(height)]
    span = array.max() - array.min()
    dx = span / 4
    bounds = np.array([[x0 - dx, x0 + dx]])
    linear_constraint = [{'type': 'ineq', 'fun': lambda x:  x - 0.5}]

    # find mode
    results = minimize(lambda x: -kernel(x)
                       [0], x0=x0, bounds=bounds, constraints=linear_constraint)

    return results.x[0]


def nanPad(array: np.ndarray, position: str = "center", padlen: int = 1):

    nans = np.full(padlen, np.nan)
    if position == "center":
        array = np.concatenate([nans, array, nans])
    if position == "left":
        array = np.concatenate([nans, array])
    if position == "right":
        array = np.concatenate([array, nans])

    return array
