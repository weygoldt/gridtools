

import numpy as np
from scipy.optimize import shgo
from scipy.stats import gaussian_kde, iqr

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



def removeOutliers(x: np.ndarray, bar: float = 1.5, fillnan: bool =False) -> np.ndarray:

    """
    Removes outliers based on the interquartile range (i.e. datapoints that would
    be considered outliers in a regular boxplot).

    Returns
    -------
    np.ndarray
        Data without outliers
    """    

    d_iqr = iqr(x)
    d_q1 = np.percentile(x, 25)
    d_q3 = np.percentile(x, 75)
    iqr_distance = np.multiply(d_iqr, bar)
    
    upper_range = d_q3 + iqr_distance
    lower_range = d_q1 - iqr_distance

    if fillnan:
        x[x<lower_range] = np.nan
        x[x>upper_range] = np.nan
        return x
    else: 
        return x[(x>lower_range)&(x<upper_range)]


def normQ10(data: np.ndarray, temp: np.ndarray, normtemp: float, q10: float) -> np.ndarray:
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


def estimateMode(array: np.ndarray, bw_method: str = 'scott') -> float:
    """
    Estimates the "mode" of continuous data using a probability density function 
    estimated by a gaussian kernel convolution.

    Parameters
    ----------
    array : np.ndarray
        The data
    bw_method : str, optional
        bandwidth estimation method, by default 'scott'

    Returns
    -------
    float
        The mode estimate
    """

    kernel = gaussian_kde(array, bw_method=bw_method)
    bounds = np.array([[array.min(), array.max()]])
    results = shgo(lambda x: -kernel(x)[0], bounds=bounds, n=100*len(array))
    
    return results.x[0]


def nanPad(array: np.ndarray, position: str = "center", padlen: int = 1) -> np.ndarray:
    """
    nanPad adds a given number of NaNs to the start and/or end of an array.

    Parameters
    ----------
    array : np.ndarray
        The array to add NaNs
    position : str, optional
        Where to add NaNs, by default "center"
    padlen : int, optional
        Number of NaNs to add, by default 1

    Returns
    -------
    np.ndarray
        Output array
    """    

    assert position in ["center", "left", "right"] and isinstance(position, str), f"Position can only be `left`, `right` or `center`! Got {position}!"

    nans = np.full(padlen, np.nan)
    if position == "center":
        array = np.concatenate([nans, array, nans])
    if position == "left":
        array = np.concatenate([nans, array])
    if position == "right":
        array = np.concatenate([array, nans])

    return array
