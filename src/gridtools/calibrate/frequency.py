"""Baseline EOD frequency, frequency differences and drift from tracks.

All functions work on a fish's frequency trace resampled to a regular time
grid (default 1 s). The *baseline* is a running low quantile of that trace:
chirps (tens of ms, mostly invisible at the 0.8-1.6 s analysis windows of the
tracks) and rises (fast upward excursions decaying over seconds to minutes)
only push the trace up, so a low quantile follows the baseline and ignores
most of them.

Drift is described the way :mod:`gridtools.sim.eod` generates it:

- a shared *relative* drift :math:`s(t)` common to all fish (temperature);
- an individual drift :math:`d_i(t)` in Hz, a stationary Gaussian process
  with autocorrelation :math:`\\exp(-\\Delta t^2 / (4\\tau^2))`, i.e. the
  semivariogram

  .. math::

     \\gamma(\\Delta t) = c_0 + \\sigma^2\\,
         \\bigl(1 - \\exp(-\\Delta t^2 / (4\\tau^2))\\bigr),

  where :math:`c_0` (the nugget) absorbs tracking jitter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import least_squares

from gridtools.io import Track

__all__ = [
    "DeltaFStats",
    "DriftFit",
    "count_crossings",
    "delta_f_stats",
    "fit_drift_variogram",
    "resample_track",
    "running_quantile",
    "semivariogram",
]

FloatArray = npt.NDArray[np.float64]


def resample_track(
    track: Track,
    t_start_s: float,
    t_stop_s: float,
    grid_s: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Median frequency of a track in regular time bins.

    Parameters
    ----------
    track : gridtools.io.Track
        One fish's detections.
    t_start_s, t_stop_s : float
        Range of the grid in seconds.
    grid_s : float
        Bin width in seconds (default 1).

    Returns
    -------
    times_s : numpy.ndarray, shape (n,)
        Bin centres.
    freqs_hz : numpy.ndarray, shape (n,)
        Median frequency per bin; NaN in bins without detections.
    """
    n = max(1, math.floor((t_stop_s - t_start_s) / grid_s))
    centres = t_start_s + (np.arange(n) + 0.5) * grid_s
    bins = np.floor((track.times_s - t_start_s) / grid_s).astype(np.int64)
    keep = (bins >= 0) & (bins < n)
    series = pd.Series(track.freqs_hz[keep]).groupby(bins[keep]).median()
    out = np.full(n, np.nan)
    out[series.index.to_numpy()] = series.to_numpy()
    return centres.astype(np.float64), out


def running_quantile(
    values: npt.ArrayLike,
    window: int,
    quantile: float,
    min_fraction: float = 0.5,
) -> FloatArray:
    """Centred running quantile that ignores NaN.

    Parameters
    ----------
    values : array_like, shape (n,)
        Regularly sampled values (NaN = missing).
    window : int
        Window length in samples.
    quantile : float
        Quantile in [0, 1].
    min_fraction : float
        Minimum fraction of valid samples in a window; NaN otherwise.

    Returns
    -------
    numpy.ndarray, shape (n,)
    """
    series = pd.Series(np.asarray(values, dtype=np.float64))
    min_periods = max(1, math.ceil(min_fraction * window))
    rolled = series.rolling(window, center=True, min_periods=min_periods)
    return rolled.quantile(quantile).to_numpy(dtype=np.float64)


def count_crossings(difference: npt.ArrayLike, hysteresis: float) -> int:
    """Count sign changes of a signal with hysteresis.

    The state becomes +1 when the signal exceeds ``hysteresis`` and -1 when
    it falls below ``-hysteresis``; NaN and values in between keep the
    state. Every change of state is one crossing.

    Parameters
    ----------
    difference : array_like, shape (n,)
        E.g. the frequency difference of two fish.
    hysteresis : float
        Half-width of the dead band (>= 0).

    Returns
    -------
    int
    """
    d = np.asarray(difference, dtype=np.float64)
    state = np.where(d > hysteresis, 1.0, np.where(d < -hysteresis, -1.0, np.nan))
    state = state[np.isfinite(state)]
    if state.size < 2:
        return 0
    return int(np.count_nonzero(np.diff(state)))


@dataclass(frozen=True)
class DeltaFStats:
    """Distribution of the absolute frequency difference of two fish.

    Attributes
    ----------
    duration_h : float
        Time with both baselines defined, in hours.
    quantiles_hz : dict of float to float
        Quantiles of :math:`|\\Delta F|`.
    fraction_below : dict of float to float
        Fraction of that time with :math:`|\\Delta F|` below each threshold.
    n_crossings : int
        Number of sign changes of :math:`\\Delta F` (with hysteresis).
    crossings_per_h : float
    """

    duration_h: float
    quantiles_hz: dict[float, float]
    fraction_below: dict[float, float]
    n_crossings: int
    crossings_per_h: float


def delta_f_stats(
    delta_f_hz: npt.ArrayLike,
    grid_s: float,
    thresholds_hz: tuple[float, ...] = (5.0, 10.0, 20.0, 50.0),
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    hysteresis_hz: float = 1.0,
) -> DeltaFStats:
    """Summarise a regularly sampled frequency difference.

    Parameters
    ----------
    delta_f_hz : array_like, shape (n,)
        Signed difference of two baselines (NaN where undefined).
    grid_s : float
        Sampling interval in seconds.
    thresholds_hz : tuple of float
        Thresholds for :attr:`DeltaFStats.fraction_below`.
    quantiles : tuple of float
        Quantiles of :math:`|\\Delta F|` to report.
    hysteresis_hz : float
        Dead band for counting crossings (default 1 Hz, above tracking
        jitter of the baselines).

    Returns
    -------
    DeltaFStats
    """
    d = np.asarray(delta_f_hz, dtype=np.float64)
    valid = np.isfinite(d)
    absolute = np.abs(d[valid])
    duration_h = valid.sum() * grid_s / 3600.0
    if absolute.size == 0:
        nan = float("nan")
        return DeltaFStats(
            0.0,
            dict.fromkeys(quantiles, nan),
            dict.fromkeys(thresholds_hz, nan),
            0,
            nan,
        )
    q = np.quantile(absolute, quantiles)
    n_cross = count_crossings(d, hysteresis_hz)
    return DeltaFStats(
        duration_h=float(duration_h),
        quantiles_hz={qq: float(v) for qq, v in zip(quantiles, q, strict=True)},
        fraction_below={t: float(np.mean(absolute < t)) for t in thresholds_hz},
        n_crossings=n_cross,
        crossings_per_h=n_cross / duration_h if duration_h > 0 else float("nan"),
    )


def semivariogram(
    values: npt.ArrayLike,
    lags: npt.ArrayLike,
    *,
    trim: float = 0.0,
) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    """Empirical semivariogram of a regularly sampled series with gaps.

    .. math:: \\gamma(k) = \\tfrac12 \\operatorname{mean}_t (x_{t+k} - x_t)^2

    over all pairs where both values are finite. With ``trim > 0`` the
    largest ``trim`` fraction of squared increments is dropped before
    averaging, which ignores a minority of large jumps (rises, tracking
    errors) at the price of a small downward bias (about 7.5 % of the
    variance for ``trim = 0.01`` and Gaussian increments).

    Parameters
    ----------
    values : array_like, shape (n,)
    lags : array_like of int
        Lags in samples (>= 1).
    trim : float
        Fraction in [0, 0.5) of the largest squared increments to drop
        (default 0).

    Returns
    -------
    gamma : numpy.ndarray, shape (len(lags),)
        NaN for lags without pairs.
    n_pairs : numpy.ndarray of int, shape (len(lags),)
    """
    x = np.asarray(values, dtype=np.float64)
    lag_array = np.asarray(lags, dtype=np.int64)
    gamma = np.full(lag_array.size, np.nan)
    counts = np.zeros(lag_array.size, dtype=np.int64)
    for i, lag in enumerate(lag_array):
        if lag < 1 or lag >= x.size:
            continue
        diff = x[lag:] - x[:-lag]
        diff = diff[np.isfinite(diff)]
        counts[i] = diff.size
        if not diff.size:
            continue
        squared = diff * diff
        if trim > 0.0:
            squared = np.sort(squared)[: max(1, math.ceil((1.0 - trim) * squared.size))]
        gamma[i] = 0.5 * float(np.mean(squared))
    return gamma, counts


Kernel = Literal["gaussian", "exponential"]


def _correlation(
    lag: FloatArray, timescale: float | FloatArray, kernel: Kernel
) -> FloatArray:
    if kernel == "gaussian":
        return np.exp(-(lag**2) / (4.0 * timescale**2))
    return np.exp(-lag / timescale)


@dataclass(frozen=True)
class DriftFit:
    """Fit of a stationary drift model to a semivariogram.

    Attributes
    ----------
    sd : float
        Stationary standard deviation (same unit as the data).
    timescale_s : float
        Correlation time-scale :math:`\\tau` in seconds. For the
        ``"gaussian"`` kernel this is the ``timescale_s`` of
        :class:`gridtools.sim.DriftParams` (autocorrelation
        :math:`\\exp(-\\Delta t^2/(4\\tau^2))`); for ``"exponential"`` the
        autocorrelation is :math:`\\exp(-\\Delta t/\\tau)` (Ornstein-Uhlenbeck).
    nugget : float
        Variance at lag 0+ (tracking jitter), unit squared.
    rms_log_residual : float
        RMS of ``log(model / data)`` over the fitted lags (fit quality).
    kernel : {"gaussian", "exponential"}
    """

    sd: float
    timescale_s: float
    nugget: float
    rms_log_residual: float
    kernel: Kernel = "gaussian"

    def model(self, lags_s: npt.ArrayLike) -> FloatArray:
        """Semivariogram of the fitted model at ``lags_s``."""
        lag = np.asarray(lags_s, dtype=np.float64)
        return self.nugget + self.sd**2 * (
            1.0 - _correlation(lag, self.timescale_s, self.kernel)
        )


def fit_drift_variogram(
    lags_s: npt.ArrayLike,
    gamma: npt.ArrayLike,
    kernel: Kernel = "gaussian",
) -> DriftFit:
    """Fit :math:`c_0 + \\sigma^2 (1 - \\rho(\\Delta t))` in log space.

    Fitting the logarithm weighs short and long lags alike, which a
    semivariogram spanning several decades needs.

    Parameters
    ----------
    lags_s : array_like, shape (k,)
        Lags in seconds (> 0).
    gamma : array_like, shape (k,)
        Semivariogram values (> 0; NaN entries are ignored).
    kernel : {"gaussian", "exponential"}
        Autocorrelation :math:`\\rho`: Gaussian (the simulator's model,
        default) or exponential.

    Returns
    -------
    DriftFit
    """
    lag = np.asarray(lags_s, dtype=np.float64)
    g = np.asarray(gamma, dtype=np.float64)
    keep = np.isfinite(g) & (g > 0) & (lag > 0)
    lag, g = lag[keep], g[keep]
    if lag.size < 3:
        nan = float("nan")
        return DriftFit(nan, nan, nan, nan, kernel)

    def model(p: FloatArray) -> FloatArray:
        log_nugget, log_var, log_tau = p
        return np.exp(log_nugget) + np.exp(log_var) * (
            1.0 - _correlation(lag, np.exp(log_tau), kernel)
        )

    def residual(p: FloatArray) -> FloatArray:
        return np.log(model(p)) - np.log(g)

    start = np.array(
        [math.log(max(g[0] * 0.5, 1e-12)), math.log(g.max()), math.log(np.median(lag))]
    )
    bounds = (
        [math.log(g.min() * 1e-4), math.log(g.max() * 1e-3), math.log(lag.min() / 4)],
        [math.log(g.max()), math.log(g.max() * 1e3), math.log(lag.max() * 20)],
    )
    start = np.clip(start, np.array(bounds[0]) + 1e-9, np.array(bounds[1]) - 1e-9)
    result = least_squares(residual, start, bounds=bounds)
    log_nugget, log_var, log_tau = result.x
    rms = float(np.sqrt(np.mean(result.fun**2)))
    return DriftFit(
        sd=float(np.exp(0.5 * log_var)),
        timescale_s=float(np.exp(log_tau)),
        nugget=float(np.exp(log_nugget)),
        rms_log_residual=rms,
        kernel=kernel,
    )
