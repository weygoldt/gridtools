"""Cut, align and taper templates.

Every function here returns new arrays and never writes into its inputs
(v1 tapered templates in place through views, so reused templates were
tapered again; audit item G5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TaperWindow:
    """A support window with raised-cosine ramps.

    Attributes
    ----------
    weights
        Window of length ``n_samples``: exactly 1 on the support, exactly 0
        outside support and ramps, raised cosine in between.
    support_start, support_stop
        Support boundaries actually used (fractional samples).
    truncated
        The requested support plus ramps did not fit into the valid range,
        so the support was shrunk.
    """

    weights: FloatArray
    support_start: float
    support_stop: float
    truncated: bool


def taper_window(
    n_samples: int,
    support_start: float,
    support_stop: float,
    ramp_samples: float,
    valid_start: int = 0,
    valid_stop: int | None = None,
) -> TaperWindow:
    """Build a window that is 1 on the support and tapers to 0 around it.

    Parameters
    ----------
    n_samples
        Window length.
    support_start, support_stop
        Requested support (fractional samples, ``start <= stop``).
    ramp_samples
        Length of each raised-cosine ramp; must be positive.
    valid_start, valid_stop
        Range of samples backed by data. Ramps are kept inside it; if the
        support plus ramps does not fit, the support is shrunk and
        ``truncated`` is set. ``valid_stop`` defaults to ``n_samples``.

    Returns
    -------
    TaperWindow
        The weights are 0 at sample 0 and at sample ``n_samples - 1``
        whenever ``ramp_samples >= 1`` and the support lies inside the valid
        range.
    """
    if ramp_samples <= 0:
        msg = "ramp_samples must be positive"
        raise ValueError(msg)
    if valid_stop is None:
        valid_stop = n_samples
    lo = float(max(valid_start, 0))
    hi = float(min(valid_stop, n_samples) - 1)
    start = float(support_start)
    stop = float(support_stop)
    truncated = False
    if start - ramp_samples < lo:
        start = lo + ramp_samples
        truncated = True
    if stop + ramp_samples > hi:
        stop = hi - ramp_samples
        truncated = True
    if stop < start:
        # nothing fits: collapse the support to the middle of the valid range
        start = stop = 0.5 * (lo + hi)
        truncated = True
    t = np.arange(n_samples, dtype=np.float64)
    rise = np.clip((t - (start - ramp_samples)) / ramp_samples, 0.0, 1.0)
    fall = np.clip(((stop + ramp_samples) - t) / ramp_samples, 0.0, 1.0)
    weights = 0.5 - 0.5 * np.cos(np.pi * np.minimum(rise, fall))
    # cos(0) and cos(pi) are exact, but make the plateaus exact by construction
    weights[np.minimum(rise, fall) <= 0.0] = 0.0
    weights[np.minimum(rise, fall) >= 1.0] = 1.0
    return TaperWindow(weights, start, stop, truncated)


def crop_centred(
    trace: FloatArray, centre: int, n_samples: int, fill: float
) -> tuple[FloatArray, int, int]:
    """Cut ``n_samples`` around ``centre`` into a new array, padding with ``fill``.

    Parameters
    ----------
    trace
        Contiguous source trace.
    centre
        Source index that becomes sample ``n_samples // 2``.
    n_samples
        Output length.
    fill
        Value for output samples outside the source.

    Returns
    -------
    crop : numpy.ndarray
        New array of length ``n_samples``.
    valid_start, valid_stop
        Output range backed by source samples.
    """
    out = np.full(n_samples, fill, dtype=np.float64)
    first = centre - n_samples // 2
    src_start = max(first, 0)
    src_stop = min(first + n_samples, trace.size)
    if src_stop <= src_start:
        return out, 0, 0
    out[src_start - first : src_stop - first] = trace[src_start:src_stop]
    return out, src_start - first, src_stop - first


def apply_taper_delta(delta: FloatArray, weights: FloatArray) -> FloatArray:
    """Taper a baseline-relative trace toward 0: ``weights * delta``."""
    w = np.asarray(weights, dtype=np.float64)
    out = w * np.asarray(delta, dtype=np.float64)
    out[w == 0.0] = 0.0  # avoid -0.0
    return out


def apply_taper_ratio(ratio: FloatArray, weights: FloatArray) -> FloatArray:
    """Taper a relative trace toward 1: ``1 + weights * (ratio - 1)``."""
    w = np.asarray(weights, dtype=np.float64)
    out = 1.0 + w * (np.asarray(ratio, dtype=np.float64) - 1.0)
    out[w == 0.0] = 1.0
    return out
