"""The MSc thesis' baseline smoothing, ported for comparison.

The thesis (Methods, "Hybrid simulations"; code
``chirp_baseline_smoothing_and_histogram_equalization.py``) made the
baseline around each chirp smooth with a cross-fade:

1. Baseline = median of the samples within ``[min(f), |min(f)|]``.
2. Start: descend the left flank from the first half-height crossing
   until the trace falls below the baseline or the slope of a Gaussian
   smoothed copy (sigma = 50 samples) stops rising.
3. Stop: descend the right flank from the second half-height crossing until
   the slope's sign has flipped twice (past the undershoot).
4. Set the trace to the baseline outside [start, stop], smooth that with a
   Gaussian (sigma = 60 samples), and blend original and smoothed with
   linear weights over 120 samples inside both ends of [start, stop].
5. Subtract the baseline.

The library keeps its own taper (:mod:`gridtools.templates.taper`); these
functions exist so the report can compare the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ThesisSupport:
    """Chirp start and stop as found by the thesis heuristic.

    Attributes
    ----------
    start, stop
        Sample indices of chirp start and stop (stop includes the undershoot).
    baseline
        Baseline value (median of samples within ``[min, |min|]``).
    """

    start: int
    stop: int
    baseline: float


def thesis_support(trace: FloatArray) -> ThesisSupport | None:
    """Find chirp start and stop with the thesis heuristic.

    Parameters
    ----------
    trace
        Contiguous frequency trace relative to its first sample (as stored),
        so the baseline lies near 0; the half-height test assumes this.

    Returns
    -------
    ThesisSupport or None
        None when the trace does not cross half its maximum exactly twice
        (the thesis discarded those chirps).
    """
    chirp = np.asarray(trace, dtype=np.float64)
    square = (chirp > np.max(chirp) / 2).astype(np.int8)
    transitions = np.flatnonzero(np.diff(square) != 0)
    if transitions.size != 2:
        return None
    cmin = float(np.min(chirp))
    inside = chirp[chirp <= -cmin]
    baseline = float(np.median(inside)) if inside.size else float(np.median(chirp))
    slope = gaussian_filter1d(chirp, 50)
    left = int(transitions[0])
    while left > 0:
        if chirp[left] < baseline:
            break
        if np.sign(slope[left] - slope[left - 1]) != 1:
            break
        left -= 1
    right = int(transitions[1])
    switches = 0
    current = np.sign(slope[right] - slope[right + 1])
    while right < chirp.size - 1:
        sign = np.sign(slope[right] - slope[right + 1])
        if sign != current:
            switches += 1
            current = sign
        if switches > 1:
            break
        right += 1
    return ThesisSupport(left, right, baseline)


def thesis_crossfade(
    trace: FloatArray, sigma: float = 60.0, blend: int = 120
) -> FloatArray | None:
    """Apply the thesis baseline cross-fade and subtract the baseline.

    Parameters
    ----------
    trace
        Contiguous frequency trace relative to its first sample.
    sigma
        Gaussian smoothing of the flattened trace, in samples.
    blend
        Length of the linear cross-fades inside both ends of the chirp.

    Returns
    -------
    numpy.ndarray or None
        Baseline-relative trace, or None where :func:`thesis_support` fails.
    """
    support = thesis_support(trace)
    if support is None:
        return None
    chirp = np.array(trace, dtype=np.float64, copy=True)
    left, right, baseline = support.start, support.stop, support.baseline
    chirp[:left] = baseline
    chirp[right:] = baseline
    smoothed = gaussian_filter1d(chirp, sigma)
    weights = np.ones_like(chirp)
    ramp = np.linspace(0.0, 1.0, blend)
    n_left = min(blend, max(chirp.size - left, 0))
    weights[left : left + n_left] = ramp[:n_left]
    n_right = min(blend, right)
    weights[right - n_right : right] = ramp[::-1][blend - n_right :]
    weights[:left] = 0.0
    weights[right:] = 0.0
    return weights * chirp + (1.0 - weights) * smoothed - baseline
