"""Amplitude modulation of a chirp derived from its frequency excursion.

This is the thesis' amplitude model and the simulator default: the chirp's
frequency excursion is inverted, scaled to a dip and used as the amplitude
envelope, with the dip drawn uniformly between 0 and 0.8 (EOD amplitude
between 100% and 20% at the excursion peak). The measured traces in the
library (``amp_rel`` with ``amp_usable``) remain available as an alternative.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: Range of the dip drawn per chirp by default: the EOD amplitude at the
#: excursion peak is between 100% and 20% of baseline (thesis).
DEFAULT_DIP_RANGE: tuple[float, float] = (0.0, 0.8)


def amplitude_from_frequency(
    delta_f_hz: npt.ArrayLike, dip: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Amplitude envelope from an inverted, scaled frequency excursion.

    ``amp = 1 - dip * clip(delta_f, 0, None) / max(delta_f)``, row by row.

    The undershoot (``delta_f < 0``) is clipped to 0, so the amplitude never
    exceeds 1 and the frequency undershoot after a chirp does not raise the
    EOD amplitude. Because templates are exactly 0 at both ends, the
    envelope is exactly 1 there.

    Parameters
    ----------
    delta_f_hz
        One template (n_samples,) or several (n, n_samples).
    dip
        Amplitude dip at the excursion peak, scalar or one per template, in
        [0, 1].

    Returns
    -------
    numpy.ndarray
        Relative amplitude with the shape of ``delta_f_hz``. Rows without a
        positive excursion are all ones.
    """
    delta = np.asarray(delta_f_hz, dtype=np.float64)
    d = np.asarray(dip, dtype=np.float64)
    if np.any((d < 0) | (d > 1)):
        msg = "dip must lie in [0, 1]"
        raise ValueError(msg)
    positive = np.clip(delta, 0.0, None)
    peak = positive.max(axis=-1, keepdims=True)
    shape = np.divide(positive, peak, out=np.zeros_like(positive), where=peak > 0)
    if delta.ndim == 2 and d.ndim == 1:
        d = d[:, None]
    return 1.0 - d * shape


def random_dips(
    rng: np.random.Generator,
    n: int,
    dip_range: tuple[float, float] = DEFAULT_DIP_RANGE,
) -> npt.NDArray[np.float64]:
    """Draw ``n`` dips uniformly from ``dip_range`` (default 0 to 0.8)."""
    low, high = dip_range
    if not 0.0 <= low <= high <= 1.0:
        msg = f"dip_range must satisfy 0 <= low <= high <= 1, got {dip_range}"
        raise ValueError(msg)
    return rng.uniform(low, high, size=n)
