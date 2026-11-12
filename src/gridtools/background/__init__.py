"""Library of real background segments for hybrid simulations.

Simulated electrode noise is Gaussian; real recordings are not (hum with
odd harmonics, clicks, non-stationary interference, other fish). As in the
hybrid data set of the author's MSc thesis, scenes can add stretches of real
recordings that contain few or no chirps, scaled to a controlled SNR.

- :mod:`gridtools.background.levels` defines background and signal levels
  and :func:`scale_for_snr`;
- :mod:`gridtools.background.select` screens candidate segments against
  chirp detections and chooses them across the recording.
"""

from gridtools.background.levels import (
    LEVEL_BAND_HZ,
    REFERENCE_BANDWIDTH_HZ,
    BackgroundLevel,
    SnrDistribution,
    background_level,
    fish_signal_level,
    fish_signal_powers,
    scale_for_snr,
    snr_db,
)
from gridtools.background.select import SelectionConfig

__all__ = [
    "LEVEL_BAND_HZ",
    "REFERENCE_BANDWIDTH_HZ",
    "BackgroundLevel",
    "SelectionConfig",
    "SnrDistribution",
    "background_level",
    "fish_signal_level",
    "fish_signal_powers",
    "scale_for_snr",
    "snr_db",
]
