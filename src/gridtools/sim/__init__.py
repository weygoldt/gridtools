"""Simulation of multi-electrode grid recordings of wave-type electric fish.

Building blocks (all random draws come from an explicit
``numpy.random.Generator``):

- :mod:`gridtools.sim.eod`: phase-continuous harmonic EODs, baseline drift;
- :mod:`gridtools.sim.movement`: correlated random walk with rest states;
- :mod:`gridtools.sim.field`: dipole field gains from fish to electrodes;
- :mod:`gridtools.sim.noise`: electrode noise, power-line hum, faults, clipping;
- :mod:`gridtools.sim.mix`: chunked mixing into a multi-electrode recording.

The mixing function is imported from its module
(``from gridtools.sim.mix import mix``) so that it does not shadow the
``gridtools.sim.mix`` submodule; everything else is re-exported here.
"""

from gridtools.sim.eod import (
    APTERONOTUS_LEPTORHYNCHUS,
    DriftParams,
    HarmonicJitterParams,
    Harmonics,
    SharedDriftParams,
    baseline_frequency,
    eod_phase,
    harmonic_waveform,
    jitter_harmonics,
    shared_frequency_drift,
    synthesize_eod,
)
from gridtools.sim.field import DipoleParams, GainTrack, dipole_gain, gain_track
from gridtools.sim.mix import Renderable
from gridtools.sim.movement import (
    Arena,
    FishPath,
    MovementParams,
    simulate_path,
    upsample_path,
    wrap_angle,
)
from gridtools.sim.noise import (
    ClipParams,
    ColoredNoiseParams,
    ElectrodeFaultParams,
    ElectrodeFaults,
    GainMismatchParams,
    Hum,
    HumParams,
    clip,
    colored_noise,
    electrode_faults,
    gain_mismatch,
    powerline_hum,
)

__all__ = [
    "APTERONOTUS_LEPTORHYNCHUS",
    "Arena",
    "ClipParams",
    "ColoredNoiseParams",
    "DipoleParams",
    "DriftParams",
    "ElectrodeFaultParams",
    "ElectrodeFaults",
    "FishPath",
    "GainMismatchParams",
    "GainTrack",
    "HarmonicJitterParams",
    "Harmonics",
    "Hum",
    "HumParams",
    "MovementParams",
    "Renderable",
    "SharedDriftParams",
    "baseline_frequency",
    "clip",
    "colored_noise",
    "dipole_gain",
    "electrode_faults",
    "eod_phase",
    "gain_mismatch",
    "gain_track",
    "harmonic_waveform",
    "jitter_harmonics",
    "powerline_hum",
    "shared_frequency_drift",
    "simulate_path",
    "synthesize_eod",
    "upsample_path",
    "wrap_angle",
]
