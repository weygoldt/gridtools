"""Simulation of multi-electrode grid recordings of wave-type electric fish.

Building blocks (all random draws come from an explicit
``numpy.random.Generator``):

- :mod:`gridtools.sim.eod`: phase-continuous harmonic EODs, baseline drift;
- :mod:`gridtools.sim.movement`: correlated random walk with rest states;
- :mod:`gridtools.sim.field`: dipole field gains from fish to electrodes;
- :mod:`gridtools.sim.noise`: electrode noise, power-line hum, faults, clipping;
- :mod:`gridtools.sim.mix`: chunked mixing into a multi-electrode recording.
"""
