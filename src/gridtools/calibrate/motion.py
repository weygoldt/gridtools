"""Movement proxies of simulated fish, comparable to the real ones.

Real recordings give no positions, only per-electrode powers. To check a
movement model against them, simulate paths with
:func:`gridtools.sim.movement.simulate_path`, turn them into ``sign_v``-like
powers with the fitted field model and noise, and compute the same
order-free statistics as for real fish (switches of the strongest electrode,
dwell times, still fraction).
"""

from __future__ import annotations

import numpy as np

from gridtools.calibrate.field import GridLayout
from gridtools.calibrate.profiles import DwellStats, dwell_stats, strongest_electrode
from gridtools.sim.field import dipole_gain
from gridtools.sim.movement import Arena, MovementParams, simulate_path

__all__ = ["layout_arena", "simulated_dwell_stats"]


def layout_arena(layout: GridLayout, margin_rel: float) -> Arena:
    """Arena spanning the electrodes plus ``margin_rel`` spacings on each side."""
    xy = layout.positions_m()
    pad = margin_rel * layout.spacing_m
    lo = xy.min(axis=0) - pad
    hi = xy.max(axis=0) + pad
    return Arena(float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))


def simulated_dwell_stats(
    rng: np.random.Generator,
    layout: GridLayout,
    movement: MovementParams,
    *,
    distance_exponent: float,
    regulariser_rel: float,
    margin_rel: float,
    duration_s: float,
    frame_dt_s: float,
    median_peak_snr_db: float,
    hysteresis_db: float = 3.0,
    still_min_s: float = 60.0,
) -> DwellStats:
    """Strongest-electrode statistics of one simulated fish.

    Parameters
    ----------
    rng : numpy.random.Generator
    layout : GridLayout
    movement : MovementParams
    distance_exponent, regulariser_rel, margin_rel : float
        Field model and arena (see :mod:`gridtools.calibrate.field`).
    duration_s : float
        Simulated duration.
    frame_dt_s : float
        Frame interval of the tracks to mimic.
    median_peak_snr_db : float
        The signal scale is set so that the median strongest-electrode SNR
        equals this value.
    hysteresis_db, still_min_s : float
        As for the real data.

    Returns
    -------
    DwellStats
    """
    arena = layout_arena(layout, margin_rel)
    path = simulate_path(rng, duration_s, arena, movement)
    times = np.arange(0.0, duration_s, frame_dt_s)
    x = np.interp(times, path.times_s, path.xy_m[:, 0])
    y = np.interp(times, path.times_s, path.xy_m[:, 1])
    heading = np.interp(times, path.times_s, np.unwrap(path.heading_rad))
    gain = dipole_gain(
        np.column_stack([x, y]),
        heading,
        layout.positions_m(),
        fish_depth_m=0.0,
        electrode_depth_m=0.0,
        min_distance_m=regulariser_rel * layout.spacing_m,
        distance_exponent=distance_exponent,
    ).astype(np.float64)
    power = gain * gain
    scale = 10.0 ** (median_peak_snr_db / 10.0) / np.median(power.max(axis=1))
    noise = (
        rng.standard_normal(power.shape) + 1j * rng.standard_normal(power.shape)
    ) / np.sqrt(2.0)
    db = 10.0 * np.log10(np.abs(np.sqrt(power * scale) + noise) ** 2)
    if layout.n_random_missing:
        drop = rng.choice(layout.n_simulated, layout.n_random_missing, replace=False)
        db = np.delete(db, drop, axis=1)
    sequence = strongest_electrode(db, hysteresis_db)
    return dwell_stats(times, sequence, max_gap_s=2.0, still_min_s=still_min_s)
