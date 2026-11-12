"""Fit the dipole falloff to real per-electrode power profiles.

Simulation-based inference that needs no absolute positions and no channel
order:

1. Draw many fish positions (uniform over the electrode grid plus a margin)
   and headings (uniform), compute gains with
   :func:`gridtools.sim.field.dipole_gain` for a candidate falloff exponent
   :math:`n` and regularisation distance :math:`a`.
2. Turn gains into ``sign_v``-like powers: the PSD of one spectrogram frame
   at the fish's frequency is :math:`|\\sqrt{S_e} + \\eta_e|^2` with signal
   power :math:`S_e \\propto g_e^2` and complex Gaussian noise
   :math:`\\eta_e` of mean power :math:`N_e` (a single-frame periodogram
   bin of noise is exponentially distributed). The signal scale is set so
   that the strongest electrode's signal-to-noise ratio follows the real
   distribution measured on the same recordings.
3. Compare quantiles of the *sorted relative profiles* (see
   :mod:`gridtools.calibrate.profiles`) between simulation and data, and
   pick the parameters that minimise the mean squared quantile difference.

Lengths are expressed in units of the electrode spacing (``*_rel``), since
with positions drawn relative to the grid the profile shapes depend on
:math:`a` and the margin only through their ratio to the spacing. A vertical
offset :math:`\\Delta z` between fish and electrodes enters the gain exactly
like :math:`a` (the denominator is :math:`r^2 + \\Delta z^2 + a^2`), so the
fit returns the *effective* regulariser
:math:`a_\\mathrm{eff} = \\sqrt{a^2 + \\Delta z^2}`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from gridtools.calibrate.profiles import profile_quantiles, relative_sorted_profiles
from gridtools.sim.field import dipole_gain

__all__ = [
    "DEFAULT_QUANTILES",
    "FieldFit",
    "GridLayout",
    "SimulationDraws",
    "draw_simulation",
    "fit_field_falloff",
    "simulate_powers_db",
    "simulate_profiles",
]

FloatArray = npt.NDArray[np.float64]

DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


@dataclass(frozen=True)
class GridLayout:
    """A rectangular electrode grid, possibly with electrodes missing.

    Attributes
    ----------
    name : str
    n_rows, n_columns : int
    spacing_m : float
        Electrode spacing (rows and columns).
    missing : tuple of int
        Row-major indices of electrodes known to have no data (excluded
        from the simulation).
    n_random_missing : int
        Further electrodes without data whose position is unknown: for each
        simulated detection, this many electrodes are dropped at random.
    """

    name: str
    n_rows: int
    n_columns: int
    spacing_m: float
    missing: tuple[int, ...] = ()
    n_random_missing: int = 0

    @property
    def n_simulated(self) -> int:
        """Electrodes simulated (all except ``missing``)."""
        return self.n_rows * self.n_columns - len(self.missing)

    @property
    def n_electrodes(self) -> int:
        """Electrodes with data in each profile."""
        return self.n_simulated - self.n_random_missing

    def positions_m(self) -> FloatArray:
        """Positions of the simulated electrodes, shape (n_simulated, 2)."""
        rows, cols = np.divmod(np.arange(self.n_rows * self.n_columns), self.n_columns)
        xy = np.column_stack([cols, rows]).astype(np.float64) * self.spacing_m
        keep = np.ones(len(xy), dtype=bool)
        keep[list(self.missing)] = False
        return xy[keep]


@dataclass(frozen=True, eq=False)
class SimulationDraws:
    """Random numbers for one layout, reused for every parameter set.

    Using the same draws (common random numbers) for every candidate makes
    the loss surface smooth in the parameters.

    Attributes
    ----------
    unit_xy : numpy.ndarray, shape (n, 2)
        Uniform [0, 1) position draws, scaled per margin.
    heading_rad : numpy.ndarray, shape (n,)
    peak_snr_db : numpy.ndarray, shape (n,)
        Strongest-electrode SNR (peak PSD over noise density) per sample.
    noise_offset_db : numpy.ndarray, shape (n, E)
        Per-electrode noise level relative to the reference noise level.
    noise : numpy.ndarray of complex128, shape (n, E)
        Standard complex Gaussian noise (unit mean power).
    drop_keys : numpy.ndarray, shape (n, E)
        Uniform keys; the electrodes with the smallest keys are dropped
        when the layout has randomly missing electrodes.
    """

    unit_xy: FloatArray
    heading_rad: FloatArray
    peak_snr_db: FloatArray
    noise_offset_db: FloatArray
    noise: npt.NDArray[np.complex128]
    drop_keys: FloatArray


def draw_simulation(
    rng: np.random.Generator,
    n_samples: int,
    n_electrodes: int,
    peak_snr_db: npt.ArrayLike,
    noise_offset_sd_db: float = 0.0,
) -> SimulationDraws:
    """Draw everything random for :func:`simulate_powers_db`.

    Parameters
    ----------
    rng : numpy.random.Generator
    n_samples : int
        Number of simulated detections.
    n_electrodes : int
        Simulated electrodes (:attr:`GridLayout.n_simulated`).
    peak_snr_db : array_like
        Real strongest-electrode SNRs (dB) to resample from.
    noise_offset_sd_db : float
        SD of the per-electrode noise level around the reference (dB).

    Returns
    -------
    SimulationDraws
    """
    snr = np.asarray(peak_snr_db, dtype=np.float64)
    snr = snr[np.isfinite(snr)]
    if snr.size == 0:
        msg = "peak_snr_db has no finite values"
        raise ValueError(msg)
    unit_xy = rng.random((n_samples, 2))
    heading = rng.uniform(-np.pi, np.pi, n_samples)
    peak = rng.choice(snr, n_samples, replace=True)
    offsets = rng.normal(0.0, noise_offset_sd_db, (n_samples, n_electrodes))
    noise = (
        rng.standard_normal((n_samples, n_electrodes))
        + 1j * rng.standard_normal((n_samples, n_electrodes))
    ) / np.sqrt(2.0)
    drop_keys = rng.random((n_samples, n_electrodes))
    return SimulationDraws(unit_xy, heading, peak, offsets, noise, drop_keys)


def simulate_powers_db(
    layout: GridLayout,
    draws: SimulationDraws,
    *,
    distance_exponent: float,
    regulariser_rel: float,
    margin_rel: float,
) -> FloatArray:
    """``sign_v``-like powers (dB re. reference noise) for random fish.

    Parameters
    ----------
    layout : GridLayout
    draws : SimulationDraws
        From :func:`draw_simulation` with ``layout.n_electrodes`` electrodes.
    distance_exponent : float
        Falloff exponent :math:`n`.
    regulariser_rel : float
        Effective regularisation distance :math:`a_\\mathrm{eff}` in units
        of the electrode spacing.
    margin_rel : float
        Fish positions extend this far beyond the outermost electrodes, in
        units of the spacing.

    Returns
    -------
    numpy.ndarray, shape (n, layout.n_simulated)
        Randomly missing electrodes are NaN.
    """
    electrodes = layout.positions_m()
    spacing = layout.spacing_m
    lo = electrodes.min(axis=0) - margin_rel * spacing
    hi = electrodes.max(axis=0) + margin_rel * spacing
    xy = lo + draws.unit_xy * (hi - lo)
    gain = dipole_gain(
        xy,
        draws.heading_rad,
        electrodes,
        fish_depth_m=0.0,
        electrode_depth_m=0.0,
        min_distance_m=regulariser_rel * spacing,
        distance_exponent=distance_exponent,
    ).astype(np.float64)
    power = gain * gain
    power /= power.max(axis=1, keepdims=True)
    power *= 10.0 ** (draws.peak_snr_db[:, None] / 10.0)
    noise_power = 10.0 ** (draws.noise_offset_db / 10.0)
    z = np.sqrt(power) + np.sqrt(noise_power) * draws.noise
    out = 10.0 * np.log10(np.abs(z) ** 2)
    if layout.n_random_missing:
        order = np.argsort(draws.drop_keys, axis=1)[:, : layout.n_random_missing]
        np.put_along_axis(out, order, np.nan, axis=1)
    return out


def simulate_profiles(
    layout: GridLayout,
    draws: SimulationDraws,
    *,
    distance_exponent: float,
    regulariser_rel: float,
    margin_rel: float,
) -> FloatArray:
    """Sorted relative profiles of :func:`simulate_powers_db`.

    Returns
    -------
    numpy.ndarray, shape (n, layout.n_electrodes)
    """
    powers = simulate_powers_db(
        layout,
        draws,
        distance_exponent=distance_exponent,
        regulariser_rel=regulariser_rel,
        margin_rel=margin_rel,
    )
    return relative_sorted_profiles(powers)[:, : layout.n_electrodes]


@dataclass(frozen=True, eq=False)
class FieldFit:
    """Result of :func:`fit_field_falloff`.

    Attributes
    ----------
    exponents, regularisers_rel, margins_rel : numpy.ndarray
        The parameter grid.
    loss : numpy.ndarray, shape (len(exponents), len(regularisers_rel), len(margins_rel))
        Mean squared quantile difference in dB**2, summed over layouts.
    loss_by_layout : dict of str to numpy.ndarray
        The same per layout.
    best : dict of str to float
        ``distance_exponent``, ``regulariser_rel``, ``margin_rel``, ``loss``.
    bootstrap : numpy.ndarray, shape (B, 3)
        Best (exponent, regulariser_rel, margin_rel) for each bootstrap
        resample of the trials.
    quantiles : tuple of float
    real_quantiles : dict of str to numpy.ndarray, shape (Q, E)
    best_quantiles : dict of str to numpy.ndarray, shape (Q, E)
        Simulated quantiles at the best parameters.
    """

    exponents: FloatArray
    regularisers_rel: FloatArray
    margins_rel: FloatArray
    loss: FloatArray
    loss_by_layout: dict[str, FloatArray]
    best: dict[str, float]
    bootstrap: FloatArray
    quantiles: tuple[float, ...]
    real_quantiles: dict[str, FloatArray]
    best_quantiles: dict[str, FloatArray] = field(default_factory=dict)

    def interval(self, name: str, level: float = 0.9) -> tuple[float, float]:
        """Bootstrap percentile interval of a fitted parameter."""
        column = {"distance_exponent": 0, "regulariser_rel": 1, "margin_rel": 2}[name]
        lo, hi = np.quantile(
            self.bootstrap[:, column], [(1 - level) / 2, (1 + level) / 2]
        )
        return float(lo), float(hi)


def fit_field_falloff(
    profiles_by_trial: Mapping[str, Sequence[npt.ArrayLike]],
    layouts: Mapping[str, GridLayout],
    draws: Mapping[str, SimulationDraws],
    *,
    exponents: npt.ArrayLike,
    regularisers_rel: npt.ArrayLike,
    margins_rel: npt.ArrayLike,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    n_bootstrap: int = 200,
    rng: np.random.Generator,
) -> FieldFit:
    """Grid-search the falloff parameters that reproduce real profiles.

    Parameters
    ----------
    profiles_by_trial : mapping of layout name to sequence of arrays
        For each layout, one array of sorted relative profiles
        (:func:`gridtools.calibrate.profiles.relative_sorted_profiles`) per
        trial, shape (n_i, E).
    layouts : mapping of layout name to GridLayout
    draws : mapping of layout name to SimulationDraws
    exponents, regularisers_rel, margins_rel : array_like
        Parameter grid.
    quantiles : sequence of float
        Quantiles compared per rank.
    n_bootstrap : int
        Resamples of the trials (with replacement, per layout) for the
        uncertainty of the best parameters.
    rng : numpy.random.Generator
        For the bootstrap.

    Returns
    -------
    FieldFit
    """
    ns = np.asarray(exponents, dtype=np.float64)
    as_ = np.asarray(regularisers_rel, dtype=np.float64)
    ms = np.asarray(margins_rel, dtype=np.float64)
    q = tuple(float(v) for v in quantiles)
    names = list(profiles_by_trial)

    sim_q: dict[str, FloatArray] = {}
    for name in names:
        grid = np.empty(
            (ns.size, as_.size, ms.size, len(q), layouts[name].n_electrodes)
        )
        for i, n in enumerate(ns):
            for j, a in enumerate(as_):
                for k, m in enumerate(ms):
                    profiles = simulate_profiles(
                        layouts[name],
                        draws[name],
                        distance_exponent=float(n),
                        regulariser_rel=float(a),
                        margin_rel=float(m),
                    )
                    grid[i, j, k] = profile_quantiles(profiles, q)
        sim_q[name] = grid

    def losses(real_by_layout: Mapping[str, FloatArray]) -> dict[str, FloatArray]:
        out = {}
        for name in names:
            real_q = real_by_layout[name]
            diff = sim_q[name][..., 1:] - real_q[None, None, None, :, 1:]
            out[name] = np.mean(diff * diff, axis=(-2, -1))
        return out

    trial_arrays = {
        name: [np.asarray(p, dtype=np.float64) for p in profiles_by_trial[name]]
        for name in names
    }
    real_q = {
        name: profile_quantiles(np.concatenate(trial_arrays[name]), q) for name in names
    }
    by_layout = losses(real_q)
    total = sum(by_layout.values(), np.zeros((ns.size, as_.size, ms.size)))
    i, j, k = np.unravel_index(int(np.argmin(total)), total.shape)
    best = {
        "distance_exponent": float(ns[i]),
        "regulariser_rel": float(as_[j]),
        "margin_rel": float(ms[k]),
        "loss": float(total[i, j, k]),
    }

    boot = np.empty((n_bootstrap, 3))
    for b in range(n_bootstrap):
        resampled = {}
        for name in names:
            arrays = trial_arrays[name]
            pick = rng.integers(0, len(arrays), len(arrays))
            resampled[name] = profile_quantiles(
                np.concatenate([arrays[p] for p in pick]), q
            )
        boot_total = sum(losses(resampled).values(), np.zeros(total.shape))
        bi, bj, bk = np.unravel_index(int(np.argmin(boot_total)), total.shape)
        boot[b] = (ns[bi], as_[bj], ms[bk])

    return FieldFit(
        exponents=ns,
        regularisers_rel=as_,
        margins_rel=ms,
        loss=total,
        loss_by_layout=by_layout,
        best=best,
        bootstrap=boot,
        quantiles=q,
        real_quantiles=real_q,
        best_quantiles={name: sim_q[name][i, j, k] for name in names},
    )
