"""Electric field gains from a fish to the electrodes.

Each fish is modelled as a point current dipole at its position, oriented
along its body axis (positive pole at the head), in an unbounded homogeneous
medium. The potential at an electrode, relative to a reference at infinity,
is

.. math::

    V = \\frac{\\mathbf{p} \\cdot \\mathbf{r}}{4\\pi\\sigma\\,|\\mathbf{r}|^3}
      \\propto \\frac{\\cos\\vartheta}{r^2},

where :math:`\\mathbf{r}` points from the fish to the electrode and
:math:`\\vartheta` is the angle between it and the body axis. The gain
returned here generalises the falloff exponent and regularises the
singularity at the fish:

.. math::

    g = r_\\mathrm{ref}^{\\,n}\\,
        \\frac{\\hat{\\mathbf{p}} \\cdot \\mathbf{r}}
             {\\left(|\\mathbf{r}|^2 + a^2\\right)^{(n+1)/2}},

with :math:`n` = ``distance_exponent`` (2 for a physical dipole),
:math:`a` = ``min_distance_m`` and :math:`r_\\mathrm{ref}` =
``reference_distance_m``. For :math:`r \\gg a` this is
:math:`(r_\\mathrm{ref}/r)^n \\cos\\vartheta`.

Units and scale. Gains are dimensionless and share one global scale for all
fish and paths: a gain of 1 is the potential on the body axis at
``reference_distance_m`` (far from the regularisation scale). Multiplying an
EOD of amplitude ``A`` by the gain therefore gives the potential in units of
"potential at 10 cm in front of a fish of amplitude ``A``". There is **no**
per-fish or per-path normalisation, so the spatial pattern of gains across
electrodes, which is the cue for assigning a chirp to its emitter, is kept
(v1 min-max normalised each fish over its own path, which left almost no
spatial contrast).

Assumptions and limitations

- Point dipole: a fish is about 15-20 cm long, so within roughly half a body
  length the real field is not dipolar. ``min_distance_m`` (default 0.1 m)
  caps the gain there instead of letting it diverge; it is a regulariser, not
  a model of the near field.
- Unbounded, homogeneous water: no tank walls (insulating, which would
  reflect current back into the tank) and no water surface or bottom (image
  charges). In shallow water the far field falls off more slowly than
  :math:`1/r^2`; ``distance_exponent`` exists to absorb this in a fit to real
  per-electrode power profiles.
- Horizontal dipole: no pitch or roll; the fish depth only sets a vertical
  offset to the electrode plane.
- Reference at infinity: if the recording reference sits in the tank, its own
  potential is subtracted from every electrode (not modelled).
- Quasi-static: all EOD harmonics share the same gain.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.interpolate import BSpline, make_interp_spline

from gridtools.sim._common import (
    Float32Array,
    FloatArray,
    as_1d_float64,
    as_points_m,
    require_nonnegative,
    require_positive,
)

__all__ = [
    "DipoleParams",
    "GainTrack",
    "dipole_gain",
    "gain_track",
]

DEFAULT_MIN_DISTANCE_M = 0.1
DEFAULT_DISTANCE_EXPONENT = 2.0
DEFAULT_REFERENCE_DISTANCE_M = 0.1


def dipole_gain(
    fish_xy_m: npt.ArrayLike,
    heading_rad: npt.ArrayLike,
    electrode_xy_m: npt.ArrayLike,
    *,
    fish_depth_m: float | npt.ArrayLike,
    electrode_depth_m: float | npt.ArrayLike,
    min_distance_m: float = DEFAULT_MIN_DISTANCE_M,
    distance_exponent: float = DEFAULT_DISTANCE_EXPONENT,
    reference_distance_m: float = DEFAULT_REFERENCE_DISTANCE_M,
) -> Float32Array:
    """Signed gain from a horizontal point dipole to each electrode.

    Parameters
    ----------
    fish_xy_m : array_like, shape (n, 2)
        Fish (dipole centre) positions in metres.
    heading_rad : array_like, shape (n,)
        Body-axis orientation in radians (dipole points to the head).
    electrode_xy_m : array_like, shape (E, 2)
        Electrode positions in metres, in the same frame.
    fish_depth_m : float or array_like of shape (n,)
        Fish depth below the surface in metres.
    electrode_depth_m : float or array_like of shape (E,)
        Electrode depth below the surface in metres. Only the vertical offset
        to the fish matters in an unbounded medium.
    min_distance_m : float
        Regularisation distance :math:`a` in metres (default 0.1, about half
        a body length): the denominator uses :math:`r^2 + a^2`.
    distance_exponent : float
        Falloff exponent :math:`n` (default 2, a physical dipole). Kept as a
        parameter for calibration against real per-electrode power profiles.
    reference_distance_m : float
        Distance on the body axis at which the gain is 1 (default 0.1).

    Returns
    -------
    numpy.ndarray of float32, shape (n, E)
        Signed, unnormalised gains (see module docstring for units).
    """
    fish = as_points_m(fish_xy_m, "fish_xy_m")
    electrodes = as_points_m(electrode_xy_m, "electrode_xy_m")
    heading = as_1d_float64(heading_rad, "heading_rad")
    n = fish.shape[0]
    n_electrodes = electrodes.shape[0]
    if heading.size != n:
        msg = f"heading_rad must have shape ({n},), got {heading.shape}"
        raise ValueError(msg)
    fish_depth = _depth(fish_depth_m, n, "fish_depth_m")
    electrode_depth = _depth(electrode_depth_m, n_electrodes, "electrode_depth_m")
    require_nonnegative(min_distance_m, "min_distance_m")
    require_positive(distance_exponent, "distance_exponent")
    require_positive(reference_distance_m, "reference_distance_m")

    dx = electrodes[None, :, 0] - fish[:, None, 0]
    dy = electrodes[None, :, 1] - fish[:, None, 1]
    dz = electrode_depth[None, :] - fish_depth[:, None]
    projection = np.cos(heading)[:, None] * dx + np.sin(heading)[:, None] * dy
    r2 = dx * dx + dy * dy + dz * dz + min_distance_m * min_distance_m
    if np.any(r2 == 0.0):
        msg = "fish coincides with an electrode; set min_distance_m > 0"
        raise ValueError(msg)
    scale = reference_distance_m**distance_exponent
    gain = scale * projection * r2 ** (-(distance_exponent + 1.0) / 2.0)
    return gain.astype(np.float32)


@dataclass(frozen=True)
class DipoleParams:
    """Parameters of the dipole field model (see :func:`dipole_gain`).

    Attributes
    ----------
    fish_depth_m : float
        Fish depth below the surface in metres (default 0.10).
    electrode_depth_m : float
        Electrode depth below the surface in metres (default 0.15). The
        default 5-cm offset assumes electrodes near the tank floor and fish
        swimming just above them; check against the setup.
    min_distance_m : float
        Regularisation distance in metres (default 0.1).
    distance_exponent : float
        Falloff exponent (default 2).
    reference_distance_m : float
        Axis distance with gain 1, in metres (default 0.1).
    """

    fish_depth_m: float = 0.10
    electrode_depth_m: float = 0.15
    min_distance_m: float = DEFAULT_MIN_DISTANCE_M
    distance_exponent: float = DEFAULT_DISTANCE_EXPONENT
    reference_distance_m: float = DEFAULT_REFERENCE_DISTANCE_M

    def __post_init__(self) -> None:
        require_nonnegative(self.fish_depth_m, "fish_depth_m")
        require_nonnegative(self.electrode_depth_m, "electrode_depth_m")
        require_nonnegative(self.min_distance_m, "min_distance_m")
        require_positive(self.distance_exponent, "distance_exponent")
        require_positive(self.reference_distance_m, "reference_distance_m")


@dataclass(frozen=True, eq=False)
class GainTrack:
    """Gains sampled at the movement rate, interpolated on demand.

    Gains change on the time-scale of movement (tens of milliseconds), so
    they are computed at the movement rate and interpolated to signal sample
    times with a cubic spline, chunk by chunk, instead of being stored at the
    signal rate.

    Attributes
    ----------
    times_s : numpy.ndarray of float64, shape (m,)
        Strictly increasing sample times in seconds (m >= 4).
    gains : numpy.ndarray of float32, shape (m, E)
        Gains at ``times_s``.
    """

    times_s: FloatArray
    gains: Float32Array
    _spline: BSpline = field(init=False, repr=False)

    def __post_init__(self) -> None:
        times = as_1d_float64(self.times_s, "times_s")
        gains = np.asarray(self.gains, dtype=np.float32)
        if gains.ndim != 2 or gains.shape[0] != times.size:
            msg = f"gains must have shape ({times.size}, E), got {gains.shape}"
            raise ValueError(msg)
        if times.size < 4 or np.any(np.diff(times) <= 0):
            msg = "times_s must be strictly increasing with at least 4 samples"
            raise ValueError(msg)
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "gains", gains)
        spline = make_interp_spline(times, gains.astype(np.float64), k=3, axis=0)
        object.__setattr__(self, "_spline", spline)

    @property
    def n_electrodes(self) -> int:
        """Number of electrodes E."""
        return self.gains.shape[1]

    def at(self, times_s: npt.ArrayLike) -> Float32Array:
        """Interpolate the gains to the given times.

        Parameters
        ----------
        times_s : array_like, shape (k,)
            Times in seconds within ``[times_s[0], times_s[-1]]`` of the track.

        Returns
        -------
        numpy.ndarray of float32, shape (k, E)
        """
        times = np.asarray(times_s, dtype=np.float64)
        if times.ndim != 1:
            msg = f"times_s must be 1-D, got shape {times.shape}"
            raise ValueError(msg)
        if times.size and (
            times.min() < self.times_s[0] or times.max() > self.times_s[-1]
        ):
            msg = (
                "times_s outside the gain track "
                f"[{self.times_s[0]}, {self.times_s[-1]}]"
            )
            raise ValueError(msg)
        return np.asarray(self._spline(times), dtype=np.float32)


def gain_track(
    times_s: npt.ArrayLike,
    fish_xy_m: npt.ArrayLike,
    heading_rad: npt.ArrayLike,
    electrode_xy_m: npt.ArrayLike,
    params: DipoleParams | None = None,
) -> GainTrack:
    """Compute gains along a path at its own (movement) rate.

    Parameters
    ----------
    times_s : array_like, shape (m,)
        Path sample times in seconds (e.g. ``FishPath.times_s``).
    fish_xy_m : array_like, shape (m, 2)
        Fish positions in metres.
    heading_rad : array_like, shape (m,)
        Body-axis orientation in radians.
    electrode_xy_m : array_like, shape (E, 2)
        Electrode positions in metres.
    params : DipoleParams, optional
        Field parameters; defaults to ``DipoleParams()``.

    Returns
    -------
    GainTrack
        Gains at ``times_s``, interpolable to any time in range.
    """
    params = DipoleParams() if params is None else params
    gains = dipole_gain(
        fish_xy_m,
        heading_rad,
        electrode_xy_m,
        fish_depth_m=params.fish_depth_m,
        electrode_depth_m=params.electrode_depth_m,
        min_distance_m=params.min_distance_m,
        distance_exponent=params.distance_exponent,
        reference_distance_m=params.reference_distance_m,
    )
    return GainTrack(times_s=as_1d_float64(times_s, "times_s"), gains=gains)


def _depth(value: float | npt.ArrayLike, n: int, name: str) -> FloatArray:
    depth = np.asarray(value, dtype=np.float64)
    if depth.ndim == 0:
        depth = np.full(n, float(depth))
    if depth.shape != (n,) or not np.all(np.isfinite(depth)):
        msg = f"{name} must be a finite scalar or have shape ({n},)"
        raise ValueError(msg)
    return depth
