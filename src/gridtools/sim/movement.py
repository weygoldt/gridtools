"""Fish movement: a correlated random walk with rest states.

The model runs at a movement rate (default 30 Hz) and has four parts:

1. **Behavioural state.** A two-state Markov process (rest / move) with
   exponentially distributed dwell times (means ``mean_rest_s`` and
   ``mean_move_s``). The long-run fraction of time at rest is
   ``mean_rest_s / (mean_rest_s + mean_move_s)``.
2. **Speed.** Each move bout gets a cruising speed from a gamma distribution
   (capped at ``speed_max_m_s``) and, with probability ``backward_prob``,
   is swum backwards (tail first, as *Apteronotus* can with its ribbon fin).
   Within a bout, speed fluctuates log-normally (Ornstein-Uhlenbeck process in
   log-speed). The signed velocity follows this target through a first-order
   lag (``speed_response_s``), so it accelerates and decelerates smoothly and
   passes through zero between forward and backward bouts.
3. **Heading.** The body axis turns with an angular velocity that is an
   Ornstein-Uhlenbeck process (SD ``turn_rate_sd_rad_s``, time-scale
   ``turn_rate_timescale_s``), damped at rest (``rest_turn_factor``). The
   heading is integrated from it, so it is continuous; the total turning
   rate is bounded by ``turn_rate_max_rad_s``.
4. **Arena.** A rectangular arena in metres. Near a wall, a fish moving
   towards it turns away (wall avoidance within ``wall_margin_m``). If it
   still reaches a wall, its position is reflected back into the arena
   (reflecting boundary) while its heading stays continuous; avoidance then
   turns it away within a few steps.

All random numbers are drawn up front from the caller's Generator, so the
random stream does not depend on the trajectory; only the wall interaction is
integrated step by step.

The defaults are plausible values for *Apteronotus leptorhynchus* in a lab
tank but are **not fitted to data**. They should be calibrated against
tracked positions (grid power centroids or video) before the simulations are
used to argue about behaviour.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator
from scipy.signal import lfilter

from gridtools.sim._common import (
    BoolArray,
    FloatArray,
    IntArray,
    as_1d_float64,
    require_nonnegative,
    require_positive,
    require_probability,
)

__all__ = [
    "Arena",
    "FishPath",
    "MovementParams",
    "simulate_path",
    "upsample_path",
    "wrap_angle",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Arena:
    """Rectangular arena in metres (tank interior, in grid coordinates).

    Attributes
    ----------
    x_min_m, x_max_m, y_min_m, y_max_m : float
        Bounds in metres; ``x_max_m > x_min_m`` and ``y_max_m > y_min_m``.
    """

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float

    def __post_init__(self) -> None:
        bounds = (self.x_min_m, self.x_max_m, self.y_min_m, self.y_max_m)
        if not all(math.isfinite(b) for b in bounds):
            msg = "arena bounds must be finite"
            raise ValueError(msg)
        if not (self.x_max_m > self.x_min_m and self.y_max_m > self.y_min_m):
            msg = f"arena bounds must satisfy max > min, got {bounds}"
            raise ValueError(msg)

    @property
    def width_m(self) -> float:
        """Extent along x in metres."""
        return self.x_max_m - self.x_min_m

    @property
    def height_m(self) -> float:
        """Extent along y in metres."""
        return self.y_max_m - self.y_min_m

    def contains(self, xy_m: npt.ArrayLike) -> BoolArray:
        """Return whether points of shape (n, 2) lie inside (bounds included)."""
        xy = np.asarray(xy_m, dtype=np.float64)
        return (
            (xy[..., 0] >= self.x_min_m)
            & (xy[..., 0] <= self.x_max_m)
            & (xy[..., 1] >= self.y_min_m)
            & (xy[..., 1] <= self.y_max_m)
        )


@dataclass(frozen=True)
class MovementParams:
    """Parameters of the movement model (not fitted to data; see module doc).

    Attributes
    ----------
    rate_hz : float
        Movement sampling rate in Hz (default 30, a typical video and
        tracking rate).
    mean_rest_s : float
        Mean dwell time at rest in seconds (default 10). ``0`` disables rest.
    mean_move_s : float
        Mean dwell time in a move bout in seconds (default 20). With the
        default rest time, the fish rests a third of the time.
    speed_mean_m_s : float
        Mean cruising speed of a move bout in m/s (default 0.1, i.e. about
        half a body length per second).
    speed_shape : float
        Shape of the gamma distribution of bout speeds (default 4; CV 0.5).
    speed_max_m_s : float
        Hard upper bound on speed in m/s (default 0.5, a fast chase).
    speed_fluct_log_sd : float
        SD of within-bout log-speed fluctuations (default 0.3).
    speed_fluct_timescale_s : float
        Time-scale of these fluctuations in seconds (default 1).
    speed_response_s : float
        Time constant of acceleration and deceleration in seconds
        (default 0.3).
    backward_prob : float
        Probability that a move bout is swum backwards (default 0.1).
    turn_rate_sd_rad_s : float
        SD of the random angular velocity in rad/s (default 0.8).
    turn_rate_timescale_s : float
        Time-scale of the angular velocity in seconds (default 0.5). Heading
        then decorrelates over roughly
        ``1 / (turn_rate_sd_rad_s**2 * turn_rate_timescale_s)`` = 3 s.
    turn_rate_max_rad_s : float
        Bound on the total turning rate in rad/s (default 6, about 340
        degrees/s).
    rest_turn_factor : float
        Factor on the random angular velocity at rest (default 0.2): resting
        fish rotate slowly in place.
    wall_margin_m : float
        Distance from a wall within which a fish moving towards it turns away
        (default 0.15). ``0`` disables avoidance; the reflecting boundary
        then does all the work.
    wall_turn_rate_rad_s : float
        Turning rate of wall avoidance at the wall in rad/s for a fish at or
        above ``speed_mean_m_s`` (default 6). It decays quadratically to zero
        at ``wall_margin_m`` and scales with speed below ``speed_mean_m_s``.
    """

    rate_hz: float = 30.0
    mean_rest_s: float = 10.0
    mean_move_s: float = 20.0
    speed_mean_m_s: float = 0.1
    speed_shape: float = 4.0
    speed_max_m_s: float = 0.5
    speed_fluct_log_sd: float = 0.3
    speed_fluct_timescale_s: float = 1.0
    speed_response_s: float = 0.3
    backward_prob: float = 0.1
    turn_rate_sd_rad_s: float = 0.8
    turn_rate_timescale_s: float = 0.5
    turn_rate_max_rad_s: float = 6.0
    rest_turn_factor: float = 0.2
    wall_margin_m: float = 0.15
    wall_turn_rate_rad_s: float = 6.0

    def __post_init__(self) -> None:
        require_positive(self.rate_hz, "rate_hz")
        require_nonnegative(self.mean_rest_s, "mean_rest_s")
        require_nonnegative(self.mean_move_s, "mean_move_s")
        if self.mean_rest_s == 0 and self.mean_move_s == 0:
            msg = "mean_rest_s and mean_move_s cannot both be 0"
            raise ValueError(msg)
        require_positive(self.speed_mean_m_s, "speed_mean_m_s")
        require_positive(self.speed_shape, "speed_shape")
        require_positive(self.speed_max_m_s, "speed_max_m_s")
        require_nonnegative(self.speed_fluct_log_sd, "speed_fluct_log_sd")
        require_positive(self.speed_fluct_timescale_s, "speed_fluct_timescale_s")
        require_positive(self.speed_response_s, "speed_response_s")
        require_probability(self.backward_prob, "backward_prob")
        require_nonnegative(self.turn_rate_sd_rad_s, "turn_rate_sd_rad_s")
        require_positive(self.turn_rate_timescale_s, "turn_rate_timescale_s")
        require_positive(self.turn_rate_max_rad_s, "turn_rate_max_rad_s")
        require_nonnegative(self.rest_turn_factor, "rest_turn_factor")
        require_nonnegative(self.wall_margin_m, "wall_margin_m")
        require_nonnegative(self.wall_turn_rate_rad_s, "wall_turn_rate_rad_s")

    @property
    def rest_fraction(self) -> float:
        """Long-run fraction of time spent at rest."""
        return self.mean_rest_s / (self.mean_rest_s + self.mean_move_s)


@dataclass(frozen=True, eq=False)
class FishPath:
    """Trajectory of one fish.

    Attributes
    ----------
    times_s : numpy.ndarray of float64, shape (n,)
        Sample times in seconds.
    xy_m : numpy.ndarray of float64, shape (n, 2)
        Position of the dipole centre in metres.
    heading_rad : numpy.ndarray of float64, shape (n,)
        Body-axis orientation (head direction) in radians, wrapped to
        [-pi, pi).
    velocity_m_s : numpy.ndarray of float64, shape (n,)
        Signed velocity along the body axis in m/s (negative = backwards).
    moving : numpy.ndarray of bool, shape (n,)
        Behavioural state (True = move bout, False = rest).
    n_wall_reflections : int
        Number of times the reflecting boundary was hit.
    """

    times_s: FloatArray
    xy_m: FloatArray
    heading_rad: FloatArray
    velocity_m_s: FloatArray
    moving: BoolArray
    n_wall_reflections: int = 0

    def __post_init__(self) -> None:
        n = self.times_s.shape[0]
        shapes_ok = (
            self.times_s.shape == (n,)
            and self.xy_m.shape == (n, 2)
            and self.heading_rad.shape == (n,)
            and self.velocity_m_s.shape == (n,)
            and self.moving.shape == (n,)
        )
        if not shapes_ok:
            msg = "FishPath arrays have inconsistent shapes"
            raise ValueError(msg)

    @property
    def speed_m_s(self) -> FloatArray:
        """Unsigned speed in m/s."""
        return np.abs(self.velocity_m_s)


def wrap_angle(angle_rad: npt.ArrayLike) -> FloatArray:
    """Wrap angles to [-pi, pi)."""
    angle = np.asarray(angle_rad, dtype=np.float64)
    return np.mod(angle + np.pi, 2.0 * np.pi) - np.pi


def simulate_path(
    rng: np.random.Generator,
    duration_s: float,
    arena: Arena,
    params: MovementParams | None = None,
    *,
    start_xy_m: tuple[float, float] | None = None,
    start_heading_rad: float | None = None,
) -> FishPath:
    """Simulate one fish's path in a rectangular arena.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of all random draws.
    duration_s : float
        Duration in seconds. The path covers at least ``[0, duration_s]``
        (the last sample time is >= ``duration_s``), so it can be
        interpolated to any signal sample in that range.
    arena : Arena
        Rectangular arena in metres.
    params : MovementParams, optional
        Defaults to ``MovementParams()``.
    start_xy_m : tuple of float, optional
        Start position in metres; uniform in the arena if omitted.
    start_heading_rad : float, optional
        Start heading in radians; uniform if omitted.

    Returns
    -------
    FishPath
        Path sampled at ``params.rate_hz``.
    """
    params = MovementParams() if params is None else params
    require_positive(duration_s, "duration_s")
    dt = 1.0 / params.rate_hz
    n = math.floor(duration_s * params.rate_hz) + 2
    times_s = np.arange(n, dtype=np.float64) * dt

    # All random draws, in a fixed order, independent of the trajectory.
    moving, bout = _state_sequence(rng, n, dt, params)
    n_bouts = int(bout[-1]) + 1
    bout_speed = np.minimum(
        rng.gamma(
            params.speed_shape, params.speed_mean_m_s / params.speed_shape, n_bouts
        ),
        params.speed_max_m_s,
    )
    bout_sign = np.where(rng.random(n_bouts) < params.backward_prob, -1.0, 1.0)
    log_speed = _unit_ou(rng, n, params.speed_fluct_timescale_s * params.rate_hz)
    turn_noise = _unit_ou(rng, n, params.turn_rate_timescale_s * params.rate_hz)
    start_draw = rng.random(3)

    sigma = params.speed_fluct_log_sd
    target = bout_speed[bout] * np.exp(sigma * log_speed - 0.5 * sigma**2)
    target = np.minimum(target, params.speed_max_m_s) * bout_sign[bout] * moving
    lag = math.exp(-dt / params.speed_response_s)
    velocity = np.asarray(
        lfilter([1.0 - lag], [1.0, -lag], target, zi=[lag * target[0]])[0],
        dtype=np.float64,
    )

    turn_rate = params.turn_rate_sd_rad_s * turn_noise
    turn_rate *= np.where(moving, 1.0, params.rest_turn_factor)

    if start_xy_m is None:
        x0 = arena.x_min_m + start_draw[0] * arena.width_m
        y0 = arena.y_min_m + start_draw[1] * arena.height_m
    else:
        x0, y0 = float(start_xy_m[0]), float(start_xy_m[1])
        if not arena.contains(np.array([[x0, y0]]))[0]:
            msg = f"start_xy_m {start_xy_m} lies outside the arena"
            raise ValueError(msg)
    h0 = (
        2.0 * math.pi * start_draw[2] - math.pi
        if start_heading_rad is None
        else float(start_heading_rad)
    )

    xy, heading, n_hits = _integrate(x0, y0, h0, velocity, turn_rate, dt, arena, params)
    if n_hits:
        logger.debug("path hit the reflecting boundary %d times", n_hits)
    return FishPath(
        times_s=times_s,
        xy_m=xy,
        heading_rad=wrap_angle(heading),
        velocity_m_s=velocity,
        moving=moving,
        n_wall_reflections=n_hits,
    )


def upsample_path(path: FishPath, times_s: npt.ArrayLike) -> FishPath:
    """Interpolate a path to other sample times (e.g. the signal rate).

    Positions and the unwrapped heading are interpolated with monotone cubic
    (PCHIP) interpolation: smooth, and free of overshoot, so interpolated
    positions stay inside the arena. The heading is unwrapped before and
    wrapped after interpolation, so crossing +/-pi does not produce a spurious
    full turn. Velocity is interpolated linearly and the behavioural state is
    held from the preceding sample.

    Parameters
    ----------
    path : FishPath
        Path at the movement rate.
    times_s : array_like, shape (m,)
        Target times in seconds, within ``[path.times_s[0], path.times_s[-1]]``.

    Returns
    -------
    FishPath
        Path at ``times_s``.
    """
    times = as_1d_float64(times_s, "times_s")
    if times.min() < path.times_s[0] or times.max() > path.times_s[-1]:
        msg = (
            "times_s must lie within the path's time range "
            f"[{path.times_s[0]}, {path.times_s[-1]}]"
        )
        raise ValueError(msg)
    xy = np.asarray(PchipInterpolator(path.times_s, path.xy_m, axis=0)(times))
    heading = np.asarray(
        PchipInterpolator(path.times_s, np.unwrap(path.heading_rad))(times)
    )
    velocity = np.interp(times, path.times_s, path.velocity_m_s)
    index = np.searchsorted(path.times_s, times, side="right") - 1
    moving = path.moving[np.clip(index, 0, path.times_s.size - 1)]
    return FishPath(
        times_s=times,
        xy_m=xy.astype(np.float64),
        heading_rad=wrap_angle(heading),
        velocity_m_s=velocity,
        moving=moving,
        n_wall_reflections=path.n_wall_reflections,
    )


def _unit_ou(rng: np.random.Generator, n: int, timescale_steps: float) -> FloatArray:
    """Stationary AR(1) (sampled Ornstein-Uhlenbeck) process with unit variance."""
    a = math.exp(-1.0 / timescale_steps)
    noise = rng.standard_normal(n)
    noise[1:] *= math.sqrt(1.0 - a * a)
    return np.asarray(lfilter([1.0], [1.0, -a], noise), dtype=np.float64)


def _state_sequence(
    rng: np.random.Generator,
    n: int,
    dt: float,
    params: MovementParams,
) -> tuple[BoolArray, IntArray]:
    """Rest/move state per step and the index of the bout it belongs to."""
    if params.mean_rest_s == 0 or params.mean_move_s == 0:
        state = params.mean_rest_s == 0
        return np.full(n, state), np.zeros(n, dtype=np.int64)
    start_moving = bool(rng.random() >= params.rest_fraction)
    total_s = n * dt
    mean_cycle_s = params.mean_rest_s + params.mean_move_s
    batch = 2 * math.ceil(total_s / mean_cycle_s) + 16
    durations: list[FloatArray] = []
    covered_s = 0.0
    while covered_s <= total_s:
        first = len(durations) * batch
        parity = (np.arange(first, first + batch) % 2).astype(bool)
        bout_moving = parity ^ start_moving
        means = np.where(bout_moving, params.mean_move_s, params.mean_rest_s)
        chunk = rng.standard_exponential(batch) * means
        durations.append(chunk)
        covered_s += float(chunk.sum())
    ends_s = np.cumsum(np.concatenate(durations))
    bout = np.searchsorted(ends_s, np.arange(n) * dt, side="right").astype(np.int64)
    moving = ((bout % 2).astype(bool)) ^ start_moving
    return moving, bout


def _integrate(
    x0: float,
    y0: float,
    h0: float,
    velocity: FloatArray,
    turn_rate: FloatArray,
    dt: float,
    arena: Arena,
    params: MovementParams,
) -> tuple[FloatArray, FloatArray, int]:
    """Integrate position and heading with wall avoidance and reflection."""
    n = velocity.size
    xs = np.empty(n)
    ys = np.empty(n)
    hs = np.empty(n)
    x, y, h = x0, y0, h0
    xs[0], ys[0], hs[0] = x, y, h
    x_lo, x_hi = arena.x_min_m, arena.x_max_m
    y_lo, y_hi = arena.y_min_m, arena.y_max_m
    margin = params.wall_margin_m
    wall_rate = params.wall_turn_rate_rad_s
    max_rate = params.turn_rate_max_rad_s
    v_ref = params.speed_mean_m_s
    hits = 0
    vel = velocity.tolist()
    turn = turn_rate.tolist()
    for i in range(1, n):
        v = vel[i - 1]
        omega = turn[i - 1]
        cos_h, sin_h = math.cos(h), math.sin(h)
        if margin > 0.0 and v != 0.0:
            # Inward repulsion from walls closer than the margin.
            rx = _proximity(x - x_lo, margin) - _proximity(x_hi - x, margin)
            ry = _proximity(y - y_lo, margin) - _proximity(y_hi - y, margin)
            if rx != 0.0 or ry != 0.0:
                ux = cos_h if v > 0.0 else -cos_h
                uy = sin_h if v > 0.0 else -sin_h
                if ux * rx + uy * ry < 0.0:  # moving towards the walls
                    cross = ux * ry - uy * rx
                    side = 1.0 if cross > 0.0 else -1.0 if cross < 0.0 else 1.0
                    # Faster fish need to turn harder; resting fish (whose
                    # velocity decays towards zero) are not spun round.
                    strength = min(1.0, math.hypot(rx, ry)) * min(1.0, abs(v) / v_ref)
                    omega += side * wall_rate * strength
        omega = max(-max_rate, min(max_rate, omega))
        h += omega * dt
        step = v * dt
        x += step * math.cos(h)
        y += step * math.sin(h)
        if x < x_lo or x > x_hi or y < y_lo or y > y_hi:
            hits += 1
            if x < x_lo:
                x = 2.0 * x_lo - x
            elif x > x_hi:
                x = 2.0 * x_hi - x
            if y < y_lo:
                y = 2.0 * y_lo - y
            elif y > y_hi:
                y = 2.0 * y_hi - y
            # A single reflection suffices unless a step exceeds the arena.
            x = min(max(x, x_lo), x_hi)
            y = min(max(y, y_lo), y_hi)
        xs[i], ys[i], hs[i] = x, y, h
    return np.column_stack([xs, ys]), hs, hits


def _proximity(distance_m: float, margin_m: float) -> float:
    """Wall proximity weight: 1 at the wall, 0 at and beyond the margin."""
    if distance_m >= margin_m:
        return 0.0
    closeness = 1.0 - max(distance_m, 0.0) / margin_m
    return closeness * closeness
