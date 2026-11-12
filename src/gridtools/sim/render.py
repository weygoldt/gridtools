"""Deterministic rendering of fish signals for any span, with or without chirps.

A :class:`SceneModel` holds everything needed to render each fish's
contribution to a scene recording: the baseline-frequency model (baseline,
individual drift, ramps, shared drift), the placed chirps, the EOD waveform,
the path (hence the dipole gains) and the channel gains. The recording
itself is mixed from exactly these renderings, so

``recording = sum_i render_fish(i) + background``

holds to float32 rounding (unless clipping is configured).

Phase
-----
The EOD phase of fish ``i`` is ``theta[n] = theta0 + 2 pi / fs * C[n]`` with
``C[n] = sum_{j<n} (f[j] + f[j+1]) / 2`` (trapezoid rule, as
:func:`gridtools.sim.eod.eod_phase`) and
``f = f_base + sum_c df_c``. Because the integral is linear, ``C`` is the
baseline's integral plus one term per chirp. The baseline term is evaluated
in fixed blocks of :data:`BLOCK` samples from checkpoints at block starts,
and each chirp term from the chirp's own cumulative sum, so every sample's
phase is computed by the same floating-point operations whatever span is
rendered: a sub-span rendering is bit-identical to the same samples of a
full rendering.

Omitting a chirp
----------------
A chirp advances the phase by its integral ``Phi_c = 2 pi z_c`` (``z_c``
cycles; e.g. about 0.7 cycles for a 64-Hz, 10-ms chirp). Without the chirp,
the phase after it would differ by ``Phi_c`` for the rest of the scene. A
re-rendering without the chirp cannot be both "like a fish that never
chirped" *and* phase-identical to the recording after the chirp. Two
definitions are offered (``phase=``):

``"aligned"`` (default)
    The omitted chirp's excursion and amplitude dip are removed and the
    fractional part of its phase advance, ``r = z_c - round(z_c)``
    (``|r| <= 1/2`` cycle), is spread linearly over the chirp's support (a
    constant offset of ``r / T`` Hz over its duration ``T``, at most
    ``1 / (2 T)``: about 7 Hz for a 70-ms support, well below the 4.9-Hz
    resolution of a 4096-point spectrogram integrated over the support).
    After the support the phase differs by ``-2 pi round(z_c)``, a whole
    number of cycles, so the signal is identical to the full rendering
    there. "With minus without" is therefore exactly zero outside the
    chirp's support, in the time domain and hence in any representation
    (complex or power spectrogram, any window, alone or mixed with other
    fish). This is the definition for ground-truth footprints.

``"natural"``
    The counterfactual fish that never produced the chirp: excursion and
    dip removed, nothing added. The phase is continuous, but lags the full
    rendering by ``Phi_c`` after the chirp, so the time-domain difference
    persists to the end of the scene. Power spectrograms of the fish alone
    still differ only around the chirp (a constant phase offset of a
    stationary harmonic signal does not change its power, up to window
    leakage between positive and negative frequencies), but mixtures with
    other fish do not (the beat phase shifts).
"""

import math
from collections import OrderedDict
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

from gridtools.sim._common import Float32Array, FloatArray
from gridtools.sim.chirps import ChirpPlacement
from gridtools.sim.eod import Harmonics, harmonic_waveform
from gridtools.sim.field import DipoleParams, GainTrack, gain_track
from gridtools.sim.mix import DEFAULT_GAIN_UPDATE_HZ, _track_chunk
from gridtools.sim.movement import FishPath

__all__ = [
    "BLOCK",
    "DriftKnots",
    "FishModel",
    "PhaseMode",
    "Ramp",
    "SceneModel",
]

#: Samples per block of the baseline phase integration.
BLOCK = 1 << 16
_BLOCK_CACHE_SIZE = 16

PhaseMode = Literal["aligned", "natural"]
ChirpSelection = Literal["all", "none"] | Collection[int]


@dataclass(frozen=True, eq=False)
class DriftKnots:
    """A smooth drift given by knots, interpolated with a cubic spline.

    Attributes
    ----------
    times_s : numpy.ndarray of float64
        Knot times in seconds (strictly increasing, covering the scene).
    values : numpy.ndarray of float64
        Values at the knots (Hz for individual drift, relative for shared).
    """

    times_s: FloatArray
    values: FloatArray

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        if times.ndim != 1 or times.shape != values.shape or times.size < 2:
            msg = "times_s and values must be equally long 1-D arrays (>= 2)"
            raise ValueError(msg)
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "values", values)

    @cached_property
    def spline(self) -> CubicSpline:
        """Cubic spline through the knots (scipy's default not-a-knot ends)."""
        return CubicSpline(self.times_s, self.values)

    def __call__(self, times_s: npt.ArrayLike) -> FloatArray:
        """Evaluate at the given times."""
        return np.asarray(self.spline(np.asarray(times_s, dtype=np.float64)))

    def to_dict(self) -> dict[str, list[float]]:
        """JSON-ready dict (floats round-trip exactly)."""
        return {"times_s": self.times_s.tolist(), "values": self.values.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Sequence[float]]) -> "DriftKnots":
        """Inverse of :meth:`to_dict`."""
        return cls(np.asarray(data["times_s"]), np.asarray(data["values"]))


@dataclass(frozen=True)
class Ramp:
    """A slow raised-cosine step of the baseline frequency.

    ``delta_hz * (1 - cos(pi x)) / 2`` with ``x = (t - start_s) / duration_s``
    clipped to [0, 1]: 0 before, ``delta_hz`` after, with zero slope at both
    ends.

    Attributes
    ----------
    start_s : float
        Start time in seconds.
    duration_s : float
        Duration in seconds (> 0).
    delta_hz : float
        Total frequency change in Hz.
    """

    start_s: float
    duration_s: float
    delta_hz: float

    def __post_init__(self) -> None:
        if not (self.duration_s > 0 and math.isfinite(self.duration_s)):
            msg = f"duration_s must be > 0, got {self.duration_s!r}"
            raise ValueError(msg)

    def __call__(self, times_s: npt.ArrayLike) -> FloatArray:
        """Frequency offset in Hz at the given times."""
        x = np.clip(
            (np.asarray(times_s, dtype=np.float64) - self.start_s) / self.duration_s,
            0.0,
            1.0,
        )
        return self.delta_hz * 0.5 * (1.0 - np.cos(np.pi * x))


@dataclass(frozen=True, eq=False)
class FishModel:
    """One fish: frequency model, waveform, path and chirps.

    Attributes
    ----------
    fish_id : int
        Identity (also the track ident).
    baseline_hz : float
        Mean baseline EODf in Hz (before drift).
    amplitude : float
        EOD amplitude (recording units at the field's reference distance).
    phase0_rad : float
        Phase of the fundamental at sample 0.
    harmonics : Harmonics
        Waveform.
    dipole : DipoleParams
        Field parameters (including the fish's depth).
    drift : DriftKnots
        Individual drift in Hz.
    ramps : tuple of Ramp
        Slow frequency ramps (crossings).
    path : FishPath
        Trajectory at the movement rate.
    chirps : tuple of ChirpPlacement
        Placed chirps, sorted by ``start``.
    """

    fish_id: int
    baseline_hz: float
    amplitude: float
    phase0_rad: float
    harmonics: Harmonics
    dipole: DipoleParams
    drift: DriftKnots
    ramps: tuple[Ramp, ...]
    path: FishPath
    chirps: tuple[ChirpPlacement, ...]

    def __post_init__(self) -> None:
        starts = [c.start for c in self.chirps]
        if starts != sorted(starts):
            msg = "chirps must be sorted by start"
            raise ValueError(msg)
        if any(c.fish_id != self.fish_id for c in self.chirps):
            msg = f"all chirps of fish {self.fish_id} must carry its id"
            raise ValueError(msg)

    @property
    def chirp_ids(self) -> list[int]:
        """Ids of this fish's chirps."""
        return [c.chirp_id for c in self.chirps]


@dataclass(frozen=True, eq=False)
class SceneModel:
    """Everything needed to render the fish of a scene.

    Attributes
    ----------
    samplerate_hz : float
        Sampling rate in Hz.
    n_samples : int
        Scene length in samples.
    electrode_xy_m : numpy.ndarray of float64, shape (E, 2)
        Electrode positions in metres (row-major grid order).
    channel_gain : numpy.ndarray of float64, shape (E,)
        Per-electrode gain applied to fish and noise (mismatch x faults).
    shared_drift : DriftKnots
        Relative EODf drift common to all fish.
    fish : tuple of FishModel
        Fish, indexed by ``fish_id``.
    gain_update_hz : float
        Rate at which gain tracks are evaluated before linear interpolation
        (as in :func:`gridtools.sim.mix.mix`).
    reference_xy_m : numpy.ndarray of float64, shape (2,), optional
        Position of an in-tank reference electrode. Its potential (same
        dipole model, per fish) is subtracted from every channel; None means
        a distant reference. The subtraction is linear, so it acts per fish
        and re-rendering stays exact.
    """

    samplerate_hz: float
    n_samples: int
    electrode_xy_m: FloatArray
    channel_gain: FloatArray
    shared_drift: DriftKnots
    fish: tuple[FishModel, ...]
    gain_update_hz: float = DEFAULT_GAIN_UPDATE_HZ
    reference_xy_m: FloatArray | None = None
    _checkpoints: dict[int, FloatArray] = field(
        default_factory=dict, init=False, repr=False
    )
    _gain_tracks: dict[int, GainTrack] = field(
        default_factory=dict, init=False, repr=False
    )
    _block_cache: "OrderedDict[tuple[int, int], FloatArray]" = field(
        default_factory=OrderedDict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        electrodes = np.asarray(self.electrode_xy_m, dtype=np.float64)
        channel = np.asarray(self.channel_gain, dtype=np.float64)
        if electrodes.ndim != 2 or electrodes.shape[1] != 2:
            msg = f"electrode_xy_m must have shape (E, 2), got {electrodes.shape}"
            raise ValueError(msg)
        if channel.shape != (electrodes.shape[0],):
            msg = f"channel_gain must have shape ({electrodes.shape[0]},)"
            raise ValueError(msg)
        object.__setattr__(self, "electrode_xy_m", electrodes)
        object.__setattr__(self, "channel_gain", channel)
        if self.reference_xy_m is not None:
            reference = np.asarray(self.reference_xy_m, dtype=np.float64)
            if reference.shape != (2,):
                msg = f"reference_xy_m must have shape (2,), got {reference.shape}"
                raise ValueError(msg)
            object.__setattr__(self, "reference_xy_m", reference)
        for i, fish in enumerate(self.fish):
            if fish.fish_id != i:
                msg = f"fish must be ordered by id, got id {fish.fish_id} at {i}"
                raise ValueError(msg)

    @property
    def n_electrodes(self) -> int:
        """Number of electrodes."""
        return int(self.electrode_xy_m.shape[0])

    @property
    def decimation(self) -> int:
        """Samples between exact gain evaluations (as in ``mix``)."""
        return max(1, math.floor(self.samplerate_hz / self.gain_update_hz))

    @cached_property
    def chirps(self) -> dict[int, ChirpPlacement]:
        """All placed chirps by id."""
        return {c.chirp_id: c for fish in self.fish for c in fish.chirps}

    # -- frequency ---------------------------------------------------------

    def base_frequency_at(self, fish_id: int, times_s: npt.ArrayLike) -> FloatArray:
        """Baseline EODf (no chirps) of a fish at arbitrary times, in Hz.

        ``(baseline + drift(t) + sum(ramps(t))) * (1 + shared_drift(t))``.
        """
        fish = self.fish[fish_id]
        times = np.asarray(times_s, dtype=np.float64)
        freq = fish.drift(times)
        freq += fish.baseline_hz
        for ramp in fish.ramps:
            freq += ramp(times)
        freq *= 1.0 + self.shared_drift(times)
        return freq

    def base_frequency(self, fish_id: int, start: int, stop: int) -> FloatArray:
        """Baseline EODf on samples ``start:stop`` in Hz."""
        return self.base_frequency_at(
            fish_id, np.arange(start, stop, dtype=np.float64) / self.samplerate_hz
        )

    def frequency(
        self,
        fish_id: int,
        start: int,
        stop: int,
        chirps: ChirpSelection = "all",
    ) -> FloatArray:
        """Instantaneous EODf (baseline plus included chirps) on ``start:stop``."""
        self._check_span(start, stop)
        freq = self.base_frequency(fish_id, start, stop)
        for chirp in self._selected(fish_id, chirps):
            lo, hi = max(start, chirp.start), min(stop, chirp.stop)
            if lo < hi:
                freq[lo - start : hi - start] += chirp.delta_f_hz[
                    lo - chirp.start : hi - chirp.start
                ]
        return freq

    # -- phase ---------------------------------------------------------------

    def _block_cycles(self, fish_id: int, block: int, offset: float) -> FloatArray:
        """Baseline cycles (Hz x samples) on block ``block`` plus the next sample.

        The last few blocks are cached (re-rendering short spans, e.g. for
        footprints, then evaluates the baseline splines only once).
        """
        key = (fish_id, block)
        cached = self._block_cache.get(key)
        if cached is not None and cached[0] == offset:
            self._block_cache.move_to_end(key)
            return cached
        cycles = self._compute_block_cycles(fish_id, block, offset)
        self._block_cache[key] = cycles
        if len(self._block_cache) > _BLOCK_CACHE_SIZE:
            self._block_cache.popitem(last=False)
        return cycles

    def _compute_block_cycles(
        self, fish_id: int, block: int, offset: float
    ) -> FloatArray:
        first = block * BLOCK
        last = min(first + BLOCK, self.n_samples - 1)  # inclusive
        freq = self.base_frequency(fish_id, first, last + 1)
        f_ref = self.fish[fish_id].baseline_hz
        residual = freq - f_ref
        cumulative = np.empty(residual.size, dtype=np.float64)
        cumulative[0] = 0.0
        np.cumsum(0.5 * (residual[:-1] + residual[1:]), out=cumulative[1:])
        ramp = np.arange(residual.size, dtype=np.float64) * f_ref
        return offset + (ramp + cumulative)

    def _checkpoints_of(self, fish_id: int) -> FloatArray:
        if fish_id not in self._checkpoints:
            n_blocks = -(-self.n_samples // BLOCK)
            points = np.empty(n_blocks, dtype=np.float64)
            points[0] = 0.0
            for block in range(n_blocks - 1):
                cycles = self._block_cycles(fish_id, block, points[block])
                points[block + 1] = cycles[BLOCK]
            self._checkpoints[fish_id] = points
        return self._checkpoints[fish_id]

    def base_cycles(self, fish_id: int, start: int, stop: int) -> FloatArray:
        """Integrated baseline frequency ``C_base`` on ``start:stop`` (Hz x samples)."""
        self._check_span(start, stop)
        points = self._checkpoints_of(fish_id)
        out = np.empty(stop - start, dtype=np.float64)
        if stop == start:
            return out
        for block in range(start // BLOCK, (stop - 1) // BLOCK + 1):
            first = block * BLOCK
            cycles = self._block_cycles(fish_id, block, points[block])
            lo, hi = max(start, first), min(stop, first + BLOCK)
            out[lo - start : hi - start] = cycles[lo - first : hi - first]
        return out

    def phase(
        self,
        fish_id: int,
        start: int,
        stop: int,
        chirps: ChirpSelection = "all",
        phase: PhaseMode = "aligned",
    ) -> FloatArray:
        """Unwrapped phase of the fundamental on ``start:stop`` in radians.

        See the module docstring for ``chirps`` and ``phase``.
        """
        if phase not in ("aligned", "natural"):
            msg = f"phase must be 'aligned' or 'natural', got {phase!r}"
            raise ValueError(msg)
        cycles = self.base_cycles(fish_id, start, stop)
        included = {c.chirp_id for c in self._selected(fish_id, chirps)}
        for chirp in self.fish[fish_id].chirps:
            first, local = chirp.local_cycles
            if first >= stop:
                break
            if chirp.chirp_id in included:
                term = local
            elif phase == "aligned":
                term = _catch_up(local, self.samplerate_hz)
            else:
                continue
            end = first + term.size  # samples >= end hold the total
            lo, hi = max(start, first), min(stop, end)
            if lo < hi:
                cycles[lo - start : hi - start] += term[lo - first : hi - first]
            lo = max(start, end)
            if lo < stop:
                cycles[lo - start :] += term[-1]
        cycles *= 2.0 * np.pi / self.samplerate_hz
        cycles += self.fish[fish_id].phase0_rad
        return cycles

    def envelope(
        self,
        fish_id: int,
        start: int,
        stop: int,
        chirps: ChirpSelection = "all",
    ) -> FloatArray:
        """Amplitude envelope (EOD amplitude times chirp AM) on ``start:stop``."""
        self._check_span(start, stop)
        env = np.full(stop - start, self.fish[fish_id].amplitude, dtype=np.float64)
        for chirp in self._selected(fish_id, chirps):
            lo, hi = max(start, chirp.start), min(stop, chirp.stop)
            if lo < hi:
                env[lo - start : hi - start] *= chirp.amp_factor[
                    lo - chirp.start : hi - chirp.start
                ]
        return env

    def waveform(
        self,
        fish_id: int,
        start: int,
        stop: int,
        chirps: ChirpSelection = "all",
        phase: PhaseMode = "aligned",
    ) -> Float32Array:
        """EOD waveform (before field and channel gains) on ``start:stop``."""
        theta = self.phase(fish_id, start, stop, chirps, phase)
        if theta.size == 0:
            return np.empty(0, dtype=np.float32)
        env = self.envelope(fish_id, start, stop, chirps)
        return harmonic_waveform(theta, self.fish[fish_id].harmonics, env)

    # -- field ---------------------------------------------------------------

    def gain_track(self, fish_id: int) -> GainTrack:
        """Dipole gains of a fish along its path, minus the reference (cached)."""
        if fish_id not in self._gain_tracks:
            fish = self.fish[fish_id]
            track = gain_track(
                fish.path.times_s,
                fish.path.xy_m,
                fish.path.heading_rad,
                self.electrode_xy_m,
                fish.dipole,
            )
            if self.reference_xy_m is not None:
                reference = gain_track(
                    fish.path.times_s,
                    fish.path.xy_m,
                    fish.path.heading_rad,
                    self.reference_xy_m[None, :],
                    fish.dipole,
                )
                track = GainTrack(
                    times_s=track.times_s, gains=track.gains - reference.gains
                )
            self._gain_tracks[fish_id] = track
        return self._gain_tracks[fish_id]

    def gains(self, fish_id: int, start: int, stop: int) -> FloatArray:
        """Field gains on ``start:stop``, interpolated exactly as ``mix`` does."""
        self._check_span(start, stop)
        if stop == start:
            return np.empty((0, self.n_electrodes), dtype=np.float64)
        return _track_chunk(
            self.gain_track(fish_id),
            start,
            stop,
            self.samplerate_hz,
            0.0,
            self.decimation,
        )

    def render(
        self,
        fish_id: int,
        start: int,
        stop: int,
        *,
        chirps: ChirpSelection = "all",
        phase: PhaseMode = "aligned",
        electrodes: Sequence[int] | None = None,
    ) -> Float32Array:
        """A fish's contribution to the recording on samples ``start:stop``.

        ``channel_gain * gains * waveform`` in float64, returned as float32;
        the same operations ``mix`` applies to this fish.

        Returns
        -------
        numpy.ndarray of float32, shape (stop - start, len(electrodes))
        """
        wave = self.waveform(fish_id, start, stop, chirps, phase)
        contribution = self.gains(fish_id, start, stop)
        contribution *= wave[:, None]
        contribution *= self.channel_gain
        if electrodes is not None:
            contribution = contribution[:, list(electrodes)]
        return contribution.astype(np.float32)

    # -- helpers -------------------------------------------------------------

    def _check_span(self, start: int, stop: int) -> None:
        if not 0 <= start <= stop <= self.n_samples:
            msg = (
                f"span [{start}, {stop}) outside the scene of {self.n_samples} samples"
            )
            raise IndexError(msg)

    def _selected(self, fish_id: int, chirps: ChirpSelection) -> list[ChirpPlacement]:
        """Chirps of a fish that are rendered under the selection."""
        own = self.fish[fish_id].chirps
        if isinstance(chirps, str):
            if chirps == "all":
                return list(own)
            if chirps == "none":
                return []
            msg = f"chirps must be 'all', 'none' or a collection of ids, got {chirps!r}"
            raise ValueError(msg)
        omit = {int(c) for c in chirps}
        unknown = omit - set(self.chirps)
        if unknown:
            msg = f"unknown chirp id(s): {sorted(unknown)}"
            raise KeyError(msg)
        return [c for c in own if c.chirp_id not in omit]


def _catch_up(local_cycles: FloatArray, samplerate_hz: float) -> FloatArray:
    """Linear phase catch-up replacing an omitted chirp (``phase="aligned"``).

    Ramps from 0 to the fractional part (nearest integer removed) of the
    chirp's total over the same samples, in the same units as
    ``local_cycles``.
    """
    total = float(local_cycles[-1])
    turns = total / samplerate_hz
    remainder = (turns - round(turns)) * samplerate_hz
    steps = local_cycles.size - 1
    return remainder * (np.arange(local_cycles.size, dtype=np.float64) / steps)
