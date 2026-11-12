"""Mix fish EODs, field gains and noise into a multi-electrode recording.

For each chunk of samples:

.. math::

    x_e(t) = \\mathrm{clip}\\Big(c_e \\big[\\textstyle\\sum_i
    \\mathrm{EOD}_i(t)\\, g_{ie}(t) + \\sum_j b_{je}(t)\\big]\\Big),

where :math:`g_{ie}` are field gains (full-rate arrays or
:class:`~gridtools.sim.field.GainTrack` objects), :math:`b_j` additive
components (coloured noise arrays, hum, real background, ...), :math:`c_e`
the channel gains and the clipping optional. Accumulation is in float64 per
chunk; the output is float32.

Gain tracks are evaluated with their cubic spline on a grid of every
``D``-th sample (``D = samplerate_hz // gain_update_hz``, 1 kHz by default)
and interpolated linearly in between. Gains vary on time-scales of 100 ms or
more (a fish at 0.5 m/s, 10 cm from an electrode), so this step changes the
gains by about 1e-5 of their peak (-100 dB) while avoiding a spline
evaluation per sample.
Pass ``gain_update_hz=None`` to evaluate the spline at every sample.

Working in chunks bounds memory to the output plus a few chunks of
temporaries: gains are never materialised at the signal rate, and ``out``
may be one of the additive arrays so the noise buffer is reused for the
result. The result does not depend on ``chunk_samples`` or ``workers``.
"""

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from gridtools.sim._common import Float32Array, FloatArray, require_positive
from gridtools.sim.field import GainTrack
from gridtools.sim.noise import ClipParams

__all__ = ["Renderable", "mix"]

DEFAULT_CHUNK_SAMPLES = 1 << 17
DEFAULT_GAIN_UPDATE_HZ = 1000.0


@runtime_checkable
class Renderable(Protocol):
    """An additive component that renders any span of samples on demand."""

    @property
    def shape(self) -> tuple[int, int]:
        """Full shape (n_samples, n_electrodes)."""
        ...

    def render(self, start: int, stop: int) -> npt.NDArray[np.floating]:
        """Samples ``start:stop`` on all electrodes, shape (stop - start, E)."""
        ...


def mix(
    eods: Sequence[npt.ArrayLike],
    gains: Sequence[npt.NDArray[np.floating] | GainTrack],
    *,
    samplerate_hz: float,
    additive: Sequence[npt.NDArray[np.floating] | Renderable] = (),
    channel_gain: npt.ArrayLike | None = None,
    clip: ClipParams | None = None,
    start_time_s: float = 0.0,
    gain_update_hz: float | None = DEFAULT_GAIN_UPDATE_HZ,
    chunk_samples: int = DEFAULT_CHUNK_SAMPLES,
    workers: int = 1,
    out: Float32Array | None = None,
) -> Float32Array:
    """Combine per-fish EODs and gains plus additive components.

    Parameters
    ----------
    eods : sequence of array_like, each shape (n,)
        EOD waveform of each fish.
    gains : sequence of arrays of shape (n, E) or GainTrack
        Field gains of each fish, either at every sample or as a
        :class:`~gridtools.sim.field.GainTrack` evaluated at sample times
        ``start_time_s + i / samplerate_hz``.
    samplerate_hz : float
        Sampling rate in Hz.
    additive : sequence of arrays of shape (n, E) or Renderable
        Components added before the channel gain (noise, hum, background).
    channel_gain : array_like of shape (E,), optional
        Per-electrode gain applied to the sum (gain mismatch times fault
        factors). Omit for unity gain.
    clip : ClipParams, optional
        Clip the result symmetrically. Omit for no clipping.
    start_time_s : float
        Time of sample 0 in seconds, for evaluating gain tracks.
    gain_update_hz : float or None
        Rate at which gain tracks are evaluated exactly before linear
        interpolation to the signal rate (default 1000). ``None`` evaluates
        them at every sample.
    chunk_samples : int
        Samples per chunk (default 2**17).
    workers : int
        Number of threads working on chunks in parallel (default 1). numpy
        releases the GIL in the arithmetic, so this scales on idle cores.
    out : numpy.ndarray of float32, shape (n, E), optional
        Array to write the result into (it is overwritten). It may be one of
        the ``additive`` arrays: each chunk is read before it is written.

    Returns
    -------
    numpy.ndarray of float32, shape (n, E)
        The recording (``out`` if given).
    """
    require_positive(samplerate_hz, "samplerate_hz")
    if isinstance(chunk_samples, bool) or chunk_samples < 1:
        msg = f"chunk_samples must be >= 1, got {chunk_samples!r}"
        raise ValueError(msg)
    if isinstance(workers, bool) or workers < 1:
        msg = f"workers must be >= 1, got {workers!r}"
        raise ValueError(msg)
    if gain_update_hz is None:
        decimation = 1
    else:
        require_positive(gain_update_hz, "gain_update_hz")
        decimation = max(1, math.floor(samplerate_hz / gain_update_hz))
    if len(eods) != len(gains):
        msg = f"got {len(eods)} EODs but {len(gains)} gains"
        raise ValueError(msg)
    waves = [np.asarray(e) for e in eods]
    n_samples, n_electrodes = _shape(waves, gains, additive, out)
    _validate(
        waves, gains, additive, n_samples, n_electrodes, samplerate_hz, start_time_s
    )

    channel = None
    if channel_gain is not None:
        channel = np.asarray(channel_gain, dtype=np.float64)
        if channel.shape != (n_electrodes,):
            msg = f"channel_gain must have shape ({n_electrodes},), got {channel.shape}"
            raise ValueError(msg)

    if out is None:
        out = np.empty((n_samples, n_electrodes), dtype=np.float32)
    elif out.shape != (n_samples, n_electrodes) or out.dtype != np.float32:
        msg = f"out must be float32 with shape ({n_samples}, {n_electrodes})"
        raise ValueError(msg)
    result = out

    def render_chunk(start: int) -> None:
        stop = min(n_samples, start + chunk_samples)
        acc = np.zeros((stop - start, n_electrodes), dtype=np.float64)
        for wave, gain in zip(waves, gains, strict=True):
            if isinstance(gain, GainTrack):
                contribution = _track_chunk(
                    gain, start, stop, samplerate_hz, start_time_s, decimation
                )
            else:
                contribution = np.array(gain[start:stop], dtype=np.float64)
            contribution *= wave[start:stop, None]
            acc += contribution
        for component in additive:
            if isinstance(component, np.ndarray):
                acc += component[start:stop]
            else:
                acc += component.render(start, stop)
        if channel is not None:
            acc *= channel
        if clip is not None:
            np.clip(acc, -clip.limit, clip.limit, out=acc)
        result[start:stop] = acc

    starts = range(0, n_samples, chunk_samples)
    if workers == 1:
        for start in starts:
            render_chunk(start)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(render_chunk, starts):
                pass
    return result


def _track_chunk(
    track: GainTrack,
    start: int,
    stop: int,
    samplerate_hz: float,
    start_time_s: float,
    decimation: int,
) -> FloatArray:
    """Gains for samples ``start:stop`` as a fresh float64 (k, E) array.

    Knots sit at absolute sample indices that are multiples of
    ``decimation``, so every sample's value depends only on its index, not on
    the chunking.
    """
    if decimation == 1:
        times_s = (
            start_time_s + np.arange(start, stop, dtype=np.float64) / samplerate_hz
        )
        return track.evaluate(times_s)
    first_knot = start // decimation
    last_knot = -(-stop // decimation)  # ceil
    knot_index = np.arange(first_knot, last_knot + 1, dtype=np.float64) * decimation
    knots = track.evaluate(start_time_s + knot_index / samplerate_hz)
    weights = np.arange(decimation, dtype=np.float64) / decimation
    slopes = knots[1:] - knots[:-1]
    block = knots[:-1, None, :] + weights[None, :, None] * slopes[:, None, :]
    offset = start - first_knot * decimation
    return block.reshape(-1, knots.shape[1])[offset : offset + stop - start]


def _validate(
    waves: Sequence[np.ndarray],
    gains: Sequence[npt.NDArray[np.floating] | GainTrack],
    additive: Sequence[npt.NDArray[np.floating] | Renderable],
    n_samples: int,
    n_electrodes: int,
    samplerate_hz: float,
    start_time_s: float,
) -> None:
    for i, (wave, gain) in enumerate(zip(waves, gains, strict=True)):
        if wave.shape != (n_samples,):
            msg = f"eods[{i}] must have shape ({n_samples},), got {wave.shape}"
            raise ValueError(msg)
        if isinstance(gain, GainTrack):
            t_first = start_time_s
            t_last = start_time_s + (n_samples - 1) / samplerate_hz
            if t_first < gain.times_s[0] or t_last > gain.times_s[-1]:
                msg = f"gains[{i}] does not cover [{t_first}, {t_last}] s"
                raise ValueError(msg)
            if gain.n_electrodes != n_electrodes:
                msg = (
                    f"gains[{i}] has {gain.n_electrodes} electrodes, "
                    f"expected {n_electrodes}"
                )
                raise ValueError(msg)
        elif np.shape(gain) != (n_samples, n_electrodes):
            msg = (
                f"gains[{i}] must have shape ({n_samples}, {n_electrodes}), "
                f"got {np.shape(gain)}"
            )
            raise ValueError(msg)
    for j, component in enumerate(additive):
        if tuple(component.shape) != (n_samples, n_electrodes):
            msg = (
                f"additive[{j}] must have shape ({n_samples}, {n_electrodes}), "
                f"got {component.shape}"
            )
            raise ValueError(msg)


def _shape(
    waves: Sequence[np.ndarray],
    gains: Sequence[npt.NDArray[np.floating] | GainTrack],
    additive: Sequence[npt.NDArray[np.floating] | Renderable],
    out: Float32Array | None,
) -> tuple[int, int]:
    """Infer (n_samples, n_electrodes) from the inputs."""
    if additive:
        n, e = additive[0].shape
        return int(n), int(e)
    if out is not None and out.ndim == 2:
        return int(out.shape[0]), int(out.shape[1])
    if waves:
        n = int(waves[0].shape[0]) if waves[0].ndim == 1 else 0
        first = gains[0]
        if isinstance(first, GainTrack):
            e = first.n_electrodes
        else:
            gain_shape = np.shape(first)
            e = int(gain_shape[1]) if len(gain_shape) == 2 else 0
        if n >= 1 and e >= 1:
            return n, e
    msg = "cannot infer the recording shape: pass at least one fish or component"
    raise ValueError(msg)
