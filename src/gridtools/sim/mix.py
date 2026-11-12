"""Mix fish EODs, field gains and noise into a multi-electrode recording.

For each chunk of samples:

.. math::

    x_e(t) = \\mathrm{clip}\\Big(c_e \\big[\\textstyle\\sum_i
    \\mathrm{EOD}_i(t)\\, g_{ie}(t) + \\sum_j b_{je}(t)\\big]\\Big),

where :math:`g_{ie}` are field gains (full-rate arrays or
:class:`~gridtools.sim.field.GainTrack` objects interpolated per chunk),
:math:`b_j` additive components (coloured noise arrays, hum, real
background, ...), :math:`c_e` the channel gains and the clipping optional.
Accumulation is in float64 per chunk; the output is float32.

Working in chunks bounds memory to the output plus one chunk of
temporaries: gains are never materialised at the signal rate, and ``out``
may be one of the additive arrays so the noise buffer is reused for the
result.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from gridtools.sim._common import Float32Array, require_positive
from gridtools.sim.field import GainTrack
from gridtools.sim.noise import ClipParams

__all__ = ["Renderable", "mix"]

DEFAULT_CHUNK_SAMPLES = 1 << 18


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
    chunk_samples: int = DEFAULT_CHUNK_SAMPLES,
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
    chunk_samples : int
        Samples per chunk (default 2**18). The result does not depend on it.
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
    if len(eods) != len(gains):
        msg = f"got {len(eods)} EODs but {len(gains)} gains"
        raise ValueError(msg)
    waves = [np.asarray(e) for e in eods]
    n_samples, n_electrodes = _shape(waves, gains, additive, out)

    for i, (wave, gain) in enumerate(zip(waves, gains, strict=True)):
        if wave.shape != (n_samples,):
            msg = f"eods[{i}] must have shape ({n_samples},), got {wave.shape}"
            raise ValueError(msg)
        gain_shape = (
            (gain.times_s.size, gain.n_electrodes)
            if isinstance(gain, GainTrack)
            else np.shape(gain)
        )
        if isinstance(gain, GainTrack):
            t_first = start_time_s
            t_last = start_time_s + (n_samples - 1) / samplerate_hz
            if t_first < gain.times_s[0] or t_last > gain.times_s[-1]:
                msg = f"gains[{i}] does not cover [{t_first}, {t_last}] s"
                raise ValueError(msg)
            if gain.n_electrodes != n_electrodes:
                msg = f"gains[{i}] has {gain.n_electrodes} electrodes, expected {n_electrodes}"
                raise ValueError(msg)
        elif gain_shape != (n_samples, n_electrodes):
            msg = f"gains[{i}] must have shape ({n_samples}, {n_electrodes}), got {gain_shape}"
            raise ValueError(msg)
    for j, component in enumerate(additive):
        if tuple(component.shape) != (n_samples, n_electrodes):
            msg = f"additive[{j}] must have shape ({n_samples}, {n_electrodes}), got {component.shape}"
            raise ValueError(msg)

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

    needs_times = any(isinstance(g, GainTrack) for g in gains)
    for start in range(0, n_samples, chunk_samples):
        stop = min(n_samples, start + chunk_samples)
        acc = np.zeros((stop - start, n_electrodes), dtype=np.float64)
        times_s = (
            start_time_s + np.arange(start, stop, dtype=np.float64) / samplerate_hz
            if needs_times
            else None
        )
        for wave, gain in zip(waves, gains, strict=True):
            if isinstance(gain, GainTrack):
                assert times_s is not None
                gain_chunk = gain.at(times_s)
            else:
                gain_chunk = gain[start:stop]
            acc += wave[start:stop, None].astype(np.float64) * gain_chunk
        for component in additive:
            if isinstance(component, np.ndarray):
                acc += component[start:stop]
            else:
                acc += component.render(start, stop)
        if channel is not None:
            acc *= channel
        if clip is not None:
            np.clip(acc, -clip.limit, clip.limit, out=acc)
        out[start:stop] = acc
    return out


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
