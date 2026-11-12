"""Ground-truth chirp boxes from each chirp's footprint in a spectrogram.

A chirp's footprint is where it changes the spectrogram (plan decision 7).
For one chirp:

1. Render the emitter alone over the chirp's support plus the analysis
   window and a margin on both sides, once with all its chirps and once
   with this chirp omitted (``phase="aligned"``: outside the support the
   two signals are identical, so other chirps of the fish cancel).
2. Compute both spectrograms with the caller's front end and take
   ``D = |P_with - P_without|`` per time-frequency cell. ``D`` holds the
   energy along the excursion and the power the chirp removes from the
   baseline band, smeared by the analysis window.
3. Keep the cells with ``D >= max(D) * 10**(-threshold_db / 10)`` (default
   30 dB) that are
   connected (8-neighbourhood) to the chirp's trajectory (baseline plus
   excursion, mapped to cells), so isolated sidelobes do not enlarge it.
4. The box spans these cells, with its edges at cell edges (half a frame
   and half a bin beyond the outermost cell centres).

Visibility compares ``D`` with a background spectrogram ``B`` over the same
span, electrodes and aggregation, as ``10 log10(sum D / sum B)``:
``snr_db`` over the whole footprint; ``snr_baseline_db`` over its cells in
the emitter's baseline band (within ``band_hz`` or three bins of the
emitter's baseline: the power the chirp takes out of the band); and
``snr_excursion_db`` over its cells outside every fish's baseline band (the
excursion where only noise competes). ``visible`` is true when either part
reaches ``visible_snr_db``: either cue reveals a chirp, and a single sum
over the whole footprint is dominated by strong baseline-band cells (in a
band crowded with other fish the whole footprint looks invisible although
the excursion is plain to see). ``visible_fraction`` is the fraction of
footprint cells where ``D > B``. By default ``B`` is everything except the
emitter (recording minus the emitter's rendering: other fish, noise, hum,
clicks). The emitter's own baseline is the canvas the chirp is drawn on, not
background. The alternative ``background="recording"`` (recording minus
only this chirp) keeps the emitter's baseline in ``B``; its strong baseline
cells then dominate the sums and hold ``snr_db`` near -2 to -5 dB even for a
clean, isolated chirp.

The front end is a parameter (:data:`SpectrogramFn`), so a detector's own
spectrogram can be used; :class:`HannSpectrogram` is a plain numpy/scipy
implementation (electrode-summed power, Hann window). By default the box
covers the fundamental; the second harmonic's footprint is returned too
when it lies in ``harmonic_band`` (for ignore regions, not labels).
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.fft
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

from gridtools.sim._common import FloatArray

__all__ = [
    "Footprint",
    "HannSpectrogram",
    "SpectrogramFn",
    "chirp_footprint",
    "scene_footprints",
]

#: ``spectrogram_fn(signal (n, E), samplerate_hz, t0_s)`` returns
#: ``(power (F, T), freqs_hz (F,), times_s (T,))``: linear, electrode-summed
#: power and the centre time of each frame (``t0_s`` is the time of
#: ``signal[0]``).
SpectrogramFn = Callable[
    [npt.NDArray[np.floating], float, float],
    tuple[FloatArray, FloatArray, FloatArray],
]

_FRAME_CHUNK = 512
NAN_BOX = (math.nan, math.nan, math.nan, math.nan)


@dataclass(frozen=True)
class HannSpectrogram:
    """Electrode-summed power spectrogram with a Hann window.

    Window ``window_s`` and hop ``hop_s`` are in seconds and converted per
    call (``round(window_s * fs)``, ``round(hop_s * fs)``), so recordings of
    any sampling rate get the same time and frequency resolution. Periodic
    Hann window, one-sided power spectral density per electrode
    (``specgram`` scaling), summed over electrodes, restricted to
    ``f_min_hz``-``f_max_hz``. Frame times are the window centres.

    Attributes
    ----------
    window_s : float
        Window length in s (default 0.2048: 4096 samples at 20 kHz).
    hop_s : float
        Frame step in s (default 0.005).
    f_min_hz, f_max_hz : float
        Band kept.
    workers : int
        Threads for the FFTs.
    """

    window_s: float = 0.2048
    hop_s: float = 0.005
    f_min_hz: float = 500.0
    f_max_hz: float = 1500.0
    workers: int = 1

    def samples(self, samplerate_hz: float) -> tuple[int, int]:
        """Window and hop in samples at a sampling rate."""
        return (
            max(8, round(self.window_s * samplerate_hz)),
            max(1, round(self.hop_s * samplerate_hz)),
        )

    def __call__(
        self, signal: npt.NDArray[np.floating], samplerate_hz: float, t0_s: float
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Spectrogram of ``signal`` (n, E); see :data:`SpectrogramFn`."""
        nfft, hop = self.samples(samplerate_hz)
        x = np.asarray(signal)
        if x.ndim == 1:
            x = x[:, None]
        n, n_electrodes = x.shape
        freqs = np.fft.rfftfreq(nfft, 1.0 / samplerate_hz)
        lo = int(np.searchsorted(freqs, self.f_min_hz))
        hi = int(np.searchsorted(freqs, self.f_max_hz, side="right"))
        if n < nfft:
            return np.zeros((hi - lo, 0)), freqs[lo:hi], np.zeros(0)
        n_frames = 1 + (n - nfft) // hop
        window = np.hanning(nfft + 1)[:nfft]
        scale = 2.0 / (samplerate_hz * float(np.sum(window**2)))
        window32 = window.astype(np.float32)
        power = np.zeros((n_frames, hi - lo), dtype=np.float64)
        for e in range(n_electrodes):
            channel = np.ascontiguousarray(x[:, e], dtype=np.float32)
            frames = sliding_window_view(channel, nfft)[::hop]
            for first in range(0, n_frames, _FRAME_CHUNK):
                block = frames[first : first + _FRAME_CHUNK] * window32
                spectrum = np.asarray(
                    scipy.fft.rfft(block, axis=1, workers=self.workers)
                )[:, lo:hi]
                power[first : first + block.shape[0]] += (
                    np.real(spectrum).astype(np.float64) ** 2
                    + np.imag(spectrum).astype(np.float64) ** 2
                )
        power *= scale
        times = t0_s + (np.arange(n_frames) * hop + nfft / 2) / samplerate_hz
        return power.T, freqs[lo:hi], times


@dataclass(frozen=True)
class Footprint:
    """A chirp's footprint, box and visibility.

    Attributes
    ----------
    chirp_id, emitter_id : int
    box : (float, float, float, float)
        ``(t0_s, t1_s, f0_hz, f1_hz)`` of the fundamental's footprint at
        cell edges; NaN if the chirp leaves no footprint in the band.
    snr_db : float
        ``10 log10(sum D / sum B)`` over the whole footprint.
    snr_baseline_db : float
        The same over the footprint cells in the emitter's baseline band;
        NaN if there are none.
    snr_excursion_db : float
        The same over the footprint cells outside every fish's baseline
        band; NaN if there are none.
    visible_fraction : float
        Fraction of footprint cells where ``D > B``.
    visible : bool
        ``max(snr_baseline_db, snr_excursion_db) >= visible_snr_db``.
    harmonic_box : (float, float, float, float)
        Box of the second harmonic's footprint, NaN if none in the band.
    energy_fraction : float
        Fraction of ``sum D`` (over the whole span and band) inside ``box``.
    n_cells : int
        Number of footprint cells.
    arrays : dict
        ``times_s``, ``freqs_hz``, ``d``, ``b``, ``mask`` and
        ``harmonic_mask`` when requested (``keep_arrays=True``), else empty.
    """

    chirp_id: int
    emitter_id: int
    box: tuple[float, float, float, float]
    snr_db: float
    snr_baseline_db: float
    snr_excursion_db: float
    visible_fraction: float
    visible: bool
    harmonic_box: tuple[float, float, float, float]
    energy_fraction: float
    n_cells: int
    arrays: dict[str, np.ndarray] = field(default_factory=dict, repr=False)


def _cells(
    times_s: FloatArray, freqs_hz: FloatArray, t_s: FloatArray, f_hz: FloatArray
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Nearest (row, column) cells of points inside the spectrogram."""
    if times_s.size == 0 or freqs_hz.size == 0:
        empty = np.zeros(0, dtype=np.intp)
        return empty, empty
    dt = times_s[1] - times_s[0] if times_s.size > 1 else 1.0
    df = freqs_hz[1] - freqs_hz[0] if freqs_hz.size > 1 else 1.0
    col = np.rint((t_s - times_s[0]) / dt).astype(np.intp)
    row = np.rint((f_hz - freqs_hz[0]) / df).astype(np.intp)
    inside = (
        (col >= 0)
        & (col < times_s.size)
        & (row >= 0)
        & (row < freqs_hz.size)
        & (np.abs(t_s - times_s[np.clip(col, 0, times_s.size - 1)]) <= dt)
    )
    return row[inside], col[inside]


def _component_mask(
    candidate: npt.NDArray[np.bool_],
    seed_rows: npt.NDArray[np.intp],
    seed_cols: npt.NDArray[np.intp],
) -> npt.NDArray[np.bool_]:
    """Union of the 8-connected components of ``candidate`` hit by the seeds."""
    labels = np.asarray(
        ndimage.label(candidate, structure=np.ones((3, 3), dtype=bool))[0]  # pyright: ignore[reportIndexIssue]
    )
    hit = np.unique(labels[seed_rows, seed_cols])
    hit = hit[hit > 0]
    return np.isin(labels, hit) if hit.size else np.zeros_like(candidate)


def _box(
    mask: npt.NDArray[np.bool_], times_s: FloatArray, freqs_hz: FloatArray
) -> tuple[float, float, float, float]:
    if not mask.any():
        return NAN_BOX
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    dt = times_s[1] - times_s[0] if times_s.size > 1 else 0.0
    df = freqs_hz[1] - freqs_hz[0] if freqs_hz.size > 1 else 0.0
    return (
        float(times_s[cols[0]] - dt / 2),
        float(times_s[cols[-1]] + dt / 2),
        float(freqs_hz[rows[0]] - df / 2),
        float(freqs_hz[rows[-1]] + df / 2),
    )


def _ratio_db(d: FloatArray, b: FloatArray) -> float:
    """``10 log10(sum d / sum b)``; NaN for no cells, +inf without background."""
    if d.size == 0:
        return math.nan
    total_d, total_b = float(d.sum()), float(b.sum())
    if total_b <= 0.0:
        return math.inf if total_d > 0.0 else math.nan
    return 10.0 * math.log10(max(total_d, 1e-300) / total_b)


def _in_box(
    box: tuple[float, float, float, float], times_s: FloatArray, freqs_hz: FloatArray
) -> npt.NDArray[np.bool_]:
    t0, t1, f0, f1 = box
    in_t = (times_s >= t0) & (times_s <= t1)
    in_f = (freqs_hz >= f0) & (freqs_hz <= f1)
    return in_f[:, None] & in_t[None, :]


def chirp_footprint(
    scene: Any,
    chirp_id: int,
    spectrogram_fn: SpectrogramFn | None = None,
    *,
    threshold_db: float = 30.0,
    visible_snr_db: float = 0.0,
    window_s: float | None = None,
    margin_s: float = 0.05,
    harmonic_band: tuple[float, float] | None = (500.0, 1500.0),
    recording: npt.NDArray[np.floating] | None = None,
    background: str = "others",
    band_hz: float = 15.0,
    electrodes: Sequence[int] | None = None,
    keep_arrays: bool = False,
) -> Footprint:
    """Footprint, box and visibility of one chirp (see the module docstring).

    Parameters
    ----------
    scene : Scene, SceneModel or path
        The scene (a :class:`~gridtools.sim.scene.Scene`, its model, or a
        scene directory).
    chirp_id : int
        Chirp to measure.
    spectrogram_fn : callable, optional
        Front end (:data:`SpectrogramFn`); default
        ``HannSpectrogram()`` (0.2048 s, hop 5 ms, 500-1500 Hz).
    threshold_db : float
        Footprint cells reach within this many dB of the largest difference
        (default 30: with a 0.2-s window the box then holds >= 99.5 % of the
        difference energy; at 20 dB 1-2.5 % and at 25 dB up to 1.5 % lie
        outside).
    visible_snr_db : float
        SNR threshold of ``visible``.
    window_s : float, optional
        Longest analysis window of the front end in s, which sets the span
        around the chirp; default ``spectrogram_fn.window_s`` (0.2048 s if
        the front end has none).
    margin_s : float
        Extra time on both sides of the span.
    harmonic_band : (float, float) or None
        Also measure the second harmonic when it falls in this band (and in
        the spectrogram); None skips it.
    recording : array, optional
        The scene's recording (n, E), for the visibility. Taken from a
        ``Scene`` or read from a scene directory if omitted; with a bare
        ``SceneModel`` and no recording the visibility is NaN.
    background : {"others", "recording"}
        What the chirp is compared with. ``"others"`` (default): the
        recording minus the emitter's whole contribution (other fish, noise,
        hum, clicks). ``"recording"``: the recording minus only this chirp's
        contribution, so ``B`` also holds the emitter's own baseline band,
        which dominates the footprint's baseline cells and pins ``snr_db``
        near -2 to -5 dB even for a perfectly clean, isolated chirp.
    band_hz : float
        Half-width of a fish's baseline band for the visibility split (at
        least three frequency bins).
    electrodes : sequence of int, optional
        Channels to use (default all).
    keep_arrays : bool
        Return the spectrograms and masks in ``Footprint.arrays``.

    Returns
    -------
    Footprint
    """
    from gridtools.sim.scene import Scene, _as_model

    model = _as_model(scene)
    if recording is None:
        if isinstance(scene, Scene):
            recording = scene.recording
        elif isinstance(scene, str | Path):
            from gridtools.io import open_recording

            recording = np.asarray(
                open_recording(Path(scene) / "recording.wav").data, dtype=np.float32
            )
    fn: SpectrogramFn = HannSpectrogram() if spectrogram_fn is None else spectrogram_fn
    if window_s is None:
        window_s = float(getattr(fn, "window_s", 0.2048))
    chirp = model.chirps[int(chirp_id)]
    fish_id = chirp.fish_id
    fs = model.samplerate_hz
    pad = round((window_s + margin_s) * fs)
    start = max(0, chirp.support_start - pad)
    stop = min(model.n_samples, chirp.support_stop + pad)
    cols = list(electrodes) if electrodes is not None else None
    x_with = model.render(fish_id, start, stop, electrodes=cols).astype(np.float64)
    x_without = model.render(
        fish_id, start, stop, chirps=[chirp.chirp_id], phase="aligned", electrodes=cols
    ).astype(np.float64)
    t0 = start / fs
    p_with, freqs, times = fn(x_with, fs, t0)
    p_without, _, _ = fn(x_without, fs, t0)
    d = np.abs(np.asarray(p_with) - np.asarray(p_without))
    freqs = np.asarray(freqs, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    # trajectory of the chirp in cells: baseline + excursion over its support
    lo, hi = max(chirp.start, start), min(chirp.stop, stop)
    sample_t = np.arange(lo, hi, dtype=np.float64) / fs
    base = model.base_frequency_at(fish_id, sample_t)
    trajectory = base + chirp.delta_f_hz[lo - chirp.start : hi - chirp.start]

    empty = np.zeros_like(d, dtype=bool)
    peak = float(d.max()) if d.size else 0.0
    if peak <= 0.0:
        return Footprint(
            chirp_id=chirp.chirp_id,
            emitter_id=fish_id,
            box=NAN_BOX,
            snr_db=math.nan,
            snr_baseline_db=math.nan,
            snr_excursion_db=math.nan,
            visible_fraction=math.nan,
            visible=False,
            harmonic_box=NAN_BOX,
            energy_fraction=0.0,
            n_cells=0,
            arrays={"times_s": times, "freqs_hz": freqs} if keep_arrays else {},
        )
    candidate = d >= peak * 10.0 ** (-threshold_db / 10.0)
    rows, cols_ = _cells(times, freqs, sample_t, trajectory)
    mask = _component_mask(candidate, rows, cols_) if rows.size else empty
    if not mask.any():  # trajectory outside the band: keep the strongest component
        r, c = np.unravel_index(int(np.argmax(d)), d.shape)
        mask = _component_mask(candidate, np.array([r]), np.array([c]))
    box = _box(mask, times, freqs)

    harmonic_mask = empty
    harmonic_box = NAN_BOX
    if harmonic_band is not None:
        rows2, cols2 = _cells(times, freqs, sample_t, 2.0 * trajectory)
        in_band = (freqs[rows2] >= harmonic_band[0]) & (
            freqs[rows2] <= harmonic_band[1]
        )
        rows2, cols2 = rows2[in_band], cols2[in_band]
        if rows2.size:
            region = np.zeros_like(d, dtype=bool)
            region[rows2, cols2] = True
            region = np.asarray(
                ndimage.binary_dilation(region, iterations=3), dtype=bool
            )
            # the harmonic lies above 1.5 x baseline and outside the fundamental's box
            allowed = (freqs >= 1.5 * float(base.min()))[:, None] & ~_in_box(
                box, times, freqs
            )
            region &= allowed
            local = float(d[region].max()) if region.any() else 0.0
            if local > 0.0:
                candidate2 = (d >= local * 10.0 ** (-threshold_db / 10.0)) & allowed
                harmonic_mask = _component_mask(candidate2, rows2, cols2)
                harmonic_box = _box(harmonic_mask, times, freqs)

    total = float(d.sum())
    energy_fraction = float(d[_in_box(box, times, freqs)].sum() / total)

    snr_db = snr_baseline_db = snr_excursion_db = math.nan
    visible_fraction = math.nan
    b = None
    if recording is not None:
        rest = np.asarray(recording[start:stop], dtype=np.float64)
        if cols is not None:
            rest = rest[:, cols]
        if background == "others":
            rest = rest - x_with
        elif background == "recording":
            rest = rest - (x_with - x_without)
        else:
            msg = f"background must be 'others' or 'recording', got {background!r}"
            raise ValueError(msg)
        b, _, _ = fn(rest, fs, t0)
        b = np.asarray(b)
        d_fp, b_fp = d[mask], b[mask]
        df = freqs[1] - freqs[0] if freqs.size > 1 else 1.0
        half_band = max(3.0 * df, band_hz)
        t_centre = chirp.center_index / fs
        bases = np.array(
            [model.base_frequency_at(j, [t_centre])[0] for j in range(len(model.fish))]
        )
        near = np.abs(freqs[:, None] - bases[None, :]) <= half_band
        own_band = near[:, fish_id][:, None] & mask
        free = ~near.any(axis=1)[:, None] & mask
        snr_db = _ratio_db(d_fp, b_fp)
        snr_baseline_db = _ratio_db(d[own_band], b[own_band])
        snr_excursion_db = _ratio_db(d[free], b[free])
        visible_fraction = float(np.mean(d_fp > b_fp))
    arrays: dict[str, np.ndarray] = {}
    if keep_arrays:
        arrays = {
            "times_s": times,
            "freqs_hz": freqs,
            "d": d,
            "mask": mask,
            "harmonic_mask": harmonic_mask,
        }
        if b is not None:
            arrays["b"] = b
    return Footprint(
        chirp_id=chirp.chirp_id,
        emitter_id=fish_id,
        box=box,
        snr_db=snr_db,
        snr_baseline_db=snr_baseline_db,
        snr_excursion_db=snr_excursion_db,
        visible_fraction=visible_fraction,
        visible=bool(
            np.nanmax([snr_baseline_db, snr_excursion_db, -np.inf]) >= visible_snr_db
        ),
        harmonic_box=harmonic_box,
        energy_fraction=energy_fraction,
        n_cells=int(mask.sum()),
        arrays=arrays,
    )


def scene_footprints(
    scene: Any,
    spectrogram_fn: SpectrogramFn | None = None,
    *,
    chirp_ids: Sequence[int] | None = None,
    **kwargs: Any,
) -> list[Footprint]:
    """Footprints of all (or the given) chirps of a scene.

    Keyword arguments go to :func:`chirp_footprint`. For a scene directory
    the recording is opened once.
    """
    from gridtools.sim.scene import Scene, _as_model

    model = _as_model(scene)
    if "recording" not in kwargs:
        if isinstance(scene, Scene):
            kwargs["recording"] = scene.recording
        elif isinstance(scene, str | Path):
            from gridtools.io import open_recording

            kwargs["recording"] = open_recording(Path(scene) / "recording.wav").data
    ids = sorted(model.chirps) if chirp_ids is None else list(chirp_ids)
    return [chirp_footprint(model, i, spectrogram_fn, **kwargs) for i in ids]
