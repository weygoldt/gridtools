"""Chirps for scenes: template access, timing, placement.

Three parts:

- :class:`TemplatePool`, the adapter to the template library
  (:mod:`gridtools.templates`): selects eligible templates, draws them with
  the library's histogram-equalised cycler, reads their traces and their
  support. Scenes only talk to the library through it.
- :func:`schedule_chirps`, the timing process: bursts of spontaneous chirps
  per fish, responses by other fish, a minimum silent interval between the
  supports of one fish's chirps, and a rate cap.
- :func:`place_chirp`, which turns a template, a centre sample and an
  amplitude envelope into a :class:`ChirpPlacement`: the frequency excursion
  and amplitude factor cropped to the samples where they differ from
  baseline, plus ground-truth measures of the trace that is actually
  rendered.

Library API versions. The template library v2 API
(``TemplateLibrary.equalization_bins``/``cycler(equalize=...)``,
``amplitude_from_frequency``, ``random_dips``) is used when the installed
:mod:`gridtools.templates` provides it. Older installs fall back to exact
local copies of the same algorithms (same random draws in the same order),
so a scene does not depend on which path ran; ``scene.json`` records it.
Template length is taken from the arrays (no fixed 0.2/0.4 s), and
everything downstream works on the chirp's support, not the template
length.
"""

import functools
import hashlib
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

import gridtools.templates as _templates
from gridtools.sim._common import Float32Array, FloatArray, IntArray
from gridtools.sim.config import ChirpConfig
from gridtools.templates import TemplateLibrary, load_library

__all__ = [
    "ChirpEvent",
    "ChirpPlacement",
    "ChirpSchedule",
    "TemplatePool",
    "amplitude_from_frequency",
    "library_api",
    "place_chirp",
    "random_dips",
    "schedule_chirps",
]

logger = logging.getLogger(__name__)

#: Metadata column used as chirp duration for equalisation (library v2 default).
EQUALIZE_DURATION_COLUMN = "duration_10_s"


def library_api() -> str:
    """Which template API is used: ``"gridtools.templates"`` or ``"local copy"``."""
    if hasattr(TemplateLibrary, "equalization_bins") and hasattr(
        _templates, "amplitude_from_frequency"
    ):
        return "gridtools.templates"
    return "local copy"


def amplitude_from_frequency(delta_f_hz: npt.ArrayLike, dip: float) -> FloatArray:
    """Chirp amplitude envelope from its frequency trace.

    ``1 - dip * clip(delta_f, 0) / max(delta_f)``: the inverted, normalised
    frequency excursion scaled by ``dip`` (MSc thesis; decision 10). The EOD
    amplitude falls to ``1 - dip`` at the frequency peak and is 1 where the
    frequency is at or below baseline. Uses
    ``gridtools.templates.amplitude_from_frequency`` when available.

    Parameters
    ----------
    delta_f_hz : array_like, shape (n,)
        Frequency excursion relative to baseline in Hz.
    dip : float
        Relative amplitude drop at the frequency peak, in [0, 1].

    Returns
    -------
    numpy.ndarray of float64
    """
    impl = getattr(_templates, "amplitude_from_frequency", None)
    if impl is not None:
        return np.asarray(impl(delta_f_hz, dip), dtype=np.float64)
    delta = np.asarray(delta_f_hz, dtype=np.float64)
    if not 0.0 <= dip <= 1.0:
        msg = "dip must lie in [0, 1]"
        raise ValueError(msg)
    positive = np.clip(delta, 0.0, None)
    peak = positive.max(axis=-1, keepdims=True)
    shape = np.divide(positive, peak, out=np.zeros_like(positive), where=peak > 0)
    return 1.0 - dip * shape


def random_dips(
    rng: np.random.Generator, n: int, dip_range: tuple[float, float]
) -> FloatArray:
    """``n`` amplitude dips, uniform in ``dip_range`` (library's ``random_dips``)."""
    impl = getattr(_templates, "random_dips", None)
    if impl is not None:
        return np.asarray(impl(rng, n, dip_range), dtype=np.float64)
    low, high = dip_range
    if not 0.0 <= low <= high <= 1.0:
        msg = f"dip_range must satisfy 0 <= low <= high <= 1, got {dip_range}"
        raise ValueError(msg)
    return rng.uniform(low, high, size=n)


# -- template access -------------------------------------------------------


class Cycler(Protocol):
    """Stateful template sampler (``gridtools.templates.TemplateCycler``)."""

    def draw(self, n: int) -> npt.NDArray[np.int64]:
        """Return the next ``n`` template ids."""
        ...


class _PermutationQueue:
    """Successive random permutations of a pool (copy of the library's)."""

    def __init__(self, pool: npt.NDArray[np.int64], rng: np.random.Generator) -> None:
        self._pool = np.array(pool, dtype=np.int64, copy=True)
        self._rng = rng
        self._queue = np.empty(0, dtype=np.int64)

    def take(self, n: int) -> npt.NDArray[np.int64]:
        out: list[npt.NDArray[np.int64]] = []
        needed = n
        while needed > 0:
            if self._queue.size == 0:
                self._queue = self._rng.permutation(self._pool)
            k = min(needed, self._queue.size)
            out.append(self._queue[:k])
            self._queue = self._queue[k:]
            needed -= k
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


class _LocalCycler:
    """Exact copy of ``gridtools.templates.TemplateCycler`` (library v2)."""

    def __init__(
        self, groups: Sequence[npt.NDArray[np.int64]], rng: np.random.Generator
    ) -> None:
        groups = [np.asarray(g, dtype=np.int64) for g in groups if np.size(g) > 0]
        if not groups:
            msg = "cannot sample from an empty template pool"
            raise ValueError(msg)
        self._groups = [_PermutationQueue(g, rng) for g in groups]
        self._group_queue = _PermutationQueue(
            np.arange(len(groups), dtype=np.int64), rng
        )

    def draw(self, n: int) -> npt.NDArray[np.int64]:
        if len(self._groups) == 1:
            return self._groups[0].take(n)
        which = self._group_queue.take(n)
        out = np.empty(n, dtype=np.int64)
        for g in np.unique(which):
            positions = np.flatnonzero(which == g)
            out[positions] = self._groups[int(g)].take(positions.size)
        return out


def _local_groups(
    view: TemplateLibrary, bins: tuple[int, int]
) -> list[npt.NDArray[np.int64]]:
    """Template ids per non-empty equalisation bin (copy of library v2).

    Equal-width bins from minimum to maximum of the pool (one template per
    duplicate group) on excursion and on width at 10 %.
    """
    n_exc, n_dur = bins
    pool = view.unique_ids()
    rows = view.rows_of(pool)
    excursion = view.metadata["excursion_hz"].to_numpy(dtype=np.float64)[rows]
    duration = view.metadata[EQUALIZE_DURATION_COLUMN].to_numpy(dtype=np.float64)[rows]
    finite = np.isfinite(excursion) & np.isfinite(duration)
    pool, excursion, duration = pool[finite], excursion[finite], duration[finite]
    if pool.size == 0:
        msg = "cannot bin an empty template pool"
        raise ValueError(msg)
    exc_edges = np.linspace(excursion.min(), excursion.max(), n_exc + 1)
    dur_edges = np.linspace(duration.min(), duration.max(), n_dur + 1)
    exc_bin = np.clip(
        np.searchsorted(exc_edges, excursion, side="right") - 1, 0, n_exc - 1
    )
    dur_bin = np.clip(
        np.searchsorted(dur_edges, duration, side="right") - 1, 0, n_dur - 1
    )
    flat = (exc_bin * n_dur + dur_bin).astype(np.int64)
    counts = np.bincount(flat, minlength=n_exc * n_dur)
    return [pool[flat == b] for b in np.flatnonzero(counts)]


@dataclass(frozen=True, eq=False)
class TemplatePool:
    """Eligible templates of a library and how to draw them.

    Use :meth:`open` to create one (cached per process). The library arrays
    are memory-mapped; only drawn rows are read. Selection and sampling work
    on a metadata-only view of the library (arrays of zero width), so no
    template arrays are copied.

    Attributes
    ----------
    library : TemplateLibrary
        The full, memory-mapped library.
    view : TemplateLibrary
        Metadata-only view of the eligible templates.
    path : pathlib.Path
        Library directory.
    manifest_sha256 : str
        SHA-256 of the library's ``manifest.json`` (identifies the build).
    equalize : bool
        Histogram-equalised sampling.
    bins : (int, int)
        Equalisation bins on excursion and width at 10 %.
    require_amp : bool
        Only templates with a usable measured amplitude trace (and their
        amplitude change counts towards the support).
    """

    library: TemplateLibrary
    view: TemplateLibrary
    path: Path
    manifest_sha256: str
    equalize: bool
    bins: tuple[int, int]
    require_amp: bool
    _rows: dict[int, int] = field(repr=False)
    _supports: dict[tuple[int, float], tuple[int, int]] = field(
        default_factory=dict, repr=False
    )

    @classmethod
    def open(cls, config: ChirpConfig) -> "TemplatePool":
        """Open (and cache per process) the pool described by ``config``."""
        return _open_pool(
            config.library,
            tuple(config.families),
            tuple(config.exclude),
            config.amplitude_mode == "measured",
            config.equalize,
            tuple(config.equalize_bins),
        )

    @property
    def samplerate_hz(self) -> float:
        """Sampling rate of the templates."""
        return self.library.samplerate_hz

    @property
    def n_samples(self) -> int:
        """Samples per template (from the arrays)."""
        return self.library.n_samples

    @property
    def centre_index(self) -> int:
        """Template sample of the alignment point."""
        return self.library.n_samples // 2

    @property
    def template_ids(self) -> IntArray:
        """Ids of the eligible templates."""
        return self.view.template_ids

    @property
    def sampling(self) -> str:
        """Description of the sampling scheme (for ``scene.json``)."""
        if not self.equalize:
            return "natural: successive permutations of the eligible pool"
        return (
            f"equalized: {self.bins[0]}x{self.bins[1]} equal-width bins over "
            f"excursion_hz x {EQUALIZE_DURATION_COLUMN}, bins from successive "
            "permutations of the non-empty bins, templates from permutations "
            f"within a bin ({library_api()})"
        )

    def cycler(self, rng: np.random.Generator) -> Cycler:
        """A stateful sampler (one per fish) drawing from ``rng``."""
        if library_api() == "gridtools.templates":
            # Equal-width bins and full equalization, as documented in
            # `sampling`; the library's own defaults (log bins, tempered
            # weights) take over when the scene config exposes them.
            return self.view.cycler(  # type: ignore[call-arg]
                rng,
                equalize=self.equalize,
                bins=self.bins,
                spacing="linear",
                alpha=1.0,
            )
        groups = (
            _local_groups(self.view, self.bins)
            if self.equalize
            else [self.view.unique_ids()]
        )
        return _LocalCycler(groups, rng)

    def traces(self, template_id: int) -> tuple[Float32Array, Float32Array]:
        """Frequency excursion (Hz) and measured amplitude factor of one template."""
        row = self._rows[int(template_id)]
        delta = np.array(self.library.delta_f_hz[row], dtype=np.float32)
        amp = np.array(self.library.amp_rel[row], dtype=np.float32)
        return delta, amp

    def scene_traces(
        self, template_id: int, scene_rate_hz: float
    ) -> tuple[Float32Array, Float32Array, int]:
        """Traces at the scene rate and the index of the alignment point."""
        delta, amp = self.traces(template_id)
        delta, centre = resample_template(
            delta, self.samplerate_hz, self.centre_index, scene_rate_hz, 0.0
        )
        amp, _ = resample_template(
            amp, self.samplerate_hz, self.centre_index, scene_rate_hz, 1.0
        )
        return delta, amp, centre

    def support(self, template_id: int, scene_rate_hz: float) -> tuple[int, int]:
        """Support of a template around its alignment point, in scene samples.

        Returns ``(before, after)``: the template differs from baseline on
        samples ``centre - before .. centre + after - 1``. The frequency
        trace defines it; with ``require_amp`` the measured amplitude change
        counts too. Cached.
        """
        key = (int(template_id), float(scene_rate_hz))
        if key not in self._supports:
            delta, amp, centre = self.scene_traces(template_id, scene_rate_hz)
            changed = delta != 0
            if self.require_amp:
                changed |= amp != 1
            nonzero = np.flatnonzero(changed)
            if nonzero.size == 0:
                msg = f"template {template_id} has no support"
                raise ValueError(msg)
            self._supports[key] = (
                centre - int(nonzero[0]),
                int(nonzero[-1]) + 1 - centre,
            )
        return self._supports[key]

    def family(self, template_id: int) -> str:
        """Library family of a template (``"small"`` or ``"big"``)."""
        meta = self.library.metadata
        return str(meta["family"].iat[self._rows[int(template_id)]])


@functools.cache
def _open_pool(
    library_path: str,
    families: tuple[str, ...],
    exclude: tuple[str, ...],
    require_amp: bool,
    equalize: bool,
    bins: tuple[int, int],
) -> TemplatePool:
    path = Path(library_path)
    if not path.is_dir():
        msg = f"template library not found: {path}"
        raise FileNotFoundError(msg)
    library = load_library(path, mmap=True)
    n = library.n_templates
    metadata_only = TemplateLibrary(
        delta_f_hz=np.empty((n, 0), dtype=np.float32),
        amp_rel=np.empty((n, 0), dtype=np.float32),
        metadata=library.metadata,
        samplerate_hz=library.samplerate_hz,
        manifest=library.manifest,
    )
    view = metadata_only.select(
        families=list(families),  # type: ignore[arg-type]
        exclude=exclude,
        require_amp=require_amp,
    )
    if view.n_templates == 0:
        msg = f"no eligible templates in {path} for families={families}"
        raise ValueError(msg)
    ids = library.template_ids
    manifest = (path / "manifest.json").read_bytes()
    pool = TemplatePool(
        library=library,
        view=view,
        path=path.resolve(),
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        equalize=equalize,
        bins=bins,
        require_amp=require_amp,
        _rows={int(t): i for i, t in enumerate(ids)},
    )
    logger.info(
        "template pool %s: %d eligible templates, %s",
        path,
        view.n_templates,
        pool.sampling,
    )
    return pool


# -- timing -----------------------------------------------------------------


#: Draws a template for a fish: ``fish_id -> (template_id, before, after)``
#: with the support in scene samples around the alignment point.
TemplateDraw = Callable[[int], tuple[int, int, int]]


@dataclass(frozen=True)
class ChirpEvent:
    """A scheduled chirp.

    Attributes
    ----------
    fish_id : int
        Emitting fish.
    center_index : int
        Scene sample of the chirp's alignment point.
    template_id : int
        Template drawn for it.
    before, after : int
        Support in samples before and after the alignment point (the chirp
        occupies ``center_index - before .. center_index + after - 1``).
    is_response : bool
        Whether the chirp is a response to another fish's chirp.
    trigger : int
        Index (in the schedule) of the triggering chirp, or -1.
    """

    fish_id: int
    center_index: int
    template_id: int
    before: int
    after: int
    is_response: bool = False
    trigger: int = -1

    @property
    def start(self) -> int:
        """First sample of the support."""
        return self.center_index - self.before

    @property
    def stop(self) -> int:
        """One past the last sample of the support."""
        return self.center_index + self.after


@dataclass(frozen=True)
class ChirpSchedule:
    """Result of :func:`schedule_chirps`.

    Attributes
    ----------
    events : tuple of ChirpEvent
        Sorted by (centre, fish); the position is the chirp id.
    n_spontaneous : int
        Spontaneous chirps kept.
    n_response_attempts : int
        Spontaneous chirps that triggered a response draw (``response_prob``).
    n_responses : int
        Responses kept (the rest violated the responder's limits or fell
        outside the scene).
    n_dropped_interval : int
        Spontaneous chirps dropped for the minimum interval.
    n_dropped_cap : int
        Spontaneous chirps dropped for the rate cap.
    """

    events: tuple[ChirpEvent, ...]
    n_spontaneous: int
    n_response_attempts: int
    n_responses: int
    n_dropped_interval: int
    n_dropped_cap: int


def _spacing_ok(first: ChirpEvent, second: ChirpEvent, min_interval: int) -> bool:
    """Chirps of one fish: centres ``min_interval`` apart, supports disjoint."""
    return (
        second.center_index - first.center_index >= min_interval
        and second.start >= first.stop
    )


def _spontaneous(
    fish_id: int,
    rng: np.random.Generator,
    rate_hz: float,
    config: ChirpConfig,
    draw: TemplateDraw,
    n_samples: int,
    samplerate_hz: float,
) -> list[ChirpEvent]:
    """Burst process on [0, duration]: candidates sorted by centre, unfiltered.

    Burst onsets are Poisson; within a burst the centre-to-centre intervals
    are log-normal (``burst_interval_s`` = median, log SD).
    """
    if rate_hz <= 0.0:
        return []
    duration_s = (n_samples - 1) / samplerate_hz
    burst_rate = rate_hz / config.burst_size_mean
    median_s, log_sd = config.burst_interval_s
    events: list[ChirpEvent] = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / burst_rate)
        if t > duration_s:
            break
        size = int(rng.geometric(1.0 / config.burst_size_mean))
        intervals = median_s * np.exp(log_sd * rng.standard_normal(size - 1))
        centres = t + np.concatenate([[0.0], np.cumsum(intervals)])
        for centre_s in centres.tolist():
            template_id, before, after = draw(fish_id)
            centre = round(centre_s * samplerate_hz)
            events.append(ChirpEvent(fish_id, centre, template_id, before, after))
    inside = [e for e in events if 0 <= e.center_index <= n_samples - 1]
    return sorted(inside, key=lambda e: e.center_index)


def schedule_chirps(
    config: ChirpConfig,
    rates_hz: Sequence[float],
    timing_rngs: Sequence[np.random.Generator],
    response_rng: np.random.Generator,
    draw: TemplateDraw,
    *,
    n_samples: int,
    samplerate_hz: float,
) -> ChirpSchedule:
    """Draw when every fish chirps and with which template.

    See :class:`~gridtools.sim.config.ChirpConfig`. Chirps of one fish are at
    least ``min_interval_s`` apart centre to centre and their supports (from
    the drawn templates) never overlap; spontaneous chirps that violate this
    are dropped (earlier ones win), and so are responses.

    Parameters
    ----------
    config : ChirpConfig
        Timing parameters.
    rates_hz : sequence of float
        Spontaneous rate of each fish in chirps/s.
    timing_rngs : sequence of numpy.random.Generator
        One generator per fish for its spontaneous chirps.
    response_rng : numpy.random.Generator
        Generator for responses (couples the fish). Five numbers are drawn
        per spontaneous chirp whatever the outcome.
    draw : callable
        ``draw(fish_id) -> (template_id, before, after)``; called for every
        candidate in a fixed order (fish by fish for spontaneous chirps,
        then for every response attempt in time order).
    n_samples : int
        Scene length in samples.
    samplerate_hz : float
        Sampling rate in Hz.

    Returns
    -------
    ChirpSchedule
    """
    n_fish = len(rates_hz)
    if len(timing_rngs) != n_fish:
        msg = "need one timing generator per fish"
        raise ValueError(msg)
    min_interval = round(config.min_interval_s * samplerate_hz)
    if config.max_rate_hz is None:
        cap = math.inf
    else:
        cap = math.floor(config.max_rate_hz * n_samples / samplerate_hz)
    per_fish: list[list[ChirpEvent]] = []
    dropped_interval = dropped_cap = 0
    for fish_id, (rng, rate) in enumerate(zip(timing_rngs, rates_hz, strict=True)):
        candidates = _spontaneous(
            fish_id, rng, rate, config, draw, n_samples, samplerate_hz
        )
        kept: list[ChirpEvent] = []
        for event in candidates:
            if kept and not _spacing_ok(kept[-1], event, min_interval):
                dropped_interval += 1
                continue
            kept.append(event)
        if len(kept) > cap:
            n_keep = int(cap)
            dropped_cap += len(kept) - n_keep
            choice = np.sort(rng.choice(len(kept), size=n_keep, replace=False))
            kept = [kept[i] for i in choice.tolist()]
        per_fish.append(kept)

    spontaneous = sorted(
        (e for events in per_fish for e in events),
        key=lambda e: (e.center_index, e.fish_id),
    )
    lag_lo, lag_hi = config.response_delay_s
    window = config.simultaneous_window_s
    responses: list[tuple[ChirpEvent, int]] = []  # (event, trigger position)
    attempts = 0
    for position, trigger in enumerate(spontaneous):
        u = response_rng.random()
        partner = int(response_rng.integers(max(n_fish - 1, 1)))
        delay_s = response_rng.uniform(lag_lo, lag_hi)
        simultaneous = response_rng.random() < config.simultaneous_fraction
        offset_s = response_rng.uniform(-window, window)
        if n_fish < 2 or u >= config.response_prob:
            continue
        attempts += 1
        responder = partner if partner < trigger.fish_id else partner + 1
        template_id, before, after = draw(responder)
        lag = offset_s if simultaneous else delay_s
        centre = trigger.center_index + round(lag * samplerate_hz)
        event = ChirpEvent(responder, centre, template_id, before, after, True)
        taken = per_fish[responder]
        if not 0 <= centre <= n_samples - 1 or len(taken) >= cap:
            continue
        slot = int(np.searchsorted([e.center_index for e in taken], centre))
        if slot > 0 and not _spacing_ok(taken[slot - 1], event, min_interval):
            continue
        if slot < len(taken) and not _spacing_ok(event, taken[slot], min_interval):
            continue
        taken.insert(slot, event)
        responses.append((event, position))

    entries = [(e, -1) for e in spontaneous] + responses
    order = sorted(
        range(len(entries)),
        key=lambda k: (entries[k][0].center_index, entries[k][0].fish_id),
    )
    final_id = {old: new for new, old in enumerate(order)}
    events = []
    for k in order:
        event, trigger_position = entries[k]
        if event.is_response:
            event = ChirpEvent(
                event.fish_id,
                event.center_index,
                event.template_id,
                event.before,
                event.after,
                True,
                final_id[trigger_position],
            )
        events.append(event)
    return ChirpSchedule(
        events=tuple(events),
        n_spontaneous=len(spontaneous),
        n_response_attempts=attempts,
        n_responses=len(responses),
        n_dropped_interval=dropped_interval,
        n_dropped_cap=dropped_cap,
    )


# -- placement --------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ChirpPlacement:
    """One chirp as rendered: traces cropped to their support.

    The excursion ``delta_f_hz`` is added to the fish's baseline frequency
    and ``amp_factor`` multiplies its amplitude on scene samples
    ``start .. start + len - 1``; outside, they are 0 and 1. Both are
    float32 and exactly what the renderer uses.

    Attributes
    ----------
    chirp_id : int
        Scene-wide id (order of centre time).
    fish_id : int
        Emitter.
    template_id : int
        Library template id.
    center_index : int
        Scene sample of the template's alignment point.
    start : int
        Scene sample of the first stored value (>= 0).
    delta_f_hz : numpy.ndarray of float32
        Frequency excursion in Hz on the stored samples.
    amp_factor : numpy.ndarray of float32
        Amplitude factor on the stored samples.
    dip : float
        Amplitude dip (frequency mode: the drawn value; measured mode:
        ``1 - min(amp_factor)``).
    support_start, support_stop : int
        Scene samples of the first and one past the last sample where the
        template differs from baseline (may lie outside the scene).
    onset_index, peak_index, offset_index : int
        Scene samples of the first and last 10 % crossing and of the
        maximum of the excursion (the 10 % crossings bracket the peak).
    excursion_hz : float
        Maximum of the excursion.
    undershoot_hz : float
        ``-min(excursion)`` (>= 0): how far the frequency drops below
        baseline, typically after the chirp.
    is_response : bool
        Generated as a response to another fish's chirp.
    response_to : int
        Chirp id of the trigger, or -1.
    """

    chirp_id: int
    fish_id: int
    template_id: int
    center_index: int
    start: int
    delta_f_hz: Float32Array
    amp_factor: Float32Array
    dip: float
    support_start: int
    support_stop: int
    onset_index: int
    peak_index: int
    offset_index: int
    excursion_hz: float
    undershoot_hz: float
    is_response: bool = False
    response_to: int = -1

    @property
    def stop(self) -> int:
        """One past the last stored scene sample."""
        return self.start + int(self.delta_f_hz.size)

    @cached_property
    def local_cycles(self) -> tuple[int, FloatArray]:
        """Trapezoid-integrated excursion around the stored samples.

        Returns ``(first, cycles)``: ``cycles[k]`` is the accumulated
        ``sum_{j < n} (df[j] + df[j+1]) / 2`` (Hz x samples) at scene sample
        ``n = first + k``. It covers the stored samples plus one zero sample on
        each side where the scene allows, so ``cycles[0] = 0`` and the last
        value is the chirp's total; before ``first`` the contribution is 0 and
        after the end it stays at the total.
        """
        delta = self.delta_f_hz.astype(np.float64)
        first = self.start - 1 if self.start >= 1 else self.start
        pad_left = [0.0] if self.start >= 1 else []
        values = np.concatenate([pad_left, delta, [0.0]])
        increments = 0.5 * (values[:-1] + values[1:])
        cycles = np.concatenate([[0.0], np.cumsum(increments)])
        return first, cycles


def _crossings(delta: FloatArray, peak: int, level: float) -> tuple[int, int]:
    """First and last samples >= level in the contiguous run around ``peak``."""
    below = delta < level
    left = np.flatnonzero(below[:peak])
    right = np.flatnonzero(below[peak:])
    onset = int(left[-1]) + 1 if left.size else 0
    offset = peak + int(right[0]) - 1 if right.size else delta.size - 1
    return onset, offset


def resample_template(
    trace: npt.ArrayLike,
    template_rate_hz: float,
    centre_index: int,
    scene_rate_hz: float,
    fill: float,
) -> tuple[Float32Array, int]:
    """Bring a template to the scene sampling rate.

    Returns the trace (float32) and the index of its alignment point. Equal
    rates return the trace unchanged.
    """
    values = np.asarray(trace, dtype=np.float32)
    if template_rate_hz == scene_rate_hz:
        return values, centre_index
    t_template = (np.arange(values.size) - centre_index) / template_rate_hz
    first = math.ceil(t_template[0] * scene_rate_hz)
    last = math.floor(t_template[-1] * scene_rate_hz)
    t_scene = np.arange(first, last + 1) / scene_rate_hz
    out = np.interp(t_scene, t_template, values.astype(np.float64), fill, fill)
    return out.astype(np.float32), -first


def place_chirp(
    *,
    chirp_id: int,
    fish_id: int,
    template_id: int,
    delta_f_hz: npt.ArrayLike,
    amp_factor: npt.ArrayLike,
    centre_index_template: int,
    center_index: int,
    dip: float,
    n_samples: int,
    is_response: bool = False,
    response_to: int = -1,
) -> ChirpPlacement | None:
    """Place a template (at the scene rate) with its alignment point at a sample.

    Parameters
    ----------
    chirp_id, fish_id, template_id : int
        Identifiers.
    delta_f_hz, amp_factor : array_like
        Template frequency excursion (Hz) and amplitude factor at the scene
        rate (amplitude already scaled by the dip if applicable).
    centre_index_template : int
        Index of the alignment point in the template.
    center_index : int
        Scene sample of the alignment point.
    dip : float
        Amplitude dip, recorded as is.
    n_samples : int
        Scene length.
    is_response, response_to
        Timing provenance.

    Returns
    -------
    ChirpPlacement or None
        None if the template has no support or its support lies entirely
        outside the scene.
    """
    delta = np.asarray(delta_f_hz, dtype=np.float32)
    amp = np.asarray(amp_factor, dtype=np.float32)
    support = np.flatnonzero((delta != 0) | (amp != 1))
    if support.size == 0:
        return None
    k0, k1 = int(support[0]), int(support[-1]) + 1
    offset = center_index - centre_index_template
    support_start, support_stop = offset + k0, offset + k1
    start, stop = max(support_start, 0), min(support_stop, n_samples)
    if stop <= start:
        return None
    delta64 = delta.astype(np.float64)
    peak = int(np.argmax(delta64))
    excursion = float(delta64[peak])
    onset, offset10 = _crossings(delta64, peak, 0.1 * excursion)
    return ChirpPlacement(
        chirp_id=chirp_id,
        fish_id=fish_id,
        template_id=int(template_id),
        center_index=int(center_index),
        start=start,
        delta_f_hz=delta[start - offset : stop - offset].copy(),
        amp_factor=amp[start - offset : stop - offset].copy(),
        dip=float(dip),
        support_start=support_start,
        support_stop=support_stop,
        onset_index=offset + onset,
        peak_index=offset + peak,
        offset_index=offset + offset10,
        excursion_hz=excursion,
        undershoot_hz=max(0.0, -float(delta64.min())),
        is_response=is_response,
        response_to=response_to,
    )
