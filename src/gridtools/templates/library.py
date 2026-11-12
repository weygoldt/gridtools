"""Load and sample the chirp template library."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

log = logging.getLogger(__name__)

SUPPORTED_FORMAT_VERSIONS = (1,)

#: Flags excluded by :meth:`TemplateLibrary.select` unless told otherwise:
#: demonstrable artefacts only (broken or incomplete traces, duplicates) plus
#: the atypical-EODf group. Excursion and duration are never exclusion
#: criteria.
DEFAULT_EXCLUDE: tuple[str, ...] = (
    "invalid",
    "has_nan",
    "duplicate",
    "truncated",
    "step_flag",
    "eodf_atypical",
)

#: Every flag name accepted by :meth:`TemplateLibrary.select`.
KNOWN_FLAGS: tuple[str, ...] = (
    *DEFAULT_EXCLUDE,
    "multi_peak",
    "ripple_flag",
    "amp_ripple_flag",
)

#: Duration measure used for histogram equalization: full width at 10% of the
#: excursion, the thesis' chirp width (crossings descended to 10% of the peak).
EQUALIZE_DURATION_COLUMN = "duration_10_s"

Family = Literal["small", "big"]


@dataclass(frozen=True, slots=True)
class TemplateSample:
    """Templates drawn for placement in a simulation.

    Attributes
    ----------
    template_ids
        Library ``template_id`` of every drawn chirp (log these).
    delta_f_hz
        Array (n, n_samples): frequency excursion to add to the baseline.
    amp_rel
        Array (n, n_samples): factor to multiply the EOD amplitude with.
    """

    template_ids: npt.NDArray[np.int64]
    delta_f_hz: npt.NDArray[np.float32]
    amp_rel: npt.NDArray[np.float32]


@dataclass(frozen=True)
class TemplateLibrary:
    """Real chirp templates with metadata.

    Row ``i`` of both arrays belongs to row ``i`` of ``metadata``. Every
    template has ``n_samples`` samples at ``samplerate_hz``; the alignment
    point (midpoint of the 50% crossings) is sample ``n_samples // 2``.
    ``delta_f_hz`` is exactly 0 and ``amp_rel`` exactly 1 at both ends.

    Attributes
    ----------
    delta_f_hz
        Array (n_templates, n_samples), Hz relative to baseline EODf.
    amp_rel
        Array (n_templates, n_samples), amplitude relative to baseline.
    metadata
        One row per template (see the library report for the columns).
    samplerate_hz
        Sampling rate of the templates.
    manifest
        Build manifest (configuration, source files, code version).
    """

    delta_f_hz: npt.NDArray[np.float32]
    amp_rel: npt.NDArray[np.float32]
    metadata: pd.DataFrame
    samplerate_hz: float
    manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.metadata)
        if self.delta_f_hz.shape != self.amp_rel.shape:
            msg = (
                f"array shapes differ: {self.delta_f_hz.shape} vs {self.amp_rel.shape}"
            )
            raise ValueError(msg)
        if self.delta_f_hz.ndim != 2 or self.delta_f_hz.shape[0] != n:
            msg = f"arrays have {self.delta_f_hz.shape[0]} rows, metadata {n}"
            raise ValueError(msg)
        if "template_id" not in self.metadata.columns:
            msg = "metadata needs a template_id column"
            raise ValueError(msg)

    @property
    def n_templates(self) -> int:
        """Number of templates."""
        return int(self.delta_f_hz.shape[0])

    @property
    def n_samples(self) -> int:
        """Samples per template."""
        return int(self.delta_f_hz.shape[1])

    @property
    def template_ids(self) -> npt.NDArray[np.int64]:
        """``template_id`` of every row."""
        return self.metadata["template_id"].to_numpy(dtype=np.int64)

    def time_s(self) -> npt.NDArray[np.float64]:
        """Template time axis in seconds, 0 at the alignment point."""
        return (np.arange(self.n_samples) - self.n_samples // 2) / self.samplerate_hz

    def flag_mask(self, flag: str) -> npt.NDArray[np.bool_]:
        """Boolean mask of templates carrying ``flag`` (see ``KNOWN_FLAGS``)."""
        meta = self.metadata
        if flag == "invalid":
            return ~meta["valid"].to_numpy(dtype=bool)
        if flag == "duplicate":
            return meta["duplicate_of"].to_numpy() >= 0
        if flag in KNOWN_FLAGS:
            return meta[flag].to_numpy(dtype=bool)
        msg = f"unknown flag {flag!r}; known: {', '.join(KNOWN_FLAGS)}"
        raise KeyError(msg)

    def select(
        self,
        *,
        families: Iterable[Family] | None = None,
        exclude: Iterable[str] = DEFAULT_EXCLUDE,
        require_amp: bool = False,
        template_ids: Sequence[int] | None = None,
    ) -> TemplateLibrary:
        """Return the subset of templates that passes the given filters.

        Parameters
        ----------
        families
            Keep only these families (``"small"``, ``"big"``); None keeps all.
        exclude
            Flags whose templates are dropped (default ``DEFAULT_EXCLUDE``:
            invalid, NaN, duplicates, truncated, step artefacts, atypical
            EODf). Pass ``()`` to keep everything.
        require_amp
            Keep only templates whose amplitude trace is usable
            (``amp_usable``).
        template_ids
            Keep only these template ids (applied before the other filters).

        Returns
        -------
        TemplateLibrary
            New library with copies of the selected rows.
        """
        keep = np.ones(self.n_templates, dtype=bool)
        if template_ids is not None:
            keep &= np.isin(self.template_ids, np.asarray(template_ids, dtype=np.int64))
        if families is not None:
            keep &= self.metadata["family"].isin(list(families)).to_numpy()
        for flag in exclude:
            keep &= ~self.flag_mask(flag)
        if require_amp:
            keep &= self.metadata["amp_usable"].to_numpy(dtype=bool)
        return self._subset(np.flatnonzero(keep))

    def _subset(self, rows: npt.NDArray[np.intp]) -> TemplateLibrary:
        return TemplateLibrary(
            delta_f_hz=np.array(self.delta_f_hz[rows], dtype=np.float32),
            amp_rel=np.array(self.amp_rel[rows], dtype=np.float32),
            metadata=self.metadata.iloc[rows].reset_index(drop=True),
            samplerate_hz=self.samplerate_hz,
            manifest=self.manifest,
        )

    def rows_of(
        self, template_ids: Sequence[int] | npt.NDArray[np.int64]
    ) -> npt.NDArray[np.intp]:
        """Row indices of the given template ids (KeyError if one is missing)."""
        lookup = {int(t): i for i, t in enumerate(self.template_ids)}
        try:
            return np.array([lookup[int(t)] for t in template_ids], dtype=np.intp)
        except KeyError as err:
            msg = f"template id {err.args[0]} is not in this library"
            raise KeyError(msg) from None

    def get(
        self, template_ids: Sequence[int] | npt.NDArray[np.int64]
    ) -> TemplateSample:
        """Return the traces of the given template ids (in the given order)."""
        rows = self.rows_of(template_ids)
        return TemplateSample(
            template_ids=np.asarray(template_ids, dtype=np.int64).copy(),
            delta_f_hz=np.array(self.delta_f_hz[rows], dtype=np.float32),
            amp_rel=np.array(self.amp_rel[rows], dtype=np.float32),
        )

    def unique_ids(self) -> npt.NDArray[np.int64]:
        """One template id per duplicate group present in this library.

        A template whose ``duplicate_of`` target is also present is dropped,
        so every distinct chirp appears once in the sampling pool.
        """
        ids = self.template_ids
        group = self.metadata["duplicate_of"].to_numpy(dtype=np.int64)
        group = np.where(group >= 0, group, ids)
        _, first = np.unique(group, return_index=True)
        return ids[np.sort(first)]

    def equalization_bins(
        self,
        bins: tuple[int, int] = (10, 10),
        duration_column: str = EQUALIZE_DURATION_COLUMN,
    ) -> EqualizationBins:
        """Bin the sampling pool on excursion and duration.

        Equal-width bins span the minimum to the maximum of the pool (one
        template per duplicate group, see :meth:`unique_ids`); the maximum
        falls into the last bin. Templates with a non-finite excursion or
        duration are left out.

        Parameters
        ----------
        bins
            Number of bins on excursion and on duration.
        duration_column
            Metadata column with the duration (default: full width at 10%).

        Returns
        -------
        EqualizationBins
        """
        n_exc, n_dur = bins
        if n_exc < 1 or n_dur < 1:
            msg = f"bins must be positive, got {bins}"
            raise ValueError(msg)
        pool = self.unique_ids()
        rows = self.rows_of(pool)
        excursion = self.metadata["excursion_hz"].to_numpy(dtype=np.float64)[rows]
        duration = self.metadata[duration_column].to_numpy(dtype=np.float64)[rows]
        finite = np.isfinite(excursion) & np.isfinite(duration)
        if not finite.all():
            log.warning(
                "%d templates without excursion or duration left out", (~finite).sum()
            )
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
        counts = np.bincount(flat, minlength=n_exc * n_dur).reshape(n_exc, n_dur)
        return EqualizationBins(
            template_ids=pool,
            excursion_edges_hz=exc_edges,
            duration_edges_s=dur_edges,
            duration_column=duration_column,
            bin_of=flat,
            counts=counts.astype(np.int64),
        )

    def cycler(
        self,
        rng: np.random.Generator,
        *,
        equalize: bool = True,
        bins: tuple[int, int] = (10, 10),
    ) -> TemplateCycler:
        """Return a stateful sampler that keeps its balance across calls.

        Use one cycler per simulation when chirps are drawn one at a time.
        See :meth:`sample` for the two modes.
        """
        if not equalize:
            return TemplateCycler([self.unique_ids()], rng)
        return TemplateCycler(self.equalization_bins(bins).groups(), rng)

    def sample(
        self,
        rng: np.random.Generator,
        n: int,
        *,
        equalize: bool = True,
        bins: tuple[int, int] = (10, 10),
    ) -> TemplateSample:
        """Draw ``n`` templates without inflating duplicates.

        The pool holds one template per duplicate group.

        ``equalize=True`` (histogram equalization, as in the thesis, but as
        sampling weights instead of duplicated rows): the pool is binned on
        excursion and duration (:meth:`equalization_bins`); every non-empty
        bin is equally likely (bins are taken from successive random
        permutations of the non-empty bins) and within a bin templates come
        from successive random permutations of its members. A template in a
        bin with ``c`` members out of ``K`` non-empty bins is drawn
        ``n / (K * c)`` times on average.

        ``equalize=False`` (natural distribution): templates come from
        successive random permutations of the whole pool, so with
        ``n <= pool size`` none repeats and otherwise each is used
        ``floor(n / pool)`` or ``ceil(n / pool)`` times.

        Parameters
        ----------
        rng
            Source of randomness; the result is a deterministic function of
            its state.
        n
            Number of templates to draw.
        equalize
            Equalize excursion and duration (default) or keep the natural
            distribution.
        bins
            Bins on excursion and duration for equalization.

        Returns
        -------
        TemplateSample
            Template ids and traces, in draw order.
        """
        ids = self.cycler(rng, equalize=equalize, bins=bins).draw(n)
        return self.get(ids)


@dataclass(frozen=True, slots=True)
class EqualizationBins:
    """Equal-width 2-D binning of a template pool on excursion and duration.

    Attributes
    ----------
    template_ids
        Pool template ids.
    excursion_edges_hz, duration_edges_s
        Bin edges.
    duration_column
        Metadata column used for the duration.
    bin_of
        Flat bin index (``excursion_bin * n_duration_bins + duration_bin``)
        of every pool template.
    counts
        Templates per bin, shape (n_excursion_bins, n_duration_bins).
    """

    template_ids: npt.NDArray[np.int64]
    excursion_edges_hz: npt.NDArray[np.float64]
    duration_edges_s: npt.NDArray[np.float64]
    duration_column: str
    bin_of: npt.NDArray[np.int64]
    counts: npt.NDArray[np.int64]

    @property
    def n_nonempty(self) -> int:
        """Number of bins holding at least one template."""
        return int(np.count_nonzero(self.counts))

    def groups(self) -> list[npt.NDArray[np.int64]]:
        """Template ids of every non-empty bin, in flat bin order."""
        return [
            self.template_ids[self.bin_of == b]
            for b in np.flatnonzero(self.counts.ravel())
        ]

    def expected_draws(self, n: float) -> npt.NDArray[np.float64]:
        """Expected number of draws of every pool template in ``n`` draws."""
        per_bin = self.counts.ravel()[self.bin_of].astype(np.float64)
        return n / (self.n_nonempty * per_bin)

    def table(self) -> pd.DataFrame:
        """Occupancy table: one row per excursion bin, one column per duration bin."""
        exc = self.excursion_edges_hz
        dur = 1e3 * self.duration_edges_s
        return pd.DataFrame(
            self.counts,
            index=[f"{exc[i]:.0f}-{exc[i + 1]:.0f} Hz" for i in range(exc.size - 1)],
            columns=[f"{dur[j]:.0f}-{dur[j + 1]:.0f} ms" for j in range(dur.size - 1)],
        )


class _PermutationQueue:
    """Successive random permutations of a fixed pool."""

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


class TemplateCycler:
    """Draw template ids with balanced repeats, across any number of calls.

    With one group, ids come from successive random permutations of it. With
    several groups (histogram bins), the group of each draw comes from
    successive random permutations of the groups, and the id from successive
    random permutations of that group's members.

    Parameters
    ----------
    groups
        Template ids per group (each id should be a distinct chirp).
    rng
        Random generator; drawing advances its state.
    """

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
        self._pool_size = int(sum(g.size for g in groups))

    @property
    def pool_size(self) -> int:
        """Number of distinct templates in the pool."""
        return self._pool_size

    @property
    def n_groups(self) -> int:
        """Number of non-empty groups."""
        return len(self._groups)

    def draw(self, n: int) -> npt.NDArray[np.int64]:
        """Return the next ``n`` template ids."""
        if n < 0:
            msg = "n must be non-negative"
            raise ValueError(msg)
        if len(self._groups) == 1:
            return self._groups[0].take(n)
        which = self._group_queue.take(n)
        out = np.empty(n, dtype=np.int64)
        for g in np.unique(which):
            positions = np.flatnonzero(which == g)
            out[positions] = self._groups[int(g)].take(positions.size)
        return out


def load_library(path: Path | str, *, mmap: bool = False) -> TemplateLibrary:
    """Load a library written by :func:`gridtools.templates.build_library`.

    Parameters
    ----------
    path
        Library directory.
    mmap
        Memory-map the template arrays instead of reading them.

    Returns
    -------
    TemplateLibrary
    """
    from gridtools.templates.build import (
        AMP_FILE,
        DELTA_FILE,
        MANIFEST_FILE,
        METADATA_FILE,
    )

    path = Path(path)
    manifest = json.loads((path / MANIFEST_FILE).read_text())
    version = manifest.get("library_format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS:
        msg = f"unsupported library format version {version!r}"
        raise ValueError(msg)
    mode: Literal["r"] | None = "r" if mmap else None
    delta = np.load(path / DELTA_FILE, mmap_mode=mode)
    amp = np.load(path / AMP_FILE, mmap_mode=mode)
    metadata = pd.read_parquet(path / METADATA_FILE)
    library = TemplateLibrary(
        delta_f_hz=delta,
        amp_rel=amp,
        metadata=metadata,
        samplerate_hz=float(manifest["samplerate_hz"]),
        manifest=manifest,
    )
    log.info("loaded %d templates from %s", library.n_templates, path)
    return library
