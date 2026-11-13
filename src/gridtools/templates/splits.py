"""Train/validation/test splits of the template library by fish.

Scenes of one split draw chirps only from templates of that split, so that
test scenes use chirps of fish the models never saw in training. The split
unit is the fish where the library knows it, else its group:

- harvested competition templates with a known individual (``individual``,
  from the competition metadata ``order_meta.csv``: group and fish number of
  the two fish of every recording) form one unit per fish across all its
  recordings, ``competition/<individual>`` (e.g. ``competition/g6f5`` holds
  three recordings);
- every other template uses its group (:func:`individual_labels`):
  ``group_label`` (harvested without a known individual:
  ``<recording>/track<id>``; chirp chamber: ``chirp_chamber/<inferred_group>``,
  which approximates one recording, not reliably one fish), else
  ``inferred_group``, else the template itself.

:func:`assign_template_splits` assigns every unit of a library to ``train``,
``val`` or ``test`` (seeded, deterministic), :func:`write_splits` stores the
assignment as JSON so that every consumer uses the same split, and
:func:`load_splits` reads it back. :func:`splits_report` documents it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from gridtools.templates.library import (
    DEFAULT_ALPHA,
    DEFAULT_BALANCE,
    DEFAULT_BINS,
    DEFAULT_EXCLUDE,
    DEFAULT_SPACING,
    EQUALIZE_DURATION_COLUMN,
    Balance,
    EqualizationBins,
    Spacing,
    TemplateLibrary,
    individual_labels,
)

SplitName = Literal["train", "val", "test"]

#: Split names, in allocation order.
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "val", "test")
#: Default fractions of units per split.
DEFAULT_FRACTIONS: tuple[float, float, float] = (0.8, 0.1, 0.1)
#: Chirps at least this long (full width at 10%) count as long.
LONG_CHIRP_S = 0.150
#: A property or bin is covered in every split only if it has this many units.
MIN_UNITS_TO_COVER = 3
#: Version of the JSON written by :func:`write_splits`.
SPLITS_FORMAT_VERSION = 1

_SIZE_TIERS = (1, 2, 10, 100)  # lower bounds of the unit-size strata
_PROPERTIES = ("long", "big", "small")
_COVER_ORDER: tuple[SplitName, ...] = ("test", "val", "train")


def fish_units(meta: pd.DataFrame) -> npt.NDArray[np.str_]:
    """Split unit of every row: the fish where known, else its group.

    ``competition/<individual>`` for harvested competition rows with a
    non-empty ``individual``; :func:`individual_labels` otherwise.
    """
    units = individual_labels(meta).astype(object)
    if {"source", "individual"} <= set(meta.columns):
        individual = meta["individual"].fillna("").astype(str).str.strip().to_numpy()
        known = (meta["source"].astype(str).to_numpy() == "competition") & (
            individual != ""
        )
        units[known] = [f"competition/{i}" for i in individual[known]]
    return np.asarray(units, dtype=str)


@dataclass(frozen=True)
class TemplateSplits:
    """Assignment of library units (fish or groups) to splits.

    Attributes
    ----------
    units : dict
        ``unit -> split``.
    groups : dict
        ``group -> unit`` (groups as in :func:`individual_labels`).
    template_ids : dict
        ``split -> sorted template ids`` of the library the split was made
        for (every row, whatever its flags).
    provenance : dict
        Seed, fractions, pins, selection and library hashes.
    """

    units: dict[str, str]
    groups: dict[str, str]
    template_ids: dict[str, list[int]]
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bad = {s for s in self.units.values() if s not in SPLIT_NAMES}
        if bad:
            msg = f"unknown split names {sorted(bad)}; known: {SPLIT_NAMES}"
            raise ValueError(msg)
        missing = sorted({u for u in self.groups.values() if u not in self.units})
        if missing:
            msg = f"groups map to units without a split: {missing[:5]}"
            raise ValueError(msg)
        seen: dict[int, str] = {}
        for split, ids in self.template_ids.items():
            for t in ids:
                if t in seen:
                    msg = f"template {t} is in {seen[t]!r} and {split!r}"
                    raise ValueError(msg)
                seen[t] = split

    def group_split(self) -> dict[str, str]:
        """``group -> split``."""
        return {g: self.units[u] for g, u in self.groups.items()}

    def split_of(self, meta: pd.DataFrame) -> npt.NDArray[np.str_]:
        """Split of every row of a library's metadata.

        Raises
        ------
        KeyError
            If a row's group is not in the assignment (another library).
        """
        lookup = self.group_split()
        groups = individual_labels(meta)
        unknown = sorted(set(groups.tolist()) - lookup.keys())
        if unknown:
            msg = (
                f"{len(unknown)} groups are not in the split assignment "
                f"(e.g. {unknown[:3]}); was it made for another library?"
            )
            raise KeyError(msg)
        return np.array([lookup[g] for g in groups], dtype=str)

    def counts(self) -> dict[str, dict[str, int]]:
        """Units, groups and templates per split."""
        group_split = self.group_split()
        return {
            s: {
                "units": sum(v == s for v in self.units.values()),
                "groups": sum(v == s for v in group_split.values()),
                "templates": len(self.template_ids.get(s, [])),
            }
            for s in SPLIT_NAMES
        }


def _largest_remainder(
    n: int, fractions: Sequence[float], rng: np.random.Generator
) -> npt.NDArray[np.int64]:
    """Integers summing to ``n`` closest to ``n * fractions`` (random ties)."""
    p = np.asarray(fractions, dtype=np.float64)
    exact = n * p / p.sum()
    base = np.floor(exact).astype(np.int64)
    rest = exact - base
    order = np.lexsort((rng.permutation(p.size), -rest))
    base[order[: n - int(base.sum())]] += 1
    return base


def _interleave(quota: npt.NDArray[np.int64], rng: np.random.Generator) -> list[int]:
    """Split indices with exact counts ``quota``, each spread evenly.

    Split ``s`` gets the positions of its ``quota[s]`` systematic "times"
    ``(j + u_s) / quota[s]`` (random phase ``u_s``) in the merged order, so
    any contiguous run of the result gets close to its proportional share.
    """
    times: list[tuple[float, int]] = []
    for s, q in enumerate(quota.tolist()):
        u = float(rng.uniform())
        times.extend(((j + u) / q, s) for j in range(q))
    return [s for _, s in sorted(times)]


def _metadata_view(meta: pd.DataFrame, samplerate_hz: float = 1.0) -> TemplateLibrary:
    n = len(meta)
    return TemplateLibrary(
        delta_f_hz=np.empty((n, 0), dtype=np.float32),
        amp_rel=np.empty((n, 0), dtype=np.float32),
        metadata=meta.reset_index(drop=True),
        samplerate_hz=samplerate_hz,
    )


def _tier(n: int) -> int:
    return int(np.searchsorted(_SIZE_TIERS, n, side="right")) - 1


def file_sha256(path: Path | str) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def assign_template_splits(
    library: TemplateLibrary,
    *,
    seed: int = 0,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    pins: Mapping[str, str] | None = None,
    families: Iterable[str] | None = None,
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
    bins: tuple[int, int] = DEFAULT_BINS,
    spacing: Spacing = DEFAULT_SPACING,
    long_chirp_s: float = LONG_CHIRP_S,
    min_units: int = MIN_UNITS_TO_COVER,
) -> TemplateSplits:
    """Assign every unit (fish or group, :func:`fish_units`) to a split.

    Stratification uses the selection that scenes draw from (``families``,
    ``exclude``) and its equalization bins (``bins``, ``spacing``):

    1. Pinned units take their split. The units with at least 100
       templates in the selection (a few of them hold most templates) are
       dealt next, sorted by size, systematically by the fractions (pinned
       ones count), so that template counts follow the unit fractions;
       those holding a bin with fewer than ``min_units`` units go to train.
    2. Coverage: for every property (``long``: a chirp of at least
       ``long_chirp_s``; ``big``; ``small``) and then for every occupied bin,
       from the bin with fewest units on, every split takes one unassigned
       unit with it unless it has one already. Test and val pick before
       train if the property or bin has at least ``min_units`` units, and
       only if a unit is left for train; with fewer units train picks
       first, then test. No split takes more than its fraction of the
       selection's units this way (rarest bins first). For ``long`` the pick is the unit with the most
       long chirps (ties at random), so that val and test hold as many long
       chirps of unseen fish as possible; otherwise it is uniform among the
       candidates with fewer than 100 templates where there are any, which
       leaves the few large groups to the size-balanced step 3 (they occur
       in many bins and would otherwise be picked far more often than
       their share).
    3. The other units are stratified by class (long, big, small, or no
       template in the selection) and size (1, 2--9, 10--99, >= 100
       templates), sorted by size within strata (random ties), and dealt
       systematically (random phase per split), so that the units with
       templates in the selection and those without each end with the
       split fractions, and every stratum and size range gets close to its
       proportional share.

    Parameters
    ----------
    library
        The library (only its metadata is used).
    seed
        Seed of the assignment.
    fractions
        Fractions of units for train, val and test.
    pins
        ``unit -> split`` fixed beforehand (e.g. ``{"competition/g6f5":
        "train"}``).
    families, exclude
        Selection used for stratification (default: the library's default
        selection).
    bins, spacing
        Equalization bins used for coverage.
    long_chirp_s
        Duration (full width at 10%) from which a chirp counts as long.
    min_units
        Units needed before a property or bin is covered in val and test.

    Returns
    -------
    TemplateSplits
    """
    fractions = tuple(float(f) for f in fractions)
    if len(fractions) != len(SPLIT_NAMES) or min(fractions) < 0 or sum(fractions) <= 0:
        msg = f"need three non-negative fractions, got {fractions}"
        raise ValueError(msg)
    pins = dict(pins or {})
    rng = np.random.default_rng(seed)
    meta = library.metadata.reset_index(drop=True)
    unit_of_row = fish_units(meta)
    group_of_row = individual_labels(meta)
    group_unit: dict[str, str] = {}
    for g, u in zip(group_of_row.tolist(), unit_of_row.tolist(), strict=True):
        if group_unit.setdefault(g, u) != u:
            msg = f"group {g!r} holds two units ({group_unit[g]!r}, {u!r})"
            raise ValueError(msg)
    all_units = sorted(set(unit_of_row.tolist()))
    for unit, split in pins.items():
        if unit not in all_units:
            msg = f"pinned unit {unit!r} is not in the library"
            raise KeyError(msg)
        if split not in SPLIT_NAMES:
            msg = f"pin {unit!r}: unknown split {split!r}"
            raise ValueError(msg)

    view = _metadata_view(meta).select(
        families=list(families) if families is not None else None,  # type: ignore[arg-type]
        exclude=exclude,
    )
    sel_units = fish_units(view.metadata)
    eq = view.equalization_bins(bins, spacing=spacing)
    pool_rows = view.rows_of(eq.template_ids)
    pool_units = sel_units[pool_rows]
    sel_meta = view.metadata
    duration = sel_meta[EQUALIZE_DURATION_COLUMN].to_numpy(dtype=np.float64)
    big = sel_meta["family"].astype(str).to_numpy() == "big"
    members: dict[str, set[str]] = {
        "long": set(sel_units[duration >= long_chirp_s].tolist()),
        "big": set(sel_units[big].tolist()),
        "small": set(sel_units[~big].tolist()),
    }
    bin_members: dict[int, set[str]] = {}
    for b, u in zip(eq.bin_of.tolist(), pool_units.tolist(), strict=True):
        bin_members.setdefault(int(b), set()).add(u)
    n_sel = Counter(sel_units.tolist())
    n_all = Counter(unit_of_row.tolist())
    n_long = Counter(sel_units[duration >= long_chirp_s].tolist())

    assigned: dict[str, str] = dict(pins)
    selected = set(n_sel)
    sel_targets: dict[str, int] = dict(
        zip(
            SPLIT_NAMES,
            _largest_remainder(len(selected), fractions, rng).tolist(),
            strict=True,
        )
    )

    def deal(units: list[str], quota: npt.NDArray[np.int64]) -> None:
        """Assign ``units`` (in order) to splits with exact counts ``quota``."""
        for u, k in zip(units, _interleave(quota, rng), strict=True):
            assigned[u] = SPLIT_NAMES[k]

    def quota_for(
        part: list[str], targets: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64]:
        have = np.array([sum(assigned.get(u) == s for u in part) for s in SPLIT_NAMES])
        n_free = sum(u not in assigned for u in part)
        quota = np.maximum(targets - have, 0)
        while quota.sum() > n_free:
            quota[int(np.argmax(quota))] -= 1
        quota[0] += n_free - int(quota.sum())
        return quota

    large = sorted(
        (u for u in selected if n_sel[u] >= _SIZE_TIERS[-1]),
        key=lambda u: (-n_sel[u], u),
    )
    rare = {u for m in bin_members.values() if len(m) < min_units for u in m}
    for u in large:
        if u in rare:
            assigned.setdefault(u, "train")
    deal(
        [u for u in large if u not in assigned],
        quota_for(large, _largest_remainder(len(large), fractions, rng)),
    )

    def full(split: str) -> bool:
        taken = sum(v == split for u, v in assigned.items() if u in selected)
        return taken >= sel_targets[split]

    def choose(free: list[str], by_long: bool) -> str:
        if by_long:
            score = np.array([-n_long[u] for u in free])
        else:
            score = np.array([int(n_sel[u] >= _SIZE_TIERS[-1]) for u in free])
        best = [
            u for u, v in zip(free, score.tolist(), strict=True) if v == score.min()
        ]
        return best[int(rng.integers(len(best)))]

    def cover(group: set[str], by_long: bool = False) -> None:
        order = _COVER_ORDER if len(group) >= min_units else ("train", "test")
        for split in order:
            if any(assigned.get(u) == split for u in group) or full(split):
                continue
            free = sorted(u for u in group if u not in assigned)
            if not free:
                return
            pick = choose(free, by_long)
            if split != "train":
                trainable = [
                    u
                    for u in group
                    if u != pick and assigned.get(u, "train") == "train"
                ]
                if not trainable:
                    continue
            assigned[pick] = split

    for name in _PROPERTIES:
        cover(members[name], by_long=name == "long")
    for b in sorted(bin_members, key=lambda b: (len(bin_members[b]), b)):
        cover(bin_members[b])

    # stratified fill to exact unit quotas
    def stratum(unit: str) -> tuple[int, int]:
        if unit not in n_sel:
            return (len(_PROPERTIES), _tier(n_all[unit]))
        cls = next(i for i, p in enumerate(_PROPERTIES) if unit in members[p])
        return (cls, _tier(n_sel[unit]))

    def size(unit: str) -> int:
        return n_sel[unit] if unit in n_sel else n_all[unit]

    for part in (
        [u for u in all_units if u in selected],
        [u for u in all_units if u not in selected],
    ):
        targets = (
            np.array([sel_targets[s] for s in SPLIT_NAMES])
            if part and part[0] in selected
            else _largest_remainder(len(part), fractions, rng)
        )
        quota = quota_for(part, targets)
        free_units = [u for u in part if u not in assigned]
        strata: dict[tuple[int, int], list[str]] = {}
        for u in free_units:
            strata.setdefault(stratum(u), []).append(u)
        ordered: list[str] = []
        for key in sorted(strata):
            members_k = strata[key]
            tiebreak = rng.permutation(len(members_k))
            sizes = np.array([-size(u) for u in members_k], dtype=np.int64)
            order = np.lexsort((tiebreak, sizes))
            ordered += [members_k[i] for i in order.tolist()]
        deal(ordered, quota)

    ids = meta["template_id"].to_numpy(dtype=np.int64)
    row_split = np.array([assigned[u] for u in unit_of_row], dtype=str)
    template_ids = {s: sorted(ids[row_split == s].tolist()) for s in SPLIT_NAMES}
    provenance = {
        "seed": int(seed),
        "fractions": dict(zip(SPLIT_NAMES, fractions, strict=True)),
        "pins": pins,
        "unit_rule": (
            "competition templates with a known individual (order_meta.csv): "
            "competition/<individual>, one unit per fish across recordings; "
            "all others: their group (gridtools.templates.library."
            "individual_labels)"
        ),
        "stratification": {
            "families": list(families) if families is not None else None,
            "exclude": list(exclude),
            "bins": list(bins),
            "spacing": spacing,
            "duration_column": EQUALIZE_DURATION_COLUMN,
            "long_chirp_s": long_chirp_s,
            "min_units": min_units,
            "n_selected": view.n_templates,
            "n_selected_units": len(n_sel),
            "excursion_edges_hz": eq.excursion_edges_hz.tolist(),
            "duration_edges_s": eq.duration_edges_s.tolist(),
        },
        "algorithm": (
            "gridtools.templates.splits.assign_template_splits: pins; "
            "units >= 100 templates dealt by size; "
            "coverage of long (most long chirps), big, small and every "
            "occupied bin (rarest first; test, val, train; units < 100 "
            "templates preferred); "
            "stratified systematic fill by class "
            "x size, size-sorted, to exact unit quotas (selected and "
            "unselected units separately)"
        ),
        "gridtools_git_sha": _git_sha(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return TemplateSplits(
        units={u: assigned[u] for u in all_units},
        groups=dict(sorted(group_unit.items())),
        template_ids=template_ids,
        provenance=provenance,
    )


def write_splits(
    splits: TemplateSplits, path: Path | str, library_path: Path | str
) -> Path:
    """Write ``splits`` (made for the library at ``library_path``) as JSON.

    The file records the library's path and the SHA-256 of its
    ``manifest.json`` and ``templates.parquet``; :class:`~gridtools.sim.chirps.
    TemplatePool` refuses a splits file made for another build.
    """
    from gridtools.templates.build import MANIFEST_FILE, METADATA_FILE

    library_path = Path(library_path).resolve()
    provenance = {
        **splits.provenance,
        "library": {
            "path": str(library_path),
            "manifest_sha256": file_sha256(library_path / MANIFEST_FILE),
            "metadata_sha256": file_sha256(library_path / METADATA_FILE),
            "n_templates": sum(len(v) for v in splits.template_ids.values()),
        },
    }
    units: dict[str, dict[str, Any]] = {
        u: {"split": s, "groups": []} for u, s in splits.units.items()
    }
    for g, u in splits.groups.items():
        units[u]["groups"].append(g)
    document = {
        "format_version": SPLITS_FORMAT_VERSION,
        "provenance": provenance,
        "counts": splits.counts(),
        "groups": splits.group_split(),
        "units": units,
        "template_ids": splits.template_ids,
    }
    path = Path(path)
    path.write_text(json.dumps(document, indent=1) + "\n")
    return path


def load_splits(path: Path | str) -> TemplateSplits:
    """Read a splits file written by :func:`write_splits`."""
    document = json.loads(Path(path).read_text())
    version = document.get("format_version")
    if version != SPLITS_FORMAT_VERSION:
        msg = f"unsupported splits format version {version!r} in {path}"
        raise ValueError(msg)
    units = {u: str(v["split"]) for u, v in document["units"].items()}
    groups = {g: u for u, v in document["units"].items() for g in v["groups"]}
    for g, s in document["groups"].items():
        if units.get(groups.get(g, "")) != s:
            msg = f"group {g!r}: split {s!r} disagrees with its unit"
            raise ValueError(msg)
    return TemplateSplits(
        units=units,
        groups=groups,
        template_ids={
            s: [int(t) for t in v] for s, v in document["template_ids"].items()
        },
        provenance=document["provenance"],
    )


# -- report -------------------------------------------------------------------


def split_bins(
    view: TemplateLibrary,
    split: npt.NDArray[np.str_],
    *,
    bins: tuple[int, int] = DEFAULT_BINS,
    spacing: Spacing = DEFAULT_SPACING,
) -> tuple[EqualizationBins, dict[str, EqualizationBins | None]]:
    """Bins of a selection and of each split's part on the same edges.

    ``split`` gives the split of every row of ``view``. A split without
    templates maps to None.
    """
    full = view.equalization_bins(bins, spacing=spacing)
    edges = (full.excursion_edges_hz, full.duration_edges_s)
    parts: dict[str, EqualizationBins | None] = {}
    for s in SPLIT_NAMES:
        ids = view.template_ids[split == s]
        parts[s] = (
            view.select(template_ids=ids.tolist(), exclude=()).equalization_bins(
                bins, spacing=spacing, edges=edges
            )
            if ids.size
            else None
        )
    return full, parts


def _bin_label(eq: EqualizationBins, b: int) -> str:
    n_dur = eq.counts.shape[1]
    i, j = divmod(b, n_dur)
    exc, dur = eq.excursion_edges_hz, 1e3 * eq.duration_edges_s
    return f"{exc[i]:.0f}-{exc[i + 1]:.0f} Hz x {dur[j]:.0f}-{dur[j + 1]:.0f} ms"


def _draw_shares(
    view: TemplateLibrary,
    eq: EqualizationBins,
    alpha: float,
    balance: Balance,
    long_chirp_s: float,
) -> dict[str, float]:
    p = eq.template_probabilities(alpha, balance)
    rows = view.rows_of(eq.template_ids)
    meta = view.metadata.iloc[rows]
    big = meta["family"].astype(str).to_numpy() == "big"
    long = meta[EQUALIZE_DURATION_COLUMN].to_numpy(dtype=np.float64) >= long_chirp_s
    harvested = (
        meta["source"].astype(str).to_numpy() == "competition"
        if "source" in meta.columns
        else np.zeros(len(meta), dtype=bool)
    )
    return {
        "big": float(p[big].sum()),
        "small": float(p[~big].sum()),
        "long": float(p[long].sum()),
        "harvested": float(p[harvested].sum()),
    }


def splits_report(
    library: TemplateLibrary,
    splits: TemplateSplits,
    *,
    families: Iterable[str] | None = None,
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
    bins: tuple[int, int] = DEFAULT_BINS,
    spacing: Spacing = DEFAULT_SPACING,
    alpha: float = DEFAULT_ALPHA,
    balance: Balance = DEFAULT_BALANCE,
    long_chirp_s: float = LONG_CHIRP_S,
    title: str = "Template splits",
) -> str:
    """Markdown report of a split assignment on a selection of the library.

    Per split: units, groups, templates, big, small, long and harvested
    templates of the selection; bin coverage and expected draw shares with
    the given sampling options (bins on the selection's edges, empty bins
    skipped); bins occupied in train but empty in val or test.
    """
    view = _metadata_view(library.metadata).select(
        families=list(families) if families is not None else None,  # type: ignore[arg-type]
        exclude=exclude,
    )
    meta = view.metadata
    split = splits.split_of(meta)
    units = fish_units(meta)
    groups = individual_labels(meta)
    duration = meta[EQUALIZE_DURATION_COLUMN].to_numpy(dtype=np.float64)
    big = meta["family"].astype(str).to_numpy() == "big"
    long = duration >= long_chirp_s
    harvested = (
        meta["source"].astype(str).to_numpy() == "competition"
        if "source" in meta.columns
        else np.zeros(len(meta), dtype=bool)
    )
    full, parts = split_bins(view, split, bins=bins, spacing=spacing)
    full_p = full.bin_probabilities(alpha)
    occupied = np.flatnonzero(full.counts.ravel())
    prov = splits.provenance
    lib = prov.get("library", {})
    lines = [
        f"# {title}",
        "",
        f"Created {prov.get('created_utc', '?')} by gridtools "
        f"`{str(prov.get('gridtools_git_sha') or '?')[:10]}` for the library "
        f"`{lib.get('path', '?')}` (manifest SHA-256 "
        f"`{str(lib.get('manifest_sha256', '?'))[:12]}`). Seed "
        f"{prov.get('seed')}, fractions of units "
        + "/".join(f"{v:g}" for v in prov.get("fractions", {}).values())
        + ", pins: "
        + (", ".join(f"`{u}` → {s}" for u, s in prov.get("pins", {}).items()) or "none")
        + ".",
        "",
        "**Units.** " + str(prov.get("unit_rule", "")) + ". A unit is in "
        "exactly one split; so is every group and template.",
        "",
        "**Assignment.** " + str(prov.get("algorithm", "")) + ".",
        "",
        f"## Splits on the default selection ({view.n_templates} templates)",
        "",
        f"Long: full width at 10% ≥ {1e3 * long_chirp_s:.0f} ms. Units and "
        "groups count those with templates in the selection; *library units* "
        "counts every unit of the library.",
        "",
        "| split | library units | units | groups | templates | big | small "
        "| long | harvested | occupied bins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    all_units = splits.units
    for s in SPLIT_NAMES:
        mask = split == s
        eq = parts[s]
        n_bins = eq.n_nonempty if eq is not None else 0
        lines.append(
            f"| {s} | {sum(v == s for v in all_units.values())} "
            f"| {len(set(units[mask].tolist()))} | {len(set(groups[mask].tolist()))} "
            f"| {int(mask.sum())} | {int((mask & big).sum())} "
            f"| {int((mask & ~big).sum())} | {int((mask & long).sum())} "
            f"| {int((mask & harvested).sum())} | {n_bins} of {occupied.size} |"
        )
    lines += [
        "",
        f"## Draw shares (log bins on the selection's edges, alpha = {alpha:g}, "
        f"{balance} balance)",
        "",
        "Every split is binned on the edges of the whole selection; bins empty "
        "in a split are skipped and the other bins' weights renormalised. "
        "*Lost share*: draw share the empty bins have on the whole selection.",
        "",
        "| pool | big | small | long | harvested | empty bins | lost share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    shares = _draw_shares(view, full, alpha, balance, long_chirp_s)
    lines.append(
        f"| all | {shares['big']:.1%} | {shares['small']:.1%} "
        f"| {shares['long']:.1%} | {shares['harvested']:.1%} | 0 | 0.0% |"
    )
    for s in SPLIT_NAMES:
        eq = parts[s]
        if eq is None:
            lines.append(f"| {s} | - | - | - | - | {occupied.size} | 100% |")
            continue
        shares = _draw_shares(view, eq, alpha, balance, long_chirp_s)
        empty = [b for b in occupied if eq.counts.ravel()[b] == 0]
        lines.append(
            f"| {s} | {shares['big']:.1%} | {shares['small']:.1%} "
            f"| {shares['long']:.1%} | {shares['harvested']:.1%} | {len(empty)} "
            f"| {float(full_p[empty].sum()):.1%} |"
        )
    lines += [
        "",
        "## Bins empty in val or test",
        "",
        "Occupied bins (on the whole selection) without a template in val or "
        "test: templates (units) per split and the bin's draw share on the "
        "whole selection.",
        "",
        "| bin | train | val | test | share |",
        "|---|---:|---:|---:|---:|",
    ]
    unit_codes = {u: i for i, u in enumerate(sorted(set(units.tolist())))}
    pool_units = {
        s: (
            np.array([unit_codes[u] for u in units[view.rows_of(eq.template_ids)]])
            if (eq := parts[s]) is not None
            else None
        )
        for s in SPLIT_NAMES
    }

    def cell(s: str, b: int) -> str:
        eq = parts[s]
        codes = pool_units[s]
        if eq is None or codes is None:
            return "·"
        n = int(eq.counts.ravel()[b])
        if n == 0:
            return "·"
        return f"{n} ({np.unique(codes[eq.bin_of == b]).size})"

    gaps = [
        int(b)
        for b in occupied
        if any(
            (eq := parts[s]) is None or eq.counts.ravel()[b] == 0
            for s in ("val", "test")
        )
    ]
    for b in gaps:
        lines.append(
            f"| {_bin_label(full, int(b))} | {cell('train', b)} | {cell('val', b)} "
            f"| {cell('test', b)} | {full_p[b]:.2%} |"
        )
    if not gaps:
        lines.append("| (none) | | | | |")
    exc, dur = full.excursion_edges_hz, 1e3 * full.duration_edges_s
    n_exc, n_dur = full.counts.shape
    header = "| | " + " | ".join(
        f"{dur[j]:.0f}-{dur[j + 1]:.0f} ms" for j in range(n_dur)
    )
    for s in SPLIT_NAMES:
        eq = parts[s]
        codes = pool_units[s]
        thin = 0
        lines += [
            "",
            f"### Units per bin: {s}",
            "",
            "Units (fish or groups) with templates in the bin; `0`: occupied in "
            "the selection but empty in this split, `·`: empty in the selection.",
            "",
            header + " |",
            "|---|" + "---:|" * n_dur,
        ]
        for i in range(n_exc):
            row = []
            for j in range(n_dur):
                b = i * n_dur + j
                if full.counts.ravel()[b] == 0:
                    row.append("·")
                    continue
                n = 0
                if eq is not None and codes is not None and eq.counts.ravel()[b]:
                    n = int(np.unique(codes[eq.bin_of == b]).size)
                thin += n == 1
                row.append(str(n))
            lines.append(
                f"| {exc[i]:.0f}-{exc[i + 1]:.0f} Hz | " + " | ".join(row) + " |"
            )
        lines += ["", f"Bins with a single unit in {s}: {thin}."]
    lines += [
        "",
        "## Harvested competition fish",
        "",
        "| unit | split | recordings (groups) | templates | long |",
        "|---|---|---|---:|---:|",
    ]
    for u in sorted({u for u, h in zip(units, harvested, strict=True) if h}):
        mask = units == u
        member_groups = sorted(set(groups[mask].tolist()))
        lines.append(
            f"| `{u}` | {all_units[u]} | {', '.join(member_groups)} "
            f"| {int(mask.sum())} | {int((mask & long).sum())} |"
        )
    lines.append("")
    return "\n".join(lines)
