"""Merge a built library with harvested templates into a new library version.

Library v3 = the v2 chirp-chamber templates plus the long chirps harvested
from the competition recordings (:mod:`gridtools.templates.harvest`), minus
those vetoed in ``review.csv``.

Template length
---------------
Every template keeps the dense, fixed-length layout of v1/v2: arrays
``(n_templates, n_samples)`` with the alignment point at ``n_samples // 2``
and exact 0 / 1 outside the support, so ``TemplateLibrary.delta_f_hz[row]``,
``get`` and ``sample`` work unchanged and return 2-D arrays. The length is
the smallest multiple of 0.1 s that holds the widest harvested support
(the v2 templates are padded on both sides; the padding is exact). Ragged
storage would save disk but break consumers that index rows of the
memory-mapped arrays directly (the scene simulator does); with
``load_library(mmap=True)`` only the drawn rows are read, so the longer rows
cost disk (float32, ``2 * n * n_samples * 4`` bytes) but not memory.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from gridtools.templates.build import (
    AMP_FILE,
    DELTA_FILE,
    LIBRARY_FORMAT_VERSION,
    MANIFEST_FILE,
    METADATA_FILE,
    PAIRS_FILE,
    _code_version,
    _file_digest,
    _json_default,
    qc_counts,
)
from gridtools.templates.harvest import (
    HARVEST_AMP_FILE,
    HARVEST_DELTA_FILE,
    HARVEST_MANIFEST_FILE,
    HARVEST_METADATA_FILE,
    read_review,
)
from gridtools.templates.library import KNOWN_FLAGS, flag_mask_of

log = logging.getLogger(__name__)

Frame = Any
REVIEW_COPY = "review.csv"


def resize_centred(
    rows: npt.NDArray[np.floating], n_samples: int, fill: float
) -> npt.NDArray[np.float32]:
    """Pad or crop templates around their centre sample.

    Sample ``n_old // 2`` of every row becomes sample ``n_samples // 2``.
    Padding uses ``fill``; cropping is allowed only where the cropped samples
    equal ``fill`` exactly.

    Parameters
    ----------
    rows
        Array (n, n_old).
    n_samples
        New length.
    fill
        0 for frequency, 1 for amplitude templates.

    Returns
    -------
    numpy.ndarray
        New float32 array (n, n_samples).

    Raises
    ------
    ValueError
        If cropping would remove samples that differ from ``fill``.
    """
    rows = np.asarray(rows)
    n, n_old = rows.shape
    shift = n_samples // 2 - n_old // 2
    out = np.full((n, n_samples), fill, dtype=np.float32)
    src_lo = max(0, -shift)
    src_hi = min(n_old, n_samples - shift)
    cropped = np.ones(n_old, dtype=bool)
    cropped[src_lo:src_hi] = False
    if cropped.any() and np.any(rows[:, cropped] != fill):
        msg = f"cropping to {n_samples} samples would cut template support"
        raise ValueError(msg)
    out[:, src_lo + shift : src_hi + shift] = rows[:, src_lo:src_hi]
    return out


def support_half_width(
    delta: npt.NDArray[np.floating], amp: npt.NDArray[np.floating]
) -> int:
    """Largest distance (samples) of a non-baseline sample from the centre."""
    centre = delta.shape[1] // 2
    active = (np.asarray(delta) != 0) | (np.asarray(amp) != 1)
    if not active.any():
        return 0
    columns = np.flatnonzero(active.any(axis=0))
    return int(max(centre - columns[0], columns[-1] - centre))


def merged_length(
    n_base: int, harvest_half_width: int, samplerate_hz: float, step_s: float = 0.1
) -> int:
    """Smallest length (multiple of ``step_s``) holding both template sets."""
    step = round(step_s * samplerate_hz)
    needed = max(n_base, 2 * harvest_half_width + 2)
    return int(np.ceil(needed / step) * step)


def build_merged_library(
    base_dir: Path | str,
    harvest_dir: Path | str,
    out_dir: Path | str,
    *,
    review_csv: Path | str | None = None,
    template_duration_s: float | None = None,
    verify_base: bool = True,
    block_rows: int = 1024,
) -> dict[str, Any]:
    """Write a library of the base templates plus accepted harvested ones.

    Base templates keep their ``template_id`` (row ``i`` is id ``i``);
    harvested ones follow (``template_id = n_base + k`` in harvest order).
    Templates vetoed in the review sheet (``accept`` False) are left out.
    Deterministic for fixed inputs.

    Parameters
    ----------
    base_dir
        Library to extend (e.g. v2); read-only.
    harvest_dir
        ``harvest/`` directory written by
        :func:`~gridtools.templates.harvest.run_harvest`.
    out_dir
        Output directory (created; existing library files are overwritten).
    review_csv
        Review sheet; default ``harvest_dir/../review.csv``. Copied into
        ``out_dir``.
    template_duration_s
        Template length; default: the smallest multiple of 0.1 s that holds
        every template.
    verify_base
        Check the base arrays against the SHA-256 in its manifest.
    block_rows
        Rows copied at a time (memory).

    Returns
    -------
    dict
        The manifest written.
    """
    base_dir, harvest_dir, out_dir = Path(base_dir), Path(harvest_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(tz=datetime.UTC)
    base_manifest = json.loads((base_dir / MANIFEST_FILE).read_text())
    if verify_base:
        for name in (DELTA_FILE, AMP_FILE, METADATA_FILE):
            expected = base_manifest["outputs"][name]["sha256"]
            if _file_digest(base_dir / name)["sha256"] != expected:
                msg = f"{base_dir / name} does not match its manifest"
                raise ValueError(msg)
    base_delta = np.load(base_dir / DELTA_FILE, mmap_mode="r")
    base_amp = np.load(base_dir / AMP_FILE, mmap_mode="r")
    base_meta: Frame = pd.read_parquet(base_dir / METADATA_FILE)
    fs = float(base_manifest["samplerate_hz"])

    harvest_manifest = json.loads((harvest_dir / HARVEST_MANIFEST_FILE).read_text())
    if float(harvest_manifest["samplerate_hz"]) != fs:
        msg = "harvest and base library differ in sampling rate"
        raise ValueError(msg)
    h_delta = np.load(harvest_dir / HARVEST_DELTA_FILE)
    h_amp = np.load(harvest_dir / HARVEST_AMP_FILE)
    h_meta: Frame = pd.read_parquet(harvest_dir / HARVEST_METADATA_FILE)

    review_path = (
        Path(review_csv)
        if review_csv is not None
        else harvest_dir.parent / "review.csv"
    )
    review: Frame = read_review(review_path)
    accept = dict(zip(review["harvest_key"], review["accept"], strict=True))
    missing = [k for k in h_meta["harvest_key"] if k not in accept]
    if missing:
        msg = f"{len(missing)} harvested templates are not in {review_path}"
        raise ValueError(msg)
    keep = np.array([bool(accept[k]) for k in h_meta["harvest_key"]], dtype=bool)
    vetoed = sorted(h_meta.loc[~keep, "harvest_key"].tolist())
    h_delta, h_amp = h_delta[keep], h_amp[keep]
    h_meta = h_meta.loc[keep].reset_index(drop=True)
    notes = dict(zip(review["harvest_key"], review["note"].fillna(""), strict=True))

    n_base = base_delta.shape[0]
    if template_duration_s is None:
        n_out = merged_length(
            base_delta.shape[1], support_half_width(h_delta, h_amp), fs
        )
    else:
        n_out = round(template_duration_s * fs)
    n_total = n_base + len(h_meta)
    log.info(
        "merging %d base + %d harvested templates (%d vetoed) at %d samples",
        n_base,
        len(h_meta),
        len(vetoed),
        n_out,
    )

    delta_out = np.lib.format.open_memmap(
        out_dir / DELTA_FILE, mode="w+", dtype=np.float32, shape=(n_total, n_out)
    )
    amp_out = np.lib.format.open_memmap(
        out_dir / AMP_FILE, mode="w+", dtype=np.float32, shape=(n_total, n_out)
    )
    for a in range(0, n_base, block_rows):
        b = min(a + block_rows, n_base)
        delta_out[a:b] = resize_centred(base_delta[a:b], n_out, 0.0)
        amp_out[a:b] = resize_centred(base_amp[a:b], n_out, 1.0)
    if len(h_meta):
        delta_out[n_base:] = resize_centred(h_delta, n_out, 0.0)
        amp_out[n_base:] = resize_centred(h_amp, n_out, 1.0)
    delta_out.flush()
    amp_out.flush()
    del delta_out, amp_out

    meta = merged_metadata(base_meta, h_meta, notes)
    meta.to_parquet(out_dir / METADATA_FILE, index=False)
    pairs = base_dir / PAIRS_FILE
    if pairs.exists():
        pd.read_parquet(pairs).to_parquet(out_dir / PAIRS_FILE, index=False)
    (out_dir / REVIEW_COPY).write_bytes(review_path.read_bytes())

    outputs = [DELTA_FILE, AMP_FILE, METADATA_FILE, REVIEW_COPY]
    if (out_dir / PAIRS_FILE).exists():
        outputs.append(PAIRS_FILE)
    manifest: dict[str, Any] = {
        "library_format_version": LIBRARY_FORMAT_VERSION,
        "created_utc": started.isoformat(),
        "samplerate_hz": fs,
        "n_templates": int(n_total),
        "n_samples": int(n_out),
        "centre_index": int(n_out // 2),
        "config": base_manifest["config"],
        "harvest_config": harvest_manifest["config"],
        "source": {
            "base_library": {
                "path": str(base_dir.resolve()),
                "manifest_sha256": _sha256(base_dir / MANIFEST_FILE),
                "n_templates": int(n_base),
                "n_samples": int(base_delta.shape[1]),
                "source": base_manifest.get("source"),
            },
            "harvest": {
                "path": str(harvest_dir.resolve()),
                "manifest_sha256": _sha256(harvest_dir / HARVEST_MANIFEST_FILE),
                "n_harvested": int(keep.size),
                "n_accepted": int(keep.sum()),
                "recordings": harvest_manifest.get("recordings"),
            },
            "review": {
                "path": str(review_path.resolve()),
                "sha256": _sha256(review_path),
                "vetoed": vetoed,
            },
        },
        "code": _code_version(),
        "outputs": {name: _file_digest(out_dir / name) for name in outputs},
        "counts": qc_counts(meta),
    }
    manifest["counts"]["per_source"] = {
        str(k): int(v) for k, v in meta["source"].value_counts().items()
    }
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, default=_json_default)
    )
    log.info("merged library written to %s", out_dir)
    return manifest


def merged_metadata(
    base_meta: Frame, harvest_meta: Frame, notes: dict[str, str] | None = None
) -> pd.DataFrame:
    """Metadata of the merged library (union of columns, provenance added).

    Base rows get ``source = "chirp_chamber"`` and keep their ids; harvested
    rows get ``source = "competition"``, ids after the base, and
    ``inferred_group`` values after the base's (one group per recording and
    emitter track; ``group_label`` names it). Every known flag becomes a bool
    column: derived for libraries built before it existed (``too_short``,
    ``brief_cycles``), False where it does not apply (harvest-only flags on
    chirp-chamber rows).
    """
    base: Frame = base_meta.copy()
    n_base = len(base)
    base["source"] = "chirp_chamber"
    base["group_label"] = [f"chirp_chamber/{int(g)}" for g in base["inferred_group"]]
    base["review_accept"] = True
    base["review_note"] = ""
    if harvest_meta is None or len(harvest_meta) == 0:
        return _complete_flags(base)
    harvest: Frame = harvest_meta.copy()
    harvest["template_id"] = n_base + np.arange(len(harvest), dtype=np.int64)
    harvest["source_row"] = -1
    first_group = int(base["inferred_group"].max()) + 1 if n_base else 0
    harvest["inferred_group"] = first_group + harvest["inferred_group"].to_numpy(
        dtype=np.int64
    )
    harvest["in_v1_clustered"] = False
    harvest["v1_cluster_label"] = -1
    harvest["max_shape_corr"] = np.nan
    harvest["max_shape_corr_partner"] = -1
    harvest["max_fingerprint_corr"] = np.nan
    harvest["review_accept"] = True
    harvest["review_note"] = [
        (notes or {}).get(k, "") for k in harvest["harvest_key"].astype(str)
    ]
    merged = pd.concat([base, harvest], ignore_index=True, sort=False)
    return _complete_flags(merged)


def _complete_flags(meta: Frame) -> pd.DataFrame:
    """Write every known flag as a bool column (derived or False where missing)."""
    meta = meta.copy()
    for flag in KNOWN_FLAGS:
        if flag in ("invalid", "duplicate"):
            continue
        if flag in ("too_short", "brief_cycles") and flag in meta.columns:
            # derived flags: fill rows of sources built before the flag existed
            derived = flag_mask_of(meta.drop(columns=flag), flag)
            given = meta[flag]
            meta[flag] = np.where(
                given.isna().to_numpy(),
                derived,
                given.fillna(False).to_numpy(dtype=bool),
            )
        else:
            meta[flag] = flag_mask_of(meta, flag)
    for column in meta.columns:
        if column == "hint_agrees":  # None = v1 had no track; keep it
            continue
        values = meta[column]
        if values.dtype == object and set(values.dropna().map(type)) <= {
            bool,
            np.bool_,
        }:
            meta[column] = values.fillna(False).astype(bool)
    return meta


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()
