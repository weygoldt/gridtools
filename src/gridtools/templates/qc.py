"""Quality control: duplicates and outliers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def exact_duplicates(hashes: Sequence[bytes]) -> IntArray:
    """Map every row to the first row with an identical hash.

    Parameters
    ----------
    hashes
        One digest per row.

    Returns
    -------
    numpy.ndarray
        ``duplicate_of[i]`` is the index of the first earlier row with the
        same hash, or -1 if row ``i`` is the first of its kind.
    """
    first: dict[bytes, int] = {}
    out = np.full(len(hashes), -1, dtype=np.int64)
    for i, digest in enumerate(hashes):
        if digest in first:
            out[i] = first[digest]
        else:
            first[digest] = i
    return out


def _standardise(rows: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    x = np.asarray(rows, dtype=np.float32)
    x = x - x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (x / norm).astype(np.float32)


def max_lag_correlation(
    a: FloatArray, b: FloatArray, max_lag: int
) -> tuple[float, int]:
    """Largest Pearson correlation of ``a`` and ``b`` over integer lags.

    Parameters
    ----------
    a, b
        Equal-length 1-D traces.
    max_lag
        Lags ``-max_lag..max_lag`` are tested; samples shifted out are
        dropped (no wrap-around).

    Returns
    -------
    corr : float
        Best correlation.
    lag : int
        Lag of ``b`` relative to ``a`` at the best correlation.
    """
    best, best_lag = -np.inf, 0
    n = a.size
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[lag:], b[: n - lag]
        else:
            x, y = a[: n + lag], b[-lag:]
        if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if c > best:
            best, best_lag = c, lag
    return float(best), best_lag


@dataclass(frozen=True, slots=True)
class NearDuplicateResult:
    """Outcome of :func:`near_duplicates`.

    Attributes
    ----------
    duplicate_of
        Index of the kept row of the duplicate group, or -1.
    max_shape_corr
        Each row's highest zero-lag shape correlation with any other row.
    shape_partner
        Row reaching ``max_shape_corr`` (-1 if none).
    max_fingerprint_corr
        Each row's best lag-searched fingerprint correlation with a candidate
        partner (NaN if it had no candidate).
    pairs
        ``(i, j, fingerprint_corr, shape_corr, lag)`` for every scored
        candidate pair (``i < j``).
    """

    duplicate_of: IntArray
    max_shape_corr: FloatArray
    shape_partner: IntArray
    max_fingerprint_corr: FloatArray
    pairs: list[tuple[int, int, float, float, int]]


def near_duplicates(
    shape: npt.NDArray[np.floating],
    fingerprint: npt.NDArray[np.floating],
    *,
    shape_corr_min: float,
    fingerprint_corr_min: float,
    candidate_corr_min: float = 0.5,
    max_lag: int = 20,
    exclude: npt.NDArray[np.bool_] | None = None,
    block_rows: int = 512,
) -> NearDuplicateResult:
    """Find rows that are re-extractions of the same chirp.

    Two aligned rows are near-duplicates when their frequency traces have
    the same shape (``shape`` correlation >= ``shape_corr_min``) *and* the
    same cycle-to-cycle jitter (``fingerprint`` correlation >=
    ``fingerprint_corr_min``). The jitter (``orig_freq`` minus its smoothed
    version) is independent between recordings, so it separates repeated
    extractions of one chirp from distinct but stereotyped chirps, which can
    reach a shape correlation of 0.999 on their own.

    Candidate pairs are those with zero-lag fingerprint correlation of at
    least ``candidate_corr_min``; each is rescored over lags
    ``-max_lag..max_lag``. Groups are connected components of duplicate
    pairs; the lowest index of a group is kept.

    Parameters
    ----------
    shape, fingerprint
        Arrays (n_rows, n_samples) of aligned traces.
    shape_corr_min, fingerprint_corr_min
        Thresholds for a duplicate pair.
    candidate_corr_min
        Zero-lag fingerprint pre-filter.
    max_lag
        Lag range for rescoring candidates.
    exclude
        Rows to ignore (e.g. exact duplicates, invalid rows).
    block_rows
        Rows per block of the correlation matrices.

    Returns
    -------
    NearDuplicateResult
    """
    n = shape.shape[0]
    keep = np.ones(n, dtype=bool) if exclude is None else ~exclude
    index = np.flatnonzero(keep)
    zs = _standardise(shape[index])
    zf = _standardise(fingerprint[index])
    max_shape = np.full(n, np.nan)
    partner = np.full(n, -1, dtype=np.int64)
    candidates: list[tuple[int, int]] = []
    for start in range(0, index.size, block_rows):
        stop = min(start + block_rows, index.size)
        local = np.arange(stop - start)
        block_s = zs[start:stop] @ zs.T
        block_s[local, start + local] = -np.inf
        best = np.argmax(block_s, axis=1)
        max_shape[index[start:stop]] = block_s[local, best]
        partner[index[start:stop]] = index[best]
        block_f = zf[start:stop] @ zf.T
        rows, cols = np.nonzero(block_f >= candidate_corr_min)
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            i, j = start + r, c
            if i < j:
                candidates.append((int(index[i]), int(index[j])))
    log.info("near-duplicate candidates: %d pairs", len(candidates))

    max_fp = np.full(n, np.nan)
    pairs: list[tuple[int, int, float, float, int]] = []
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for i, j in candidates:
        fp_corr, lag = max_lag_correlation(
            np.asarray(fingerprint[i], dtype=np.float64),
            np.asarray(fingerprint[j], dtype=np.float64),
            max_lag,
        )
        shape_corr = _lagged_corr(
            np.asarray(shape[i], dtype=np.float64),
            np.asarray(shape[j], dtype=np.float64),
            lag,
        )
        pairs.append((i, j, fp_corr, shape_corr, lag))
        max_fp[i] = np.fmax(max_fp[i], fp_corr)
        max_fp[j] = np.fmax(max_fp[j], fp_corr)
        if fp_corr >= fingerprint_corr_min and shape_corr >= shape_corr_min:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

    duplicate_of = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        root = find(i)
        if root != i:
            duplicate_of[i] = root
    return NearDuplicateResult(duplicate_of, max_shape, partner, max_fp, pairs)


def _lagged_corr(a: FloatArray, b: FloatArray, lag: int) -> float:
    n = a.size
    x, y = (a[lag:], b[: n - lag]) if lag >= 0 else (a[: n + lag], b[-lag:])
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def robust_z(values: npt.ArrayLike) -> FloatArray:
    """Robust z-score: ``(x - median) / (1.4826 * MAD)``; NaN stays NaN."""
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() < 3:
        return np.full_like(x, np.nan)
    med = np.median(x[finite])
    mad = 1.4826 * np.median(np.abs(x[finite] - med))
    if mad == 0:
        return np.where(finite, 0.0, np.nan)
    return (x - med) / mad
