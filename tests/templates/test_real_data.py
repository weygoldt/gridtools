"""Build the library from the real extraction file, if it is available."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gridtools.templates import BuildConfig, build_library, load_library

REAL_H5 = Path(
    "/home/weygoldt/wrk/analyses/mscthesis_chirpdetection/data/interrim/"
    "new_chirpextractions/extracted_chirps.h5"
)


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.skipif(not REAL_H5.exists(), reason="real chirp extractions not available")
def test_build_from_real_extractions(tmp_path: Path) -> None:
    before = REAL_H5.stat()
    build_library(REAL_H5, tmp_path, BuildConfig())
    after = REAL_H5.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)

    lib = load_library(tmp_path, mmap=True)
    assert lib.n_templates == 13015
    assert lib.samplerate_hz == 20000.0
    assert lib.n_samples == 8000  # the full source window: no length crop
    assert np.all(lib.delta_f_hz[:, 0] == 0.0)
    assert np.all(lib.delta_f_hz[:, -1] == 0.0)
    assert np.all(lib.amp_rel[:, 0] == 1.0)
    assert np.all(lib.amp_rel[:, -1] == 1.0)
    meta = lib.metadata
    # the v1 clustered set is an exact subset of the source rows
    assert np.count_nonzero(meta["in_v1_clustered"].to_numpy(dtype=bool)) == 9082
    assert np.count_nonzero(meta["duplicate_kind"].to_numpy() == "exact") == 90
    assert np.isfinite(meta["baseline_hz"].to_numpy(dtype=float)).mean() > 0.99
    clean = lib.select()
    assert clean.n_templates > 12000
    cleaned = clean.metadata
    # long and large chirps are kept; nothing in the default set is cut by the template
    assert np.max(cleaned["duration_10_s"].to_numpy(dtype=float)) > 0.1
    assert np.max(cleaned["excursion_hz"].to_numpy(dtype=float)) > 550
    assert not np.any(cleaned["template_cut"].to_numpy(dtype=bool))
    natural = clean.sample(np.random.default_rng(0), 100, equalize=False)
    assert len(set(natural.template_ids.tolist())) == 100
    eq = clean.equalization_bins()
    ids = clean.cycler(np.random.default_rng(1)).draw(50_000)
    rows = clean.rows_of(ids)
    flat = eq.bin_of[np.searchsorted(eq.template_ids, clean.template_ids[rows])]
    per_bin = np.bincount(flat, minlength=eq.counts.size)[eq.counts.ravel() > 0]
    assert per_bin.max() - per_bin.min() <= 1
