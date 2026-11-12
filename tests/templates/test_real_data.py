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
    assert clean.n_templates > 10000
    draw = clean.sample(np.random.default_rng(0), 100)
    assert len(set(draw.template_ids.tolist())) == 100
