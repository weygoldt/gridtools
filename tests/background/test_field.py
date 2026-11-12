"""Fish-free scan of field logger deployments (synthetic WavPack chunks)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from gridtools.background.build import BackgroundConfig, build_background_library
from gridtools.background.field import (
    FieldFolder,
    FieldScanConfig,
    decode_wv,
    fish_free_test,
    probe_wv,
)
from gridtools.background.library import BackgroundLibrary
from gridtools.background.select import SelectionConfig

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe",
)

FS = 8000
N_CH = 4
CHUNK_S = 30.0


def _write_wv(path: Path, x: np.ndarray) -> None:
    wav = path.with_suffix(".wav")
    wavfile.write(wav, FS, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-c:a", "wavpack", str(path)],
        check=True,
    )
    wav.unlink()


def _deployment(root: Path, fish_in: set[int], n_chunks: int = 4) -> FieldFolder:
    folder = root / "site_x" / "logger1-20260101T1200-1"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    n = int(CHUNK_S * FS)
    t = np.arange(n) / FS
    for k in range(n_chunks):
        x = rng.normal(0, 0.01, (n, N_CH))
        if k in fish_in:
            x[:, 2] += 0.002 * np.sin(2 * np.pi * 733.0 * t)  # weak fish, one electrode
        _write_wv(folder / f"logger1-20260101T12{k:02d}00.wv", x)
    layout = root / "site_x" / "electrode_layout.json"
    layout.write_text(
        json.dumps(
            {"coordinates": [[0, 0, -20], [0, 60, -20], [60, 0, -20], [60, 60, -20]]}
        )
    )
    return FieldFolder(
        dataset="field_test",
        site="site_x",
        name=f"site_x/{folder.name}",
        path=folder,
        layout_path=layout,
        group="field_test/site_x",
    )


def test_decode_and_probe(tmp_path: Path) -> None:
    folder = _deployment(tmp_path, fish_in=set(), n_chunks=1)
    path = folder.chunks()[0]
    rate, channels, duration = probe_wv(path)
    assert (rate, channels) == (FS, N_CH)
    assert duration == pytest.approx(CHUNK_S)
    x = decode_wv(path, 5.0, 2.0, N_CH)
    assert x.shape == (2 * FS, N_CH)
    assert x.dtype == np.float32
    assert 0.005 < float(x.std()) < 0.02


def test_fish_free_test_finds_a_weak_line_on_one_electrode() -> None:
    rng = np.random.default_rng(1)
    n = 20 * FS
    x = rng.normal(0, 0.01, (n, N_CH))
    ok, peaks, _ = fish_free_test(x, FS)
    assert ok
    assert peaks == []
    x[:, 3] += 0.001 * np.sin(2 * np.pi * 512.3 * np.arange(n) / FS)
    ok, peaks, excess = fish_free_test(x, FS)
    assert not ok
    assert any(abs(p - 512.3) < 3 for p in peaks)
    assert excess > 8


def test_scan_keeps_only_fish_free_windows(tmp_path: Path) -> None:
    folder = _deployment(tmp_path, fish_in={1})
    config = BackgroundConfig(
        selection=SelectionConfig(segment_s=20.0),
        field=FieldScanConfig(probes_per_folder=4, segments_per_folder=2),
        workers=1,
    )
    root = tmp_path / "lib"
    manifest = build_background_library([folder], root, config)
    info = (root / "recordings.csv").read_text()
    assert "field_test" in info
    assert len(manifest) == 2
    assert set(manifest["segment_class"]) == {"fish_free"}
    assert bool(manifest["screened"].all())
    assert bool(manifest["geometry_known"].all())
    for row in manifest.to_dict("records"):
        chunk = int(row["start_s"] // CHUNK_S)
        assert chunk != 1  # the chunk with a fish
        np.testing.assert_allclose(
            row["electrode_positions_m"], [[0, 0], [0, 0.6], [0.6, 0], [0.6, 0.6]]
        )
        assert row["samplerate_hz"] == FS
        assert row["duration_s"] == pytest.approx(20.0)
    lib = BackgroundLibrary.load(root)
    seg = lib.sample(
        np.random.default_rng(0), 10.0, samplerate_hz=4000.0, resample=True
    )
    assert seg.samplerate_hz == 4000.0
    assert seg.n_samples == 40_000
    np.testing.assert_allclose(seg.geometry.positions_m[3], [0.6, 0.6])
