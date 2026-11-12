"""Smoke test of the library report on a synthetic source."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridtools.templates import BuildConfig, build_library

from .conftest import SourceRow

pytest.importorskip("matplotlib")


def test_write_report(
    synthetic_h5: tuple[Path, list[SourceRow]], tmp_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from gridtools.templates.report import write_report

    h5_path, _ = synthetic_h5
    out = tmp_path / "library"
    build_library(h5_path, out, BuildConfig())
    path = write_report(
        out, h5_path=h5_path, compare_with=tmp_path / "missing", n_evidence_rows=5
    )
    text = path.read_text()
    assert path == out / "report.md"
    for heading in (
        "## 1. Consistency with the thesis",
        "## 4. Exclusions",
        "## 5. Histogram equalization",
        "## 6. Amplitude",
        "## 8. Open questions",
    ):
        assert heading in text
    for name in (
        "distributions",
        "families",
        "flags",
        "equalization",
        "v1_comparison",
        "eodf",
        "amplitude_evidence",
    ):
        assert (out / "figures" / f"{name}.png").stat().st_size > 0
