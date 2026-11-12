"""Review plots, contact sheets and the report on a synthetic library."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from gridtools.background.build import BackgroundConfig, build_background_library
from gridtools.background.report import write_report
from gridtools.background.review import read_review, write_review
from gridtools.background.select import SelectionConfig

from .conftest import Trial

pytest.importorskip("matplotlib")


def test_review_and_report(make_trial: Callable[..., Trial], tmp_path: Path) -> None:
    root = tmp_path / "lib"
    config = BackgroundConfig(
        selection=SelectionConfig(
            segment_s=10.0,
            step_s=2.5,
            margin_s=1.0,
            edge_s=5.0,
            segments_per_recording=2,
        ),
        fish_free_per_recording=1,
        workers=1,
    )
    manifest = build_background_library([make_trial().source], root, config)
    stats = write_review(root, workers=2)
    assert stats["n_plotted"] == len(manifest)
    for sid in manifest["segment_id"]:
        assert (root / "review" / f"{sid}.png").exists()
        assert (root / "review" / f"{sid}.thumb.jpg").exists()
    assert (root / "review" / "synthetic.html").exists()
    assert write_review(root)["n_plotted"] == 0  # existing PNGs are kept
    review = read_review(root)
    assert review is not None
    assert len(review) == len(manifest)
    report = write_report(root, n_demo_per_dataset=1).read_text()
    for heading in ("## Inventory", "## Levels", "## SNR", "## Demo"):
        assert heading in report
    assert (root / "figures" / "snr.png").exists()
    assert list((root / "figures").glob("demo_*.png"))
