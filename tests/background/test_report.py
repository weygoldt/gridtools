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


@pytest.mark.parametrize("command", ["review", "report"])
def test_a_library_without_segments_exits_with_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    from gridtools.background.__main__ import main

    root = tmp_path / "lib"
    assert build_background_library([], root, BackgroundConfig(workers=1)).empty
    with pytest.raises(SystemExit) as exit_info:
        main([command, str(root)])
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "has no segments" in err
    assert "Traceback" not in err
    with pytest.raises(SystemExit) as exit_info:
        main([command, str(tmp_path / "missing")])
    assert exit_info.value.code == 1
    assert "not a background library" in capsys.readouterr().err
    with pytest.raises(ValueError, match="has no segments"):
        write_review(root)


def test_build_needs_the_roots_of_its_datasets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from gridtools.background.__main__ import main

    with pytest.raises(SystemExit) as exit_info:
        main(["build", str(tmp_path / "lib"), "--datasets", "competition2019"])
    assert exit_info.value.code == 2
    assert "--root-2019" in capsys.readouterr().err
    assert not (tmp_path / "lib").exists()
