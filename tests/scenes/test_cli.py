"""The ``gridtools sim`` command line."""

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from gridtools.cli import app
from gridtools.sim.cli import parse_seeds, resolve_config

runner = CliRunner()


def test_parse_seeds() -> None:
    assert parse_seeds("0:3") == [0, 1, 2]
    assert parse_seeds("2:10:4") == [2, 6]
    assert parse_seeds("5") == [5]
    assert parse_seeds("1, 4,7") == [1, 4, 7]
    for bad in ("", "a:b", "3:1", "-1", "1:2:3:4"):
        with pytest.raises(typer.BadParameter):
            parse_seeds(bad)


def test_resolve_shipped_config() -> None:
    assert resolve_config("default").name == "default"
    with pytest.raises(typer.BadParameter):
        resolve_config("no-such-config")


def test_run_and_plot(library_dir: Path, tmp_path: Path) -> None:
    config = tmp_path / "tiny.toml"
    config.write_text(
        "duration_s = 1.0\n"
        "[population]\nn_fish = [2, 2]\n"
        f'[chirps]\nlibrary = "{library_dir}"\n'
        "[grid]\nlayouts = [[2, 2]]\n"
    )
    out = tmp_path / "scenes"
    result = runner.invoke(
        app,
        ["sim", "run", str(config), "--seeds", "0:2", "--out", str(out), "--no-plots"],
    )
    assert result.exit_code == 0, result.output
    for seed in ("000000", "000001"):
        assert (out / seed / "recording.wav").exists()
        assert not (out / seed / "spectrogram_overview.png").exists()
    assert (out / "config.json").exists()
    pytest.importorskip("matplotlib")
    result = runner.invoke(app, ["sim", "plot", str(out / "000001")])
    assert result.exit_code == 0, result.output
    assert (out / "000001" / "spectrogram_overview.png").exists()
