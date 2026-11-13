"""The ``gridtools sim`` command line."""

import re
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from gridtools.cli import app
from gridtools.sim.cli import parse_seeds, resolve_config
from gridtools.sim.config import TEMPLATES_ENV

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


def test_default_scene_without_templates_says_what_to_provide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(TEMPLATES_ENV, raising=False)
    result = runner.invoke(
        app, ["sim", "run", "default", "--seeds", "0:1", "--out", str(tmp_path / "s")]
    )
    assert result.exit_code == 1
    assert TEMPLATES_ENV in result.output
    assert "[chirps] library" in result.output
    assert not (tmp_path / "s" / "000000").exists()


def test_hybrid_scenes_need_a_background_library(
    library_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TEMPLATES_ENV, str(library_dir))
    config = tmp_path / "hybrid.toml"
    config.write_text("[background]\nprob = 1.0\n")
    result = runner.invoke(app, ["sim", "run", str(config), "--out", str(tmp_path)])
    assert result.exit_code == 2
    assert "background.library" in result.output
    config.write_text(f'[background]\nprob = 1.0\nlibrary = "{tmp_path / "nope"}"\n')
    result = runner.invoke(app, ["sim", "run", str(config), "--out", str(tmp_path)])
    assert result.exit_code == 1
    assert "background library" in result.output


def test_run_defaults_to_scenes_in_the_working_directory(
    library_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TEMPLATES_ENV, str(library_dir))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tiny.toml").write_text(
        "duration_s = 1.0\n[population]\nn_fish = [1, 1]\n[grid]\nlayouts = [[2, 2]]\n"
    )
    result = runner.invoke(app, ["sim", "run", "tiny.toml", "--no-plots"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "scenes" / "tiny" / "000000" / "recording.wav").exists()


def test_compare_needs_a_calibration(tmp_path: Path) -> None:
    result = runner.invoke(app, ["sim", "compare", str(tmp_path)])
    assert result.exit_code == 2
    assert "--calibration" in result.output


def _commands(command: object, path: tuple[str, ...] = ()):
    yield path, command
    for name, sub in getattr(command, "commands", {}).items():
        yield from _commands(sub, (*path, name))


def test_help_texts_survive_rich_markup() -> None:
    # rich drops "[word]" as a style tag: "[background] library" -> " library"
    tag = re.compile(r"(?<!\\)\[[a-z#/@][^\[\]]*\]")
    for path, command in _commands(typer.main.get_command(app)):
        texts = [getattr(command, "help", None)]
        texts += [getattr(p, "help", None) for p in getattr(command, "params", [])]
        for text in filter(None, texts):
            assert not tag.search(text), (path, text)
