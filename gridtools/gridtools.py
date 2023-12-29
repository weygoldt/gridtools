"""Gridtools - A command line tool for electrode grid recordings."""

from pathlib import Path
from typing import Callable

import rich_click as click
import toml

from gridtools.datasets.subsetters import subset_cli
from gridtools.simulations.fakegrid import fakegrid_cli, hybridgrid_cli
from .utils.configfiles import copy_config

click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True


pyproject = toml.load(Path(__file__).parent.parent / "pyproject.toml")
__version__ = pyproject["tool"]["poetry"]["version"]


def add_version(f: Callable) -> Callable:
    """Add the version of the gridtools to the help heading."""
    doc = f.__doc__
    f.__doc__ = "Welcome to Gridtools Version: " + __version__ + "\n\n" + doc

    return f


@click.group()
@click.version_option(
    __version__, "-V", "--version", message="Gridtools, version %(version)s"
)
@add_version
def cli() -> None:
    """Interact with electrode grid recordings a bit more easily.

    The gridtools command line tool is a collection of commands that
    make it easier to work with electrode grid recordings. It provides
    a carefully designed set of classes and functions that can be used
    load, subset, and save electrode grid recordings including frequency
    tracks, position estimates, and other metadata.

    Additional functionality includes plotting commands to interactively
    visualize datasets, a suite of preprocessing functions, and a suite
    to render videos of the recordings.

    For more information including a tutorial, see the documentation at
    https://weygoldt.com/gridtools

    Have fun exploring the recordings :fish::zap:
    """
    pass


@cli.group()
def io() -> None:
    """Dataset managing operations, such as conversion, subsetting, etc."""
    pass


@cli.group()
def show() -> None:
    """Visualize datasets as spectrograms, position estimates, etc."""
    pass


@cli.group()
def render() -> None:
    """Render animated plots of the dataset."""
    pass


@cli.group()
def prepro() -> None:
    """Preprocess datasets according to your prepro.toml file."""
    pass


@cli.group()
def simulate() -> None:
    """Simulate full grid datasets."""
    pass


@cli.command()
@click.option(
    "--config_path",
    "-c",
    type=Path,
    required=True,
    help="Path to the config file.",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["prepro", "simulations"]),
    required=True,
    help="Mode of operation.",
)
def copyconfig(config_path: Path, mode: str) -> None:
    """Copy a config file to a directory."""
    copy_config(config_path, mode)


@io.command()
@click.option(
    "--input_path",
    "-i",
    type=Path,
    required=True,
    help="Path to the input dataset.",
)
@click.option(
    "--output_path",
    "-o",
    type=Path,
    required=True,
    help="Path to the output dataset.",
)
@click.option(
    "--start_time",
    "-s",
    type=float,
    required=True,
    help="Start time of the subset.",
)
@click.option(
    "--end_time",
    "-e",
    type=float,
    required=True,
    help="End time of the subset.",
)
def subset(
        input_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
) -> None:
    """Create a subset of a dataset and save it to a new location."""
    subset_cli(input_path, output_path, start_time, end_time)


@show.command()
@click.option(
    "--input_path",
    "-i",
    type=Path,
    required=True,
    help="Path to the input dataset.",
)
def spec(input_path: Path) -> None:
    """Show a spectrogram of the dataset."""
    print(f"{input_path}")
    print("Sorry, not implemented yet.")


@show.command()
@click.argument("input_path", type=Path)
def tracks(input_path: Path) -> None:
    """Show the position estimates of the dataset."""
    print(f"{input_path}")
    print("Sorry, not implemented yet.")
    pass


@simulate.command()
@click.option(
    "--output_path",
    "-o",
    type=Path,
    required=True,
    help="Path to the output dataset.",
)
def grid(output_path: Path) -> None:
    """Simulate a grid dataset."""
    fakegrid_cli(output_path)


@simulate.command()
@click.option(
    "--input_path",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to the simulated input dataset.",
)
@click.option(
    "--real_path",
    "-r",
    type=click.Path(),
    required=True,
    help="Path to a real dataset to grap background noise from.",
)
@click.option(
    "--output_path",
    "-o",
    type=click.Path(),
    required=True,
    help="Path to the output dataset.",
)
def noise(
    input_path: Path,
    real_path: Path,
    output_path: Path,
) -> None:
    """Add real noise to a simulated dataset."""
    hybridgrid_cli(input_path, real_path, output_path)


if __name__ == "__main__":
    cli()
