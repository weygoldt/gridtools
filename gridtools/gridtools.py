#!/usr/bin/env python

"""
Gridtools - A command line tool for electrode grid recordings.
This is the main entry point of the gridtools command line tool.
"""

from pathlib import Path

import rich_click as click
import toml

from .datasets import subset_cli
from .fakegrid import fakegrid_cli, hybridgrid_cli
from .utils.configfiles import copy_config

click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True


pyproject = toml.load(Path(__file__).parent.parent / "pyproject.toml")
__version__ = pyproject["tool"]["poetry"]["version"]


def add_version(f):
    """
    Add the version of the gridtools to the help heading.
    """
    doc = f.__doc__
    f.__doc__ = "Welcome to Gridtools Version: " + __version__ + "\n\n" + doc

    return f


@click.group()
@click.version_option(
    __version__, "-V", "--version", message="Gridtools, version %(version)s"
)
@add_version
def cli():
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
def io():
    """Dataset managing operations, such as conversion, subsetting, etc."""
    pass


@cli.group()
def show():
    """Visualize datasets as spectrograms, position estimates, etc."""
    pass


@cli.group()
def render():
    """Render videos"""
    pass


@cli.group()
def prepro():
    """Preprocess datasets according to your prepro.toml file"""
    pass


@cli.group()
def simulate():
    """Simulate full grid datasets including wavetracker tracks, position estimates, etc."""
    pass


@cli.command()
@click.option(
    "--config_path",
    "-c",
    type=click.Path(),
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
def copyconfig(config_path, mode):
    """Copy a config file to a directory."""
    copy_config(config_path, mode)


@io.command()
@click.option(
    "--input_path",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input dataset.",
)
@click.option(
    "--output_path",
    "-o",
    type=click.Path(),
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
def subset(input_path, output_path, start_time, end_time):
    """Create a subset of a dataset and save it to a new location."""
    subset_cli(input_path, output_path, start_time, end_time)


@show.command()
@click.option(
    "--input_path",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input dataset.",
)
def spec(input_path):
    """Show a spectrogram of the dataset."""


@show.command()
@click.argument("input_path")
def tracks(input_path):
    """Show the position estimates of the dataset."""
    pass


@simulate.command()
@click.option(
    "--output_path",
    "-o",
    type=click.Path(),
    required=True,
    help="Path to the output dataset.",
)
def grid(output_path):
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
def noise(input_path, real_path, output_path):
    """Add real noise to a simulated dataset (`input_path`) from a dataset of real
    recordings (`real_path`) and save the result to `output_path`."""
    hybridgrid_cli(input_path, real_path, output_path)


if __name__ == "__main__":
    cli()
