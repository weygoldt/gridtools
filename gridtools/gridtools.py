#!/usr/bin/env python

"""
Gridtools - A command line tool for electrode grid recordings.
This is the main entry point of the gridtools command line tool.
"""

from pathlib import Path

import rich_click as click
import toml

from .datasets import subset_cli

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


@cli.group()
def show():
    """Visualize datasets as spectrograms, position estimates, etc."""
    pass


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


@cli.group()
def render():
    """Render videos"""
    pass


@cli.group()
def prepro():
    """Preprocess datasets according to your prepro.toml file"""
    pass


if __name__ == "__main__":
    cli()
