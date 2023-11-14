from pathlib import Path

import rich_click as click
import toml

from .datasets import subset_cli

click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True


pyproject = toml.load(Path(__file__).parent.parent / "pyproject.toml")
__version__ = pyproject["tool"]["poetry"]["version"]

hepstr = f"#Welcome to gridtools {__version__}!"


@click.group()
def cli():
    __doc__ = hepstr
    pass


@cli.group()
def io():
    """Dataset managing operations, such as conversion, subsetting, etc."""
    pass


@io.command()
@click.argument("input_path")
@click.argument("output_path")
@click.argument("start_time")
@click.argument("end_time")
def subset(input_path, output_path, start_time, end_time):
    """Create a subset of a dataset and save it to a new location."""
    subset_cli(input_path, output_path, start_time, end_time)


@cli.group()
def show():
    """Visualize datasets as spectrograms, position estimates, etc."""
    pass


@show.command()
@click.argument("input_path")
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


if __name__ == "__main__":
    cli()
