"""Command line interface: ``gridtools <group> <command>``.

Sub-apps live next to their code and are registered here.
"""

import typer

from gridtools.sim.cli import app as sim_app

app = typer.Typer(
    name="gridtools",
    help="Read and simulate grid recordings of wave-type electric fish.",
    no_args_is_help=True,
)
app.add_typer(sim_app, name="sim")


if __name__ == "__main__":  # pragma: no cover
    app()
