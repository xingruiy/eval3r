"""eval3r CLI entry point — typer Typer app with subcommands."""

from __future__ import annotations

import typer

from eval3r._version import __version__
from eval3r.cli import benchmark as benchmark_cmd
from eval3r.cli import inspect as inspect_cmd
from eval3r.cli import mask as mask_cmd
from eval3r.cli import metric as metric_cmd
from eval3r.cli import render as render_cmd
from eval3r.cli import validate as validate_cmd

app = typer.Typer(
    name="e3r",
    help="Handy toolkit for saving, evaluating, and visualizing 3D reconstruction predictions.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(metric_cmd.app, name="metric", help="Compute geometry metrics.")
app.add_typer(render_cmd.app, name="render", help="Render mesh / point cloud / comparisons.")
app.add_typer(mask_cmd.app, name="mask", help="Generate / inspect occlusion masks.")
app.add_typer(
    benchmark_cmd.app,
    name="benchmark",
    help="Run a method against a dataset split.",
)



def _version_callback(value: bool) -> None:
    if not value:
        return
    typer.echo(__version__)
    raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Print the eval3r version and exit.",
        is_eager=True,
        callback=_version_callback,
    ),
) -> None:
    """e3r command group."""


@app.command()
def validate(
    path: str = typer.Argument(..., help="Path to a prediction directory."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Validate a prediction directory against its manifest."""
    validate_cmd.run(path, json_out=json_out)


@app.command()
def inspect(
    path: str = typer.Argument(..., help="Path to a prediction directory."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Print a summary of a prediction directory."""
    inspect_cmd.run(path, json_out=json_out)


@app.command(name="version")
def version_cmd() -> None:
    """Print the eval3r version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
