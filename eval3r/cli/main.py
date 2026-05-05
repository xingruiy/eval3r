"""eval3r CLI entry point — typer Typer app with subcommands."""

from __future__ import annotations

import typer

from eval3r._version import __version__
from eval3r.cli import benchmark as benchmark_cmd
from eval3r.cli import datasets as datasets_cmd
from eval3r.cli import inspect as inspect_cmd
from eval3r.cli import metric as metric_cmd
from eval3r.cli import preset as preset_cmd
from eval3r.cli import render as render_cmd
from eval3r.cli import validate as validate_cmd

app = typer.Typer(
    name="e3r",
    help="Handy toolkit for saving, evaluating, and visualizing 3D reconstruction predictions.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(metric_cmd.app, name="metric", help="Compute geometry metrics.")
app.add_typer(render_cmd.app, name="render", help="Render mesh / point cloud / comparisons.")
app.add_typer(preset_cmd.app, name="preset", help="Inspect dataset presets.")
app.add_typer(datasets_cmd.app, name="datasets", help="Inspect / validate dataset adapters.")
app.add_typer(benchmark_cmd.app, name="benchmark", help="Run a method against a dataset split.")


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
