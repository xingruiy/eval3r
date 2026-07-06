"""eval3r command-line entry point (`e3r`).

Assembles the command tree so `e3r --help` lists every command group. Each command
group is wired in from its own module.
"""

from __future__ import annotations

import typer

from eval3r.cli.align import align_command
from eval3r.cli.benchmark import benchmark_app
from eval3r.cli.dataset import dataset_app
from eval3r.cli.diff import diff_command
from eval3r.cli.metric import metric_app
from eval3r.cli.protocol import protocol_app

app = typer.Typer(
    name="e3r",
    help=(
        "eval3r: dataset-aware 3D reconstruction evaluation under explicit protocols. "
        "Evaluate meshes, point clouds, depth predictions, and trajectories."
    ),
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(metric_app, name="metric")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(dataset_app, name="dataset")
app.add_typer(protocol_app, name="protocol")
app.command("diff")(diff_command)
app.command("align")(align_command)


if __name__ == "__main__":  # pragma: no cover
    app()
