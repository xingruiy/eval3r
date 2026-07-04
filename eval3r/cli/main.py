"""eval3r command-line entry point (`e3r`).

Assembles the command tree so `e3r --help` lists every command group. Implemented
groups (e.g. `protocol`) are wired in from their own modules; commands owned by
later task slices are stubs that fail loudly with an explicit reason naming the
slice that implements them (per CLAUDE.md's error and CLI verbosity rules).
"""

from __future__ import annotations

import typer

from eval3r.cli.benchmark import benchmark_app
from eval3r.cli.dataset import dataset_app
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


def _not_yet(command: str, task: str) -> None:
    """Fail explicitly, naming the command and the task slice that implements it."""
    raise NotImplementedError(
        f"`{command}` is not implemented yet in this skeleton. "
        f"It is delivered by {task}. See .agent/tasks/ for the implementation order."
    )


# --- top-level diff ------------------------------------------------------------


@app.command("diff")
def diff() -> None:
    """Compare two run result directories (refuses mismatched protocol hashes)."""
    _not_yet("e3r diff", "task 016 (reports and diffing)")


if __name__ == "__main__":  # pragma: no cover
    app()
