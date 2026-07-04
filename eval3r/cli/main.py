"""eval3r command-line entry point (`e3r`).

Assembles the command tree so `e3r --help` lists every command group. Implemented
groups (e.g. `protocol`) are wired in from their own modules; commands owned by
later task slices are stubs that fail loudly with an explicit reason naming the
slice that implements them (per CLAUDE.md's error and CLI verbosity rules).
"""

from __future__ import annotations

import typer

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

metric_app = typer.Typer(
    help="Evaluate a single prediction against ground truth (geometry, depth, pose).",
    no_args_is_help=True,
)
benchmark_app = typer.Typer(
    help="Run or validate a dataset benchmark under a named protocol.",
    no_args_is_help=True,
)
dataset_app = typer.Typer(
    help="Inspect dataset adapters and their capabilities.",
    no_args_is_help=True,
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


# --- metric group --------------------------------------------------------------


@metric_app.command("geometry")
def metric_geometry() -> None:
    """Evaluate a mesh or point cloud against ground-truth geometry."""
    _not_yet("e3r metric geometry", "task 007 (single-file geometry runner)")


@metric_app.command("depth")
def metric_depth() -> None:
    """Evaluate a predicted depth map/sequence against ground-truth depth."""
    _not_yet("e3r metric depth", "task 014 (depth metrics)")


@metric_app.command("pose")
def metric_pose() -> None:
    """Evaluate a predicted trajectory against a ground-truth trajectory."""
    _not_yet("e3r metric pose", "task 015 (pose metrics)")


# --- benchmark group -----------------------------------------------------------


@benchmark_app.command("run")
def benchmark_run() -> None:
    """Run a dataset benchmark under a named protocol."""
    _not_yet("e3r benchmark run", "task 008 (benchmark run plumbing)")


@benchmark_app.command("validate")
def benchmark_validate() -> None:
    """Validate a prediction manifest against a dataset/protocol without evaluating."""
    _not_yet("e3r benchmark validate", "task 008 (benchmark run plumbing)")


# --- dataset group -------------------------------------------------------------


@dataset_app.command("inspect")
def dataset_inspect() -> None:
    """Show a dataset adapter's capabilities and resolved layout."""
    _not_yet("e3r dataset inspect", "task 008 (benchmark run plumbing)")


# --- top-level diff ------------------------------------------------------------


@app.command("diff")
def diff() -> None:
    """Compare two run result directories (refuses mismatched protocol hashes)."""
    _not_yet("e3r diff", "task 016 (reports and diffing)")


if __name__ == "__main__":  # pragma: no cover
    app()
