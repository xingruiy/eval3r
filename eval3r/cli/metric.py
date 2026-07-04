"""`e3r metric` command group.

`geometry` runs the single-file geometry pipeline (task 007); `depth` and `pose`
are stubs owned by tasks 014 / 015. The geometry command is deliberately verbose
(``rich``): it echoes the resolved protocol (name + hash), the alignment / masking /
sampling / confidence / failure policies actually in effect, the resolved input and
output paths, and the per-scene outcome — never a bare exit code (CLAUDE.md CLI
verbosity rules).
"""

from __future__ import annotations

import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from eval3r.api import evaluate_geometry
from eval3r.core.errors import Eval3rError
from eval3r.pipeline.runner import GeometryRunOutput
from eval3r.reports.run_directory import default_run_dir_name, write_run_directory

metric_app = typer.Typer(
    help="Evaluate a single prediction against ground truth (geometry, depth, pose).",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


def _not_yet(command: str, task: str) -> None:
    raise NotImplementedError(
        f"`{command}` is not implemented yet in this skeleton. "
        f"It is delivered by {task}. See .agent/tasks/ for the implementation order."
    )


_KIND_CHOICES = ("pointcloud", "mesh")


def _echo_config(run: GeometryRunOutput, pred: Path, gt: Path, out_dir: Path) -> None:
    proto = run.protocol
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("protocol", f"{proto.name}  ({run.protocol_hash})")
    table.add_row("protocol version", proto.protocol_version)
    table.add_row("fidelity", proto.fidelity)
    table.add_row(
        "dataset / variant",
        f"{proto.dataset.dataset} / {proto.dataset.variant or '-'} "
        f"(split {proto.dataset.split or '-'})",
    )
    table.add_row("prediction", f"{pred}  [{run.overrides['input_type']}]")
    table.add_row("ground truth", f"{gt}  [{run.overrides['gt_type']}]")
    table.add_row("output dir", str(out_dir))
    table.add_row(
        "alignment",
        f"mode={proto.alignment.mode} solver={proto.alignment.solver} "
        f"estimate_on={proto.alignment.estimate_on}",
    )
    table.add_row(
        "masking",
        f"pred={proto.masking.pred_culling.method} gt={proto.masking.gt_culling.method}",
    )
    table.add_row(
        "sampling",
        f"pred={proto.sampling.pred.method}/{proto.sampling.pred.n_points} "
        f"gt={proto.sampling.gt.method}/{proto.sampling.gt.n_points}",
    )
    table.add_row("confidence", proto.confidence.policy)
    table.add_row("failure policy", proto.failure_policy.policy)
    if len(run.overrides) > 2:  # more than the always-present input/gt types
        table.add_row("overrides", str(run.overrides))
    console.print(Panel(table, title="resolved evaluation configuration", expand=False))


def _echo_outcome(run: GeometryRunOutput, out_dir: Path) -> None:
    result = run.result
    if result.failed_scenes:
        for failure in result.failed_scenes:
            err_console.print(
                f"[bold red]scene '{failure.scene_id}' FAILED at stage "
                f"'{failure.stage}':[/] {failure.reason}"
            )
        console.print(
            f"[yellow]partial coverage:[/] {result.n_scenes_evaluated}/"
            f"{result.n_scenes_expected} scenes evaluated "
            f"(policy: {result.failure_policy.policy})"
        )
    else:
        metrics = Table(title="metrics", show_header=True, header_style="bold")
        metrics.add_column("metric")
        metrics.add_column("value", justify="right")
        for name, value in result.metrics.items():
            metrics.add_row(name, "-" if value is None else f"{value:.6g}")
        console.print(metrics)
    console.print(f"[green]run directory written:[/] {out_dir}")


@metric_app.command("geometry")
def geometry(
    pred: Path = typer.Argument(..., help="Predicted mesh or point cloud (e.g. pred.ply)."),
    gt: Path = typer.Option(..., "--gt", help="Ground-truth mesh or point cloud."),
    threshold: float | None = typer.Option(
        None, "--threshold", help="Distance threshold τ for precision/recall/F-score (metres)."
    ),
    sample: int | None = typer.Option(
        None, "--sample", help="Sample count (surface points for meshes; subsample for clouds)."
    ),
    input_type: str = typer.Option(
        "pointcloud", "--input", help="Prediction type: pointcloud | mesh."
    ),
    gt_type: str = typer.Option(
        "pointcloud", "--gt-input", help="Ground-truth type: pointcloud | mesh."
    ),
    protocol: str = typer.Option(
        "single_geometry", "--protocol", help="Built-in protocol name or path to a protocol YAML."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Run directory to write (default: runs/<timestamp>_<dataset>_<method>)."
    ),
    method: str | None = typer.Option(
        None, "--method", help="Method name recorded in the result (optional)."
    ),
) -> None:
    """Evaluate a mesh or point cloud against ground-truth geometry."""
    for label, value in (("--input", input_type), ("--gt-input", gt_type)):
        if value not in _KIND_CHOICES:
            err_console.print(
                f"[bold red]invalid {label} '{value}'[/]: choose one of "
                f"{', '.join(_KIND_CHOICES)}."
            )
            raise typer.Exit(code=2)

    command = "e3r " + shlex.join(sys.argv[1:]) if len(sys.argv) > 1 else "e3r metric geometry"

    try:
        run = evaluate_geometry(
            pred, gt,
            input_type=input_type,  # type: ignore[arg-type]
            gt_type=gt_type,  # type: ignore[arg-type]
            threshold=threshold, sample=sample, protocol=protocol,
            method=method, command=command, return_run=True,
        )
    except Eval3rError as exc:
        err_console.print(Panel(str(exc), title="evaluation failed", style="red", expand=False))
        err_console.print_exception()
        raise typer.Exit(code=1) from exc

    assert isinstance(run, GeometryRunOutput)
    # Use UTC for the default directory name so the local timezone is not leaked.
    out_dir = (
        Path(out)
        if out is not None
        else Path("runs") / default_run_dir_name(run.result, now=datetime.now(timezone.utc))
    )

    _echo_config(run, Path(pred), Path(gt), out_dir)
    write_run_directory(
        run.result, out_dir,
        protocol=run.protocol, config=run.config,
        environment=run.result.environment, backend_versions=run.result.backend_versions,
        alignment_transforms=run.alignment_transforms,
    )
    _echo_outcome(run, out_dir)


@metric_app.command("depth")
def depth() -> None:
    """Evaluate a predicted depth map/sequence against ground-truth depth."""
    _not_yet("e3r metric depth", "task 014 (depth metrics)")


@metric_app.command("pose")
def pose() -> None:
    """Evaluate a predicted trajectory against a ground-truth trajectory."""
    _not_yet("e3r metric pose", "task 015 (pose metrics)")
