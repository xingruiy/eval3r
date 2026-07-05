"""`e3r metric` command group.

`geometry` runs the single-file geometry pipeline (task 007); `depth` runs the
single-file / sequence depth pipeline (task 014); `pose` is a stub owned by task
015. Both commands are deliberately verbose (``rich``): they echo the resolved
protocol (name + hash), the alignment / masking / sampling / confidence / failure
policies actually in effect, the resolved input and output paths, and the
per-scene outcome — never a bare exit code (CLAUDE.md CLI verbosity rules).
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

from eval3r.api import evaluate_depth, evaluate_geometry
from eval3r.core.errors import Eval3rError
from eval3r.metrics.depth import DEPTH_ALIGNMENT_GRANULARITIES, DEPTH_ALIGNMENT_MODES
from eval3r.pipeline.depth_runner import DepthRunOutput
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


def _echo_depth_config(run: DepthRunOutput, pred: Path, gt: Path, out_dir: Path) -> None:
    proto = run.protocol
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("protocol", f"{proto.name}  ({run.protocol_hash})")
    table.add_row("protocol version", proto.protocol_version)
    table.add_row("fidelity", proto.fidelity)
    table.add_row("modality", f"{proto.prediction_modality}  ({run.n_frames} frame(s))")
    table.add_row("prediction", str(pred))
    table.add_row("ground truth", str(gt))
    table.add_row("output dir", str(out_dir))
    table.add_row(
        "depth units",
        f"pred={run.overrides.get('pred_depth_unit', 1.0)} "
        f"gt={run.overrides.get('gt_depth_unit', 1.0)} (metres per stored unit)",
    )
    table.add_row(
        "scale alignment",
        f"mode={proto.alignment.mode} granularity={proto.alignment.granularity}",
    )
    table.add_row(
        "masking",
        f"invalid_values={proto.masking.invalid_depth_values} "
        f"ignore_invalid_depth={proto.masking.ignore_invalid_depth} "
        f"valid_region={proto.masking.valid_region.method}",
    )
    table.add_row("confidence", proto.confidence.policy)
    table.add_row("failure policy", proto.failure_policy.policy)
    if len(run.overrides) > 1:  # more than the always-present modality
        table.add_row("overrides", str(run.overrides))
    console.print(Panel(table, title="resolved evaluation configuration", expand=False))


def _echo_depth_outcome(run: DepthRunOutput, out_dir: Path) -> None:
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
        metrics = Table(title="metrics (per-frame mean)", show_header=True, header_style="bold")
        metrics.add_column("metric")
        metrics.add_column("value", justify="right")
        metrics.add_column("n_pixels_valid", justify="right")
        metrics.add_column("valid_fraction", justify="right")
        for m in [m for m in result.per_scene_metrics if m.frame_id is None]:
            metrics.add_row(
                m.name,
                "-" if m.value is None else f"{m.value:.6g}",
                "-" if m.n_pixels_valid is None else str(m.n_pixels_valid),
                "-" if m.valid_fraction is None else f"{m.valid_fraction:.4f}",
            )
        console.print(metrics)
        for record in run.alignment_records:
            frame = record["frame_id"] or "(pooled)"
            console.print(
                f"[dim]alignment {frame}: mode={record['mode']} "
                f"granularity={record['granularity']} scale={record['scale']:.6g} "
                f"shift={record['shift']:.6g} n_pixels={record['n_pixels_used']}[/]"
            )
    console.print(f"[green]run directory written:[/] {out_dir}")


@metric_app.command("depth")
def depth(
    pred: Path = typer.Argument(
        ..., help="Predicted depth: one file (PNG/PFM/npy) or a directory of frames."
    ),
    gt: Path = typer.Option(
        ..., "--gt", help="Ground-truth depth: one file or a directory of frames."
    ),
    depth_unit: float | None = typer.Option(
        None, "--depth-unit",
        help="Metres per stored unit of the *prediction* (e.g. 0.001 for millimetre "
        "PNGs). Required for integer depth files.",
    ),
    gt_depth_unit: float | None = typer.Option(
        None, "--gt-depth-unit",
        help="Metres per stored unit of the *ground truth*. Required for integer "
        "depth files.",
    ),
    align: str | None = typer.Option(
        None, "--align",
        help="Scale alignment override: none | scale_median | scale_least_squares | "
        "scale_affine (default: the protocol's mode).",
    ),
    align_granularity: str | None = typer.Option(
        None, "--align-granularity",
        help="Scale alignment granularity override: per_frame | per_sequence | "
        "per_scene (default: the protocol's granularity).",
    ),
    protocol: str = typer.Option(
        "single_depth", "--protocol", help="Built-in protocol name or path to a protocol YAML."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Run directory to write (default: runs/<timestamp>_<dataset>_<method>)."
    ),
    method: str | None = typer.Option(
        None, "--method", help="Method name recorded in the result (optional)."
    ),
) -> None:
    """Evaluate a predicted depth map/sequence against ground-truth depth."""
    if align is not None and align not in DEPTH_ALIGNMENT_MODES:
        err_console.print(
            f"[bold red]invalid --align '{align}'[/]: choose one of "
            f"{', '.join(DEPTH_ALIGNMENT_MODES)}."
        )
        raise typer.Exit(code=2)
    if align_granularity is not None and align_granularity not in DEPTH_ALIGNMENT_GRANULARITIES:
        err_console.print(
            f"[bold red]invalid --align-granularity '{align_granularity}'[/]: choose one of "
            f"{', '.join(DEPTH_ALIGNMENT_GRANULARITIES)}."
        )
        raise typer.Exit(code=2)

    command = "e3r " + shlex.join(sys.argv[1:]) if len(sys.argv) > 1 else "e3r metric depth"

    try:
        run = evaluate_depth(
            pred, gt,
            depth_unit=depth_unit, gt_depth_unit=gt_depth_unit,
            align=align, align_granularity=align_granularity,
            protocol=protocol, method=method, command=command, return_run=True,
        )
    except Eval3rError as exc:
        err_console.print(Panel(str(exc), title="evaluation failed", style="red", expand=False))
        err_console.print_exception()
        raise typer.Exit(code=1) from exc

    assert isinstance(run, DepthRunOutput)
    out_dir = (
        Path(out)
        if out is not None
        else Path("runs") / default_run_dir_name(run.result, now=datetime.now(timezone.utc))
    )

    _echo_depth_config(run, Path(pred), Path(gt), out_dir)
    write_run_directory(
        run.result, out_dir,
        protocol=run.protocol, config=run.config,
        environment=run.result.environment, backend_versions=run.result.backend_versions,
        alignment_transforms=run.alignment_records,
    )
    _echo_depth_outcome(run, out_dir)


@metric_app.command("pose")
def pose() -> None:
    """Evaluate a predicted trajectory against a ground-truth trajectory."""
    _not_yet("e3r metric pose", "task 015 (pose metrics)")
