"""`e3r metric` command group.

`geometry` runs the single-file geometry pipeline (task 007); `depth` runs the
single-file / sequence depth pipeline (task 014); `pose` runs the
single-trajectory pose pipeline (task 015). All commands are deliberately
verbose (``rich``): they echo the resolved protocol (name + hash), the
alignment / masking / sampling / confidence / failure policies actually in
effect, the resolved input and output paths, and the per-scene outcome — never
a bare exit code (CLAUDE.md CLI verbosity rules).
"""

from __future__ import annotations

import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from eval3r.api import evaluate_depth, evaluate_geometry, evaluate_pose
from eval3r.core.errors import Eval3rError
from eval3r.core.protocol import EvalProtocol
from eval3r.core.types import NormalizedConvention, SourcePoseFormat, WorldAxes
from eval3r.metrics.depth import DEPTH_ALIGNMENT_GRANULARITIES, DEPTH_ALIGNMENT_MODES
from eval3r.pipeline.depth_runner import DepthRunOutput
from eval3r.pipeline.pose_runner import POSE_ALIGN_ALIASES, PoseRunOutput
from eval3r.pipeline.runner import GeometryRunOutput
from eval3r.reports.run_directory import default_run_dir_name, write_run_directory

metric_app = typer.Typer(
    help="Evaluate a single prediction against ground truth (geometry, depth, pose).",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


_KIND_CHOICES = ("pointcloud", "mesh")
_POSE_FORMAT_CHOICES = get_args(SourcePoseFormat)
_WORLD_FRAME_CHOICES = get_args(WorldAxes)


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
    pred_world = run.overrides.get("pred_world_frame", "opencv")
    adaptation = run.result.adaptation
    table.add_row(
        "world frame",
        f"pred={pred_world} -> internal=opencv "
        f"[{'transformed' if pred_world != 'opencv' else 'passthrough'}]",
    )
    if adaptation is not None:
        table.add_row(
            "adaptation",
            f"source={adaptation.source} reason={adaptation.reason} "
            f"alignment={adaptation.alignment} "
            f"[{'transformed' if adaptation.transformed else 'passthrough'}]",
        )
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
    pred_world_frame: str = typer.Option(
        "opencv", "--pred-world-frame",
        help="World-frame convention the prediction geometry was built in: opencv | "
        "opengl. An opengl prediction is rotated into eval3r's internal opencv world "
        "frame before metrics.",
    ),
    adapt: str | None = typer.Option(
        None, "--as", "--adapt",
        help="Prediction adaptation override, e.g. opengl@sim3 or cw@opencv@sim3.",
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
    if pred_world_frame not in _WORLD_FRAME_CHOICES:
        err_console.print(
            f"[bold red]invalid --pred-world-frame '{pred_world_frame}'[/]: choose one "
            f"of {', '.join(_WORLD_FRAME_CHOICES)}."
        )
        raise typer.Exit(code=2)

    command = "e3r " + shlex.join(sys.argv[1:]) if len(sys.argv) > 1 else "e3r metric geometry"

    try:
        run = evaluate_geometry(
            pred, gt,
            input_type=input_type,  # type: ignore[arg-type]
            gt_type=gt_type,  # type: ignore[arg-type]
            threshold=threshold, sample=sample, adapt=adapt, pred_world_frame=pred_world_frame,
            protocol=protocol, method=method, command=command, return_run=True,
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
    if run.result.adaptation is not None:
        adaptation = run.result.adaptation
        table.add_row(
            "adaptation",
            f"source={adaptation.source} reason={adaptation.reason} "
            f"alignment={adaptation.alignment} "
            f"[{'transformed' if adaptation.transformed else 'passthrough'}]",
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
    adapt: str | None = typer.Option(
        None, "--as", "--adapt",
        help="Prediction adaptation override, e.g. relative@scale_median.",
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
            align=align, adapt=adapt, align_granularity=align_granularity,
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


def _pose_target(proto: EvalProtocol) -> str:
    """Human-readable internal pose-convention target for the config echo."""
    from eval3r.datasets.conventions import normalized_convention_target

    nc: NormalizedConvention = (
        proto.ground_truth.normalized_convention or "cam_to_world_opencv_meters"
    )
    target = normalized_convention_target(nc)
    return f"{target.axes}/{target.direction}"


def _echo_pose_config(run: PoseRunOutput, pred: Path, gt: Path, out_dir: Path) -> None:
    proto = run.protocol
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("protocol", f"{proto.name}  ({run.protocol_hash})")
    table.add_row("protocol version", proto.protocol_version)
    table.add_row("fidelity", proto.fidelity)
    table.add_row("modality", proto.prediction_modality)
    table.add_row("prediction", f"{pred}  [tum]")
    table.add_row("ground truth", f"{gt}  [tum]")
    table.add_row("output dir", str(out_dir))
    inputs = run.config.get("inputs", {})
    pred_fmt = inputs.get("pred_pose_format") or "passthrough"
    gt_fmt = inputs.get("gt_pose_format") or "passthrough"
    table.add_row(
        "convention",
        f"pred={pred_fmt} gt={gt_fmt} -> target={_pose_target(proto)}",
    )
    table.add_row(
        "alignment",
        f"mode={proto.alignment.mode} solver={proto.alignment.solver} "
        f"granularity={proto.alignment.granularity}",
    )
    if run.result.adaptation is not None:
        adaptation = run.result.adaptation
        table.add_row(
            "adaptation",
            f"source={adaptation.source} reason={adaptation.reason} "
            f"pose={adaptation.pose_convention or 'passthrough'} "
            f"alignment={adaptation.alignment} "
            f"[{'transformed' if adaptation.transformed else 'passthrough'}]",
        )
    table.add_row(
        "association",
        f"nearest_timestamp max_diff="
        f"{proto.alignment.parameters.get('associate_max_diff')}s "
        f"offset={proto.alignment.parameters.get('offset', 0.0)}s",
    )
    table.add_row(
        "trajectory backend", proto.backend_preferences.get("trajectory", "evo")
    )
    table.add_row("confidence", proto.confidence.policy)
    table.add_row("failure policy", proto.failure_policy.policy)
    if run.overrides:
        table.add_row("overrides", str(run.overrides))
    console.print(Panel(table, title="resolved evaluation configuration", expand=False))


def _echo_pose_outcome(run: PoseRunOutput, out_dir: Path) -> None:
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
        metrics.add_column("unit", justify="right")
        metrics.add_column("statistic", justify="right")
        for m in result.per_scene_metrics:
            metrics.add_row(
                m.name,
                "-" if m.value is None else f"{m.value:.6g}",
                m.unit or "-",
                m.statistic or "-",
            )
        console.print(metrics)
        meta = result.metadata
        conv = meta.get("convention")
        if conv is not None:
            console.print(
                f"[dim]convention: pred={conv['pred_pose_format'] or 'passthrough'} "
                f"(transformed={conv['pred_transformed']}) "
                f"gt={conv['gt_pose_format'] or 'passthrough'} "
                f"(transformed={conv['gt_transformed']}) -> {conv['target']}[/]"
            )
        console.print(
            f"[dim]association: {meta['n_associated']} pose(s) associated "
            f"(pred {meta['n_pred_poses']}, gt {meta['n_gt_poses']}; dropped "
            f"pred {meta['n_dropped_pred']}, gt {meta['n_dropped_gt']}) at "
            f"max_diff={meta['association']['associate_max_diff']}s[/]"
        )
        for record in run.alignment_records:
            console.print(
                f"[dim]alignment: mode={record['mode']} scale={record['scale']:.6g} "
                f"|ln s|={record['scale_error']:.6g} "
                f"n_poses={record['n_poses_used']}[/]"
            )
    console.print(f"[green]run directory written:[/] {out_dir}")


@metric_app.command("pose")
def pose(
    pred: Path = typer.Argument(
        ..., help="Predicted trajectory, TUM format (timestamp x y z qx qy qz qw)."
    ),
    gt: Path = typer.Option(..., "--gt", help="Ground-truth trajectory, TUM format."),
    align: str | None = typer.Option(
        None, "--align",
        help="Trajectory alignment override: none | se3 | sim3 (or trajectory_se3 / "
        "trajectory_sim3; default: the protocol's mode).",
    ),
    adapt: str | None = typer.Option(
        None, "--as", "--adapt",
        help="Prediction adaptation override, e.g. wc@opengl@trajectory_sim3.",
    ),
    associate_max_diff: float | None = typer.Option(
        None, "--associate-max-diff",
        help="Timestamp association tolerance override in seconds "
        "(default: the protocol's associate_max_diff).",
    ),
    backend: str | None = typer.Option(
        None, "--backend", help="Trajectory backend (default: the protocol's, evo)."
    ),
    pred_pose_convention: str | None = typer.Option(
        None, "--pred-pose-convention",
        help="Source pose convention of the *prediction* trajectory (e.g. "
        "cam_to_world_opengl, world_to_cam_colmap); transformed to the protocol's "
        "internal convention before metrics. Default: assume already internal.",
    ),
    gt_pose_convention: str | None = typer.Option(
        None, "--gt-pose-convention",
        help="Source pose convention of the *ground-truth* trajectory (default: the "
        "protocol's ground_truth.source_pose_format).",
    ),
    protocol: str = typer.Option(
        "single_pose", "--protocol", help="Built-in protocol name or path to a protocol YAML."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Run directory to write (default: runs/<timestamp>_<dataset>_<method>)."
    ),
    method: str | None = typer.Option(
        None, "--method", help="Method name recorded in the result (optional)."
    ),
) -> None:
    """Evaluate a predicted trajectory against a ground-truth trajectory."""
    if align is not None and align not in POSE_ALIGN_ALIASES:
        err_console.print(
            f"[bold red]invalid --align '{align}'[/]: choose one of "
            f"{', '.join(POSE_ALIGN_ALIASES)}."
        )
        raise typer.Exit(code=2)
    for label, value in (
        ("--pred-pose-convention", pred_pose_convention),
        ("--gt-pose-convention", gt_pose_convention),
    ):
        if value is not None and value not in _POSE_FORMAT_CHOICES:
            err_console.print(
                f"[bold red]invalid {label} '{value}'[/]: choose one of "
                f"{', '.join(_POSE_FORMAT_CHOICES)}."
            )
            raise typer.Exit(code=2)

    command = "e3r " + shlex.join(sys.argv[1:]) if len(sys.argv) > 1 else "e3r metric pose"

    try:
        run = evaluate_pose(
            pred, gt,
            align=align, adapt=adapt, associate_max_diff=associate_max_diff, backend=backend,
            pred_pose_format=pred_pose_convention, gt_pose_format=gt_pose_convention,
            protocol=protocol, method=method, command=command, return_run=True,
        )
    except Eval3rError as exc:
        err_console.print(Panel(str(exc), title="evaluation failed", style="red", expand=False))
        err_console.print_exception()
        raise typer.Exit(code=1) from exc

    assert isinstance(run, PoseRunOutput)
    out_dir = (
        Path(out)
        if out is not None
        else Path("runs") / default_run_dir_name(run.result, now=datetime.now(timezone.utc))
    )

    _echo_pose_config(run, Path(pred), Path(gt), out_dir)
    write_run_directory(
        run.result, out_dir,
        protocol=run.protocol, config=run.config,
        environment=run.result.environment, backend_versions=run.result.backend_versions,
        alignment_transforms=run.alignment_records,
    )
    _echo_pose_outcome(run, out_dir)
