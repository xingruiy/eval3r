"""eval3r metric ... — compute geometry metrics from a prediction directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

import typer

import numpy as np

from eval3r.align import AlignMode
from eval3r.io.geometry import load_mesh, load_point_cloud
from eval3r.io.trajectory import Trajectory, load_trajectory_auto
from eval3r.metrics.depth import depth_metrics
from eval3r.metrics.geometry import ChamferVariant, evaluate_geometry
from eval3r.metrics.sampling import SampleMethod
from eval3r.prediction.reader import PredictionReader
from eval3r.report.table import print_depth_result, print_geometry_result
from eval3r.utils.errors import MissingArtifactError
from eval3r.utils.optional import optional_import

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _load_depth_image(path: str) -> np.ndarray:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        return np.load(p).astype(np.float32)
    if suffix == ".png":
        imageio = optional_import("imageio.v3", extra="render")
        return imageio.imread(p).astype(np.float32)
    raise typer.BadParameter(f"Unsupported depth image format: {suffix}. Use .png or .npy.")


def _load_geom(path: str):  # type: ignore[no-untyped-def]
    p = Path(path)
    if p.is_dir():
        reader = PredictionReader(p)
        try:
            return reader.mesh
        except MissingArtifactError:
            return reader.points
    suffix = p.suffix.lower()
    if suffix not in {".ply", ".obj", ".stl", ".off", ".glb"}:
        raise typer.BadParameter(f"Unsupported geometry file extension: {suffix}")
    try:
        return load_mesh(p)
    except Exception:
        return load_point_cloud(p)


def _load_poses(path: str, convention: str) -> Trajectory:
    """Load trajectory from text file, auto-detecting the format."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix != ".txt":
        raise typer.BadParameter(
            f"Unsupported trajectory file extension: {suffix}. Use .txt."
        )
    try:
        return load_trajectory_auto(p, convention=convention)
    except ValueError as e:
        raise typer.BadParameter(str(e))


def _load_pred_poses(
    pred_path: str, pred_poses_arg: str | None, convention: str,
) -> Trajectory | None:
    """Load prediction poses from manifest or explicit file.

    When *pred_path* is a directory containing a manifest, try to load
    poses from the manifest first.  Otherwise fall back to the explicit
    ``--pred-poses`` file argument.
    """
    p = Path(pred_path)
    if p.is_dir():
        try:
            reader = PredictionReader(p)
            return reader.poses
        except MissingArtifactError:
            pass
    if pred_poses_arg is not None:
        return load_trajectory_auto(Path(pred_poses_arg), convention=convention)
    return None


@app.command("all")
def all_cmd(
    pred: str = typer.Argument(..., help="Prediction directory or geometry file."),
    gt: str = typer.Option(..., "--gt", help="Ground-truth geometry file (ply/obj/...)."),
    samples: int = typer.Option(200_000, help="Number of samples for metric evaluation."),
    seed: int = typer.Option(42, help="RNG seed for sampling."),
    sample_method: str = typer.Option("area", help="area | vertex | uniform"),
    align: str = typer.Option("none", help="Alignment mode: " + " | ".join(get_args(AlignMode))),
    thresholds: list[float] = typer.Option([0.05], "--thresholds", help="F-score thresholds."),
    chamfer_variant: str = typer.Option(
        "l1_mean_bidirectional",
        help="Chamfer variant: " + " | ".join(get_args(ChamferVariant)),
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
    debug_plot: bool = typer.Option(
        False, "--debug-plot", help="Export a 3D scatter plot of aligned point clouds."
    ),
    pred_poses: str | None = typer.Option(
        None, "--pred-poses", help="Prediction trajectory file (.txt) for traj_* alignment."
    ),
    gt_poses: str | None = typer.Option(
        None, "--gt-poses", help="GT trajectory file (.txt) for traj_* alignment."
    ),
    pred_pose_convention: str = typer.Option(
        "unspecified", "--pred-pose-convention", help="Pose convention: T_wc | T_cw."
    ),
    gt_pose_convention: str = typer.Option(
        "unspecified", "--gt-pose-convention", help="Pose convention: T_wc | T_cw."
    ),
) -> None:
    """Compute chamfer, accuracy, completeness, and F-score for a prediction vs. GT."""
    pred_geom = _load_geom(pred)
    gt_geom = _load_geom(gt)
    if align not in get_args(AlignMode):
        raise typer.BadParameter(f"--align must be one of {get_args(AlignMode)}")
    if sample_method not in get_args(SampleMethod):
        raise typer.BadParameter(f"--sample-method must be one of {get_args(SampleMethod)}")
    if chamfer_variant not in get_args(ChamferVariant):
        raise typer.BadParameter(f"--chamfer-variant must be one of {get_args(ChamferVariant)}")

    pred_traj = _load_pred_poses(pred, pred_poses, pred_pose_convention)
    gt_traj = _load_poses(gt_poses, gt_pose_convention) if gt_poses else None

    if isinstance(align, str) and align.startswith("traj_"):
        if pred_traj is None:
            raise typer.BadParameter(
                "Trajectory alignment requires prediction poses. "
                "If the prediction directory contains a manifest with "
                "trajectory data it is used automatically; otherwise "
                "provide --pred-poses."
            )
        if gt_traj is None:
            raise typer.BadParameter(
                "Trajectory alignment requires GT poses. Provide --gt-poses."
            )

    result = evaluate_geometry(
        pred_geom,
        gt_geom,
        samples=samples,
        seed=seed,
        sample_method=sample_method,  # type: ignore[arg-type]
        align_mode=align,  # type: ignore[arg-type]
        thresholds=thresholds,
        chamfer_variant=chamfer_variant,  # type: ignore[arg-type]
        debug_plot_path="debug_plot.png" if debug_plot else None,
        pred_poses=pred_traj.poses if pred_traj else None,
        gt_poses=gt_traj.poses if gt_traj else None,
        pred_convention=pred_traj.convention if pred_traj else pred_pose_convention,
        gt_convention=gt_traj.convention if gt_traj else gt_pose_convention,
        pred_timestamps=pred_traj.timestamps if pred_traj else None,
        gt_timestamps=gt_traj.timestamps if gt_traj else None,
    )
    print_geometry_result(result, as_json=json_out)


@app.command("chamfer")
def chamfer_cmd(
    pred: str = typer.Argument(...),
    gt: str = typer.Option(..., "--gt"),
    samples: int = typer.Option(200_000),
    seed: int = typer.Option(42),
    align: str = typer.Option("none"),
    chamfer_variant: str = typer.Option("l1_mean_bidirectional"),
    json_out: bool = typer.Option(False, "--json"),
    debug_plot: bool = typer.Option(
        False, "--debug-plot", help="Export a 3D scatter plot of aligned point clouds."
    ),
    pred_poses: str | None = typer.Option(
        None, "--pred-poses", help="Prediction trajectory file (.txt) for traj_* alignment."
    ),
    gt_poses: str | None = typer.Option(
        None, "--gt-poses", help="GT trajectory file (.txt) for traj_* alignment."
    ),
    pred_pose_convention: str = typer.Option(
        "unspecified", "--pred-pose-convention", help="Pose convention: T_wc | T_cw."
    ),
    gt_pose_convention: str = typer.Option(
        "unspecified", "--gt-pose-convention", help="Pose convention: T_wc | T_cw."
    ),
) -> None:
    """Just chamfer distance, with thresholds=[]."""
    pred_geom = _load_geom(pred)
    gt_geom = _load_geom(gt)
    pred_traj = _load_pred_poses(pred, pred_poses, pred_pose_convention)
    gt_traj = _load_poses(gt_poses, gt_pose_convention) if gt_poses else None

    if isinstance(align, str) and align.startswith("traj_"):
        if pred_traj is None:
            raise typer.BadParameter(
                "Trajectory alignment requires prediction poses. "
                "If the prediction directory contains a manifest with "
                "trajectory data it is used automatically; otherwise "
                "provide --pred-poses."
            )
        if gt_traj is None:
            raise typer.BadParameter(
                "Trajectory alignment requires GT poses. Provide --gt-poses."
            )

    result = evaluate_geometry(
        pred_geom,
        gt_geom,
        samples=samples,
        seed=seed,
        align_mode=align,  # type: ignore[arg-type]
        thresholds=[],
        chamfer_variant=chamfer_variant,  # type: ignore[arg-type]
        debug_plot_path="debug_plot.png" if debug_plot else None,
        pred_poses=pred_traj.poses if pred_traj else None,
        gt_poses=gt_traj.poses if gt_traj else None,
        pred_convention=pred_traj.convention if pred_traj else pred_pose_convention,
        gt_convention=gt_traj.convention if gt_traj else gt_pose_convention,
        pred_timestamps=pred_traj.timestamps if pred_traj else None,
        gt_timestamps=gt_traj.timestamps if gt_traj else None,
    )
    if json_out:
        print(json.dumps({"chamfer": result.chamfer, "variant": result.chamfer_variant}, indent=2))
    else:
        print_geometry_result(result, as_json=False)


@app.command("fscore")
def fscore_cmd(
    pred: str = typer.Argument(...),
    gt: str = typer.Option(..., "--gt"),
    threshold: float = typer.Option(0.05, help="F-score distance threshold."),
    samples: int = typer.Option(200_000),
    seed: int = typer.Option(42),
    align: str = typer.Option("none"),
    json_out: bool = typer.Option(False, "--json"),
    debug_plot: bool = typer.Option(
        False, "--debug-plot", help="Export a 3D scatter plot of aligned point clouds."
    ),
    pred_poses: str | None = typer.Option(
        None, "--pred-poses", help="Prediction trajectory file (.txt) for traj_* alignment."
    ),
    gt_poses: str | None = typer.Option(
        None, "--gt-poses", help="GT trajectory file (.txt) for traj_* alignment."
    ),
    pred_pose_convention: str = typer.Option(
        "unspecified", "--pred-pose-convention", help="Pose convention: T_wc | T_cw."
    ),
    gt_pose_convention: str = typer.Option(
        "unspecified", "--gt-pose-convention", help="Pose convention: T_wc | T_cw."
    ),
) -> None:
    """F-score / precision / recall at a single threshold."""
    pred_geom = _load_geom(pred)
    gt_geom = _load_geom(gt)
    pred_traj = _load_pred_poses(pred, pred_poses, pred_pose_convention)
    gt_traj = _load_poses(gt_poses, gt_pose_convention) if gt_poses else None

    if isinstance(align, str) and align.startswith("traj_"):
        if pred_traj is None:
            raise typer.BadParameter(
                "Trajectory alignment requires prediction poses. "
                "If the prediction directory contains a manifest with "
                "trajectory data it is used automatically; otherwise "
                "provide --pred-poses."
            )
        if gt_traj is None:
            raise typer.BadParameter(
                "Trajectory alignment requires GT poses. Provide --gt-poses."
            )

    result = evaluate_geometry(
        pred_geom,
        gt_geom,
        samples=samples,
        seed=seed,
        align_mode=align,  # type: ignore[arg-type]
        thresholds=[threshold],
        debug_plot_path="debug_plot.png" if debug_plot else None,
        pred_poses=pred_traj.poses if pred_traj else None,
        gt_poses=gt_traj.poses if gt_traj else None,
        pred_convention=pred_traj.convention if pred_traj else pred_pose_convention,
        gt_convention=gt_traj.convention if gt_traj else gt_pose_convention,
        pred_timestamps=pred_traj.timestamps if pred_traj else None,
        gt_timestamps=gt_traj.timestamps if gt_traj else None,
    )
    print_geometry_result(result, as_json=json_out)


@app.command("depth")
def depth_cmd(
    pred: str = typer.Argument(..., help="Predicted depth image (.png or .npy)."),
    gt: str = typer.Option(..., "--gt", help="Ground-truth depth image (.png or .npy)."),
    mask: str = typer.Option(None, "--mask", help="Optional boolean mask (.npy)."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Compute AbsRel, SqRel, RMSE, RMSE log, and δ accuracy for depth maps."""
    pred_depth = _load_depth_image(pred)
    gt_depth = _load_depth_image(gt)
    mask_arr = None
    if mask is not None:
        mask_arr = np.load(mask).astype(bool)
    result = depth_metrics(pred_depth, gt_depth, mask_arr)
    print_depth_result(result, as_json=json_out)
