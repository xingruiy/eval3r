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
from eval3r.metrics.geometry import ChamferVariant, MaskMode, evaluate_geometry
from eval3r.metrics.sampling import SampleMethod
from eval3r.prediction.reader import PredictionReader
from eval3r.report.table import print_depth_result, print_geometry_result
from eval3r.utils.errors import MissingArtifactError
from eval3r.utils.optional import optional_import

app = typer.Typer(no_args_is_help=True, add_completion=False)


METRIC_PRESETS: dict[str, dict[str, object]] = {
    "dtu": {
        "thresholds": [1.0, 2.0, 5.0],
        "units": "mm",
        "sample_method": "area",
        "align": "none",
        "mask_mode": "pred",
        "gt_geometry": "point_cloud",
    },
    "tanks_temples": {
        "thresholds": [0.01, 0.02, 0.05],
        "units": "m",
        "sample_method": "area",
        "align": "none",
        "mask_mode": "pred",
        "gt_geometry": "mesh",
    },
    "scannet": {
        "thresholds": [0.05],
        "units": "m",
        "sample_method": "area",
        "align": "none",
        "mask_mode": "pred",
        "gt_geometry": "mesh",
    },
}


def _load_depth_image(path: str, scale: float = 1.0) -> np.ndarray:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        return np.load(p).astype(np.float32) * np.float32(scale)
    if suffix == ".png":
        imageio = optional_import("imageio.v3", extra="render")
        return imageio.imread(p).astype(np.float32) * np.float32(scale)
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


def _load_pred_mask(
    mask_path: str | None = None,
    t_mask_scene_path: str | None = None,
) -> object | None:  # OcclusionMask | None
    """Load occlusion mask from explicit file paths."""
    if (mask_path is None) != (t_mask_scene_path is None):
        raise typer.BadParameter(
            "Masking requires both --mask and --t-mask-scene, or neither."
        )
    if mask_path is not None and t_mask_scene_path is not None:
        from eval3r.metrics.occlusion import load_occlusion_mask
        return load_occlusion_mask(mask_path, t_mask_scene_path)
    return None


def _resolve_masks(mask: object | None, mask_mode: str) -> tuple[object | None, object | None]:
    if mask_mode == "pred":
        return mask, None
    if mask_mode == "gt":
        return None, mask
    if mask_mode == "both":
        return mask, mask
    raise typer.BadParameter(f"--mask-mode must be one of {get_args(MaskMode)}")


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



def _validate_metric_options(
    *,
    align: str,
    sample_method: str | None = None,
    chamfer_variant: str | None = None,
) -> None:
    if align not in get_args(AlignMode):
        raise typer.BadParameter(f"--align must be one of {get_args(AlignMode)}")
    if sample_method is not None and sample_method not in get_args(SampleMethod):
        raise typer.BadParameter(f"--sample-method must be one of {get_args(SampleMethod)}")
    if chamfer_variant is not None and chamfer_variant not in get_args(ChamferVariant):
        raise typer.BadParameter(f"--chamfer-variant must be one of {get_args(ChamferVariant)}")


def _load_gt_for_policy(path: str, gt_geometry: str):
    if gt_geometry == "mesh":
        return load_mesh(Path(path))
    if gt_geometry == "point_cloud":
        return load_point_cloud(Path(path))
    return _load_geom(path)

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
    mask_path: str | None = typer.Option(
        None, "--mask", help="Explicit path to occlusion mask .npy for this scene.",
    ),
    t_mask_scene_path: str | None = typer.Option(
        None, "--t-mask-scene", help="Explicit path to T_mask_scene .txt for this scene.",
    ),
    mask_mode: str = typer.Option("pred", "--mask-mode", help="pred | gt | both"),
    preset: str | None = typer.Option(
        None,
        "--preset",
        help="Metric preset: dtu | tanks_temples | scannet. Overrides thresholds, units, sampling, alignment, masking, and GT geometry policy.",
    ),
) -> None:
    """Compute chamfer, accuracy, completeness, and F-score for a prediction vs. GT."""
    config: dict[str, object] = {
        "thresholds": thresholds,
        "sample_method": sample_method,
        "align": align,
        "mask_mode": mask_mode,
        "units": "native",
        "gt_geometry": "auto",
    }
    if preset is not None:
        if preset not in METRIC_PRESETS:
            raise typer.BadParameter(
                f"unknown preset: {preset}; available: {sorted(METRIC_PRESETS)}"
            )
        config.update(METRIC_PRESETS[preset])

    pred_geom = _load_geom(pred)
    gt_geom = _load_gt_for_policy(gt, str(config["gt_geometry"]))
    _validate_metric_options(
        align=str(config["align"]), sample_method=str(config["sample_method"]), chamfer_variant=chamfer_variant
    )

    pred_traj = _load_pred_poses(pred, pred_poses, pred_pose_convention)
    gt_traj = _load_poses(gt_poses, gt_pose_convention) if gt_poses else None

    if str(config["align"]).startswith("traj_"):
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

    mask = _load_pred_mask(mask_path, t_mask_scene_path)
    pred_mask, gt_mask = _resolve_masks(mask, str(config["mask_mode"]))

    result = evaluate_geometry(
        pred_geom,
        gt_geom,
        samples=samples,
        seed=seed,
        sample_method=str(config["sample_method"]),  # type: ignore[arg-type]
        align_mode=str(config["align"]),  # type: ignore[arg-type]
        thresholds=list(config["thresholds"]),
        chamfer_variant=chamfer_variant,  # type: ignore[arg-type]
        debug_plot_path="debug_plot.png" if debug_plot else None,
        pred_poses=pred_traj.poses if pred_traj else None,
        gt_poses=gt_traj.poses if gt_traj else None,
        pred_convention=pred_traj.convention if pred_traj else pred_pose_convention,
        gt_convention=gt_traj.convention if gt_traj else gt_pose_convention,
        pred_timestamps=pred_traj.timestamps if pred_traj else None,
        gt_timestamps=gt_traj.timestamps if gt_traj else None,
        pred_mask=pred_mask,
        gt_mask=gt_mask,
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
    mask_path: str | None = typer.Option(
        None, "--mask", help="Explicit path to occlusion mask .npy for this scene.",
    ),
    t_mask_scene_path: str | None = typer.Option(
        None, "--t-mask-scene", help="Explicit path to T_mask_scene .txt for this scene.",
    ),
    mask_mode: str = typer.Option("pred", "--mask-mode", help="pred | gt | both"),
) -> None:
    """Just chamfer distance, with thresholds=[]."""
    pred_geom = _load_geom(pred)
    gt_geom = _load_geom(gt)
    _validate_metric_options(align=align, chamfer_variant=chamfer_variant)
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

    mask = _load_pred_mask(mask_path, t_mask_scene_path)
    pred_mask, gt_mask = _resolve_masks(mask, mask_mode)

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
        pred_mask=pred_mask,
        gt_mask=gt_mask,
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
    mask_path: str | None = typer.Option(
        None, "--mask", help="Explicit path to occlusion mask .npy for this scene.",
    ),
    t_mask_scene_path: str | None = typer.Option(
        None, "--t-mask-scene", help="Explicit path to T_mask_scene .txt for this scene.",
    ),
    mask_mode: str = typer.Option("pred", "--mask-mode", help="pred | gt | both"),
) -> None:
    """F-score / precision / recall at a single threshold."""
    pred_geom = _load_geom(pred)
    gt_geom = _load_geom(gt)
    _validate_metric_options(align=align)
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

    mask = _load_pred_mask(mask_path, t_mask_scene_path)
    pred_mask, gt_mask = _resolve_masks(mask, mask_mode)

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
        pred_mask=pred_mask,
        gt_mask=gt_mask,
    )
    print_geometry_result(result, as_json=json_out)


@app.command("depth")
def depth_cmd(
    pred: str = typer.Argument(..., help="Predicted depth image (.png or .npy)."),
    gt: str = typer.Option(..., "--gt", help="Ground-truth depth image (.png or .npy)."),
    mask: str = typer.Option(None, "--mask", help="Optional boolean mask (.npy)."),
    pred_scale: float = typer.Option(
        1.0, "--pred-scale", help="Scale factor applied to predicted depth values."
    ),
    gt_scale: float = typer.Option(
        1.0, "--gt-scale", help="Scale factor applied to ground-truth depth values."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Compute AbsRel, SqRel, RMSE, RMSE log, and δ accuracy for depth maps."""
    pred_depth = _load_depth_image(pred, scale=pred_scale)
    gt_depth = _load_depth_image(gt, scale=gt_scale)
    mask_arr = None
    if mask is not None:
        mask_arr = np.load(mask).astype(bool)
    result = depth_metrics(pred_depth, gt_depth, mask_arr)
    print_depth_result(result, as_json=json_out)
