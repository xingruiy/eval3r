"""eval3r metric ... — compute geometry metrics from a prediction directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import math

import typer

import numpy as np

from eval3r.alignment import AlignMode
from eval3r.datasets import Asset, get_dataset
from eval3r.io.geometry import load_mesh, load_point_cloud
from eval3r.io.trajectory import Trajectory, load_trajectory_auto
from eval3r.metric.depth import depth_metrics
from eval3r.metric.geometry import ChamferVariant, evaluate_geometry
from eval3r.sampling import SampleMethod
from eval3r.prediction.reader import PredictionReader
from eval3r.presets import PRESETS
from eval3r.report.table import print_depth_result, print_geometry_result
from eval3r.utils.errors import MissingArtifactError, NotSupportedError
from eval3r.utils.optional import optional_import

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _load_depth_image(path: str, scale: float = 1.0) -> np.ndarray:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        return np.load(p).astype(np.float32) * np.float32(scale)
    if suffix == ".png":
        imageio = optional_import("imageio.v3", extra="render")
        return imageio.imread(p).astype(np.float32) * np.float32(scale)
    raise typer.BadParameter(f"Unsupported depth image format: {suffix}. Use .png or .npy.")




def _warn_if_png_default_scale(path: str, scale: float, scale_flag: str) -> None:
    if Path(path).suffix.lower() != ".png":
        return
    if not math.isclose(scale, 1.0):
        return
    typer.echo(
        (
            "WARNING: PNG depth loaded with scale=1.0. "
            "If this is a uint16 depth map in millimeters, use "
            f"--{scale_flag} 0.001."
        ),
        err=True,
    )

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


@dataclass
class GTResolution:
    geom: object  # MeshData | PointCloudData
    poses: Trajectory | None = None


def _resolve_gt(
    gt: str,
    dataset: str | None,
    scene_id: str | None,
) -> GTResolution:
    """Resolve --gt to (geometry, optional poses) using --dataset/--scene-id.

    File path → load via :func:`_load_geom` (preset still fills metric defaults
    upstream). Folder path → require both --dataset and --scene-id; instantiate
    the registered adapter rooted at the folder, confirm the scene, and load
    geometry (mesh-then-points) plus poses if the adapter supports them.
    """
    p = Path(gt)
    if p.is_file():
        return GTResolution(geom=_load_geom(gt))
    if not p.exists():
        raise typer.BadParameter(f"--gt path does not exist: {gt}")

    if dataset is None or scene_id is None:
        # Backward-compatible behavior: allow a prediction artifact directory.
        # If the directory is not a prediction folder, keep the explicit
        # dataset-root validation error for missing flags.
        try:
            return GTResolution(geom=_load_geom(gt))
        except MissingArtifactError as exc:
            missing = [
                flag
                for flag, val in (("--dataset", dataset), ("--scene-id", scene_id))
                if val is None
            ]
            raise typer.BadParameter(
                "--gt is a folder; the following options are required: "
                f"{', '.join(missing)}."
            ) from exc

    try:
        adapter_cls = get_dataset(dataset)
    except KeyError:
        # Preserve legacy directory handling when dataset name is unknown.
        return GTResolution(geom=_load_geom(gt))
    adapter = adapter_cls(root=str(p), validate_on_init=False)
    scenes = adapter.list_scenes()
    if scene_id not in scenes:
        sample = ", ".join(scenes[:5]) + (", ..." if len(scenes) > 5 else "")
        raise typer.BadParameter(
            f"scene {scene_id!r} not found in dataset {dataset!r}. "
            f"Available ({len(scenes)}): {sample}"
        )
    try:
        geom = adapter.load_mesh(scene_id)
    except (NotSupportedError, MissingArtifactError):
        geom = adapter.load_point_cloud(scene_id)
    poses: Trajectory | None = None
    if adapter.supports(Asset.POSES):
        try:
            poses = adapter.load_poses(scene_id)
        except (NotSupportedError, MissingArtifactError):
            poses = None
    return GTResolution(geom=geom, poses=poses)


def _apply_preset(
    dataset: str | None,
    *,
    samples: int | None = None,
    seed: int | None = None,
    align: str | None = None,
    thresholds: list[float] | None = None,
    chamfer_variant: str | None = None,
) -> dict:
    """Resolve sentinels (None) against the dataset preset, then defaults.

    Returns a dict with keys ``samples``, ``seed``, ``align``, ``thresholds``,
    ``chamfer_variant`` — only keys whose argument was passed in are populated.
    CLI explicit values win; preset fills the rest; hardcoded defaults last.
    """
    if dataset is not None and dataset not in PRESETS:
        raise typer.BadParameter(
            f"unknown dataset: {dataset!r}. Available: {sorted(PRESETS)}"
        )
    preset = PRESETS.get(dataset, {}) if dataset else {}
    return {
        "samples": samples if samples is not None else preset.get("samples", 200_000),
        "seed": seed if seed is not None else preset.get("seed", 42),
        "align": align if align is not None else preset.get("align", "none"),
        "thresholds": (
            list(thresholds) if thresholds else list(preset.get("thresholds", [0.05]))
        ),
        "chamfer_variant": (
            chamfer_variant
            if chamfer_variant is not None
            else preset.get("chamfer_variant", "l1_mean_bidirectional")
        ),
    }


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
        from eval3r.filtering.occlusion import load_occlusion_mask
        return load_occlusion_mask(mask_path, t_mask_scene_path)
    return None


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

@app.command("geometry")
def geometry_cmd(
    pred: str = typer.Argument(..., help="Prediction directory or geometry file."),
    gt: str = typer.Option(..., "--gt", help="Ground-truth geometry file or dataset folder."),
    dataset: str | None = typer.Option(
        None, "--dataset",
        help=(
            "Registered dataset name: " + " | ".join(sorted(PRESETS)) + ". "
            "Required when --gt is a folder. When --gt is a file, the preset still "
            "fills metric defaults (chamfer variant, thresholds, samples, seed, align)."
        ),
    ),
    scene_id: str | None = typer.Option(
        None, "--scene-id",
        help="Scene id within the dataset. Required when --gt is a folder.",
    ),
    samples: int | None = typer.Option(None, help="Number of samples for metric evaluation."),
    seed: int | None = typer.Option(None, help="RNG seed for sampling."),
    sample_method: str = typer.Option("area", help="area | vertex | uniform"),
    align: str | None = typer.Option(
        None, help="Alignment mode: " + " | ".join(get_args(AlignMode))
    ),
    thresholds: list[float] | None = typer.Option(None, "--thresholds", help="F-score thresholds."),
    chamfer_variant: str | None = typer.Option(
        None,
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
) -> None:
    """Compute chamfer, accuracy, completeness, and F-score for a prediction vs. GT."""
    pred_geom = _load_geom(pred)
    gt_resolution = _resolve_gt(gt, dataset, scene_id)
    gt_geom = gt_resolution.geom
    resolved = _apply_preset(
        dataset,
        samples=samples,
        seed=seed,
        align=align,
        thresholds=thresholds,
        chamfer_variant=chamfer_variant,
    )
    samples = resolved["samples"]
    seed = resolved["seed"]
    align = resolved["align"]
    thresholds = resolved["thresholds"]
    chamfer_variant = resolved["chamfer_variant"]
    _validate_metric_options(
        align=align, sample_method=sample_method, chamfer_variant=chamfer_variant
    )

    pred_traj = _load_pred_poses(pred, pred_poses, pred_pose_convention)
    if gt_resolution.poses is not None:
        gt_traj = gt_resolution.poses
    elif gt_poses is not None:
        gt_traj = _load_poses(gt_poses, gt_pose_convention)
    else:
        gt_traj = None

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

    pred_mask = _load_pred_mask(mask_path, t_mask_scene_path)

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
        pred_mask=pred_mask,
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
    _warn_if_png_default_scale(pred, pred_scale, "pred-scale")
    _warn_if_png_default_scale(gt, gt_scale, "gt-scale")

    pred_depth = _load_depth_image(pred, scale=pred_scale)
    gt_depth = _load_depth_image(gt, scale=gt_scale)
    mask_arr = None
    if mask is not None:
        mask_arr = np.load(mask).astype(bool)
    result = depth_metrics(pred_depth, gt_depth, mask_arr)
    print_depth_result(result, as_json=json_out)
