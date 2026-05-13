"""ScanNet benchmark CLI wiring."""

from __future__ import annotations

import typer

from eval3r.benchmark._cli_common import (
    build_config,
    parse_adapter_opts,
    run_and_emit,
    validate_relative_pattern,
)
from eval3r.datasets.scannet import ScanNetAdapter
from eval3r.presets.scannet import SCANNET_PRESET


def command(
    pred_root: str = typer.Option(..., "--pred-root", help="Predictions root."),
    gt_root: str = typer.Option(..., "--gt-root", help="ScanNet dataset root."),
    split: str | None = typer.Option(None, "--split", help="Scene-id list."),
    pred_pattern: list[str] = typer.Option(
        [],
        "--pred-pattern",
        help="Custom prediction filename pattern. Repeatable; supports {scene_id}.",
    ),
    scene_subdir: str = typer.Option("scans/{scene_id}", "--scene-subdir"),
    mesh_filename: str = typer.Option("{scene_id}_vh_clean_2.ply", "--mesh-filename"),
    color_subdir: str = typer.Option("color", "--color-subdir"),
    color_format: str = typer.Option("{frame}.jpg", "--color-format"),
    depth_subdir: str = typer.Option("depth", "--depth-subdir"),
    depth_format: str = typer.Option("{frame}.png", "--depth-format"),
    depth_scale: float = typer.Option(1000.0, "--depth-scale"),
    pose_subdir: str = typer.Option("pose", "--pose-subdir"),
    pose_format: str = typer.Option("{frame}.txt", "--pose-format"),
    intrinsics_subdir: str = typer.Option("intrinsic", "--intrinsics-subdir"),
    intrinsics_depth_filename: str = typer.Option(
        "intrinsic_depth.txt", "--intrinsics-depth-filename"
    ),
    intrinsics_color_filename: str = typer.Option(
        "intrinsic_color.txt", "--intrinsics-color-filename"
    ),
    adapter_opt: list[str] = typer.Option([], "-o", "--adapter-opt"),
    samples: int | None = typer.Option(None, "--samples"),
    seed: int | None = typer.Option(None, "--seed"),
    align: str | None = typer.Option(None, "--align"),
    thresholds: list[float] | None = typer.Option(None, "--thresholds"),
    chamfer_variant: str | None = typer.Option(None, "--chamfer-variant"),
    crop: bool = typer.Option(False, "--crop/--no-crop"),
    crop_margin: float = typer.Option(0.10, "--crop-margin"),
    crop_to_eval_region: bool = typer.Option(
        True, "--crop-eval-region/--no-crop-eval-region"
    ),
    use_dataset_thresholds: bool = typer.Option(
        True, "--dataset-thresholds/--no-dataset-thresholds"
    ),
    threshold_multiplier: float = typer.Option(1.0, "--threshold-multiplier"),
    workers: int | None = typer.Option(None, "--workers"),
    missing_distance_default: float = typer.Option(1.0, "--missing-distance-default"),
    missing_fscore_default: float = typer.Option(0.0, "--missing-fscore-default"),
    fail_on_missing: bool = typer.Option(False, "--fail-on-missing"),
    debug_plot: bool = typer.Option(False, "--debug-plot"),
    out: str | None = typer.Option(None, "--out"),
    csv: str | None = typer.Option(None, "--csv"),
    json_out: bool = typer.Option(False, "--json"),
    pred_pose_dir: str | None = typer.Option(None, "--pred-pose-dir"),
    pred_pose_file: str = typer.Option("{scene_id}.txt", "--pred-pose-file"),
    pred_pose_convention: str = typer.Option("unspecified", "--pred-pose-convention"),
    mask_dir: str | None = typer.Option(None, "--mask-dir"),
    mask_pattern: str = typer.Option(
        "{scene_id}/occlusion_mask.npy", "--mask-pattern"
    ),
    t_mask_scene_pattern: str = typer.Option(
        "{scene_id}/T_mask_scene.txt", "--t-mask-scene-pattern"
    ),
) -> None:
    """Run a ScanNet geometry benchmark."""
    validate_relative_pattern(mask_pattern, "--mask-pattern", "--mask-dir")
    validate_relative_pattern(
        t_mask_scene_pattern, "--t-mask-scene-pattern", "--mask-dir"
    )

    adapter_kwargs = {
        "scene_subdir": scene_subdir,
        "color_subdir": color_subdir,
        "color_format": color_format,
        "depth_subdir": depth_subdir,
        "depth_format": depth_format,
        "depth_scale": depth_scale,
        "pose_subdir": pose_subdir,
        "pose_format": pose_format,
        "intrinsics_subdir": intrinsics_subdir,
        "intrinsics_depth_filename": intrinsics_depth_filename,
        "intrinsics_color_filename": intrinsics_color_filename,
        "mesh_filename": mesh_filename,
    }
    adapter_kwargs.update(parse_adapter_opts(adapter_opt))
    ds = ScanNetAdapter(gt_root, split=split, validate_on_init=False, **adapter_kwargs)
    cfg = build_config(
        SCANNET_PRESET,
        require_thresholds=False,
        samples=samples,
        seed=seed,
        align=align,
        thresholds=thresholds,
        chamfer_variant=chamfer_variant,
        crop=crop,
        crop_margin=crop_margin,
        crop_to_eval_region=crop_to_eval_region,
        use_dataset_thresholds=use_dataset_thresholds,
        threshold_multiplier=threshold_multiplier,
        workers=workers,
        fail_on_missing=fail_on_missing,
        missing_distance_default=missing_distance_default,
        missing_fscore_default=missing_fscore_default,
        debug_plot=debug_plot,
        pred_pose_dir=pred_pose_dir,
        pred_pose_file=pred_pose_file,
        pred_pose_convention=pred_pose_convention,
        mask_dir=mask_dir,
        mask_pattern=mask_pattern,
        t_mask_scene_pattern=t_mask_scene_pattern,
    )
    run_and_emit(
        ds,
        pred_root=pred_root,
        split=split,
        cfg=cfg,
        pred_patterns=pred_pattern,
        out=out,
        csv=csv,
        json_out=json_out,
    )
