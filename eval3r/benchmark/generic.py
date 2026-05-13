"""Generic-layout benchmark CLI wiring."""

from __future__ import annotations

import typer

from eval3r.benchmark._cli_common import (
    build_config,
    parse_scenes,
    run_and_emit,
    validate_relative_pattern,
)
from eval3r.datasets.generic import GenericAdapter


def command(
    pred_root: str = typer.Option(..., "--pred-root", help="Predictions root."),
    gt_root: str = typer.Option(..., "--gt-root", help="Ground-truth root."),
    gt_path: str = typer.Option(
        ...,
        "--gt-path",
        help="Ground-truth geometry template relative to --gt-root.",
    ),
    split: str | None = typer.Option(None, "--split", help="Scene-id list."),
    scenes_file: str | None = typer.Option(None, "--scenes-file"),
    scenes: str | None = typer.Option(None, "--scenes"),
    pred_pattern: list[str] = typer.Option([], "--pred-pattern"),
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
    """Run a benchmark against a manual `{scene_id}` ground-truth layout."""
    if scenes is None and scenes_file is None and split is None:
        raise typer.BadParameter(
            "generic benchmark needs a scene source: pass --scenes, "
            "--scenes-file, or --split."
        )
    validate_relative_pattern(mask_pattern, "--mask-pattern", "--mask-dir")
    validate_relative_pattern(
        t_mask_scene_pattern, "--t-mask-scene-pattern", "--mask-dir"
    )

    ds = GenericAdapter(
        gt_root,
        gt_path=gt_path,
        scenes_file=scenes_file,
        scenes_list=parse_scenes(scenes),
        split=split,
        validate_on_init=False,
    )
    cfg = build_config(
        None,
        require_thresholds=True,
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
