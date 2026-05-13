"""Tanks & Temples benchmark CLI wiring."""

from __future__ import annotations

import typer

from eval3r.benchmark._cli_common import build_config, parse_adapter_opts, run_and_emit
from eval3r.datasets.tanks_temples import TanksTemplesAdapter
from eval3r.presets.tanks_temples import TANKS_TEMPLES_PRESET


def command(
    pred_root: str = typer.Option(..., "--pred-root", help="Predictions root."),
    gt_root: str = typer.Option(..., "--gt-root", help="Tanks & Temples root."),
    split: str | None = typer.Option(None, "--split", help="Scene-name list."),
    subset: str | None = typer.Option(None, "--subset"),
    pred_pattern: list[str] = typer.Option([], "--pred-pattern"),
    point_cloud_filename: str = typer.Option("{scene_id}.ply", "--point-cloud-filename"),
    pose_filename: str = typer.Option("{scene_id}_COLMAP_SfM.log", "--pose-filename"),
    alignment_filename: str = typer.Option("{scene_id}_trans.txt", "--alignment-filename"),
    crop_filename: str = typer.Option("{scene_id}.json", "--crop-filename"),
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
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a Tanks & Temples geometry benchmark."""
    adapter_kwargs = {
        "point_cloud_filename": point_cloud_filename,
        "pose_filename": pose_filename,
        "alignment_filename": alignment_filename,
        "crop_filename": crop_filename,
        "subset": subset,
    }
    adapter_kwargs.update(parse_adapter_opts(adapter_opt))
    ds = TanksTemplesAdapter(
        gt_root, split=split, validate_on_init=False, **adapter_kwargs
    )
    cfg = build_config(
        TANKS_TEMPLES_PRESET,
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
        verbose=verbose,
        mask_dir=None,
        mask_pattern="{scene_id}/occlusion_mask.npy",
        t_mask_scene_pattern="{scene_id}/T_mask_scene.txt",
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
