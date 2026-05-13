"""e3r benchmark dtu — DTU geometry benchmark."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from eval3r.benchmark.dtu import DTUBenchmark, DTUBenchmarkConfig
from eval3r.cli.benchmark._common import run_and_emit
from eval3r.manifest.discovery import PredictionLocator


def command(
    pred_root: Annotated[str, typer.Option("--pred-root", help="Predictions root.")] = ...,
    gt_root: Annotated[str, typer.Option("--gt-root", help="DTU dataset root.")] = ...,
    split: Annotated[str | None, typer.Option("--split", help="Scene-id list file.")] = None,
    pred_pattern: Annotated[list[str], typer.Option("--pred-pattern")] = [],
    scan_subdir: Annotated[str, typer.Option("--scan-subdir")] = "scans",
    scan_format: Annotated[str, typer.Option("--scan-format")] = "scan{scan_id}",
    point_cloud_filename: Annotated[str, typer.Option("--point-cloud-filename")] = "points.ply",
    aligner: Annotated[str, typer.Option("--aligner")] = "none",
    sampler: Annotated[str, typer.Option("--sampler")] = "area",
    metrics: Annotated[list[str], typer.Option("--metric")] = ["chamfer", "fscore@1.0", "fscore@2.0", "fscore@5.0"],
    samples: Annotated[int, typer.Option("--samples")] = 200_000,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    workers: Annotated[int | None, typer.Option("--workers")] = None,
    fail_on_missing: Annotated[bool, typer.Option("--fail-on-missing")] = False,
    missing_distance_default: Annotated[float, typer.Option("--missing-distance-default")] = 1.0,
    missing_fscore_default: Annotated[float, typer.Option("--missing-fscore-default")] = 0.0,
    pred_pose_dir: Annotated[str | None, typer.Option("--pred-pose-dir")] = None,
    pred_pose_file: Annotated[str, typer.Option("--pred-pose-file")] = "{scene_id}.txt",
    pred_pose_convention: Annotated[str, typer.Option("--pred-pose-convention")] = "unspecified",
    out: Annotated[str | None, typer.Option("--out")] = None,
    csv: Annotated[str | None, typer.Option("--csv")] = None,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Run a DTU geometry benchmark."""
    cfg = DTUBenchmarkConfig(
        sampler=sampler,
        aligner=aligner,
        metrics=list(metrics),
        samples=samples,
        seed=seed,
        workers=workers if workers is not None else min(8, os.cpu_count() or 1),
        fail_on_missing=fail_on_missing,
        missing_distance_default=missing_distance_default,
        missing_fscore_default=missing_fscore_default,
        pred_pose_dir=pred_pose_dir,
        pred_pose_file=pred_pose_file,
        pred_pose_convention=pred_pose_convention,
        scan_subdir=scan_subdir,
        scan_format=scan_format,
        point_cloud_filename=point_cloud_filename,
    )
    locator = PredictionLocator(
        preds_root=Path(pred_root),
        extra_patterns=tuple(pred_pattern),
    )
    result = DTUBenchmark(gt_root=gt_root, pred_root=pred_root, cfg=cfg).run(
        split=split, locator=locator
    )
    run_and_emit(result, out=out, csv=csv, json_out=json_out)
