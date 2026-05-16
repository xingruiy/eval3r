"""e3r benchmark replica — Replica geometry benchmark."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from eval3r.benchmark.replica import ReplicaBenchmark, ReplicaBenchmarkConfig
from eval3r.cli.benchmark._common import run_and_emit
from eval3r.manifest.discovery import PredictionLocator


def command(
    pred_root: Annotated[str, typer.Option("--pred-root", help="Predictions root.")] = ...,  # type: ignore[assignment]
    gt_root: Annotated[str, typer.Option("--gt-root", help="Replica dataset root.")] = ...,  # type: ignore[assignment]
    split: Annotated[str | None, typer.Option("--split", help="Scene-id list file.")] = None,
    pred_pattern: Annotated[list[str] | None, typer.Option("--pred-pattern")] = None,
    mesh_filename: Annotated[str, typer.Option("--mesh-filename")] = "mesh.ply",
    trajectory_filename: Annotated[str, typer.Option("--trajectory-filename")] = "trajectory.txt",
    aligner: Annotated[str, typer.Option("--aligner")] = "none",
    sampler: Annotated[str, typer.Option("--sampler")] = "area",
    metrics: Annotated[list[str] | None, typer.Option("--metric")] = None,
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
    """Run a Replica geometry benchmark."""
    cfg = ReplicaBenchmarkConfig(
        sampler=sampler,
        aligner=aligner,
        metrics=list(metrics or ["chamfer", "accuracy", "completeness", "fscore@0.05"]),
        samples=samples,
        seed=seed,
        workers=workers if workers is not None else min(8, os.cpu_count() or 1),
        fail_on_missing=fail_on_missing,
        missing_distance_default=missing_distance_default,
        missing_fscore_default=missing_fscore_default,
        pred_pose_dir=pred_pose_dir,
        pred_pose_file=pred_pose_file,
        pred_pose_convention=pred_pose_convention,
        mesh_filename=mesh_filename,
        trajectory_filename=trajectory_filename,
    )
    locator = PredictionLocator(
        preds_root=Path(pred_root),
        extra_patterns=tuple(pred_pattern or []),
    )
    result = ReplicaBenchmark(gt_root=gt_root, pred_root=pred_root, cfg=cfg).run(
        split=split, locator=locator
    )
    run_and_emit(result, out=out, csv=csv, json_out=json_out)
