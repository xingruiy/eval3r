"""Shared helpers for dataset-specific benchmark CLI commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, get_args

import typer
from rich.console import Console
from rich.table import Table

from eval3r.alignment import AlignMode
from eval3r.benchmark.core import BenchmarkConfig, run_benchmark
from eval3r.datasets.base import Asset, DatasetAdapter
from eval3r.metric.geometry import ChamferVariant
from eval3r.prediction.discovery import PredictionLocator


def summary_table(result_dict: dict, *, summary_key: str, title: str) -> Table:
    table = Table(title=title)
    table.add_column("metric", style="cyan")
    table.add_column("mean", style="white")
    table.add_column("median", style="white")
    table.add_column("std", style="white")
    table.add_column("n", style="white")
    for metric, stats in result_dict[summary_key].items():
        table.add_row(
            metric,
            f"{stats['mean']:.6f}",
            f"{stats['median']:.6f}",
            f"{stats['std']:.6f}",
            str(stats["n"]),
        )
    return table


def coverage_table(result_dict: dict) -> Table:
    table = Table(title="coverage")
    table.add_column("category", style="cyan")
    table.add_column("count", style="white")
    for k, v in result_dict["coverage"].items():
        table.add_row(k, str(v))
    return table


def parse_adapter_opts(raw: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise typer.BadParameter(f"Adapter opt must be key=value, got: {item!r}")
        k, v = item.split("=", 1)
        result[k.strip()] = v.strip()
    return result


def parse_scenes(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    items = [s.strip() for s in raw.split(",") if s.strip()]
    return items or None


def validate_relative_pattern(pattern: str, option_name: str, root_option_name: str) -> None:
    if Path(pattern).is_absolute():
        typer.echo(f"{option_name} must be relative to {root_option_name}", err=True)
        raise typer.Exit(code=2)


def build_config(
    preset: dict | None,
    *,
    require_thresholds: bool,
    samples: int | None,
    seed: int | None,
    align: str | None,
    thresholds: list[float] | None,
    chamfer_variant: str | None,
    crop: bool,
    crop_margin: float,
    crop_to_eval_region: bool,
    use_dataset_thresholds: bool,
    threshold_multiplier: float,
    workers: int | None,
    fail_on_missing: bool,
    missing_distance_default: float,
    missing_fscore_default: float,
    debug_plot: bool,
    pred_pose_dir: str | None,
    pred_pose_file: str,
    pred_pose_convention: str,
    mask_dir: str | None,
    mask_pattern: str,
    t_mask_scene_pattern: str,
) -> BenchmarkConfig:
    if preset is None:
        if require_thresholds and not thresholds:
            raise typer.BadParameter(
                "--thresholds is required when no dataset preset is available. "
                "Example: --thresholds 0.05"
            )
        defaults: dict[str, Any] = {
            "align": "none",
            "chamfer_variant": "l1_mean_bidirectional",
            "samples": 200_000,
            "seed": 42,
            "thresholds": (0.05,),
        }
    else:
        defaults = preset

    align_value = align or defaults["align"]
    chamfer_value = chamfer_variant or defaults["chamfer_variant"]
    if align_value not in get_args(AlignMode):
        raise typer.BadParameter(f"--align must be one of {get_args(AlignMode)}")
    if chamfer_value not in get_args(ChamferVariant):
        raise typer.BadParameter(
            f"--chamfer-variant must be one of {get_args(ChamferVariant)}"
        )

    return BenchmarkConfig(
        samples=samples if samples is not None else defaults["samples"],
        seed=seed if seed is not None else defaults["seed"],
        align=align_value,  # type: ignore[arg-type]
        thresholds=tuple(thresholds) if thresholds else tuple(defaults["thresholds"]),
        chamfer_variant=chamfer_value,  # type: ignore[arg-type]
        crop_to_gt_bbox=crop,
        bbox_margin=crop_margin,
        crop_to_eval_region=crop_to_eval_region,
        use_dataset_thresholds=use_dataset_thresholds,
        threshold_multiplier=threshold_multiplier,
        fail_on_missing=fail_on_missing,
        workers=workers if workers is not None else min(8, os.cpu_count() or 1),
        missing_distance_default=missing_distance_default,
        missing_fscore_default=missing_fscore_default,
        debug_plot=debug_plot,
        pred_pose_dir=pred_pose_dir,
        pred_pose_file=pred_pose_file,
        pred_pose_convention=pred_pose_convention,
        verbose=True,
        mask_dir=mask_dir,
        mask_pattern=mask_pattern,
        t_mask_scene_pattern=t_mask_scene_pattern,
    )


def run_and_emit(
    ds: DatasetAdapter,
    *,
    pred_root: str,
    split: str | None,
    cfg: BenchmarkConfig,
    pred_patterns: list[str],
    out: str | None,
    csv: str | None,
    json_out: bool,
) -> None:
    if not (ds.supports(Asset.MESH) or ds.supports(Asset.POINT_CLOUD)):
        typer.secho(
            f"Error: dataset {ds.name!r} supports neither mesh nor point_cloud GT. "
            f"Supported assets: {sorted(a.value for a in ds.supported_assets)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    scenes = ds.list_scenes(split)
    _emit_run_context(
        ds,
        pred_root=pred_root,
        split=split,
        scene_count=len(scenes),
        cfg=cfg,
        pred_patterns=pred_patterns,
        out=out,
        csv=csv,
        json_out=json_out,
    )

    locator = PredictionLocator(
        preds_root=Path(pred_root),
        extra_patterns=tuple(pred_patterns),
    )
    result = run_benchmark(ds, pred_root, locator=locator, config=cfg, split=split)
    payload = result.to_dict()

    if out:
        Path(out).write_text(json.dumps(payload, indent=2))
        typer.echo(f"benchmark: wrote JSON results to {out}", err=True)
    if csv:
        write_csv(Path(csv), payload)
        typer.echo(f"benchmark: wrote CSV results to {csv}", err=True)
    if json_out:
        print(json.dumps(payload, indent=2))
        return

    console = Console()
    console.print(coverage_table(payload))
    console.print(
        summary_table(
            payload,
            summary_key="summary",
            title=f"summary — available scenes ({payload['dataset']} / {payload['split']})",
        )
    )
    cov = payload["coverage"]
    n_missing = cov["n_total"] - cov["n_evaluated"]
    if n_missing > 0:
        console.print(
            summary_table(
                payload,
                summary_key="summary_all",
                title=(
                    f"summary — all {cov['n_total']} scenes "
                    f"(missing → distance={cfg.missing_distance_default}, "
                    f"fscore={cfg.missing_fscore_default})"
                ),
            )
        )


def _emit_run_context(
    ds: DatasetAdapter,
    *,
    pred_root: str,
    split: str | None,
    scene_count: int,
    cfg: BenchmarkConfig,
    pred_patterns: list[str],
    out: str | None,
    csv: str | None,
    json_out: bool,
) -> None:
    gt_root = getattr(ds, "root", None)
    split_label = split if split is not None else getattr(ds, "_split", None)
    split_text = str(split_label) if split_label is not None else "auto"
    typer.echo(
        f"benchmark: dataset={ds.name} split={split_text} scenes={scene_count}",
        err=True,
    )
    typer.echo(f"benchmark: pred_root={pred_root}", err=True)
    if gt_root is not None:
        typer.echo(f"benchmark: gt_root={gt_root}", err=True)
    typer.echo(
        "benchmark: "
        f"workers={cfg.workers} samples={cfg.samples} seed={cfg.seed} "
        f"align={cfg.align} chamfer={cfg.chamfer_variant}",
        err=True,
    )
    typer.echo(
        "benchmark: "
        f"thresholds={list(cfg.thresholds)} "
        f"dataset_thresholds={cfg.use_dataset_thresholds} "
        f"threshold_multiplier={cfg.threshold_multiplier}",
        err=True,
    )
    if cfg.crop_to_gt_bbox or cfg.crop_to_eval_region:
        typer.echo(
            "benchmark: "
            f"crop_gt_bbox={cfg.crop_to_gt_bbox} "
            f"crop_eval_region={cfg.crop_to_eval_region} "
            f"bbox_margin={cfg.bbox_margin}",
            err=True,
        )
    if cfg.mask_dir is not None:
        typer.echo(
            "benchmark: "
            f"mask_dir={cfg.mask_dir} mask_pattern={cfg.mask_pattern} "
            f"t_mask_scene_pattern={cfg.t_mask_scene_pattern}",
            err=True,
        )
    if cfg.pred_pose_dir is not None:
        typer.echo(
            "benchmark: "
            f"pred_pose_dir={cfg.pred_pose_dir} "
            f"pred_pose_file={cfg.pred_pose_file} "
            f"pred_pose_convention={cfg.pred_pose_convention}",
            err=True,
        )
    if pred_patterns:
        typer.echo(f"benchmark: extra prediction patterns={pred_patterns}", err=True)
    targets = []
    if out:
        targets.append(f"json_file={out}")
    if csv:
        targets.append(f"csv_file={csv}")
    if json_out:
        targets.append("stdout=json")
    if targets:
        typer.echo(f"benchmark: outputs {' '.join(targets)}", err=True)


def write_csv(path: Path, payload: dict) -> None:
    import csv as _csv

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    fieldnames = [
        "scene_id",
        "status",
        "mask_missing",
        "chamfer",
        "accuracy",
        "completeness",
    ]
    any_masked = any(
        s["result"] and s["result"].get("masked") for s in payload["scenes"]
    )
    if any_masked:
        fieldnames.extend(["masked", "visible_points", "total_pred_points"])

    extra_keys: set[str] = set()
    for s in payload["scenes"]:
        if s["result"]:
            for thr in s["result"]["fscore"]:
                extra_keys.update({f"f@{thr}", f"precision@{thr}", f"recall@{thr}"})
    fieldnames.extend(sorted(extra_keys))

    for s in payload["scenes"]:
        row = {
            "scene_id": s["scene_id"],
            "status": s["status"],
            "mask_missing": s.get("mask_missing", False),
        }
        if s["result"]:
            row["chamfer"] = s["result"]["chamfer"]
            row["accuracy"] = s["result"]["accuracy"]
            row["completeness"] = s["result"]["completeness"]
            if any_masked:
                row["masked"] = s["result"].get("masked", False)
                row["visible_points"] = s["result"].get("visible_points", 0)
                row["total_pred_points"] = s["result"].get("total_pred_points", 0)
            for thr, v in s["result"]["fscore"].items():
                row[f"f@{thr}"] = v["f"]
                row[f"precision@{thr}"] = v["precision"]
                row[f"recall@{thr}"] = v["recall"]
        rows.append(row)

    with path.open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
