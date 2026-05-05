"""e3r benchmark — compare a method's predictions against a dataset split."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import get_args

import typer
from rich.console import Console
from rich.table import Table

from eval3r.align import AlignMode
from eval3r.benchmark import BenchmarkConfig, run_benchmark
from eval3r.datasets import get_dataset
from eval3r.metrics.geometry import ChamferVariant
from eval3r.prediction.discovery import PredictionLocator

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _summary_table(result_dict: dict) -> Table:
    table = Table(title=f"benchmark {result_dict['dataset']} ({result_dict['split']})")
    table.add_column("metric", style="cyan")
    table.add_column("mean", style="white")
    table.add_column("median", style="white")
    table.add_column("std", style="white")
    table.add_column("n", style="white")
    for metric, stats in result_dict["summary"].items():
        table.add_row(
            metric,
            f"{stats['mean']:.6f}",
            f"{stats['median']:.6f}",
            f"{stats['std']:.6f}",
            str(stats["n"]),
        )
    return table


def _coverage_table(result_dict: dict) -> Table:
    table = Table(title="coverage")
    table.add_column("category", style="cyan")
    table.add_column("count", style="white")
    for k, v in result_dict["coverage"].items():
        table.add_row(k, str(v))
    return table


@app.command("scannet")
def scannet_cmd(
    preds_root: str = typer.Argument(..., help="Predictions root (one subdir per scene)."),
    root: str = typer.Option(..., "--root", help="ScanNet filesystem root."),
    split: str | None = typer.Option(
        None,
        "--split",
        help="Path to a split file (one scene id per line). Omit to auto-discover scenes.",
    ),
    samples: int | None = typer.Option(None, help="Samples per scene; default from preset."),
    seed: int | None = typer.Option(None, help="Sampling seed; default from preset."),
    align: str | None = typer.Option(None, help="Alignment mode."),
    thresholds: list[float] | None = typer.Option(None, "--thresholds"),
    chamfer_variant: str | None = typer.Option(None, "--chamfer-variant"),
    crop: bool = typer.Option(False, "--crop/--no-crop", help="Crop pred to GT bbox."),
    crop_margin: float = typer.Option(0.10, help="Bbox crop margin in metres."),
    workers: int | None = typer.Option(None, help="Process workers; default min(8, ncpu)."),
    geometry_pattern: list[str] = typer.Option(
        [], "--geometry-pattern",
        help="Custom prediction filename patterns (repeatable, prepended to defaults).",
    ),
    color_subdir: str | None = typer.Option(None, "--color-subdir"),
    depth_subdir: str | None = typer.Option(None, "--depth-subdir"),
    mesh_filename: str | None = typer.Option(None, "--mesh-filename"),
    fail_on_missing: bool = typer.Option(False, "--fail-on-missing"),
    out: str | None = typer.Option(None, "--out", help="Write full result JSON to this path."),
    csv: str | None = typer.Option(None, "--csv", help="Write per-scene CSV to this path."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON to stdout (no table)."),
) -> None:
    """Run the ScanNet benchmark for a method's predictions directory."""
    from eval3r.presets.scannet import SCANNET_PRESET

    cls = get_dataset("scannet")
    overrides = {
        k: v
        for k, v in {
            "color_subdir": color_subdir,
            "depth_subdir": depth_subdir,
            "mesh_filename": mesh_filename,
        }.items()
        if v is not None
    }
    ds = cls(root, split=split, validate_on_init=False, **overrides)  # type: ignore[arg-type]

    align_value = align or SCANNET_PRESET["align"]
    chamfer_value = chamfer_variant or SCANNET_PRESET["chamfer_variant"]
    if align_value not in get_args(AlignMode):
        raise typer.BadParameter(f"--align must be one of {get_args(AlignMode)}")
    if chamfer_value not in get_args(ChamferVariant):
        raise typer.BadParameter(f"--chamfer-variant must be one of {get_args(ChamferVariant)}")

    cfg = BenchmarkConfig(
        samples=samples if samples is not None else SCANNET_PRESET["samples"],
        seed=seed if seed is not None else SCANNET_PRESET["seed"],
        align=align_value,  # type: ignore[arg-type]
        thresholds=tuple(thresholds) if thresholds else tuple(SCANNET_PRESET["thresholds"]),
        chamfer_variant=chamfer_value,  # type: ignore[arg-type]
        crop_to_gt_bbox=crop,
        bbox_margin=crop_margin,
        fail_on_missing=fail_on_missing,
        workers=workers if workers is not None else min(8, os.cpu_count() or 1),
    )
    locator = PredictionLocator(
        preds_root=Path(preds_root),
        extra_patterns=tuple(geometry_pattern),
    )
    result = run_benchmark(ds, preds_root, locator=locator, config=cfg, split=split)
    payload = result.to_dict()

    if out:
        Path(out).write_text(json.dumps(payload, indent=2))
    if csv:
        _write_csv(Path(csv), payload)
    if json_out:
        print(json.dumps(payload, indent=2))
    else:
        console = Console()
        console.print(_coverage_table(payload))
        console.print(_summary_table(payload))


def _write_csv(path: Path, payload: dict) -> None:
    import csv as _csv

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    fieldnames = [
        "scene_id",
        "status",
        "chamfer",
        "accuracy",
        "completeness",
    ]
    # Add f-score / precision / recall columns per threshold encountered.
    extra_keys: set[str] = set()
    for s in payload["scenes"]:
        if s["result"]:
            for thr in s["result"]["fscore"]:
                extra_keys.update({f"f@{thr}", f"precision@{thr}", f"recall@{thr}"})
    fieldnames.extend(sorted(extra_keys))

    for s in payload["scenes"]:
        row = {"scene_id": s["scene_id"], "status": s["status"]}
        if s["result"]:
            row["chamfer"] = s["result"]["chamfer"]
            row["accuracy"] = s["result"]["accuracy"]
            row["completeness"] = s["result"]["completeness"]
            for thr, v in s["result"]["fscore"].items():
                row[f"f@{thr}"] = v["f"]
                row[f"precision@{thr}"] = v["precision"]
                row[f"recall@{thr}"] = v["recall"]
        rows.append(row)

    with path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
