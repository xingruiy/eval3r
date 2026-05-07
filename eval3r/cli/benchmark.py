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
from eval3r.datasets.base import Asset
from eval3r.metrics.geometry import ChamferVariant
from eval3r.prediction.discovery import PredictionLocator
from eval3r.presets import PRESETS
from eval3r.presets.scannet import SCANNET_PRESET

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _summary_table(
    result_dict: dict, *, summary_key: str, title: str
) -> Table:
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


def _coverage_table(result_dict: dict) -> Table:
    table = Table(title="coverage")
    table.add_column("category", style="cyan")
    table.add_column("count", style="white")
    for k, v in result_dict["coverage"].items():
        table.add_row(k, str(v))
    return table


def _parse_adapter_opts(raw: list[str]) -> dict[str, str]:
    """Parse ``-o key=value`` pairs into a dict."""
    result: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise typer.BadParameter(
                f"Adapter opt must be key=value, got: {item!r}"
            )
        k, v = item.split("=", 1)
        result[k.strip()] = v.strip()
    return result


def _build_config(
    preset: dict,
    *,
    samples: int | None,
    seed: int | None,
    align: str | None,
    thresholds: list[float] | None,
    chamfer_variant: str | None,
    crop: bool,
    crop_margin: float,
    workers: int | None,
    fail_on_missing: bool,
    missing_distance_default: float,
    missing_fscore_default: float,
    debug_plot: bool,
    pred_pose_dir: str | None,
    pred_pose_file: str,
    pred_pose_convention: str,
    verbose: bool,
    mask_dir: str | None,
    mask_name: str,
    world2grid_name: str,
) -> BenchmarkConfig:
    """Merge CLI options with preset defaults into a BenchmarkConfig."""
    align_value = align or preset["align"]
    chamfer_value = chamfer_variant or preset["chamfer_variant"]
    if align_value not in get_args(AlignMode):
        raise typer.BadParameter(f"--align must be one of {get_args(AlignMode)}")
    if chamfer_value not in get_args(ChamferVariant):
        raise typer.BadParameter(f"--chamfer-variant must be one of {get_args(ChamferVariant)}")
    return BenchmarkConfig(
        samples=samples if samples is not None else preset["samples"],
        seed=seed if seed is not None else preset["seed"],
        align=align_value,  # type: ignore[arg-type]
        thresholds=tuple(thresholds) if thresholds else tuple(preset["thresholds"]),
        chamfer_variant=chamfer_value,  # type: ignore[arg-type]
        crop_to_gt_bbox=crop,
        bbox_margin=crop_margin,
        fail_on_missing=fail_on_missing,
        workers=workers if workers is not None else min(8, os.cpu_count() or 1),
        missing_distance_default=missing_distance_default,
        missing_fscore_default=missing_fscore_default,
        debug_plot=debug_plot,
        pred_pose_dir=pred_pose_dir,
        pred_pose_file=pred_pose_file,
        pred_pose_convention=pred_pose_convention,
        verbose=verbose,
        mask_dir=mask_dir,
        mask_name=mask_name,
        world2grid_name=world2grid_name,
    )


@app.command("run")
def run_cmd(
    dataset: str = typer.Argument(..., help="Dataset name (e.g., scannet, replica, dtu)."),
    preds_root: str = typer.Argument(..., help="Predictions root (one subdir per scene)."),
    root: str = typer.Option(..., "--root", help="Dataset filesystem root."),
    split: str | None = typer.Option(
        None,
        "--split",
        help="Path to a split file (one scene id per line). Omit to auto-discover scenes.",
    ),
    samples: int | None = typer.Option(None, help="Samples per scene; default from preset."),
    seed: int | None = typer.Option(None, help="Sampling seed; default from preset."),
    align: str | None = typer.Option(None, help="Alignment mode: " + " | ".join(get_args(AlignMode))),
    thresholds: list[float] | None = typer.Option(None, "--thresholds"),
    chamfer_variant: str | None = typer.Option(None, "--chamfer-variant"),
    crop: bool = typer.Option(False, "--crop/--no-crop", help="Crop pred to GT bbox."),
    crop_margin: float = typer.Option(0.10, help="Bbox crop margin in metres."),
    workers: int | None = typer.Option(None, help="Process workers; default min(8, ncpu)."),
    missing_distance_default: float = typer.Option(
        1.0,
        "--missing-distance-default",
        help="Penalty in metres applied to chamfer/accuracy/completeness for missing or failed scenes (used in summary_all).",
    ),
    missing_fscore_default: float = typer.Option(
        0.0,
        "--missing-fscore-default",
        help="Default applied to f-score / precision / recall for missing or failed scenes (used in summary_all).",
    ),
    geometry_pattern: list[str] = typer.Option(
        [], "--geometry-pattern",
        help="Custom prediction filename patterns (repeatable, prepended to defaults).",
    ),
    adapter_opt: list[str] = typer.Option(
        [], "-o", "--adapter-opt",
        help="Adapter-specific override in key=value form (repeatable). "
             "e.g. -o depth_scale=5000 -o mesh_filename=mesh.ply",
    ),
    fail_on_missing: bool = typer.Option(False, "--fail-on-missing"),
    debug_plot: bool = typer.Option(
        False,
        "--debug-plot",
        help="Export a 3D scatter plot of aligned point clouds per scene.",
    ),
    out: str | None = typer.Option(None, "--out", help="Write full result JSON to this path."),
    csv: str | None = typer.Option(None, "--csv", help="Write per-scene CSV to this path."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON to stdout (no table)."),
    pred_pose_dir: str | None = typer.Option(
        None,
        "--pred-pose-dir",
        help="Directory containing per-scene pose files for non-manifest predictions.",
    ),
    pred_pose_file: str = typer.Option(
        "{scene_id}.txt",
        "--pred-pose-file",
        help="Pose filename pattern (supports {scene_id}) within --pred-pose-dir.",
    ),
    pred_pose_convention: str = typer.Option(
        "unspecified",
        "--pred-pose-convention",
        help="Pose convention for external poses (T_wc or T_cw).",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show error details for failed scenes.",
    ),
    mask_dir: str | None = typer.Option(
        None, "--mask-dir",
        help="Directory with per-scene subdirs containing occlusion_mask.npy + world2grid.txt.",
    ),
    mask_name: str = typer.Option(
        "occlusion_mask.npy", "--mask-name",
        help="Filename of the occlusion mask .npy file within each scene subdir.",
    ),
    world2grid_name: str = typer.Option(
        "world2grid.txt", "--world2grid-name",
        help="Filename of the world2grid .txt file within each scene subdir.",
    ),
) -> None:
    """Run a geometry benchmark against a registered dataset."""
    cls = get_dataset(dataset)
    preset = PRESETS.get(dataset, SCANNET_PRESET)

    # Parse adapter kwargs from -o key=value pairs.
    adapter_kwargs = _parse_adapter_opts(adapter_opt)

    # Build config from CLI + preset.
    cfg = _build_config(
        preset,
        samples=samples,
        seed=seed,
        align=align,
        thresholds=thresholds,
        chamfer_variant=chamfer_variant,
        crop=crop,
        crop_margin=crop_margin,
        workers=workers,
        fail_on_missing=fail_on_missing,
        missing_distance_default=missing_distance_default,
        missing_fscore_default=missing_fscore_default,
        debug_plot=debug_plot,
        pred_pose_dir=pred_pose_dir,
        pred_pose_file=pred_pose_file,
        pred_pose_convention=pred_pose_convention,
        verbose=verbose,
        mask_dir=mask_dir,
        mask_name=mask_name,
        world2grid_name=world2grid_name,
    )

    ds = cls(root, split=split, validate_on_init=False, **adapter_kwargs)  # type: ignore[arg-type]

    # Verify the adapter provides geometry GT.
    if not (ds.supports(Asset.MESH) or ds.supports(Asset.POINT_CLOUD)):
        typer.secho(
            f"Error: dataset {dataset!r} supports neither mesh nor point_cloud GT. "
            f"Supported assets: {sorted(a.value for a in ds.supported_assets)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

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
        console.print(
            _summary_table(
                payload,
                summary_key="summary",
                title=f"summary — available scenes ({payload['dataset']} / {payload['split']})",
            )
        )
        cov = payload["coverage"]
        n_missing = cov["n_total"] - cov["n_evaluated"]
        if n_missing > 0:
            console.print(
                _summary_table(
                    payload,
                    summary_key="summary_all",
                    title=(
                        f"summary — all {cov['n_total']} scenes "
                        f"(missing → distance={cfg.missing_distance_default}, "
                        f"fscore={cfg.missing_fscore_default})"
                    ),
                )
            )


@app.command("scannet", hidden=True)
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
    align: str | None = typer.Option(None, help="Alignment mode: " + " | ".join(get_args(AlignMode))),
    thresholds: list[float] | None = typer.Option(None, "--thresholds"),
    chamfer_variant: str | None = typer.Option(None, "--chamfer-variant"),
    crop: bool = typer.Option(False, "--crop/--no-crop", help="Crop pred to GT bbox."),
    crop_margin: float = typer.Option(0.10, help="Bbox crop margin in metres."),
    workers: int | None = typer.Option(None, help="Process workers; default min(8, ncpu)."),
    missing_distance_default: float = typer.Option(
        1.0,
        "--missing-distance-default",
        help="Penalty in metres applied to chamfer/accuracy/completeness for missing or failed scenes (used in summary_all).",
    ),
    missing_fscore_default: float = typer.Option(
        0.0,
        "--missing-fscore-default",
        help="Default applied to f-score / precision / recall for missing or failed scenes (used in summary_all).",
    ),
    geometry_pattern: list[str] = typer.Option(
        [], "--geometry-pattern",
        help="Custom prediction filename patterns (repeatable, prepended to defaults).",
    ),
    color_subdir: str | None = typer.Option(None, "--color-subdir"),
    depth_subdir: str | None = typer.Option(None, "--depth-subdir"),
    mesh_filename: str | None = typer.Option(None, "--mesh-filename"),
    fail_on_missing: bool = typer.Option(False, "--fail-on-missing"),
    debug_plot: bool = typer.Option(
        False,
        "--debug-plot",
        help="Export a 3D scatter plot of aligned point clouds per scene.",
    ),
    out: str | None = typer.Option(None, "--out", help="Write full result JSON to this path."),
    csv: str | None = typer.Option(None, "--csv", help="Write per-scene CSV to this path."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON to stdout (no table)."),
    pred_pose_dir: str | None = typer.Option(
        None,
        "--pred-pose-dir",
        help="Directory containing per-scene pose files for non-manifest predictions.",
    ),
    pred_pose_file: str = typer.Option(
        "{scene_id}.txt",
        "--pred-pose-file",
        help="Pose filename pattern (supports {scene_id}) within --pred-pose-dir.",
    ),
    pred_pose_convention: str = typer.Option(
        "unspecified",
        "--pred-pose-convention",
        help="Pose convention for external poses (T_wc or T_cw).",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show error details for failed scenes.",
    ),
    mask_dir: str | None = typer.Option(
        None, "--mask-dir",
        help="Directory with per-scene subdirs containing occlusion_mask.npy + world2grid.txt.",
    ),
    mask_name: str = typer.Option(
        "occlusion_mask.npy", "--mask-name",
        help="Filename of the occlusion mask .npy file within each scene subdir.",
    ),
    world2grid_name: str = typer.Option(
        "world2grid.txt", "--world2grid-name",
        help="Filename of the world2grid .txt file within each scene subdir.",
    ),
) -> None:
    """Run the ScanNet benchmark (backward-compatible alias for `e3r benchmark run scannet`)."""
    adapter_opts = []
    if color_subdir is not None:
        adapter_opts.append(f"color_subdir={color_subdir}")
    if depth_subdir is not None:
        adapter_opts.append(f"depth_subdir={depth_subdir}")
    if mesh_filename is not None:
        adapter_opts.append(f"mesh_filename={mesh_filename}")
    run_cmd(
        dataset="scannet",
        preds_root=preds_root,
        root=root,
        split=split,
        samples=samples,
        seed=seed,
        align=align,
        thresholds=thresholds,
        chamfer_variant=chamfer_variant,
        crop=crop,
        crop_margin=crop_margin,
        workers=workers,
        missing_distance_default=missing_distance_default,
        missing_fscore_default=missing_fscore_default,
        geometry_pattern=geometry_pattern,
        adapter_opt=adapter_opts,
        fail_on_missing=fail_on_missing,
        debug_plot=debug_plot,
        out=out,
        csv=csv,
        json_out=json_out,
        pred_pose_dir=pred_pose_dir,
        pred_pose_file=pred_pose_file,
        pred_pose_convention=pred_pose_convention,
        verbose=verbose,
        mask_dir=mask_dir,
        mask_name=mask_name,
        world2grid_name=world2grid_name,
    )


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
    # Check if any scene used masking.
    any_masked = any(
        s["result"] and s["result"].get("masked") for s in payload["scenes"]
    )
    if any_masked:
        fieldnames.extend(["masked", "visible_points", "total_pred_points"])
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
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
