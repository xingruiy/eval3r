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
from eval3r.datasets.base import Asset, DatasetAdapter
from eval3r.datasets.generic import GenericAdapter
from eval3r.metrics.geometry import ChamferVariant
from eval3r.prediction.discovery import PredictionLocator
from eval3r.presets import PRESETS


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


def _parse_scenes(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    items = [s.strip() for s in raw.split(",") if s.strip()]
    return items or None


def _resolve_adapter(
    *,
    dataset: str | None,
    root: str,
    split: str | None,
    adapter_kwargs: dict[str, str],
    gt_path: str | None,
    scenes_file: str | None,
    scenes: list[str] | None,
) -> DatasetAdapter:
    """Build the right adapter for either registered or manual mode."""
    if dataset is not None:
        try:
            cls = get_dataset(dataset)
        except KeyError as e:
            raise typer.BadParameter(str(e)) from e
        return cls(
            root,
            split=split,
            validate_on_init=False,
            **adapter_kwargs,
        )  # type: ignore[arg-type]

    if not gt_path:
        raise typer.BadParameter(
            "Either pass --dataset <name> or --gt-path '<template>' for "
            "manual mode (e.g. --gt-path '{scene_id}/gt.ply')."
        )
    if scenes is None and scenes_file is None and split is None:
        raise typer.BadParameter(
            "Manual mode needs a scene source: pass --scenes 'id1,id2' or "
            "--scenes-file <path> or --split <path>."
        )
    if adapter_kwargs:
        raise typer.BadParameter(
            "Adapter -o overrides require --dataset; the manual-mode "
            "adapter is configured through --gt-path / --scenes."
        )
    return GenericAdapter(
        root,
        gt_path=gt_path,
        scenes_file=scenes_file,
        scenes_list=scenes,
        split=split,
        validate_on_init=False,
    )


def _build_config(
    preset: dict | None,
    *,
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
    verbose: bool,
    mask_dir: str | None,
    mask_pattern: str,
    t_mask_scene_pattern: str,
) -> BenchmarkConfig:
    """Merge CLI options with preset defaults into a BenchmarkConfig.

    When ``preset`` is None (manual mode), fall back to the literal
    ``BenchmarkConfig`` defaults but require ``thresholds`` to be set
    explicitly so the user picks an evaluation policy on purpose.
    """
    if preset is None:
        if not thresholds:
            raise typer.BadParameter(
                "--thresholds is required in manual mode (no preset to fall "
                "back to). Example: --thresholds 0.05"
            )
        defaults: dict = {
            "align": "none",
            "chamfer_variant": "l1_mean_bidirectional",
            "samples": 200_000,
            "seed": 42,
        }
    else:
        defaults = preset

    align_value = align or defaults["align"]
    chamfer_value = chamfer_variant or defaults["chamfer_variant"]
    if align_value not in get_args(AlignMode):
        raise typer.BadParameter(f"--align must be one of {get_args(AlignMode)}")
    if chamfer_value not in get_args(ChamferVariant):
        raise typer.BadParameter(f"--chamfer-variant must be one of {get_args(ChamferVariant)}")
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
        verbose=verbose,
        mask_dir=mask_dir,
        mask_pattern=mask_pattern,
        t_mask_scene_pattern=t_mask_scene_pattern,
    )


def run_cmd(
    preds_root: str = typer.Argument(..., help="Predictions root (one subdir per scene)."),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        help="Registered dataset name (e.g. scannet, replica, dtu). "
             "Omit to use --gt-path manually.",
    ),
    root: str = typer.Option(..., "--root", help="Dataset filesystem root."),
    split: str | None = typer.Option(
        None,
        "--split",
        help="Path to a split file (one scene id per line). Omit to auto-discover.",
    ),
    samples: int | None = typer.Option(None, help="Samples per scene; default from preset."),
    seed: int | None = typer.Option(None, help="Sampling seed; default from preset."),
    align: str | None = typer.Option(None, help="Alignment mode: " + " | ".join(get_args(AlignMode))),
    thresholds: list[float] | None = typer.Option(None, "--thresholds"),
    chamfer_variant: str | None = typer.Option(None, "--chamfer-variant"),
    crop: bool = typer.Option(False, "--crop/--no-crop", help="Crop pred to GT bbox."),
    crop_margin: float = typer.Option(0.10, help="Bbox crop margin in metres."),
    crop_to_eval_region: bool = typer.Option(
        True,
        "--crop-eval-region/--no-crop-eval-region",
        help="If the dataset adapter ships a per-scene crop volume "
             "(e.g. T&T `{scene}.json`), clip prediction points to it before "
             "metrics. No-op for datasets without one.",
    ),
    use_dataset_thresholds: bool = typer.Option(
        True,
        "--dataset-thresholds/--no-dataset-thresholds",
        help="Use per-scene F-score thresholds from the dataset adapter "
             "(e.g. T&T scene-specific τ) when available. Falls back to "
             "--thresholds for adapters without one.",
    ),
    threshold_multiplier: float = typer.Option(
        1.0,
        "--threshold-multiplier",
        help="Scalar applied to every threshold (per-scene τ from the "
             "adapter or the --thresholds fallback). Useful for sensitivity "
             "studies — e.g. `--threshold-multiplier 2.0` scores T&T at 2× "
             "the published τ. 1.0 disables.",
    ),
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
    pred_filename: list[str] = typer.Option(
        [],
        "--pred-filename",
        help=(
            "Custom prediction filename patterns (repeatable, prepended to defaults). "
            "Use {scene_id} in the pattern, e.g. '{scene_id}/mesh.ply' or "
            "'predictions/{scene_id}.ply'."
        ),
    ),
    adapter_opt: list[str] = typer.Option(
        [], "-o", "--adapter-opt",
        help="Adapter-specific override in key=value form (repeatable). "
             "Only valid with --dataset. e.g. -o depth_scale=5000 -o mesh_filename=mesh.ply",
    ),
    gt_path: str | None = typer.Option(
        None, "--gt-path",
        help="Manual mode: path template for ground-truth geometry, relative "
             "to --root. Supports {scene_id}. e.g. '{scene_id}/gt.ply'.",
    ),
    scenes_file: str | None = typer.Option(
        None, "--scenes-file",
        help="Manual mode: text file with one scene id per line.",
    ),
    scenes: str | None = typer.Option(
        None, "--scenes",
        help="Manual mode: comma-separated scene ids, e.g. 's1,s2,s3'.",
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
        help="Directory for occlusion mask path patterns.",
    ),
    mask_pattern: str = typer.Option(
        "{scene_id}/occlusion_mask.npy", "--mask-pattern",
        help="Mask .npy path pattern relative to --mask-dir. Supports {scene_id}.",
    ),
    t_mask_scene_pattern: str = typer.Option(
        "{scene_id}/T_mask_scene.txt", "--t-mask-scene-pattern",
        help="T_mask_scene .txt path pattern relative to --mask-dir. Supports {scene_id}.",
    ),
) -> None:
    """Run a geometry benchmark.

    Two modes:

    - Registered: pass --dataset <name>. The matching adapter handles GT
      layout. Use -o key=value for per-adapter overrides. Preset values supply
      sensible thresholds / sampling defaults.

    - Manual: omit --dataset and pass --gt-path '<template>' plus a scene
      source (--scenes / --scenes-file / --split). No preset is applied;
      --thresholds is required.
    """
    preset = PRESETS.get(dataset) if dataset else None
    adapter_kwargs = _parse_adapter_opts(adapter_opt)
    scenes_list = _parse_scenes(scenes)

    ds = _resolve_adapter(
        dataset=dataset,
        root=root,
        split=split,
        adapter_kwargs=adapter_kwargs,
        gt_path=gt_path,
        scenes_file=scenes_file,
        scenes=scenes_list,
    )

    if Path(mask_pattern).is_absolute():
        typer.echo("--mask-pattern must be relative to --mask-dir", err=True)
        raise typer.Exit(code=2)
    if Path(t_mask_scene_pattern).is_absolute():
        typer.echo("--t-mask-scene-pattern must be relative to --mask-dir", err=True)
        raise typer.Exit(code=2)


    cfg = _build_config(
        preset,
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
        mask_dir=mask_dir,
        mask_pattern=mask_pattern,
        t_mask_scene_pattern=t_mask_scene_pattern,
    )

    if not (ds.supports(Asset.MESH) or ds.supports(Asset.POINT_CLOUD)):
        typer.secho(
            f"Error: dataset {ds.name!r} supports neither mesh nor point_cloud GT. "
            f"Supported assets: {sorted(a.value for a in ds.supported_assets)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    locator = PredictionLocator(
        preds_root=Path(preds_root),
        extra_patterns=tuple(pred_filename),
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


def _write_csv(path: Path, payload: dict) -> None:
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
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
