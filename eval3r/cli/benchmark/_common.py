"""Shared display / output helpers for benchmark CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from eval3r.benchmark.base import BenchmarkResult
from eval3r.manifest.discovery import PredictionLocator


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


def parse_scenes(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    items = [s.strip() for s in raw.split(",") if s.strip()]
    return items or None


def validate_relative_pattern(pattern: str, option_name: str, root_option_name: str) -> None:
    if Path(pattern).is_absolute():
        typer.echo(f"{option_name} must be relative to {root_option_name}", err=True)
        raise typer.Exit(code=2)


def write_csv(path: Path, payload: dict) -> None:
    import csv as _csv

    path.parent.mkdir(parents=True, exist_ok=True)

    all_metric_keys: set[str] = set()
    for s in payload["scenes"]:
        r = s.get("result")
        if r:
            for k in r:
                if k not in ("n_samples", "n_visible", "n_total", "align_mode", "align_scale"):
                    all_metric_keys.add(k)

    base_fields = ["scene_id", "status", "pred_path", "gt_path"]
    metric_fields = sorted(all_metric_keys)
    extra_fields = ["n_visible", "n_total", "align_mode", "align_scale"]
    fieldnames = base_fields + metric_fields + extra_fields

    rows = []
    for s in payload["scenes"]:
        row: dict[str, Any] = {
            "scene_id": s["scene_id"],
            "status": s["status"],
            "pred_path": s.get("pred_path") or "",
            "gt_path": s.get("gt_path") or "",
        }
        r = s.get("result")
        if r:
            for k in metric_fields:
                v = r.get(k)
                if isinstance(v, dict):
                    row[k] = v.get("f", "")
                else:
                    row[k] = v if v is not None else ""
            row["n_visible"] = r.get("n_visible", "")
            row["n_total"] = r.get("n_total", "")
            row["align_mode"] = r.get("align_mode", "")
            row["align_scale"] = r.get("align_scale", "")
        rows.append(row)

    with path.open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_and_emit(
    result: BenchmarkResult,
    *,
    out: str | None,
    csv: str | None,
    json_out: bool,
) -> None:
    payload = result.to_dict()

    if out:
        Path(out).write_text(json.dumps(payload, indent=2))
        typer.echo(f"benchmark: wrote JSON results to {out}", err=True)
    if csv:
        write_csv(Path(csv), payload)
        typer.echo(f"benchmark: wrote CSV results to {csv}", err=True)
    else:
        auto_csv = result.work_dir / "results.csv"
        write_csv(auto_csv, payload)
        typer.echo(f"benchmark: wrote CSV results to {auto_csv}", err=True)

    typer.echo(f"benchmark: run artefacts in {result.work_dir}", err=True)
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
    cfg_dict = payload.get("config", {})
    n_missing = cov["n_total"] - cov["n_evaluated"]
    if n_missing > 0:
        console.print(
            summary_table(
                payload,
                summary_key="summary_all",
                title=(
                    f"summary — all {cov['n_total']} scenes "
                    f"(missing → distance={cfg_dict.get('missing_distance_default', 1.0)}, "
                    f"fscore={cfg_dict.get('missing_fscore_default', 0.0)})"
                ),
            )
        )
