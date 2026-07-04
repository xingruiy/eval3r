"""CSV writers for aggregate (``results.csv``) and per-scene (``per_scene.csv``) output.

Core status and coverage columns are stable (present regardless of which metrics ran)
so partial coverage is always visible, per ``.agent/reproducibility.md``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from eval3r.core.result import RunResult

# Stable core columns for per_scene.csv (metric-dependent extras may follow later).
PER_SCENE_COLUMNS: list[str] = [
    "scene_id",
    "status",
    "failure_reason",
    "accuracy",
    "completeness",
    "chamfer",
    "precision",
    "recall",
    "fscore",
    "n_points_pred",
    "n_points_gt",
    "n_pixels_valid",
    "valid_fraction",
    "culled_fraction",
    "alignment_mode",
    "alignment_scale",
    "backend_nn",
    "backend_mesh",
    "runtime_seconds",
]

_METRIC_COLUMNS = {"accuracy", "completeness", "chamfer", "precision", "recall", "fscore"}


def write_results_csv(result: RunResult, path: Path) -> None:
    """Aggregate metrics as ``metric,value`` rows (blank value for missing metrics)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for name, value in result.metrics.items():
            writer.writerow([name, "" if value is None else value])


def _backend_name(result: RunResult, kind: str) -> str:
    entry = result.backend_versions.get(kind)
    if isinstance(entry, dict):
        return str(entry.get("name", ""))
    if isinstance(entry, str):
        return entry
    return ""


def per_scene_rows(result: RunResult) -> list[dict[str, Any]]:
    """Build one stable-column row per scene from per-scene metrics and failures."""
    rows: dict[str, dict[str, Any]] = {}
    nn_backend = _backend_name(result, "nearest_neighbor")
    mesh_backend = _backend_name(result, "mesh")
    alignment_mode = result.alignment.mode

    def _blank_row(scene_id: str) -> dict[str, Any]:
        row = {col: "" for col in PER_SCENE_COLUMNS}
        row["scene_id"] = scene_id
        row["alignment_mode"] = alignment_mode
        row["backend_nn"] = nn_backend
        row["backend_mesh"] = mesh_backend
        return row

    # Successful / evaluated scenes: fill from per-scene MetricResults.
    for mr in result.per_scene_metrics:
        scene_id = mr.scene_id or "<unknown>"
        row = rows.setdefault(scene_id, _blank_row(scene_id))
        row["status"] = "ok"
        if mr.name in _METRIC_COLUMNS and mr.value is not None:
            row[mr.name] = mr.value
        for attr in ("n_points_pred", "n_points_gt", "n_pixels_valid", "valid_fraction"):
            val = getattr(mr, attr)
            if val is not None and row[attr] == "":
                row[attr] = val
        if mr.culled_fraction is not None and row["culled_fraction"] == "":
            row["culled_fraction"] = mr.culled_fraction
        runtime = mr.metadata.get("runtime_seconds")
        if runtime is not None and row["runtime_seconds"] == "":
            row["runtime_seconds"] = runtime

    # Failed scenes: status + reason (override any partial row).
    for failure in result.failed_scenes:
        row = rows.setdefault(failure.scene_id, _blank_row(failure.scene_id))
        row["status"] = "failed"
        row["failure_reason"] = failure.reason

    return [rows[k] for k in sorted(rows)]


def write_per_scene_csv(result: RunResult, path: Path) -> None:
    """Write ``per_scene.csv`` with stable core columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PER_SCENE_COLUMNS)
        writer.writeheader()
        for row in per_scene_rows(result):
            writer.writerow(row)
