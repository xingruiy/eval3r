"""eval3r inspect <path> — print a summary of a prediction directory."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from eval3r.manifest.reader import PredictionReader
from eval3r.report.table import print_dict


def run(path: str, *, json_out: bool = False) -> None:
    reader = PredictionReader(path, verify_hashes=False)
    m = reader.manifest
    summary: dict[str, object] = {
        "path": str(Path(path).resolve()),
        "scene_id": m.scene_id,
        "dataset": m.dataset,
        "method": m.method,
        "unit": m.unit.value,
        "coordinate_system": m.coordinate_system.value,
        "pose_convention": m.pose_convention.value,
        "format_version": m.format_version,
        "eval3r_version": m.eval3r_version,
    }
    if m.geometry.mesh is not None:
        summary["mesh.vertices"] = m.geometry.mesh.count
        summary["mesh.faces"] = m.geometry.mesh.extra.get("faces")
    if m.geometry.point_cloud is not None:
        summary["points.count"] = m.geometry.point_cloud.count
    if m.trajectory.tum is not None:
        summary["poses.count"] = m.trajectory.tum.count
    if m.cameras.intrinsics is not None:
        summary["intrinsics"] = "present"
    if m.metadata:
        summary["metadata"] = m.metadata
    if json_out:
        print(json.dumps(summary, indent=2, default=str))
    else:
        Console().print(f"[bold]inspect[/] {path}")
        print_dict(summary, title="prediction summary")
