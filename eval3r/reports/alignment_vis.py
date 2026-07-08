"""Mandatory alignment visualization: overlay PLYs + orthographic projections.

Most real-world alignments are quirky — a flipped, mirrored, or locally-stuck
registration can still produce a plausible-looking residual number. Every alignment
eval3r estimates therefore comes with artifacts a human can look at:

```text
<prefix>_before.ply       pred (untransformed) + gt merged, two fixed colors
<prefix>_after.ply        pred (transformed) + gt merged, same colors
<prefix>_projections.png  before/after x XY/XZ/YZ orthographic scatter, annotated
                          with mode/solver, scale, residual, and fitness
alignment_vis.json        manifest: colors, subsample seed/count, the transform
                          shown, and the per-scene file names
```

The PLYs open in MeshLab/CloudCompare for real inspection; the PNG is the quick
glance. Everything that shaped the artifacts (colors, subsample seed and counts)
is recorded so they are interpretable later. The capture itself
(:class:`~eval3r.pipeline.stages.align.AlignmentVisData`) is produced next to the
align stage; this module only writes files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from eval3r.pipeline.stages.align import AlignmentVisData, TrajectoryAlignmentVisData
from eval3r.reports.json import dump_json

#: Fixed overlay colors (RGB uint8): prediction in red-orange, ground truth in blue.
PRED_COLOR = (227, 74, 51)
GT_COLOR = (49, 130, 189)

#: Point count per side in the projection PNG (the PLYs keep the captured counts).
PNG_MAX_POINTS = 20_000

_PROJECTIONS = (("XY", 0, 1), ("XZ", 0, 2), ("YZ", 1, 2))


def _scene_debug_dir(run_dir: Path, scene_id: str) -> Path:
    return Path(run_dir) / "debug" / "scenes" / scene_id


def _relative_debug_path(run_dir: Path, path: Path) -> str:
    return path.relative_to(Path(run_dir) / "debug").as_posix()


def _update_debug_index(run_dir: Path, key: str, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    debug_dir = Path(run_dir) / "debug"
    index_path = debug_dir / "debug_index.json"
    if index_path.is_file():
        import json

        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"debug_artifacts": {}}
    artifacts = index.setdefault("debug_artifacts", {})
    artifacts[key] = records
    dump_json(index, index_path)


def _merged_overlay(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.vstack([pred, gt])
    colors = np.vstack(
        [
            np.tile(np.asarray(PRED_COLOR, dtype=np.uint8), (pred.shape[0], 1)),
            np.tile(np.asarray(GT_COLOR, dtype=np.uint8), (gt.shape[0], 1)),
        ]
    )
    return points, colors


def _png_subsample(points: np.ndarray, seed: int) -> np.ndarray:
    if points.shape[0] <= PNG_MAX_POINTS:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=PNG_MAX_POINTS, replace=False)
    return points[np.sort(idx)]


def write_alignment_projections_png(vis: AlignmentVisData, path: Path) -> None:
    """Write the before/after x XY/XZ/YZ orthographic scatter overview."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    alignment = vis.alignment
    fig = Figure(figsize=(12, 8))
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
    rows = (
        ("before", _png_subsample(vis.pred_before, vis.subsample_seed)),
        ("after", _png_subsample(vis.pred_after, vis.subsample_seed)),
    )
    gt_png = _png_subsample(vis.gt, vis.subsample_seed)
    for row, (label, pred_png) in enumerate(rows):
        for col, (plane, i, j) in enumerate(_PROJECTIONS):
            ax = axes[row][col]
            ax.scatter(gt_png[:, i], gt_png[:, j], s=0.5, c=[np.asarray(GT_COLOR) / 255.0])
            ax.scatter(
                pred_png[:, i], pred_png[:, j], s=0.5, c=[np.asarray(PRED_COLOR) / 255.0]
            )
            ax.set_title(f"{label} ({plane})")
            ax.set_aspect("equal", adjustable="datalim")

    residual = alignment.get("residual_rmse")
    fitness = alignment.get("fitness")
    annotation = (
        f"scene {vis.scene_id} | mode {alignment.get('mode')} / solver "
        f"{alignment.get('solver')} | scale {alignment.get('scale'):.6g}"
        + (f" | residual RMSE {residual:.6g} m" if residual is not None else "")
        + (f" | fitness {fitness:.4f}" if fitness is not None else "")
        + " | pred = red, gt = blue"
    )
    fig.suptitle(annotation, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(str(path), format="png", dpi=110)


def write_alignment_visualization(
    vis: AlignmentVisData,
    out_dir: Path,
    *,
    pointcloud_backend: Any,
    prefix: str = "alignment",
) -> dict[str, Any]:
    """Write one capture's overlay PLYs + projection PNG; return its manifest record."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    before_ply = out_dir / f"{prefix}_before.ply"
    after_ply = out_dir / f"{prefix}_after.ply"
    png = out_dir / f"{prefix}_projections.png"

    points, colors = _merged_overlay(vis.pred_before, vis.gt)
    pointcloud_backend.save_pointcloud(points, before_ply, colors=colors)
    points, colors = _merged_overlay(vis.pred_after, vis.gt)
    pointcloud_backend.save_pointcloud(points, after_ply, colors=colors)
    write_alignment_projections_png(vis, png)

    return {
        "scene_id": vis.scene_id,
        "before_ply": before_ply.name,
        "after_ply": after_ply.name,
        "projections_png": png.name,
        "pred_color": list(PRED_COLOR),
        "gt_color": list(GT_COLOR),
        "n_points_pred": int(vis.pred_before.shape[0]),
        "n_points_gt": int(vis.gt.shape[0]),
        "subsample_seed": vis.subsample_seed,
        "subsample_max_points": vis.max_points,
        "png_max_points": PNG_MAX_POINTS,
        "alignment": vis.alignment,
    }


def write_alignment_vis_outputs(
    captures: list[AlignmentVisData],
    run_dir: Path,
    *,
    pointcloud_backend: Any,
) -> list[dict[str, Any]]:
    """Write every captured scene's artifacts into ``debug/scenes/<scene_id>/``.

    Returns the manifest records, also written to ``debug/alignment_vis.json``.
    A no-op when nothing was captured (i.e. every scene's alignment mode was
    ``none``).
    """
    if not captures:
        return []
    run_dir = Path(run_dir)
    debug_dir = run_dir / "debug"
    records = []
    for vis in captures:
        scene_dir = _scene_debug_dir(run_dir, vis.scene_id)
        record = write_alignment_visualization(
            vis, scene_dir,
            pointcloud_backend=pointcloud_backend,
            prefix="alignment",
        )
        record["scene_debug_dir"] = _relative_debug_path(run_dir, scene_dir)
        for key in ("before_ply", "after_ply", "projections_png"):
            record[key] = f"{record['scene_debug_dir']}/{record[key]}"
        records.append(record)
    dump_json({"alignment_visualizations": records}, debug_dir / "alignment_vis.json")
    _update_debug_index(run_dir, "alignment_visualizations", records)
    return records


def write_trajectory_alignment_json(
    vis: TrajectoryAlignmentVisData,
    path: Path,
    record: dict[str, Any],
) -> None:
    """Write the per-scene trajectory alignment metadata and position counts."""
    dump_json(
        {
            "scene_id": vis.scene_id,
            "alignment": vis.alignment,
            "association": vis.association,
            "counts": {
                "n_pred_poses": vis.n_pred_poses,
                "n_gt_poses": vis.n_gt_poses,
                "n_associated": vis.n_associated,
                "n_dropped_pred": vis.n_dropped_pred,
                "n_dropped_gt": vis.n_dropped_gt,
            },
            "artifacts": record,
        },
        path,
    )


def write_trajectory_alignment_visualization(
    vis: TrajectoryAlignmentVisData,
    out_dir: Path,
    *,
    pointcloud_backend: Any,
) -> dict[str, Any]:
    """Write one trajectory alignment capture and return its manifest record."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    before_ply = out_dir / "trajectory_alignment_before.ply"
    after_ply = out_dir / "trajectory_alignment_after.ply"
    png = out_dir / "trajectory_alignment_projections.png"
    meta_json = out_dir / "trajectory_alignment.json"

    points, colors = _merged_overlay(vis.pred_before, vis.gt)
    pointcloud_backend.save_pointcloud(points, before_ply, colors=colors)
    points, colors = _merged_overlay(vis.pred_after, vis.gt)
    pointcloud_backend.save_pointcloud(points, after_ply, colors=colors)

    projection_vis = AlignmentVisData(
        scene_id=vis.scene_id,
        pred_before=vis.pred_before,
        pred_after=vis.pred_after,
        gt=vis.gt,
        alignment=vis.alignment,
        subsample_seed=0,
        max_points=int(vis.pred_before.shape[0]),
    )
    write_alignment_projections_png(projection_vis, png)

    record = {
        "scene_id": vis.scene_id,
        "before_ply": before_ply.name,
        "after_ply": after_ply.name,
        "projections_png": png.name,
        "metadata_json": meta_json.name,
        "pred_color": list(PRED_COLOR),
        "gt_color": list(GT_COLOR),
        "n_points_pred": int(vis.pred_before.shape[0]),
        "n_points_gt": int(vis.gt.shape[0]),
        "alignment": vis.alignment,
        "association": vis.association,
        "n_pred_poses": vis.n_pred_poses,
        "n_gt_poses": vis.n_gt_poses,
        "n_associated": vis.n_associated,
        "n_dropped_pred": vis.n_dropped_pred,
        "n_dropped_gt": vis.n_dropped_gt,
    }
    write_trajectory_alignment_json(vis, meta_json, record)
    return record


def write_trajectory_alignment_vis_outputs(
    captures: list[TrajectoryAlignmentVisData],
    run_dir: Path,
    *,
    pointcloud_backend: Any,
) -> list[dict[str, Any]]:
    """Write trajectory alignment debug artifacts into per-scene directories."""
    if not captures:
        return []
    run_dir = Path(run_dir)
    debug_dir = run_dir / "debug"
    records = []
    for vis in captures:
        scene_dir = _scene_debug_dir(run_dir, vis.scene_id)
        record = write_trajectory_alignment_visualization(
            vis, scene_dir, pointcloud_backend=pointcloud_backend
        )
        record["scene_debug_dir"] = _relative_debug_path(run_dir, scene_dir)
        for key in ("before_ply", "after_ply", "projections_png", "metadata_json"):
            record[key] = f"{record['scene_debug_dir']}/{record[key]}"
        records.append(record)
    dump_json(
        {"trajectory_alignment_visualizations": records},
        debug_dir / "trajectory_alignment_vis.json",
    )
    _update_debug_index(run_dir, "trajectory_alignment_visualizations", records)
    return records
