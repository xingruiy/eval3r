"""Per-vertex error map: write a vertex-coloured PLY of pred coloured by distance to gt."""

from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import numpy as np
from scipy.spatial import cKDTree

from eval3r.io.geometry import MeshData, PointCloudData, save_mesh_ply, save_point_cloud_ply
from eval3r.utils.typing import PathLike


def _viridis(t: np.ndarray) -> np.ndarray:
    """Map t in [0, 1] to viridis uint8 RGB."""
    rgba = cm.viridis(np.clip(t, 0.0, 1.0))  # type: ignore[attr-defined]
    return (rgba[:, :3] * 255).astype(np.uint8)


def render_error_ply(
    pred: MeshData | PointCloudData,
    gt: MeshData | PointCloudData,
    *,
    out_path: PathLike,
    threshold: float = 0.05,
) -> Path:
    """Write a vertex-coloured PLY of `pred` shaded by distance to `gt`.

    Distances are clamped to [0, threshold] for the colour ramp; numeric
    distances are not rescaled.
    """
    pred_pts = pred.vertices if isinstance(pred, MeshData) else pred.points
    gt_pts = gt.vertices if isinstance(gt, MeshData) else gt.points
    tree = cKDTree(gt_pts)
    d, _ = tree.query(pred_pts, k=1)
    t = d / max(threshold, 1e-12)
    colors = _viridis(t)

    out = Path(out_path)
    if isinstance(pred, MeshData):
        return save_mesh_ply(out, pred.vertices, pred.faces, vertex_colors=colors)
    return save_point_cloud_ply(out, pred.points, colors=colors)
