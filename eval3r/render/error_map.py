"""Per-vertex error map: write a vertex-coloured PLY of pred coloured by distance to gt."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from eval3r.io.geometry import MeshData, PointCloudData, save_mesh_ply, save_point_cloud_ply
from eval3r.utils.typing import PathLike


def _viridis_like(t: np.ndarray) -> np.ndarray:
    """Simple piecewise viridis-ish colormap, t in [0, 1] -> uint8 RGB."""
    t = np.clip(t, 0.0, 1.0)
    # rough viridis approximation: 4-stop linear interpolation
    stops = np.array(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=np.float64,
    )
    n = stops.shape[0] - 1
    pos = t * n
    lo = np.floor(pos).astype(int).clip(0, n - 1)
    hi = (lo + 1).clip(0, n)
    frac = (pos - lo)[:, None]
    rgb = stops[lo] * (1 - frac) + stops[hi] * frac
    return rgb.astype(np.uint8)


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
    colors = _viridis_like(t)

    out = Path(out_path)
    if isinstance(pred, MeshData):
        return save_mesh_ply(out, pred.vertices, pred.faces, vertex_colors=colors)
    return save_point_cloud_ply(out, pred.points, colors=colors)
