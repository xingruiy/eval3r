"""Open3D ``SelectionPolygonVolume`` crop files (extruded-polygon prisms).

Used by the Tanks & Temples evaluation toolkit to clip the prediction
mesh/cloud to the laser-scanned region before metrics. Pure NumPy — no
Open3D dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval3r.utils.errors import MissingArtifactError
from eval3r.utils.typing import PathLike

_AXIS_LABEL_TO_INDEX: dict[str, int] = {"X": 0, "Y": 1, "Z": 2}


@dataclass
class CropVolume:
    """Extruded polygon prism in 3D.

    The polygon is given in 2D (the two axes that are NOT the orthogonal
    axis); the prism extends from ``axis_min`` to ``axis_max`` along
    ``orthogonal_axis``. Boundary is inclusive (matches Open3D).
    """

    polygon_2d: np.ndarray
    axis_min: float
    axis_max: float
    orthogonal_axis: int  # 0=X, 1=Y, 2=Z
    source: str = ""

    def filter_points(self, points: np.ndarray) -> tuple[np.ndarray, int, int]:
        inside = crop_points_inside(self, points)
        pts = np.asarray(points, dtype=np.float64)
        n_kept = int(inside.sum())
        return pts[inside], n_kept, len(pts)


def load_crop_volume_json(path: PathLike) -> CropVolume:
    """Parse an Open3D ``SelectionPolygonVolume`` JSON file."""
    p = Path(path)
    if not p.exists():
        raise MissingArtifactError(f"Crop volume JSON not found: {p}")
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise MissingArtifactError(f"Crop volume {p} is not valid JSON: {e}") from e

    cls_name = data.get("class_name")
    if cls_name != "SelectionPolygonVolume":
        raise MissingArtifactError(
            f"Crop volume {p}: expected class_name 'SelectionPolygonVolume', got {cls_name!r}"
        )

    axis_label = data.get("orthogonal_axis")
    if axis_label not in _AXIS_LABEL_TO_INDEX:
        raise MissingArtifactError(
            f"Crop volume {p}: orthogonal_axis must be 'X', 'Y', or 'Z', got {axis_label!r}"
        )
    axis = _AXIS_LABEL_TO_INDEX[axis_label]

    if "axis_min" not in data or "axis_max" not in data:
        raise MissingArtifactError(
            f"Crop volume {p}: missing axis_min/axis_max"
        )
    axis_min = float(data["axis_min"])
    axis_max = float(data["axis_max"])
    if not axis_max >= axis_min:
        raise MissingArtifactError(
            f"Crop volume {p}: axis_max ({axis_max}) must be >= axis_min ({axis_min})"
        )

    polygon = data.get("bounding_polygon")
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise MissingArtifactError(
            f"Crop volume {p}: bounding_polygon must list >= 3 vertices, got {polygon!r}"
        )
    poly3d = np.asarray(polygon, dtype=np.float64)
    if poly3d.ndim != 2 or poly3d.shape[1] != 3:
        raise MissingArtifactError(
            f"Crop volume {p}: bounding_polygon must be shape (N, 3), got {poly3d.shape}"
        )
    keep_axes = [i for i in (0, 1, 2) if i != axis]
    polygon_2d = poly3d[:, keep_axes].copy()

    return CropVolume(
        polygon_2d=polygon_2d,
        axis_min=axis_min,
        axis_max=axis_max,
        orthogonal_axis=axis,
        source=str(p),
    )


def _point_in_polygon_2d(points_2d: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorised ray-casting point-in-polygon (handles non-convex).

    Returns a boolean mask of shape ``(N,)``. Points exactly on an edge are
    counted as inside (matches Open3D's behaviour).
    """
    n_points = points_2d.shape[0]
    n_edges = polygon.shape[0]
    if n_points == 0 or n_edges < 3:
        return np.zeros(n_points, dtype=bool)

    px = points_2d[:, 0][:, None]
    py = points_2d[:, 1][:, None]
    x0 = polygon[:, 0][None, :]
    y0 = polygon[:, 1][None, :]
    x1 = np.roll(polygon[:, 0], -1)[None, :]
    y1 = np.roll(polygon[:, 1], -1)[None, :]

    # Horizontal ray to +x. An edge contributes a crossing iff its y-range
    # straddles py and the x-coordinate of the intersection lies > px.
    cond_y = (y0 > py) != (y1 > py)
    # Avoid division-by-zero: cond_y already excludes y0 == y1 edges.
    denom = np.where(cond_y, y1 - y0, 1.0)
    x_at_py = x0 + (py - y0) * (x1 - x0) / denom
    cross = cond_y & (x_at_py > px)
    inside = (np.sum(cross, axis=1) % 2) == 1

    # Inclusive boundary: any point lying exactly on an edge is inside.
    edge_dx = x1 - x0
    edge_dy = y1 - y0
    rx = px - x0
    ry = py - y0
    cross_z = edge_dx * ry - edge_dy * rx
    on_line = np.isclose(cross_z, 0.0)
    seg_len_sq = edge_dx * edge_dx + edge_dy * edge_dy
    t = np.where(seg_len_sq > 0, (rx * edge_dx + ry * edge_dy) / np.where(seg_len_sq > 0, seg_len_sq, 1.0), 0.0)
    on_segment = on_line & (t >= -1e-12) & (t <= 1.0 + 1e-12) & (seg_len_sq > 0)
    on_boundary = np.any(on_segment, axis=1)

    return inside | on_boundary


def crop_points_inside(volume: CropVolume, points: np.ndarray) -> np.ndarray:
    """Boolean mask: which rows of ``points`` lie inside ``volume``.

    ``points`` has shape ``(N, 3)``. Boundary inclusive on both the polygon
    edges and the [axis_min, axis_max] band.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    axis = volume.orthogonal_axis
    along = pts[:, axis]
    band = (along >= volume.axis_min) & (along <= volume.axis_max)

    keep_axes = [i for i in (0, 1, 2) if i != axis]
    pts_2d = pts[:, keep_axes]
    inside = _point_in_polygon_2d(pts_2d, volume.polygon_2d)

    return band & inside


__all__ = ["CropVolume", "load_crop_volume_json", "crop_points_inside"]
