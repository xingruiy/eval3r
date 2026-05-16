"""Mesh and point-cloud loading/saving via trimesh."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from eval3r.utils.errors import EmptyGeometryError, NaNGeometryError
from eval3r.utils.typing import Colors, Faces, PathLike, Points


@dataclass
class MeshData:
    vertices: Points
    faces: Faces
    vertex_colors: Colors | None = None


@dataclass
class PointCloudData:
    points: Points
    colors: Colors | None = None


def _check_finite(arr: np.ndarray, name: str) -> None:
    if arr.size == 0:
        raise EmptyGeometryError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise NaNGeometryError(f"{name} contains NaN or Inf values")


def load_mesh(path: PathLike) -> MeshData:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mesh file not found: {p}")
    obj = trimesh.load(p, force="mesh", process=False)
    if not isinstance(obj, trimesh.Trimesh) or len(obj.faces) == 0:
        raise EmptyGeometryError(f"{p} did not load as a non-empty mesh")
    vertices = np.asarray(obj.vertices, dtype=np.float64)
    faces = np.asarray(obj.faces, dtype=np.int64)
    _check_finite(vertices, f"mesh vertices in {p}")
    colors: Colors | None = None
    if hasattr(obj.visual, "kind") and obj.visual.kind == "vertex":  # type: ignore[union-attr]
        colors = np.asarray(obj.visual.vertex_colors[:, :3], dtype=np.uint8)  # type: ignore[union-attr]
    return MeshData(vertices=vertices, faces=faces, vertex_colors=colors)


def load_point_cloud(path: PathLike) -> PointCloudData:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Point cloud file not found: {p}")
    obj = trimesh.load(p, process=False)
    if isinstance(obj, trimesh.points.PointCloud):
        points = np.asarray(obj.vertices, dtype=np.float64)
        colors_attr = getattr(obj, "colors", None)
        colors: Colors | None = (
            np.asarray(colors_attr[:, :3], dtype=np.uint8)
            if colors_attr is not None and len(colors_attr) == len(points)
            else None
        )
    elif isinstance(obj, trimesh.Trimesh):
        points = np.asarray(obj.vertices, dtype=np.float64)
        colors = None
        if hasattr(obj.visual, "kind") and obj.visual.kind == "vertex":  # type: ignore[union-attr]
            colors = np.asarray(obj.visual.vertex_colors[:, :3], dtype=np.uint8)  # type: ignore[union-attr]
    else:
        raise EmptyGeometryError(f"{p} did not load as a point cloud or mesh")
    _check_finite(points, f"point cloud in {p}")
    return PointCloudData(points=points, colors=colors)


def save_mesh_ply(
    path: PathLike,
    vertices: Points,
    faces: Faces,
    vertex_colors: Colors | None = None,
) -> Path:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    _check_finite(vertices, "vertices")
    if faces.size == 0:
        raise EmptyGeometryError("faces array is empty")
    mesh_kwargs: dict = {"vertices": vertices, "faces": faces, "process": False}
    if vertex_colors is not None:
        mesh_kwargs["vertex_colors"] = np.asarray(vertex_colors, dtype=np.uint8)
    mesh = trimesh.Trimesh(**mesh_kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out, file_type="ply")
    return out


def save_point_cloud_ply(
    path: PathLike,
    points: Points,
    colors: Colors | None = None,
) -> Path:
    points = np.asarray(points, dtype=np.float64)
    _check_finite(points, "points")
    pc_kwargs: dict = {"vertices": points}
    if colors is not None:
        pc_kwargs["colors"] = np.asarray(colors, dtype=np.uint8)
    pc = trimesh.points.PointCloud(**pc_kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pc.export(out, file_type="ply")
    return out
