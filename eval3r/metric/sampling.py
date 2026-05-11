"""Deterministic point sampling from meshes and point clouds."""

from __future__ import annotations

from typing import Literal

import numpy as np

from eval3r.io.geometry import MeshData, PointCloudData
from eval3r.utils.errors import EmptyGeometryError
from eval3r.utils.typing import Points

SampleMethod = Literal["area", "vertex", "uniform"]


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def sample_mesh_area_weighted(mesh: MeshData, n: int, *, seed: int) -> Points:
    if len(mesh.faces) == 0:
        raise EmptyGeometryError("Cannot area-sample a mesh with zero faces")
    rng = np.random.default_rng(seed)
    areas = _triangle_areas(mesh.vertices, mesh.faces)
    total = float(areas.sum())
    if total <= 0:
        raise EmptyGeometryError("Mesh has total area 0 — cannot sample")
    probs = areas / total
    face_idx = rng.choice(len(mesh.faces), size=n, p=probs)
    u = rng.random(n)
    v = rng.random(n)
    over = u + v > 1.0
    u[over] = 1.0 - u[over]
    v[over] = 1.0 - v[over]
    w = 1.0 - u - v
    tri = mesh.vertices[mesh.faces[face_idx]]
    return (
        tri[:, 0] * w[:, None]
        + tri[:, 1] * u[:, None]
        + tri[:, 2] * v[:, None]
    ).astype(np.float64)


def sample_points(
    geom: MeshData | PointCloudData | np.ndarray,
    n: int,
    *,
    method: SampleMethod = "area",
    seed: int = 42,
) -> Points:
    """Return ``n`` deterministic samples from ``geom``.

    For point clouds (or raw arrays), 'area' is treated as 'uniform'.
    When ``n`` is at least the number of available points/vertices, return all
    original points/vertices exactly once.
    """
    rng = np.random.default_rng(seed)

    if isinstance(geom, MeshData):
        if method == "area":
            return sample_mesh_area_weighted(geom, n, seed=seed)
        if method == "vertex":
            if n >= len(geom.vertices):
                return geom.vertices.astype(np.float64)
            idx = rng.choice(len(geom.vertices), size=n, replace=False)
            return geom.vertices[idx].astype(np.float64)
        if method == "uniform":
            return sample_mesh_area_weighted(geom, n, seed=seed)
        raise ValueError(f"Unknown sample method: {method!r}")

    points: np.ndarray
    if isinstance(geom, PointCloudData):
        points = geom.points
    else:
        points = np.asarray(geom, dtype=np.float64)
    if len(points) == 0:
        raise EmptyGeometryError("Cannot sample from an empty point set")
    if n >= len(points):
        return points.astype(np.float64)
    idx = rng.choice(len(points), size=n, replace=False)
    return points[idx].astype(np.float64)
