"""Area-weighted (importance) sampler for triangle meshes."""
from __future__ import annotations

import numpy as np

from eval3r.io.geometry import MeshData, PointCloudData
from eval3r.sampling.base import PointSampler
from eval3r.sampling.uniform import UniformSampler
from eval3r.utils.errors import EmptyGeometryError
from eval3r.utils.typing import Points


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


class ImportanceSampler(PointSampler):
    """Area-weighted barycentric sampler for triangle meshes.

    For :class:`~eval3r.io.geometry.PointCloudData` and raw arrays where
    no face/area data is available, delegates to
    :class:`~eval3r.sampling.uniform.UniformSampler`.
    """

    def __init__(self) -> None:
        self._uniform = UniformSampler()

    def sample(
        self,
        geom: MeshData | PointCloudData | np.ndarray,
        n: int,
        *,
        seed: int = 42,
    ) -> Points:
        if isinstance(geom, MeshData):
            return self._sample_mesh(geom, n, seed=seed)
        return self._uniform.sample(geom, n, seed=seed)

    def _sample_mesh(self, mesh: MeshData, n: int, *, seed: int) -> Points:
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


__all__ = ["ImportanceSampler"]
