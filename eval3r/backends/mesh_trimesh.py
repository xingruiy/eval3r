"""trimesh mesh backend (default).

Loads meshes, samples their surface deterministically for a given seed, and
exports. Raw vertices are never used as a metric surface here — surface sampling is
always explicit (``.agent/backends.md`` mesh rules). No mesh repair is performed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from eval3r.core.errors import InvalidGeometryError
from eval3r.core.registry import BackendInfo


class TrimeshMeshBackend:
    """Mesh loading and deterministic surface sampling via trimesh."""

    name = "trimesh"

    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            kind="mesh",
            name=self.name,
            library="trimesh",
            version=trimesh.__version__ or "unknown",
            approximate=False,
        )

    def load_mesh(self, path: Path) -> trimesh.Trimesh:
        mesh = trimesh.load(Path(path), process=False, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise InvalidGeometryError(
                f"file did not load as a single triangle mesh: {path} "
                f"(loaded {type(mesh).__name__})."
            )
        if mesh.faces is None or len(mesh.faces) == 0:
            raise InvalidGeometryError(f"mesh has no faces to sample: {path}.")
        return mesh

    def sample_surface(
        self,
        mesh: Any,
        n_points: int,
        seed: int,
        return_normals: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Deterministically sample ``n_points`` on the mesh surface for ``seed``.

        Determinism is provided by trimesh's ``seed`` argument, so the same mesh and
        seed always yield the same points.
        """
        if n_points <= 0:
            raise InvalidGeometryError(f"n_points must be positive; got {n_points}.")
        points, face_idx = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
        points = np.asarray(points, dtype=np.float64)
        if not return_normals:
            return points
        normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float64)
        return points, normals

    def export_mesh(self, mesh: Any, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(Path(path))
