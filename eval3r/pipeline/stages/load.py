"""Load stage: read a prediction or ground-truth geometry file.

A :class:`LoadedGeometry` wraps either a point cloud (an ``(N, 3)`` array) or a mesh
handle from the mesh backend. Loading delegates entirely to the registered
``pointcloud`` / ``mesh`` backends (task 004); no file parsing lives here. Meshes are
never reduced to their raw vertices as a metric surface at load time — surface
sampling is an explicit later stage (``.agent/metrics.md`` mesh rules).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from eval3r.core.errors import InvalidGeometryError
from eval3r.core.registry import MeshBackend, PointCloudBackend

GeometryKind = Literal["mesh", "pointcloud"]


@dataclass
class LoadedGeometry:
    """A loaded prediction/GT geometry: exactly one of ``points`` or ``mesh`` is set."""

    kind: GeometryKind
    path: Path
    points: np.ndarray | None = None
    mesh: Any | None = None

    def alignment_points(self) -> np.ndarray:
        """Points used to estimate an alignment transform.

        Point clouds use their points; meshes use their vertices. Umeyama alignment
        needs 1:1 correspondence, so this is only meaningful when pred and gt expose
        equal, corresponding point counts (see the align stage).
        """
        if self.kind == "pointcloud":
            assert self.points is not None
            return self.points
        assert self.mesh is not None
        return np.asarray(self.mesh.vertices, dtype=np.float64)

    def transformed(self, matrix: np.ndarray) -> LoadedGeometry:
        """Return a copy with a 4x4 similarity/rigid transform applied.

        Point clouds are transformed as coordinates; meshes are transformed on a
        copy so the original stays untouched. An identity transform returns ``self``.
        """
        if _is_identity(matrix):
            return self
        if self.kind == "pointcloud":
            assert self.points is not None
            return LoadedGeometry(
                kind="pointcloud",
                path=self.path,
                points=apply_transform_points(self.points, matrix),
            )
        assert self.mesh is not None
        moved = self.mesh.copy()
        moved.apply_transform(np.asarray(matrix, dtype=np.float64))
        return LoadedGeometry(kind="mesh", path=self.path, mesh=moved)


def _is_identity(matrix: np.ndarray) -> bool:
    return bool(np.allclose(np.asarray(matrix, dtype=np.float64), np.eye(4)))


def apply_transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 (scale·rotation | translation) transform to ``(N, 3)`` points."""
    pts = np.asarray(points, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    return pts @ m[:3, :3].T + m[:3, 3]


def load_geometry(
    path: Path,
    kind: GeometryKind,
    *,
    mesh_backend: MeshBackend,
    pointcloud_backend: PointCloudBackend,
) -> LoadedGeometry:
    """Load ``path`` as a mesh or point cloud via the appropriate backend."""
    path = Path(path)
    if not path.is_file():
        raise InvalidGeometryError(
            f"geometry file does not exist: {path} (expected a {kind} file). "
            f"Check the path passed for this input."
        )
    if kind == "mesh":
        mesh = mesh_backend.load_mesh(path)
        return LoadedGeometry(kind="mesh", path=path, mesh=mesh)
    if kind == "pointcloud":
        points = pointcloud_backend.load_pointcloud(path)
        return LoadedGeometry(kind="pointcloud", path=path, points=np.asarray(points, float))
    raise InvalidGeometryError(
        f"unsupported geometry input kind '{kind}' for {path}; expected 'mesh' or 'pointcloud'."
    )
