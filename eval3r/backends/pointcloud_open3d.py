"""Open3D point-cloud IO backend.

Used where a protocol explicitly asks for ``pointcloud: open3d`` (e.g. the ScanNet
geometry protocols). Loads/saves point coordinates as an ``(N, 3)`` float array with
the same validation contract as the plyfile backend: coordinates are validated finite
before metric computation; colors are optional debug metadata only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from eval3r.core.errors import InvalidGeometryError
from eval3r.core.registry import BackendInfo


class Open3dPointCloudBackend:
    """Point-cloud loading/saving via Open3D."""

    name = "open3d"

    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            kind="pointcloud",
            name=self.name,
            library="open3d",
            version=o3d.__version__,
            approximate=False,
        )

    def load_pointcloud(self, path: Path) -> np.ndarray:
        path = Path(path)
        if not path.is_file():
            raise InvalidGeometryError(f"point-cloud file does not exist: {path}.")
        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise InvalidGeometryError(
                f"Open3D read no (N, 3) points from {path}; is it a point cloud?"
            )
        if points.shape[0] == 0:
            raise InvalidGeometryError(f"point cloud is empty: {path}.")
        if not np.isfinite(points).all():
            raise InvalidGeometryError(f"point cloud contains non-finite values: {path}.")
        return points

    def save_pointcloud(
        self,
        points: np.ndarray,
        path: Path,
        colors: np.ndarray | None = None,
    ) -> None:
        arr = np.asarray(points)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise InvalidGeometryError(f"points to save must be (N, 3); got shape {arr.shape}.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(arr.astype(np.float64))
        if colors is not None:
            col = np.asarray(colors)
            if col.shape != arr.shape:
                raise InvalidGeometryError(
                    f"colors shape {col.shape} must match points shape {arr.shape}."
                )
            pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(str(path), pcd)
