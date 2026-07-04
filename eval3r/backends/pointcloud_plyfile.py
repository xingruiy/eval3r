"""plyfile point-cloud IO backend (default).

Loads/saves point coordinates as an ``(N, 3)`` float array. Coordinates are
validated finite before metric computation; colors are handled only as optional
debug metadata (``.agent/backends.md`` point-cloud rules).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from eval3r.core.errors import InvalidGeometryError
from eval3r.core.registry import BackendInfo


class PlyfilePointCloudBackend:
    """Point-cloud loading/saving via plyfile (PLY format)."""

    name = "plyfile"

    def backend_info(self) -> BackendInfo:
        # plyfile does not expose __version__ reliably; report via importlib.metadata.
        try:
            from importlib.metadata import version

            ver = version("plyfile")
        except Exception:  # pragma: no cover - metadata always present in this env
            ver = "unknown"
        return BackendInfo(
            kind="pointcloud",
            name=self.name,
            library="plyfile",
            version=ver,
            approximate=False,
        )

    def load_pointcloud(self, path: Path) -> np.ndarray:
        path = Path(path)
        if not path.is_file():
            raise InvalidGeometryError(f"point-cloud file does not exist: {path}.")
        ply = PlyData.read(str(path))
        if "vertex" not in ply:
            raise InvalidGeometryError(f"PLY file has no 'vertex' element: {path}.")
        vertex = ply["vertex"].data
        for axis in ("x", "y", "z"):
            if axis not in vertex.dtype.names:
                raise InvalidGeometryError(
                    f"PLY vertex element is missing coordinate '{axis}': {path}."
                )
        points = np.stack(
            [vertex["x"], vertex["y"], vertex["z"]], axis=-1
        ).astype(np.float64)
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
            raise InvalidGeometryError(
                f"points to save must be (N, 3); got shape {arr.shape}."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        if colors is not None:
            col = np.asarray(colors)
            if col.shape != arr.shape:
                raise InvalidGeometryError(
                    f"colors shape {col.shape} must match points shape {arr.shape}."
                )
            dtype += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

        structured = np.empty(arr.shape[0], dtype=dtype)
        structured["x"] = arr[:, 0]
        structured["y"] = arr[:, 1]
        structured["z"] = arr[:, 2]
        if colors is not None:
            col = np.asarray(colors)
            structured["red"] = col[:, 0]
            structured["green"] = col[:, 1]
            structured["blue"] = col[:, 2]

        element = PlyElement.describe(structured, "vertex")
        PlyData([element], text=False).write(str(path))
