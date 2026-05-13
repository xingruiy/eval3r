"""PredictionWriter — context manager for saving 3D reconstruction predictions."""

from __future__ import annotations

import json
import os
import shutil
import warnings
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from eval3r.io.geometry import save_mesh_ply, save_point_cloud_ply
from eval3r.io.trajectory import save_trajectory_kitti, save_trajectory_tum
from eval3r.manifest._hash import relpath, sha256_file
from eval3r.manifest.manifest import (
    MANIFEST_FILENAME,
    Artifact,
    CameraSection,
    CoordinateSystem,
    GeometrySection,
    Manifest,
    PoseConvention,
    TrajectorySection,
    Unit,
)
from eval3r.utils.errors import EvalAssumptionWarning
from eval3r.utils.typing import Colors, Faces, PathLike, Points, Poses


class PredictionWriter:
    """Context manager that lays out a prediction directory and writes the manifest on exit.

    Example::

        with PredictionWriter("outputs/scene", scene_id="s", dataset="scannet",
                              method="m", unit="m", pose_convention="T_wc") as pred:
            pred.save_point_cloud(points)
            pred.save_mesh(verts, faces)
    """

    def __init__(
        self,
        out_dir: PathLike,
        *,
        scene_id: str,
        dataset: str,
        method: str,
        unit: str | Unit = Unit.UNSPECIFIED,
        coordinate_system: str | CoordinateSystem = CoordinateSystem.UNSPECIFIED,
        pose_convention: str | PoseConvention = PoseConvention.UNSPECIFIED,
        overwrite: bool = False,
    ) -> None:
        self.out_dir = Path(out_dir)
        self._overwrite = overwrite
        if self.out_dir.exists():
            manifest_path = self.out_dir / MANIFEST_FILENAME
            if overwrite:
                shutil.rmtree(self.out_dir)
            elif manifest_path.exists():
                raise FileExistsError(
                    f"Prediction already exists at {self.out_dir}. "
                    f"Pass overwrite=True to replace it."
                )
            elif any(self.out_dir.iterdir()):
                raise FileExistsError(
                    f"{self.out_dir} exists and is non-empty. "
                    "Pass overwrite=True to replace it."
                )
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._unit = Unit(unit)
        self._coord = CoordinateSystem(coordinate_system)
        self._pose_convention = PoseConvention(pose_convention)
        self._warn_unspecified()

        self._manifest = Manifest(
            scene_id=scene_id,
            dataset=dataset,
            method=method,
            unit=self._unit,
            coordinate_system=self._coord,
            pose_convention=self._pose_convention,
        )
        self._closed = False

    # -- context manager --------------------------------------------------
    def __enter__(self) -> "PredictionWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()

    # -- save methods -----------------------------------------------------
    def save_point_cloud(
        self,
        points: Points,
        colors: Colors | None = None,
        *,
        filename: str = "pred_points.ply",
    ) -> Path:
        out = self.out_dir / "geometry" / filename
        save_point_cloud_ply(out, points, colors)
        self._manifest.geometry.point_cloud = self._artifact(out, "ply", count=len(points))
        return out

    def save_mesh(
        self,
        vertices: Points,
        faces: Faces,
        vertex_colors: Colors | None = None,
        *,
        filename: str = "pred_mesh.ply",
    ) -> Path:
        out = self.out_dir / "geometry" / filename
        save_mesh_ply(out, vertices, faces, vertex_colors)
        self._manifest.geometry.mesh = self._artifact(
            out,
            "ply",
            count=len(vertices),
            extra={"faces": int(len(faces))},
        )
        return out

    def save_poses(
        self,
        poses: Poses,
        timestamps: np.ndarray | None = None,
        *,
        convention: str | PoseConvention | None = None,
    ) -> tuple[Path, Path]:
        if convention is not None:
            self._pose_convention = PoseConvention(convention)
            self._manifest.pose_convention = self._pose_convention
        if self._pose_convention is PoseConvention.UNSPECIFIED:
            warnings.warn(
                "save_poses called without a pose_convention. "
                "Specify convention='T_wc' or convention='T_cw'.",
                EvalAssumptionWarning,
                stacklevel=2,
            )
        tum_out = self.out_dir / "trajectory" / "pred_trajectory_tum.txt"
        kitti_out = self.out_dir / "trajectory" / "pred_trajectory_kitti.txt"
        save_trajectory_tum(tum_out, poses, timestamps)
        save_trajectory_kitti(kitti_out, poses)
        self._manifest.trajectory.tum = self._artifact(tum_out, "tum", count=len(poses))
        self._manifest.trajectory.kitti = self._artifact(kitti_out, "kitti", count=len(poses))
        return tum_out, kitti_out

    def save_intrinsics(
        self,
        K: np.ndarray,
        image_size: tuple[int, int],
        *,
        filename: str = "intrinsics.json",
    ) -> Path:
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        out = self.out_dir / "cameras" / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"K": K.tolist(), "image_size": [int(image_size[0]), int(image_size[1])]},
                indent=2,
            )
        )
        self._manifest.cameras.intrinsics = self._artifact(out, "json")
        return out

    def save_metadata(self, data: dict[str, Any]) -> None:
        self._manifest.metadata.update(data)

    def save_aux(self, name: str, value: Any) -> Path:
        out_dir = self.out_dir / "aux"
        out_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(value, np.ndarray):
            out = out_dir / f"{name}.npy"
            np.save(out, value)
        elif isinstance(value, (dict, list)):
            out = out_dir / f"{name}.json"
            out.write_text(json.dumps(value, indent=2, default=_json_default))
        else:
            raise TypeError(
                f"save_aux: unsupported value type {type(value).__name__} for '{name}'"
            )
        return out

    # -- internals --------------------------------------------------------
    def close(self) -> None:
        if self._closed:
            return
        manifest_path = self.out_dir / MANIFEST_FILENAME
        tmp = manifest_path.with_suffix(".json.tmp")
        tmp.write_text(self._manifest.model_dump_json(indent=2))
        os.replace(tmp, manifest_path)
        self._closed = True

    def _artifact(
        self,
        path: Path,
        fmt: str,
        *,
        count: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Artifact:
        return Artifact(
            path=relpath(path, self.out_dir),
            format=fmt,
            sha256=sha256_file(path),
            count=count,
            extra=extra or {},
        )

    def _warn_unspecified(self) -> None:
        missing = []
        if self._unit is Unit.UNSPECIFIED:
            missing.append("unit")
        if self._coord is CoordinateSystem.UNSPECIFIED:
            missing.append("coordinate_system")
        if self._pose_convention is PoseConvention.UNSPECIFIED:
            missing.append("pose_convention")
        if missing:
            warnings.warn(
                "PredictionWriter created without explicit "
                + ", ".join(missing)
                + ". This is allowed but downstream evaluation will be ambiguous. "
                "Pass these as keyword arguments to silence this warning.",
                EvalAssumptionWarning,
                stacklevel=3,
            )


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")
