"""DTU dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np

from eval3r.datasets.base import Asset, DatasetAdapter
from eval3r.datasets.layout import LayoutEntry, format_path, raise_missing
from eval3r.io.geometry import PointCloudData, load_point_cloud
from eval3r.io.trajectory import Trajectory
from eval3r.utils.errors import MissingArtifactError, NotSupportedError
from eval3r.utils.typing import PathLike

_LAYOUT: list[LayoutEntry] = [
    LayoutEntry(
        path="<root>/<scan_subdir>/<scan_id>/<point_cloud_filename>",
        overrides=("scan_subdir", "scan_format", "point_cloud_filename"),
    ),
    LayoutEntry(
        path="<root>/<scan_subdir>/<scan_id>/image/{frame:06d}.png",
        overrides=("image_subdir", "color_format"),
    ),
    LayoutEntry(
        path="<root>/<cameras_subdir>/<scan_id>/<pose_filename>",
        overrides=("cameras_subdir", "pose_filename"),
    ),
]

# Standard DTU evaluation subset (scan IDs).
_DTU_EVAL_SCANS = [
    1, 4, 9, 15, 24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122,
]


def _scan_dir_name(scan_id: str | int) -> str:
    """DTU scan directories are named ``scan<id>``."""
    return f"scan{int(scan_id)}"


def _decompose_projection(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a 3x4 projection matrix ``P = K [R | t]`` into K (3x3), R, t.

    Returns ``(K, T_cw)`` where ``T_cw`` is a 4x4 world-to-camera matrix.
    """
    M = P[:3, :3]
    K, R = np.linalg.qr(np.linalg.inv(M))  # M = K @ R in RQ decomposition
    K = np.linalg.inv(K)
    R = np.linalg.inv(R)
    # Ensure positive diagonal of K.
    D = np.diag(np.sign(np.diag(K)))
    K = K @ D
    R = D @ R
    # Ensure det(R) = 1.
    if np.linalg.det(R) < 0:
        R = -R
        K = -K
    K = K / K[2, 2]
    t = np.linalg.inv(K) @ P[:3, 3]
    T_cw = np.eye(4, dtype=np.float64)
    T_cw[:3, :3] = R
    T_cw[:3, 3] = t
    return K, T_cw


class DTUAdapter(DatasetAdapter):
    """DTU MVS dataset adapter.

    Provides point cloud GT (structured-light scans). The default scene
    list is the standard 19-scan evaluation subset.
    """

    name: ClassVar[str] = "dtu"
    expected_layout: ClassVar[str] = "\n".join(e.render() for e in _LAYOUT)

    def __init__(
        self,
        root: PathLike,
        *,
        split: str | PathLike | None = None,
        scan_subdir: str = "scans",
        scan_format: str = "scan{scan_id}",
        point_cloud_filename: str = "points.ply",
        image_subdir: str = "image",
        color_format: str = "{frame:06d}.png",
        cameras_subdir: str = "Cameras",
        pose_filename: str = "camera_poses.txt",
        intrinsics_width: int = 1600,
        intrinsics_height: int = 1200,
        eval_scans: list[int] | None = None,
        validate_on_init: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise MissingArtifactError(f"DTU root not found: {self.root}")
        self._split = split
        self._scan_subdir = scan_subdir
        self._scan_format = scan_format
        self._point_cloud_filename = point_cloud_filename
        self._image_subdir = image_subdir
        self._color_format = color_format
        self._cameras_subdir = cameras_subdir
        self._pose_filename = pose_filename
        self._intrinsics_width = intrinsics_width
        self._intrinsics_height = intrinsics_height
        self._eval_scans = eval_scans or _DTU_EVAL_SCANS

        self._scenes = self._load_split(split)

        if validate_on_init:
            if self._scenes:
                self._probe_layout(self._scenes[0])

    # ------------------------------------------------------------------
    # split handling
    # ------------------------------------------------------------------
    def _load_split(self, split: str | PathLike | None) -> list[str]:
        if split is None:
            # Return standard eval subset as string IDs.
            return [str(s) for s in self._eval_scans]
        p = Path(split)
        if not p.exists():
            raise MissingArtifactError(
                f"DTU split file not found: {p}. "
                f"Pass a path to a text file with one scan id per line."
            )
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]

    def list_scenes(self, split: str | PathLike | None = None) -> list[str]:
        if split is None or split == self._split:
            return list(self._scenes)
        return self._load_split(split)

    # ------------------------------------------------------------------
    # path helpers
    # ------------------------------------------------------------------
    def _scan_dir(self, scene_id: str) -> Path:
        name = format_path(self._scan_format, scan_id=int(scene_id))
        return self.root / self._scan_subdir / name

    def asset_path(self, scene_id: str, asset: Asset, **kw: object) -> Path:
        sd = self._scan_dir(scene_id)
        if asset is Asset.POINT_CLOUD:
            return sd / self._point_cloud_filename
        if asset is Asset.COLOR:
            return sd / self._image_subdir / format_path(
                self._color_format, frame=int(kw["frame"])  # type: ignore[arg-type]
            )
        if asset is Asset.DEPTH:
            return sd / self._image_subdir / format_path(
                self._color_format, frame=int(kw["frame"])  # type: ignore[arg-type]
            )
        if asset is Asset.POSES:
            return self.root / self._cameras_subdir / _scan_dir_name(scene_id) / self._pose_filename
        if asset is Asset.INTRINSICS:
            return self.root / self._cameras_subdir / _scan_dir_name(scene_id) / "intrinsics.txt"
        raise NotSupportedError(f"dtu: asset_path({asset}) not implemented")

    def _probe_layout(self, scene_id: str) -> None:
        pc = self.asset_path(scene_id, Asset.POINT_CLOUD)
        if not pc.exists():
            raise_missing(
                dataset="DTU",
                asset="point_cloud",
                tried=pc,
                overrides=("scan_subdir", "scan_format", "point_cloud_filename"),
                layout=_LAYOUT,
            )

    # ------------------------------------------------------------------
    # loaders
    # ------------------------------------------------------------------
    def load_point_cloud(self, scene_id: str) -> PointCloudData:
        path = self.asset_path(scene_id, Asset.POINT_CLOUD)
        if not path.exists():
            raise_missing(
                dataset="DTU",
                asset="point_cloud",
                tried=path,
                overrides=("scan_subdir", "scan_format", "point_cloud_filename"),
                layout=_LAYOUT,
            )
        return load_point_cloud(path)

    def load_color(self, scene_id: str, frame: int) -> np.ndarray:
        path = self.asset_path(scene_id, Asset.COLOR, frame=frame)
        if not path.exists():
            raise_missing(
                dataset="DTU",
                asset="color",
                tried=path,
                overrides=("image_subdir", "color_format"),
                layout=_LAYOUT,
            )
        from eval3r.utils.optional import optional_import

        imageio = optional_import("imageio.v3", extra="render")
        return np.asarray(imageio.imread(path))

    def load_depth(self, scene_id: str, frame: int) -> np.ndarray:
        path = self.asset_path(scene_id, Asset.DEPTH, frame=frame)
        if not path.exists():
            raise_missing(
                dataset="DTU",
                asset="depth",
                tried=path,
                overrides=("image_subdir", "color_format"),
                layout=_LAYOUT,
            )
        from eval3r.utils.optional import optional_import

        imageio = optional_import("imageio.v3", extra="render")
        depth_raw = imageio.imread(path)
        return depth_raw.astype(np.float32)

    def load_intrinsics(self, scene_id: str) -> np.ndarray:
        # DTU uses a known intrinsics matrix (standard DTU camera).
        # Focal length varies slightly per scan but defaults to ~2892 pixels.
        path = self.asset_path(scene_id, Asset.INTRINSICS)
        if path.exists():
            K4 = np.loadtxt(path)
            if K4.shape == (4, 4):
                return K4[:3, :3].astype(np.float64)
            if K4.shape == (3, 3):
                return K4.astype(np.float64)
        # Fallback: standard DTU intrinsics.
        fx = 2892.33
        fy = 2883.18
        cx = self._intrinsics_width / 2.0
        cy = self._intrinsics_height / 2.0
        return np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
        )

    def load_poses(self, scene_id: str) -> Trajectory:
        path = self.asset_path(scene_id, Asset.POSES)
        if not path.exists():
            raise_missing(
                dataset="DTU",
                asset="poses",
                tried=path,
                overrides=("cameras_subdir", "pose_filename"),
                layout=_LAYOUT,
            )
        # DTU camera files contain one 3x4 projection matrix per line.
        arr = np.loadtxt(path)
        if arr.size == 0:
            raise MissingArtifactError(f"DTU pose file is empty: {path}")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != 12:
            raise MissingArtifactError(
                f"DTU poses: expected 12 cols per row (3x4 projection), got {arr.shape[1]}"
            )
        poses_4x4 = np.tile(np.eye(4), (arr.shape[0], 1, 1))
        for i in range(arr.shape[0]):
            P = arr[i].reshape(3, 4)
            _, T_cw = _decompose_projection(P)
            poses_4x4[i] = T_cw
        return Trajectory(poses=poses_4x4, timestamps=None, convention="T_cw")
