"""Tanks & Temples dataset adapter."""

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
        path="<root>/<scene_name>/<point_cloud_filename>",
        overrides=("point_cloud_filename",),
    ),
    LayoutEntry(
        path="<root>/<scene_name>/<image_subdir>/<color_format>",
        overrides=("image_subdir", "color_format"),
    ),
    LayoutEntry(
        path="<root>/<scene_name>/<pose_filename>",
        overrides=("pose_filename",),
    ),
]

_TRAINING_SCENES = [
    "Barn", "Caterpillar", "Church", "Courthouse",
    "Ignatius", "Meetingroom", "Truck",
]
_INTERMEDIATE_SCENES = [
    "Family", "Francis", "Horse", "Lighthouse",
    "M60", "Panther", "Playground", "Train",
]
_ADVANCED_SCENES = [
    "Auditorium", "Ballroom", "Courtroom", "Museum", "Palace", "Temple",
]

_SUBSETS: dict[str, list[str]] = {
    "training": _TRAINING_SCENES,
    "intermediate": _INTERMEDIATE_SCENES,
    "advanced": _ADVANCED_SCENES,
    "all": _TRAINING_SCENES + _INTERMEDIATE_SCENES + _ADVANCED_SCENES,
}


class TanksTemplesAdapter(DatasetAdapter):
    """Tanks & Temples dataset adapter.

    Provides point cloud GT (laser scans). Supports training, intermediate,
    and advanced subsets. Only training scenes ship ground truth.
    """

    name: ClassVar[str] = "tanks_temples"
    expected_layout: ClassVar[str] = "\n".join(e.render() for e in _LAYOUT)

    def __init__(
        self,
        root: PathLike,
        *,
        split: str | PathLike | None = None,
        point_cloud_filename: str = "{scene_id}.ply",
        image_subdir: str = "image",
        color_format: str = "{image_id:06d}.jpg",
        pose_filename: str = "{scene_id}_COLMAP_SfM.log",
        intrinsics_filename: str = "intrinsics.txt",
        intrinsics_fx: float | None = None,
        intrinsics_fy: float | None = None,
        intrinsics_cx: float | None = None,
        intrinsics_cy: float | None = None,
        subset: str | None = None,
        validate_on_init: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise MissingArtifactError(f"Tanks & Temples root not found: {self.root}")
        self._split = split
        self._point_cloud_filename = point_cloud_filename
        self._image_subdir = image_subdir
        self._color_format = color_format
        self._pose_filename = pose_filename
        self._intrinsics_filename = intrinsics_filename
        self._intrinsics_fx = intrinsics_fx
        self._intrinsics_fy = intrinsics_fy
        self._intrinsics_cx = intrinsics_cx
        self._intrinsics_cy = intrinsics_cy
        self._subset = subset

        self._scenes = self._load_split(split)

        if validate_on_init:
            if self._scenes:
                self._probe_layout(self._scenes[0])

    # ------------------------------------------------------------------
    # split handling
    # ------------------------------------------------------------------
    def _load_split(self, split: str | PathLike | None) -> list[str]:
        if split is not None:
            p = Path(split)
            if not p.exists():
                raise MissingArtifactError(
                    f"Tanks & Temples split file not found: {p}. "
                    f"Pass a path to a text file with one scene name per line."
                )
            return [
                line.strip()
                for line in p.read_text().splitlines()
                if line.strip()
            ]
        if self._subset is not None:
            subset_key = self._subset.lower()
            if subset_key not in _SUBSETS:
                raise MissingArtifactError(
                    f"Tanks & Temples: unknown subset {self._subset!r}. "
                    f"Available: {sorted(_SUBSETS)}"
                )
            return list(_SUBSETS[subset_key])
        # Enumerate directories that contain a point cloud file.
        return sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir()
            and (p / format_path(self._point_cloud_filename, scene_id=p.name)).exists()
        )

    def list_scenes(self, split: str | PathLike | None = None) -> list[str]:
        if split is None or split == self._split:
            return list(self._scenes)
        return self._load_split(split)

    # ------------------------------------------------------------------
    # path helpers
    # ------------------------------------------------------------------
    def _scene_dir(self, scene_id: str) -> Path:
        return self.root / scene_id

    def asset_path(self, scene_id: str, asset: Asset, **kw: object) -> Path:
        sd = self._scene_dir(scene_id)
        if asset is Asset.POINT_CLOUD:
            return sd / format_path(self._point_cloud_filename, scene_id=scene_id)
        if asset is Asset.COLOR:
            frame = int(kw["frame"])
            return sd / self._image_subdir / format_path(
                self._color_format, frame=frame, image_id=frame + 1
            )
        if asset is Asset.DEPTH:
            frame = int(kw["frame"])
            return sd / self._image_subdir / format_path(
                self._color_format, frame=frame, image_id=frame + 1
            )
        if asset is Asset.POSES:
            return sd / format_path(self._pose_filename, scene_id=scene_id)
        if asset in (Asset.INTRINSICS, Asset.INTRINSICS_DEPTH, Asset.INTRINSICS_COLOR):
            return sd / self._intrinsics_filename
        raise NotSupportedError(f"tanks_temples: asset_path({asset}) not implemented")

    def _probe_layout(self, scene_id: str) -> None:
        pc = self.asset_path(scene_id, Asset.POINT_CLOUD)
        if not pc.exists():
            raise_missing(
                dataset="Tanks & Temples",
                asset="point_cloud",
                tried=pc,
                overrides=("point_cloud_filename",),
                layout=_LAYOUT,
            )

    # ------------------------------------------------------------------
    # loaders
    # ------------------------------------------------------------------
    def load_point_cloud(self, scene_id: str) -> PointCloudData:
        path = self.asset_path(scene_id, Asset.POINT_CLOUD)
        if not path.exists():
            raise_missing(
                dataset="Tanks & Temples",
                asset="point_cloud",
                tried=path,
                overrides=("point_cloud_filename",),
                layout=_LAYOUT,
            )
        return load_point_cloud(path)

    def load_color(self, scene_id: str, frame: int) -> np.ndarray:
        path = self.asset_path(scene_id, Asset.COLOR, frame=frame)
        if not path.exists():
            raise_missing(
                dataset="Tanks & Temples",
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
                dataset="Tanks & Temples",
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
        return self.load_intrinsics_depth(scene_id)

    def load_intrinsics_depth(self, scene_id: str) -> np.ndarray:
        if self._intrinsics_fx is not None:
            fx = self._intrinsics_fx
            fy = self._intrinsics_fy if self._intrinsics_fy is not None else fx
            cx = self._intrinsics_cx if self._intrinsics_cx is not None else 0.0
            cy = self._intrinsics_cy if self._intrinsics_cy is not None else 0.0
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        path = self.asset_path(scene_id, Asset.INTRINSICS_DEPTH)
        if path.exists():
            K4 = np.loadtxt(path)
            if K4.shape == (4, 4):
                return K4[:3, :3].astype(np.float64)
            if K4.shape == (3, 3):
                return K4.astype(np.float64)

        # Fallback for common Tanks & Temples releases that omit intrinsics.txt.
        color_path = self.asset_path(scene_id, Asset.COLOR, frame=0)
        if color_path.exists():
            from eval3r.utils.optional import optional_import

            imageio = optional_import("imageio.v3", extra="render")
            image = np.asarray(imageio.imread(color_path))
            h, w = image.shape[:2]
            fx = float(max(w, h))
            fy = fx
            cx = w / 2.0
            cy = h / 2.0
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        raise MissingArtifactError(
            f"Tanks & Temples: intrinsics not found at {path}. "
            f"Provide intrinsics.txt or pass intrinsics_fx=... (optionally fy/cx/cy)."
        )

    def load_intrinsics_color(self, scene_id: str) -> np.ndarray:
        return self.load_intrinsics_depth(scene_id)

    def load_poses(self, scene_id: str) -> Trajectory:
        pose_file = self.asset_path(scene_id, Asset.POSES)
        if not pose_file.exists():
            raise_missing(
                dataset="Tanks & Temples",
                asset="poses",
                tried=pose_file,
                overrides=("pose_filename",),
                layout=_LAYOUT,
            )
        lines = [ln.strip() for ln in pose_file.read_text().splitlines() if ln.strip()]
        if not lines:
            raise MissingArtifactError(
                f"Tanks & Temples: pose log is empty at {pose_file}"
            )
        if len(lines) % 5 != 0:
            raise MissingArtifactError(
                f"Tanks & Temples: pose log {pose_file} expected blocks of 5 lines "
                f"(header + 4x4 matrix), got {len(lines)} lines"
            )
        poses_list: list[np.ndarray] = []
        timestamps: list[float] = []
        for i in range(0, len(lines), 5):
            hdr = lines[i].split()
            if len(hdr) < 2:
                raise MissingArtifactError(
                    f"Tanks & Temples: malformed pose header at line {i + 1} in {pose_file}"
                )
            frame_idx = float(hdr[0])
            mat = np.array(
                [[float(x) for x in lines[i + j].split()] for j in range(1, 5)],
                dtype=np.float64,
            )
            if mat.shape != (4, 4):
                raise MissingArtifactError(
                    f"Tanks & Temples: pose matrix at block {i // 5} in {pose_file} "
                    f"expected 4x4, got {mat.shape}"
                )
            timestamps.append(frame_idx)
            poses_list.append(mat)
        poses = np.stack(poses_list, axis=0)
        timestamps_arr = np.asarray(timestamps, dtype=np.float64)
        return Trajectory(poses=poses, timestamps=timestamps_arr, convention="T_wc")
