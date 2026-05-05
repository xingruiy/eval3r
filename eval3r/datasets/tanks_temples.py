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
        path="<root>/<scene_name>/images/<color_format>",
        overrides=("image_subdir", "color_format"),
    ),
    LayoutEntry(
        path="<root>/<scene_name>/<pose_subdir>/<pose_format>",
        overrides=("pose_subdir", "pose_format"),
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
        point_cloud_filename: str = "point_cloud.ply",
        image_subdir: str = "images",
        color_format: str = "{frame:04d}.jpg",
        pose_subdir: str = "poses",
        pose_format: str = "{frame:04d}.txt",
        intrinsics_filename: str = "intrinsics.txt",
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
        self._pose_subdir = pose_subdir
        self._pose_format = pose_format
        self._intrinsics_filename = intrinsics_filename
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
            and (p / self._point_cloud_filename).exists()
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
            return sd / self._pose_subdir / format_path(
                self._pose_format, frame=int(kw["frame"])  # type: ignore[arg-type]
            )
        if asset is Asset.INTRINSICS:
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
        path = self.asset_path(scene_id, Asset.INTRINSICS)
        if path.exists():
            K4 = np.loadtxt(path)
            if K4.shape == (4, 4):
                return K4[:3, :3].astype(np.float64)
            if K4.shape == (3, 3):
                return K4.astype(np.float64)
        raise MissingArtifactError(
            f"Tanks & Temples: intrinsics not found at {path}. "
            f"Pass intrinsics_fx=... to the adapter if intrinsics are known."
        )

    def load_poses(self, scene_id: str) -> Trajectory:
        pose_dir = self._scene_dir(scene_id) / self._pose_subdir
        if not pose_dir.is_dir():
            raise_missing(
                dataset="Tanks & Temples",
                asset="poses",
                tried=pose_dir,
                overrides=("pose_subdir", "pose_format"),
                layout=_LAYOUT,
            )
        files = sorted(
            (p for p in pose_dir.iterdir() if p.suffix == ".txt"),
            key=lambda p: int(p.stem),
        )
        if not files:
            raise_missing(
                dataset="Tanks & Temples",
                asset="poses",
                tried=pose_dir / format_path(self._pose_format, frame=0),
                overrides=("pose_subdir", "pose_format"),
                layout=_LAYOUT,
            )
        poses = np.stack([np.loadtxt(p) for p in files], axis=0)
        if poses.shape[1:] != (4, 4):
            raise MissingArtifactError(
                f"Tanks & Temples: pose files in {pose_dir} expected 4x4 matrices, "
                f"got {poses.shape[1:]}"
            )
        timestamps = np.array([float(p.stem) for p in files], dtype=np.float64)
        return Trajectory(poses=poses, timestamps=timestamps, convention="T_cw")
