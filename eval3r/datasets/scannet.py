"""ScanNet v2 dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np

from eval3r.datasets.base import Asset, DatasetAdapter
from eval3r.datasets.layout import LayoutEntry, format_path, raise_missing
from eval3r.io.geometry import MeshData, PointCloudData, load_mesh
from eval3r.io.trajectory import Trajectory
from eval3r.utils.errors import MissingArtifactError, NotSupportedError
from eval3r.utils.typing import PathLike

_LAYOUT: list[LayoutEntry] = [
    LayoutEntry(
        path="<root>/<scene_subdir>/<scene_id>_vh_clean_2.ply",
        overrides=("scene_subdir", "mesh_filename"),
    ),
    LayoutEntry(
        path="<root>/<scene_subdir>/color/{frame}.jpg",
        overrides=("color_subdir", "color_format"),
    ),
    LayoutEntry(
        path="<root>/<scene_subdir>/depth/{frame}.png",
        overrides=("depth_subdir", "depth_format"),
    ),
    LayoutEntry(
        path="<root>/<scene_subdir>/pose/{frame}.txt",
        overrides=("pose_subdir", "pose_format"),
    ),
    LayoutEntry(
        path="<root>/<scene_subdir>/intrinsic/intrinsic_depth.txt",
        overrides=("intrinsics_subdir", "intrinsics_depth_filename"),
    ),
    LayoutEntry(
        path="<root>/<scene_subdir>/intrinsic/intrinsic_color.txt",
        overrides=("intrinsics_subdir", "intrinsics_color_filename"),
    ),
]


class ScanNetAdapter(DatasetAdapter):
    """ScanNet v2 layout adapter.

    Defaults match ``/mnt/dataset/ScanNet/scans/<scene_id>/`` with non-zero-padded
    frame indices (``0.jpg``, ``1.jpg``, ``1000.jpg``). Override any subfolder or
    filename pattern via the constructor kwargs.
    """

    name: ClassVar[str] = "scannet"
    expected_layout: ClassVar[str] = "\n".join(e.render() for e in _LAYOUT)

    def __init__(
        self,
        root: PathLike,
        *,
        split: str | PathLike | None = None,
        scene_subdir: str = "scans/{scene_id}",
        color_subdir: str = "color",
        color_format: str = "{frame}.jpg",
        depth_subdir: str = "depth",
        depth_format: str = "{frame}.png",
        depth_scale: float = 1000.0,
        pose_subdir: str = "pose",
        pose_format: str = "{frame}.txt",
        intrinsics_subdir: str = "intrinsic",
        intrinsics_depth_filename: str = "intrinsic_depth.txt",
        intrinsics_color_filename: str = "intrinsic_color.txt",
        mesh_filename: str = "{scene_id}_vh_clean_2.ply",
        validate_on_init: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise MissingArtifactError(f"ScanNet root not found: {self.root}")
        self._split = split
        self._scene_subdir = scene_subdir
        self._color_subdir = color_subdir
        self._color_format = color_format
        self._depth_subdir = depth_subdir
        self._depth_format = depth_format
        self._depth_scale = depth_scale
        self._pose_subdir = pose_subdir
        self._pose_format = pose_format
        self._intrinsics_subdir = intrinsics_subdir
        self._intrinsics_depth_filename = intrinsics_depth_filename
        self._intrinsics_color_filename = intrinsics_color_filename
        self._mesh_filename = mesh_filename

        self._scenes = self._load_split(split)

        if validate_on_init:
            # Touch the first scene's mesh + intrinsics to surface problems early.
            if self._scenes:
                self._probe_layout(self._scenes[0])

    # ------------------------------------------------------------------
    # split handling
    # ------------------------------------------------------------------
    def _load_split(self, split: str | PathLike | None) -> list[str]:
        """Resolve a split selector to a list of scene ids.

        - ``None`` → enumerate ``<root>/<scene_subdir>``-shaped directories.
        - anything else → treat as a path to a text file with one id per line.

        eval3r does not bundle dataset splits — pass a path to your own list
        (e.g. ``/data/scannet/splits/scannetv2_test.txt``) to pin a benchmark.
        """
        if split is None:
            scans = self.root / self._scene_subdir.split("/", 1)[0]
            if not scans.is_dir():
                raise MissingArtifactError(
                    f"ScanNet auto-discovery: {scans} does not exist. "
                    f"Pass split=<path-to-list> or fix the dataset root."
                )
            return sorted(p.name for p in scans.iterdir() if p.is_dir())
        p = Path(split)
        if not p.exists():
            raise MissingArtifactError(
                f"ScanNet split file not found: {p}. "
                f"eval3r does not bundle dataset splits; pass a path to a "
                f"text file with one scene id per line."
            )
        text = p.read_text()
        return [line.strip() for line in text.splitlines() if line.strip()]

    def list_scenes(self, split: str | PathLike | None = None) -> list[str]:
        if split is None or split == self._split:
            return list(self._scenes)
        return self._load_split(split)

    # ------------------------------------------------------------------
    # path helpers
    # ------------------------------------------------------------------
    def _scene_dir(self, scene_id: str) -> Path:
        return self.root / format_path(self._scene_subdir, scene_id=scene_id)

    def asset_path(self, scene_id: str, asset: Asset, **kw: object) -> Path:
        sd = self._scene_dir(scene_id)
        if asset is Asset.MESH:
            return sd / format_path(self._mesh_filename, scene_id=scene_id)
        if asset is Asset.POINT_CLOUD:
            return sd / format_path(self._mesh_filename, scene_id=scene_id)
        if asset is Asset.COLOR:
            return sd / self._color_subdir / format_path(
                self._color_format, frame=int(kw["frame"])  # type: ignore[arg-type]
            )
        if asset is Asset.DEPTH:
            return sd / self._depth_subdir / format_path(
                self._depth_format, frame=int(kw["frame"])  # type: ignore[arg-type]
            )
        if asset is Asset.POSES:
            return sd / self._pose_subdir
        if asset in (Asset.INTRINSICS, Asset.INTRINSICS_DEPTH):
            return sd / self._intrinsics_subdir / self._intrinsics_depth_filename
        if asset is Asset.INTRINSICS_COLOR:
            return sd / self._intrinsics_subdir / self._intrinsics_color_filename
        raise NotSupportedError(f"scannet: asset_path({asset}) not implemented")

    def _probe_layout(self, scene_id: str) -> None:
        # Just check the mesh exists; detailed checks happen in validate().
        mesh = self.asset_path(scene_id, Asset.MESH)
        if not mesh.exists():
            raise_missing(
                dataset="ScanNet",
                asset="mesh",
                tried=mesh,
                overrides=("scene_subdir", "mesh_filename"),
                layout=_LAYOUT,
            )

    # ------------------------------------------------------------------
    # loaders
    # ------------------------------------------------------------------
    def load_mesh(self, scene_id: str) -> MeshData:
        path = self.asset_path(scene_id, Asset.MESH)
        if not path.exists():
            raise_missing(
                dataset="ScanNet",
                asset="mesh",
                tried=path,
                overrides=("scene_subdir", "mesh_filename"),
                layout=_LAYOUT,
            )
        return load_mesh(path)

    def load_point_cloud(self, scene_id: str) -> PointCloudData:
        # ScanNet ships meshes; expose vertices as a point cloud for convenience.
        mesh = self.load_mesh(scene_id)
        return PointCloudData(points=mesh.vertices, colors=mesh.vertex_colors)

    def load_depth(self, scene_id: str, frame: int) -> np.ndarray:
        path = self.asset_path(scene_id, Asset.DEPTH, frame=frame)
        if not path.exists():
            raise_missing(
                dataset="ScanNet",
                asset="depth",
                tried=path,
                overrides=("depth_subdir", "depth_format"),
                layout=_LAYOUT,
            )
        from eval3r.utils.optional import optional_import

        imageio = optional_import("imageio.v3", extra="render")
        depth_mm = imageio.imread(path)
        return (depth_mm.astype(np.float32) / float(self._depth_scale))

    def load_color(self, scene_id: str, frame: int) -> np.ndarray:
        path = self.asset_path(scene_id, Asset.COLOR, frame=frame)
        if not path.exists():
            raise_missing(
                dataset="ScanNet",
                asset="color",
                tried=path,
                overrides=("color_subdir", "color_format"),
                layout=_LAYOUT,
            )
        from eval3r.utils.optional import optional_import

        imageio = optional_import("imageio.v3", extra="render")
        return np.asarray(imageio.imread(path))

    def _load_intrinsics_file(
        self,
        scene_id: str,
        asset: Asset,
        *,
        override: str,
    ) -> np.ndarray:
        path = self.asset_path(scene_id, asset)
        if not path.exists():
            raise_missing(
                dataset="ScanNet",
                asset="intrinsics",
                tried=path,
                overrides=("intrinsics_subdir", override),
                layout=_LAYOUT,
            )
        K4 = np.loadtxt(path)
        if K4.shape == (4, 4):
            return K4[:3, :3].astype(np.float64)
        if K4.shape == (3, 3):
            return K4.astype(np.float64)
        raise MissingArtifactError(
            f"ScanNet: intrinsics at {path} expected 3x3 or 4x4 matrix, got shape {K4.shape}"
        )

    def load_intrinsics(self, scene_id: str) -> np.ndarray:
        return self.load_intrinsics_depth(scene_id)

    def load_intrinsics_depth(self, scene_id: str) -> np.ndarray:
        return self._load_intrinsics_file(
            scene_id,
            Asset.INTRINSICS_DEPTH,
            override="intrinsics_depth_filename",
        )

    def load_intrinsics_color(self, scene_id: str) -> np.ndarray:
        return self._load_intrinsics_file(
            scene_id,
            Asset.INTRINSICS_COLOR,
            override="intrinsics_color_filename",
        )

    def load_poses(self, scene_id: str) -> Trajectory:
        pose_dir = self.asset_path(scene_id, Asset.POSES)
        if not pose_dir.is_dir():
            raise_missing(
                dataset="ScanNet",
                asset="poses",
                tried=pose_dir,
                overrides=("pose_subdir", "pose_format"),
                layout=_LAYOUT,
            )
        # Sort by integer frame index extracted from the filename.
        files = sorted(
            (p for p in pose_dir.iterdir() if p.suffix == ".txt"),
            key=lambda p: int(p.stem),
        )
        if not files:
            raise_missing(
                dataset="ScanNet",
                asset="poses",
                tried=pose_dir / format_path(self._pose_format, frame=0),
                overrides=("pose_subdir", "pose_format"),
                layout=_LAYOUT,
            )
        poses = np.stack([np.loadtxt(p) for p in files], axis=0)
        if poses.shape[1:] != (4, 4):
            raise MissingArtifactError(
                f"ScanNet: pose files in {pose_dir} expected 4x4 matrices, got {poses.shape[1:]}"
            )
        timestamps = np.array([float(p.stem) for p in files], dtype=np.float64)
        # ScanNet poses are camera-to-world (T_wc).
        return Trajectory(poses=poses, timestamps=timestamps, convention="T_wc")
