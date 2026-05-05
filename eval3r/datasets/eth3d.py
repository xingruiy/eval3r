"""ETH3D dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np

from eval3r.datasets.base import Asset, DatasetAdapter
from eval3r.datasets.layout import LayoutEntry, format_path, raise_missing
from eval3r.io.geometry import MeshData, PointCloudData, load_mesh, load_point_cloud
from eval3r.io.trajectory import Trajectory, _quat_to_rot
from eval3r.utils.errors import MissingArtifactError, NotSupportedError
from eval3r.utils.typing import PathLike

_LAYOUT: list[LayoutEntry] = [
    LayoutEntry(
        path="<root>/<track>/<scene_name>/<mesh_filename>",
        overrides=("track", "mesh_filename"),
    ),
    LayoutEntry(
        path="<root>/<track>/<scene_name>/images/<image_subdir>/<color_format>",
        overrides=("image_subdir", "color_format"),
    ),
    LayoutEntry(
        path="<root>/<track>/<scene_name>/<calibration_subdir>/calibration.txt",
        overrides=("track", "calibration_subdir"),
    ),
]

_DSLR_TRAINING_SCENES = [
    "courtyard", "delivery_area", "electro", "facade", "kicker",
    "meadow", "office", "pipes", "playground", "relief",
    "relief_2", "terrace", "terrains",
]
_RIG_TRAINING_SCENES = [
    "courtyard", "delivery_area", "electro", "facade", "kicker",
    "meadow", "office", "playground", "relief", "relief_2", "terrace", "terrains",
]
_TRAINING_SCENES: dict[str, list[str]] = {
    "dslr": _DSLR_TRAINING_SCENES,
    "rig": _RIG_TRAINING_SCENES,
}


def _parse_colmap_images_txt(path: Path) -> Trajectory:
    """Parse a COLMAP ``images.txt`` file into a Trajectory.

    COLMAP format: odd lines are data, even lines are empty comments.
    Each data line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    """
    lines = path.read_text().splitlines()
    data_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    n = len(data_lines)
    if n == 0:
        raise MissingArtifactError(f"COLMAP images.txt is empty: {path}")
    timestamps = np.arange(n, dtype=np.float64)
    poses = np.tile(np.eye(4), (n, 1, 1))
    for i, line in enumerate(data_lines):
        parts = line.split()
        if len(parts) < 9:
            continue
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        poses[i, :3, :3] = _quat_to_rot(qx, qy, qz, qw)
        poses[i, :3, 3] = [tx, ty, tz]
    return Trajectory(poses=poses, timestamps=timestamps, convention="T_cw")


def _parse_colmap_cameras_txt(path: Path) -> np.ndarray | None:
    """Parse a COLMAP ``cameras.txt`` file, returning the first 3x3 intrinsics.

    Format: CAMERA_ID, MODEL, WIDTH, HEIGHT, params...
    """
    lines = path.read_text().splitlines()
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 8:
            continue
        w = float(parts[2])
        h = float(parts[3])
        fx = float(parts[4])
        fy = float(parts[5])
        cx = float(parts[6])
        cy = float(parts[7])
        # Could also be SIMPLE_RADIAL (fx, cx, cy, k1) etc.
        # PINHOLE: fx, fy, cx, cy
        # SIMPLE_RADIAL: f, cx, cy, k1
        model = parts[1]
        if model == "SIMPLE_RADIAL":
            return np.array(
                [[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64
            )
        return np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
        )
    return None


class ETH3DAdapter(DatasetAdapter):
    """ETH3D dataset adapter.

    Supports high-res (``track="dslr"``) and low-res (``track="rig"``) tracks.
    Provides mesh and point cloud GT for training scenes.
    """

    name: ClassVar[str] = "eth3d"
    expected_layout: ClassVar[str] = "\n".join(e.render() for e in _LAYOUT)

    def __init__(
        self,
        root: PathLike,
        *,
        split: str | PathLike | None = None,
        track: str = "dslr",
        mesh_filename: str = "scan.ply",
        point_cloud_filename: str = "scan_points.ply",
        image_subdir: str = "images",
        color_format: str = "{frame:04d}.jpg",
        calibration_subdir: str = "dslr_calibration_jpg",
        calibration_filename: str = "calibration.txt",
        validate_on_init: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise MissingArtifactError(f"ETH3D root not found: {self.root}")
        self._split = split
        self._track = track
        self._mesh_filename = mesh_filename
        self._point_cloud_filename = point_cloud_filename
        self._image_subdir = image_subdir
        self._color_format = color_format
        self._calibration_subdir = calibration_subdir
        self._calibration_filename = calibration_filename
        self._training_scenes = _TRAINING_SCENES.get(
            track, _DSLR_TRAINING_SCENES
        )

        self._scenes = self._load_split(split)

        if validate_on_init:
            if self._scenes:
                self._probe_layout(self._scenes[0])

    # ------------------------------------------------------------------
    # split handling
    # ------------------------------------------------------------------
    def _load_split(self, split: str | PathLike | None) -> list[str]:
        if split is None:
            # Enumerate directories under <root>/<track>/ that contain scan.ply.
            track_dir = self.root / self._track
            if not track_dir.is_dir():
                raise MissingArtifactError(
                    f"ETH3D track directory not found: {track_dir}. "
                    f"Pass track='dslr' or track='rig'."
                )
            return sorted(
                p.name
                for p in track_dir.iterdir()
                if p.is_dir()
                and ((p / self._mesh_filename).exists() or (p / self._point_cloud_filename).exists())
            )
        p = Path(split)
        if not p.exists():
            raise MissingArtifactError(
                f"ETH3D split file not found: {p}. "
                f"Pass a path to a text file with one scene name per line."
            )
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]

    def list_scenes(self, split: str | PathLike | None = None) -> list[str]:
        if split is None or split == self._split:
            return list(self._scenes)
        return self._load_split(split)

    # ------------------------------------------------------------------
    # path helpers
    # ------------------------------------------------------------------
    def _scene_dir(self, scene_id: str) -> Path:
        return self.root / self._track / scene_id

    def asset_path(self, scene_id: str, asset: Asset, **kw: object) -> Path:
        sd = self._scene_dir(scene_id)
        if asset is Asset.MESH:
            return sd / self._mesh_filename
        if asset is Asset.POINT_CLOUD:
            pc = sd / self._point_cloud_filename
            if pc.exists():
                return pc
            return sd / self._mesh_filename
        if asset is Asset.COLOR:
            return sd / self._image_subdir / format_path(
                self._color_format, frame=int(kw["frame"])  # type: ignore[arg-type]
            )
        if asset is Asset.DEPTH:
            return sd / self._image_subdir / format_path(
                self._color_format, frame=int(kw["frame"])  # type: ignore[arg-type]
            )
        if asset is Asset.POSES:
            return sd / self._calibration_subdir / "images.txt"
        if asset is Asset.INTRINSICS:
            return sd / self._calibration_subdir / self._calibration_filename
        raise NotSupportedError(f"eth3d: asset_path({asset}) not implemented")

    def _probe_layout(self, scene_id: str) -> None:
        mesh_path = self.asset_path(scene_id, Asset.MESH)
        pc_path = self._scene_dir(scene_id) / self._point_cloud_filename
        if not mesh_path.exists() and not pc_path.exists():
            raise_missing(
                dataset="ETH3D",
                asset="mesh or point_cloud",
                tried=mesh_path,
                overrides=("track", "mesh_filename", "point_cloud_filename"),
                layout=_LAYOUT,
            )

    # ------------------------------------------------------------------
    # loaders
    # ------------------------------------------------------------------
    def load_mesh(self, scene_id: str) -> MeshData:
        path = self.asset_path(scene_id, Asset.MESH)
        if not path.exists():
            raise_missing(
                dataset="ETH3D",
                asset="mesh",
                tried=path,
                overrides=("track", "mesh_filename"),
                layout=_LAYOUT,
            )
        return load_mesh(path)

    def load_point_cloud(self, scene_id: str) -> PointCloudData:
        pc_path = self._scene_dir(scene_id) / self._point_cloud_filename
        if pc_path.exists():
            return load_point_cloud(pc_path)
        # Fall back to mesh vertices.
        mesh = self.load_mesh(scene_id)
        return PointCloudData(points=mesh.vertices, colors=mesh.vertex_colors)

    def load_color(self, scene_id: str, frame: int) -> np.ndarray:
        path = self.asset_path(scene_id, Asset.COLOR, frame=frame)
        if not path.exists():
            raise_missing(
                dataset="ETH3D",
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
                dataset="ETH3D",
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
        calibration_dir = self._scene_dir(scene_id) / self._calibration_subdir
        # Try COLMAP cameras.txt first.
        cameras_txt = calibration_dir / "cameras.txt"
        if cameras_txt.exists():
            K = _parse_colmap_cameras_txt(cameras_txt)
            if K is not None:
                return K
        # Fall back to calibration.txt.
        calib_path = calibration_dir / self._calibration_filename
        if calib_path.exists():
            K4 = np.loadtxt(calib_path)
            if K4.shape == (4, 4):
                return K4[:3, :3].astype(np.float64)
            if K4.shape == (3, 3):
                return K4.astype(np.float64)
        raise MissingArtifactError(
            f"ETH3D: intrinsics not found in {calibration_dir}. "
            f"Pass calibration_subdir=... or calibration_filename=..."
        )

    def load_poses(self, scene_id: str) -> Trajectory:
        images_txt = self.asset_path(scene_id, Asset.POSES)
        if not images_txt.exists():
            raise_missing(
                dataset="ETH3D",
                asset="poses",
                tried=images_txt,
                overrides=("calibration_subdir",),
                layout=_LAYOUT,
            )
        return _parse_colmap_images_txt(images_txt)
