"""Tanks & Temples dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np

from eval3r.datasets.base import Asset, DatasetAdapter
from eval3r.datasets.layout import LayoutEntry, format_path, raise_missing
from eval3r.io.crop import CropVolume, load_crop_volume_json
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
    LayoutEntry(
        path="<root>/<scene_name>/<alignment_filename>",
        overrides=("alignment_filename",),
    ),
    LayoutEntry(
        path="<root>/<scene_name>/<crop_filename>",
        overrides=("crop_filename",),
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


# Official Tanks & Temples per-scene F-score thresholds (metres). Only the
# training subset has published τ values — intermediate/advanced are scored
# on the leaderboard, not via the open-source toolkit.
_SCENES_TAU_DICT: dict[str, float] = {
    "Barn": 0.01,
    "Caterpillar": 0.005,
    "Church": 0.025,
    "Courthouse": 0.025,
    "Ignatius": 0.003,
    "Meetingroom": 0.01,
    "Truck": 0.005,
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
        color_format: str = "{frame:06d}.jpg",
        pose_filename: str = "{scene_id}_COLMAP_SfM.log",
        alignment_filename: str = "{scene_id}_trans.txt",
        crop_filename: str = "{scene_id}.json",
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
        self._alignment_filename = alignment_filename
        self._crop_filename = crop_filename
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
            return sd / self._image_subdir / format_path(
                self._color_format, frame=int(kw["frame"]) + 1  # type: ignore[arg-type]
            )
        if asset is Asset.DEPTH:
            return sd / self._image_subdir / format_path(
                self._color_format, frame=int(kw["frame"]) + 1  # type: ignore[arg-type]
            )
        if asset is Asset.POSES:
            return sd / format_path(self._pose_filename, scene_id=scene_id)
        if asset in (Asset.INTRINSICS, Asset.INTRINSICS_DEPTH, Asset.INTRINSICS_COLOR):
            # Tanks & Temples does not ship canonical intrinsics; these are
            # synthesized from image size in `load_intrinsics_depth`.
            return sd / self._image_subdir
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

    def _load_alignment(self, scene_id: str) -> np.ndarray | None:
        # Empty filename ⇒ caller has opted out of SfM→laser alignment.
        if not self._alignment_filename:
            return None
        path = self._scene_dir(scene_id) / format_path(
            self._alignment_filename, scene_id=scene_id
        )
        if not path.exists():
            raise_missing(
                dataset="Tanks & Temples",
                asset="alignment (pass alignment_filename='' to skip)",
                tried=path,
                overrides=("alignment_filename",),
                layout=_LAYOUT,
            )
        mat = np.loadtxt(path, dtype=np.float64)
        if mat.shape != (4, 4):
            raise MissingArtifactError(
                f"Tanks & Temples: alignment matrix at {path} expected 4x4, got {mat.shape}"
            )
        return mat

    # ------------------------------------------------------------------
    # loaders
    # ------------------------------------------------------------------
    def load_thresholds(self, scene_id: str) -> tuple[float, ...]:
        if scene_id not in _SCENES_TAU_DICT:
            raise NotSupportedError(
                f"tanks_temples: no published τ for scene {scene_id!r} "
                f"(only training scenes have one). "
                f"Available: {sorted(_SCENES_TAU_DICT)}"
            )
        return (_SCENES_TAU_DICT[scene_id],)

    def load_crop_volume(self, scene_id: str) -> CropVolume:
        # Empty filename ⇒ adapter advertises "no crop volume" so the
        # benchmark layer treats us like adapters without crop support.
        if not self._crop_filename:
            raise NotSupportedError(
                "tanks_temples: crop volume disabled (crop_filename=\"\")"
            )
        path = self._scene_dir(scene_id) / format_path(
            self._crop_filename, scene_id=scene_id
        )
        if not path.exists():
            raise_missing(
                dataset="Tanks & Temples",
                asset="crop volume (pass crop_filename='' to skip)",
                tried=path,
                overrides=("crop_filename",),
                layout=_LAYOUT,
            )
        return load_crop_volume_json(path)

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
        img_path = self.asset_path(scene_id, Asset.COLOR, frame=0)
        if not img_path.exists():
            raise_missing(
                dataset="Tanks & Temples",
                asset="color",
                tried=img_path,
                overrides=("image_subdir", "color_format"),
                layout=_LAYOUT,
            )
        from eval3r.utils.optional import optional_import

        imageio = optional_import("imageio.v3", extra="render")
        image = np.asarray(imageio.imread(img_path))
        if image.ndim < 2:
            raise MissingArtifactError(
                f"Tanks & Temples: expected an image-like array at {img_path}, got shape {image.shape}"
            )
        height, width = int(image.shape[0]), int(image.shape[1])
        focal = 0.7 * float(width)
        cx = float(width) / 2.0
        cy = float(height) / 2.0
        return np.array(
            [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
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
        # The .log poses live in the COLMAP/SfM world; the GT .ply lives in
        # the laser-scan world. T&T ships a per-scene SfM→laser transform
        # (`{scene}_trans.txt`) used by the official toolkit's
        # `mesh.transform(gt_trans)` step. Apply it here so cameras and the
        # GT cloud share a frame.
        T_align = self._load_alignment(scene_id)
        if T_align is not None:
            poses = T_align @ poses
        return Trajectory(poses=poses, timestamps=timestamps_arr, convention="T_wc")
