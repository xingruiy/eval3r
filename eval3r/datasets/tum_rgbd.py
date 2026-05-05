"""TUM RGB-D dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np

from eval3r.datasets.base import Asset, DatasetAdapter
from eval3r.datasets.layout import LayoutEntry, format_path, raise_missing
from eval3r.io.trajectory import Trajectory, load_trajectory_tum
from eval3r.utils.errors import MissingArtifactError, NotSupportedError
from eval3r.utils.typing import PathLike

_LAYOUT: list[LayoutEntry] = [
    LayoutEntry(
        path="<root>/<sequence_name>/rgb/{timestamp}.png",
        overrides=("rgb_subdir", "color_format"),
    ),
    LayoutEntry(
        path="<root>/<sequence_name>/depth/{timestamp}.png",
        overrides=("depth_subdir", "depth_format"),
    ),
    LayoutEntry(
        path="<root>/<sequence_name>/groundtruth.txt",
        overrides=("pose_filename",),
    ),
]

# Known TUM intrinsics (approximate, resolution-dependent).
_TUM_INTRINSICS: dict[str, tuple[float, float, float, float]] = {
    "fr1": (517.3, 517.3, 318.6, 255.3),
    "fr2": (520.9, 520.9, 325.1, 249.7),
    "fr3": (535.4, 535.4, 320.1, 247.6),
}


def _guess_intrinsics(seq_name: str) -> np.ndarray | None:
    """Return a 3x3 intrinsics matrix for known TUM sequence prefixes."""
    for prefix, (fx, fy, cx, cy) in _TUM_INTRINSICS.items():
        if seq_name.startswith(prefix):
            return np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
            )
    return None


class TumRGBDAdapter(DatasetAdapter):
    """TUM RGB-D dataset adapter.

    Supports depth, color, poses, and intrinsics. Does **not** provide mesh
    or point cloud GT — use this for depth-metric evaluation or trajectory
    evaluation.
    """

    name: ClassVar[str] = "tum_rgbd"
    expected_layout: ClassVar[str] = "\n".join(e.render() for e in _LAYOUT)

    def __init__(
        self,
        root: PathLike,
        *,
        split: str | PathLike | None = None,
        rgb_subdir: str = "rgb",
        color_format: str = "{ts:.6f}.png",
        depth_subdir: str = "depth",
        depth_format: str = "{ts:.6f}.png",
        depth_scale: float = 5000.0,
        pose_filename: str = "groundtruth.txt",
        associations_file: str | None = None,
        intrinsics_fx: float | None = None,
        intrinsics_fy: float | None = None,
        intrinsics_cx: float | None = None,
        intrinsics_cy: float | None = None,
        validate_on_init: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise MissingArtifactError(f"TUM RGB-D root not found: {self.root}")
        self._split = split
        self._rgb_subdir = rgb_subdir
        self._color_format = color_format
        self._depth_subdir = depth_subdir
        self._depth_format = depth_format
        self._depth_scale = depth_scale
        self._pose_filename = pose_filename
        self._associations_file = associations_file
        self._intrinsics_fx = intrinsics_fx
        self._intrinsics_fy = intrinsics_fy
        self._intrinsics_cx = intrinsics_cx
        self._intrinsics_cy = intrinsics_cy

        self._scenes = self._load_split(split)

        if validate_on_init:
            if self._scenes:
                self._probe_layout(self._scenes[0])

    # ------------------------------------------------------------------
    # split handling
    # ------------------------------------------------------------------
    def _load_split(self, split: str | PathLike | None) -> list[str]:
        if split is None:
            return sorted(
                p.name
                for p in self.root.iterdir()
                if p.is_dir()
                and (
                    (p / self._rgb_subdir).is_dir()
                    or (p / self._pose_filename).exists()
                )
            )
        p = Path(split)
        if not p.exists():
            raise MissingArtifactError(
                f"TUM RGB-D split file not found: {p}. "
                f"Pass a path to a text file with one sequence name per line."
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
        return self.root / scene_id

    def asset_path(self, scene_id: str, asset: Asset, **kw: object) -> Path:
        sd = self._scene_dir(scene_id)
        if asset is Asset.COLOR:
            return sd / self._rgb_subdir / format_path(
                self._color_format, ts=float(kw["frame"])  # type: ignore[arg-type]
            )
        if asset is Asset.DEPTH:
            return sd / self._depth_subdir / format_path(
                self._depth_format, ts=float(kw["frame"])  # type: ignore[arg-type]
            )
        if asset is Asset.POSES:
            return sd / self._pose_filename
        if asset is Asset.INTRINSICS:
            return sd / "intrinsics.txt"
        raise NotSupportedError(f"tum_rgbd: asset_path({asset}) not implemented")

    def _probe_layout(self, scene_id: str) -> None:
        pose_file = self.asset_path(scene_id, Asset.POSES)
        rgb_dir = self._scene_dir(scene_id) / self._rgb_subdir
        if not pose_file.exists() and not rgb_dir.is_dir():
            raise_missing(
                dataset="TUM RGB-D",
                asset="poses or rgb dir",
                tried=rgb_dir,
                overrides=("rgb_subdir", "pose_filename"),
                layout=_LAYOUT,
            )

    # ------------------------------------------------------------------
    # loaders
    # ------------------------------------------------------------------
    def load_depth(self, scene_id: str, frame: int) -> np.ndarray:
        path = self.asset_path(scene_id, Asset.DEPTH, frame=frame)
        if not path.exists():
            raise_missing(
                dataset="TUM RGB-D",
                asset="depth",
                tried=path,
                overrides=("depth_subdir", "depth_format"),
                layout=_LAYOUT,
            )
        from eval3r.utils.optional import optional_import

        imageio = optional_import("imageio.v3", extra="render")
        depth_raw = imageio.imread(path)
        return (depth_raw.astype(np.float32) / float(self._depth_scale))

    def load_color(self, scene_id: str, frame: int) -> np.ndarray:
        path = self.asset_path(scene_id, Asset.COLOR, frame=frame)
        if not path.exists():
            raise_missing(
                dataset="TUM RGB-D",
                asset="color",
                tried=path,
                overrides=("rgb_subdir", "color_format"),
                layout=_LAYOUT,
            )
        from eval3r.utils.optional import optional_import

        imageio = optional_import("imageio.v3", extra="render")
        return np.asarray(imageio.imread(path))

    def load_intrinsics(self, scene_id: str) -> np.ndarray:
        if self._intrinsics_fx is not None:
            fx = self._intrinsics_fx
            fy = self._intrinsics_fy if self._intrinsics_fy is not None else fx
            cx = self._intrinsics_cx if self._intrinsics_cx is not None else 0.0
            cy = self._intrinsics_cy if self._intrinsics_cy is not None else 0.0
            return np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
            )
        guessed = _guess_intrinsics(scene_id)
        if guessed is not None:
            return guessed
        raise MissingArtifactError(
            f"TUM RGB-D: no intrinsics for {scene_id!r}. "
            f"Pass `intrinsics_fx=` (and optionally fy, cx, cy) or use a "
            f"known sequence prefix (fr1/fr2/fr3)."
        )

    def load_poses(self, scene_id: str) -> Trajectory:
        path = self.asset_path(scene_id, Asset.POSES)
        if not path.exists():
            raise_missing(
                dataset="TUM RGB-D",
                asset="poses",
                tried=path,
                overrides=("pose_filename",),
                layout=_LAYOUT,
            )
        return load_trajectory_tum(path, convention="T_wc")
