"""Replica dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np

from eval3r.datasets.base import Asset, DatasetAdapter
from eval3r.datasets.layout import LayoutEntry, format_path, raise_missing
from eval3r.io.geometry import MeshData, PointCloudData, load_mesh
from eval3r.io.trajectory import Trajectory, load_trajectory_tum
from eval3r.utils.errors import MissingArtifactError, NotSupportedError
from eval3r.utils.typing import PathLike

_LAYOUT: list[LayoutEntry] = [
    LayoutEntry(
        path="<root>/<scene_name>/mesh.ply",
        overrides=("mesh_filename",),
    ),
    LayoutEntry(
        path="<root>/<scene_name>/results/depth{frame}.png",
        overrides=("results_subdir", "depth_format"),
    ),
    LayoutEntry(
        path="<root>/<scene_name>/results/rgb{frame}.png",
        overrides=("results_subdir", "color_format"),
    ),
    LayoutEntry(
        path="<root>/<scene_name>/trajectory.txt",
        overrides=("trajectory_filename",),
    ),
]


class ReplicaAdapter(DatasetAdapter):
    """Replica dataset adapter.

    Defaults match the standard Replica layout with per-scene directories
    containing ``mesh.ply`` and optionally rendered frames in ``results/``.
    """

    name: ClassVar[str] = "replica"
    expected_layout: ClassVar[str] = "\n".join(e.render() for e in _LAYOUT)

    def __init__(
        self,
        root: PathLike,
        *,
        split: str | PathLike | None = None,
        mesh_filename: str = "mesh.ply",
        results_subdir: str = "results",
        depth_format: str = "depth{frame}.png",
        color_format: str = "rgb{frame}.png",
        depth_scale: float = 1000.0,
        trajectory_filename: str = "trajectory.txt",
        validate_on_init: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise MissingArtifactError(f"Replica root not found: {self.root}")
        self._split = split
        self._mesh_filename = mesh_filename
        self._results_subdir = results_subdir
        self._depth_format = depth_format
        self._color_format = color_format
        self._depth_scale = depth_scale
        self._trajectory_filename = trajectory_filename

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
                if p.is_dir() and (p / self._mesh_filename).exists()
            )
        p = Path(split)
        if not p.exists():
            raise MissingArtifactError(
                f"Replica split file not found: {p}. "
                f"Pass a path to a text file with one scene id per line."
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
        if asset is Asset.MESH:
            return sd / self._mesh_filename
        if asset is Asset.POINT_CLOUD:
            return sd / self._mesh_filename
        if asset is Asset.COLOR:
            return sd / self._results_subdir / format_path(
                self._color_format, frame=int(kw["frame"])  # type: ignore[arg-type]
            )
        if asset is Asset.DEPTH:
            return sd / self._results_subdir / format_path(
                self._depth_format, frame=int(kw["frame"])  # type: ignore[arg-type]
            )
        if asset is Asset.POSES:
            return sd / self._trajectory_filename
        raise NotSupportedError(f"replica: asset_path({asset}) not implemented")

    def _probe_layout(self, scene_id: str) -> None:
        mesh = self.asset_path(scene_id, Asset.MESH)
        if not mesh.exists():
            raise_missing(
                dataset="Replica",
                asset="mesh",
                tried=mesh,
                overrides=("mesh_filename",),
                layout=_LAYOUT,
            )

    # ------------------------------------------------------------------
    # loaders
    # ------------------------------------------------------------------
    def load_mesh(self, scene_id: str) -> MeshData:
        path = self.asset_path(scene_id, Asset.MESH)
        if not path.exists():
            raise_missing(
                dataset="Replica",
                asset="mesh",
                tried=path,
                overrides=("mesh_filename",),
                layout=_LAYOUT,
            )
        return load_mesh(path)

    def load_point_cloud(self, scene_id: str) -> PointCloudData:
        mesh = self.load_mesh(scene_id)
        return PointCloudData(points=mesh.vertices, colors=mesh.vertex_colors)

    def load_depth(self, scene_id: str, frame: int) -> np.ndarray:
        path = self.asset_path(scene_id, Asset.DEPTH, frame=frame)
        if not path.exists():
            raise_missing(
                dataset="Replica",
                asset="depth",
                tried=path,
                overrides=("results_subdir", "depth_format"),
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
                dataset="Replica",
                asset="color",
                tried=path,
                overrides=("results_subdir", "color_format"),
                layout=_LAYOUT,
            )
        from eval3r.utils.optional import optional_import

        imageio = optional_import("imageio.v3", extra="render")
        return np.asarray(imageio.imread(path))

    def load_poses(self, scene_id: str) -> Trajectory:
        path = self.asset_path(scene_id, Asset.POSES)
        if not path.exists():
            raise_missing(
                dataset="Replica",
                asset="poses",
                tried=path,
                overrides=("trajectory_filename",),
                layout=_LAYOUT,
            )
        return load_trajectory_tum(path, convention="T_wc")
