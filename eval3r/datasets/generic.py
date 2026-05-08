"""Generic dataset adapter for ad-hoc layouts.

Used when the user wants to evaluate against a custom folder structure without
writing a Python adapter. The CLI exposes this through ``e3r benchmark`` when
``--dataset`` is omitted: the user passes ``--gt-path '{scene_id}/gt.ply'`` plus
a scene source (``--scenes-file`` or ``--scenes``) and we synthesize an adapter
on the fly.

Only ground-truth geometry is supported (mesh OR point cloud, picked by probing
the first scene's file). Intrinsics, poses, depth, and color are intentionally
not supported here — those need a real adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from eval3r.datasets.base import Asset, DatasetAdapter
from eval3r.datasets.layout import LayoutEntry, format_path, raise_missing
from eval3r.io.geometry import (
    MeshData,
    PointCloudData,
    load_mesh,
    load_point_cloud,
)
from eval3r.utils.errors import MissingArtifactError, NotSupportedError
from eval3r.utils.typing import PathLike

_LAYOUT: list[LayoutEntry] = [
    LayoutEntry(
        path="<root>/<gt_path>",
        overrides=("gt_path",),
    ),
]


class GenericAdapter(DatasetAdapter):
    """Adapter for arbitrary `<root>/<template>` ground-truth layouts.

    Parameters
    ----------
    root:
        Directory the ``gt_path`` template is resolved against.
    gt_path:
        Path template with ``{scene_id}`` substitution. Resolved file may be a
        mesh (any trimesh-loadable format with faces) or a point cloud. The
        first scene is probed at construction to decide which asset to expose.
    scenes_file:
        Path to a text file with one scene id per line.
    scenes_list:
        Explicit list of scene ids. Takes precedence over ``scenes_file``.
    split:
        Same semantics as ``scenes_file`` (text file path). Used only if
        ``scenes_list`` and ``scenes_file`` are both ``None``.
    validate_on_init:
        If True, also probe the first scene for missing-file errors at init
        time. The kind probe (mesh vs. point cloud) always runs when at least
        one scene is available.
    """

    name: ClassVar[str] = "generic"
    expected_layout: ClassVar[str] = "\n".join(e.render() for e in _LAYOUT)

    def __init__(
        self,
        root: PathLike,
        *,
        gt_path: str | None = None,
        scenes_file: PathLike | None = None,
        scenes_list: list[str] | None = None,
        split: str | PathLike | None = None,
        validate_on_init: bool = True,
    ) -> None:
        if not gt_path:
            raise MissingArtifactError(
                "GenericAdapter requires gt_path (e.g. '{scene_id}/gt.ply')."
            )
        self.root = Path(root)
        if not self.root.exists():
            raise MissingArtifactError(f"root not found: {self.root}")
        self._gt_path = gt_path
        self._split = split
        self._scenes = self._resolve_scenes(scenes_list, scenes_file, split)
        self._is_mesh: bool = self._probe_kind() if self._scenes else True

        if validate_on_init and self._scenes:
            self._probe_layout(self._scenes[0])

    # ------------------------------------------------------------------
    # scene discovery
    # ------------------------------------------------------------------
    def _resolve_scenes(
        self,
        scenes_list: list[str] | None,
        scenes_file: PathLike | None,
        split: str | PathLike | None,
    ) -> list[str]:
        if scenes_list:
            return [s.strip() for s in scenes_list if s and s.strip()]
        path = scenes_file or split
        if path is None:
            raise MissingArtifactError(
                "GenericAdapter requires a scene source: pass scenes_list, "
                "scenes_file, or split."
            )
        p = Path(path)
        if not p.exists():
            raise MissingArtifactError(
                f"Scenes file not found: {p}. Pass a text file with one "
                f"scene id per line."
            )
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]

    def list_scenes(self, split: str | PathLike | None = None) -> list[str]:
        if split is None or split == self._split:
            return list(self._scenes)
        return self._resolve_scenes(None, None, split)

    # ------------------------------------------------------------------
    # path helpers
    # ------------------------------------------------------------------
    def _gt_file(self, scene_id: str) -> Path:
        return self.root / format_path(self._gt_path, scene_id=scene_id)

    def asset_path(self, scene_id: str, asset: Asset, **kw: object) -> Path:
        if asset in (Asset.MESH, Asset.POINT_CLOUD):
            return self._gt_file(scene_id)
        raise NotSupportedError(
            f"generic: asset_path({asset}) not implemented. GenericAdapter "
            f"only supports mesh / point_cloud GT."
        )

    def _probe_layout(self, scene_id: str) -> None:
        path = self._gt_file(scene_id)
        if not path.exists():
            raise_missing(
                dataset="generic",
                asset="mesh or point_cloud",
                tried=path,
                overrides=("gt_path",),
                layout=_LAYOUT,
            )

    def _probe_kind(self) -> bool:
        """Return True if the first scene's GT file loads as a mesh."""
        first = self._scenes[0]
        path = self._gt_file(first)
        if not path.exists():
            # Layout error will surface from _probe_layout when validate_on_init
            # is True; otherwise defer to load-time. Default to point cloud:
            # `load_point_cloud` works for both meshes and point clouds, so it
            # is the safer default if we haven't been able to probe yet.
            return False
        try:
            load_mesh(path)
            return True
        except Exception:  # noqa: BLE001 — probe falls back to point-cloud kind
            return False

    # ------------------------------------------------------------------
    # asset advertisement
    # ------------------------------------------------------------------
    @property
    def supported_assets(self) -> set[Asset]:
        return {Asset.MESH if self._is_mesh else Asset.POINT_CLOUD}

    # ------------------------------------------------------------------
    # loaders
    # ------------------------------------------------------------------
    def load_mesh(self, scene_id: str) -> MeshData:
        if not self._is_mesh:
            raise NotSupportedError(
                f"generic: GT file at {self._gt_file(scene_id)} is a point "
                f"cloud, not a mesh."
            )
        path = self._gt_file(scene_id)
        if not path.exists():
            raise_missing(
                dataset="generic",
                asset="mesh",
                tried=path,
                overrides=("gt_path",),
                layout=_LAYOUT,
            )
        return load_mesh(path)

    def load_point_cloud(self, scene_id: str) -> PointCloudData:
        path = self._gt_file(scene_id)
        if not path.exists():
            raise_missing(
                dataset="generic",
                asset="point_cloud",
                tried=path,
                overrides=("gt_path",),
                layout=_LAYOUT,
            )
        return load_point_cloud(path)
