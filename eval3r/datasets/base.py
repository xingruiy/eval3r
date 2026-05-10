"""DatasetAdapter abstract base class and Asset enum."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import ClassVar

import numpy as np

from eval3r.io.crop import CropVolume
from eval3r.io.geometry import MeshData, PointCloudData
from eval3r.io.trajectory import Trajectory
from eval3r.utils.errors import NotSupportedError
from eval3r.utils.typing import PathLike


class Asset(str, Enum):
    MESH = "mesh"
    POINT_CLOUD = "point_cloud"
    DEPTH = "depth"
    COLOR = "color"
    INTRINSICS = "intrinsics"
    INTRINSICS_DEPTH = "intrinsics_depth"
    INTRINSICS_COLOR = "intrinsics_color"
    POSES = "poses"


@dataclass
class AdapterValidationReport:
    ok: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))
        if not ok:
            self.ok = False
            self.errors.append(f"{name}: {detail}" if detail else name)


class DatasetAdapter(ABC):
    """Owns per-dataset filesystem layout + asset loaders.

    Subclasses set the ``name`` and ``expected_layout`` class attributes,
    implement :meth:`list_scenes`, and override the ``load_*`` methods for
    assets the dataset ships. Loaders raise :class:`MissingArtifactError`
    with a hint pointing at the override knob the user can flip.
    """

    name: ClassVar[str]
    expected_layout: ClassVar[str] = ""

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------
    @abstractmethod
    def list_scenes(self, split: str | PathLike | None = None) -> list[str]: ...

    @property
    def supported_assets(self) -> set[Asset]:
        """Assets the adapter implements; default = whatever subclasses override."""
        impls: set[Asset] = set()
        for asset, fn in (
            (Asset.MESH, self.load_mesh),
            (Asset.POINT_CLOUD, self.load_point_cloud),
            (Asset.DEPTH, self.load_depth),
            (Asset.COLOR, self.load_color),
            (Asset.INTRINSICS_DEPTH, self.load_intrinsics_depth),
            (Asset.INTRINSICS_COLOR, self.load_intrinsics_color),
            (Asset.POSES, self.load_poses),
        ):
            base = getattr(DatasetAdapter, fn.__name__)
            if fn.__func__ is not base:  # type: ignore[attr-defined]
                impls.add(asset)
        return impls

    def supports(self, asset: Asset) -> bool:
        return asset in self.supported_assets

    # ------------------------------------------------------------------
    # loaders — default raises NotSupported. Subclasses override what they have.
    # Each must raise MissingArtifactError with override hints when paths are missing.
    # ------------------------------------------------------------------
    def load_mesh(self, scene_id: str) -> MeshData:
        raise NotSupportedError(f"{self.name}: load_mesh not supported")

    def load_point_cloud(self, scene_id: str) -> PointCloudData:
        raise NotSupportedError(f"{self.name}: load_point_cloud not supported")

    def load_depth(self, scene_id: str, frame: int) -> np.ndarray:
        raise NotSupportedError(f"{self.name}: load_depth not supported")

    def load_color(self, scene_id: str, frame: int) -> np.ndarray:
        raise NotSupportedError(f"{self.name}: load_color not supported")

    def load_intrinsics(self, scene_id: str) -> np.ndarray:
        return self.load_intrinsics_depth(scene_id)

    def load_intrinsics_depth(self, scene_id: str) -> np.ndarray:
        raise NotSupportedError(f"{self.name}: load_intrinsics_depth not supported")

    def load_intrinsics_color(self, scene_id: str) -> np.ndarray:
        raise NotSupportedError(f"{self.name}: load_intrinsics_color not supported")

    def load_poses(self, scene_id: str) -> Trajectory:
        raise NotSupportedError(f"{self.name}: load_poses not supported")

    def load_crop_volume(self, scene_id: str) -> CropVolume:
        """Per-scene evaluation crop region (e.g. T&T ``{scene}.json``).

        Optional: only datasets that ship a crop file override this.
        Callers should treat ``NotSupportedError`` as "no crop applies".
        """
        raise NotSupportedError(f"{self.name}: load_crop_volume not supported")

    def load_thresholds(self, scene_id: str) -> tuple[float, ...]:
        """Per-scene F-score thresholds (e.g. T&T's scene-specific τ).

        Optional: only datasets with published per-scene τ override this.
        Callers should treat ``NotSupportedError`` as "fall back to the
        global ``BenchmarkConfig.thresholds``".
        """
        raise NotSupportedError(f"{self.name}: load_thresholds not supported")

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def asset_path(self, scene_id: str, asset: Asset, **kw: object) -> Path:
        """Where the adapter looks for ``asset`` for ``scene_id`` (without IO)."""
        raise NotSupportedError(
            f"{self.name}: asset_path({asset}) not implemented"
        )

    def validate(self, scenes: int = 1) -> AdapterValidationReport:
        """Sample-check the first ``scenes`` scenes; subclasses can override."""
        report = AdapterValidationReport(ok=True)
        try:
            ids = self.list_scenes()[:scenes]
        except Exception as e:  # pragma: no cover - defensive
            report.add("list_scenes", False, str(e))
            return report
        report.add("list_scenes", True, f"{len(ids)} scenes")
        for sid in ids:
            for asset in self.supported_assets:
                try:
                    self._probe(sid, asset)
                    report.add(f"{sid}/{asset.value}", True)
                except Exception as e:
                    report.add(f"{sid}/{asset.value}", False, str(e))
        return report

    def _probe(self, scene_id: str, asset: Asset) -> None:
        """Touch the on-disk path for ``asset`` without doing heavy IO."""
        if asset is Asset.MESH:
            self.load_mesh(scene_id)
        elif asset is Asset.POINT_CLOUD:
            self.load_point_cloud(scene_id)
        elif asset is Asset.INTRINSICS:
            self.load_intrinsics(scene_id)
        elif asset is Asset.INTRINSICS_DEPTH:
            self.load_intrinsics_depth(scene_id)
        elif asset is Asset.INTRINSICS_COLOR:
            self.load_intrinsics_color(scene_id)
        elif asset is Asset.POSES:
            self.load_poses(scene_id)
        elif asset is Asset.DEPTH:
            self.load_depth(scene_id, 0)
        elif asset is Asset.COLOR:
            self.load_color(scene_id, 0)
