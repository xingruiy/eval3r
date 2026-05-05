"""Find a method's prediction for a scene id, manifest-first then patterned."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict

from eval3r.prediction.manifest import MANIFEST_FILENAME
from eval3r.prediction.reader import PredictionReader
from eval3r.utils.typing import PathLike


class ResolvedPrediction(TypedDict):
    scene_id: str
    kind: Literal["manifest", "mesh_file", "point_cloud_file"]
    path: Path
    reader: PredictionReader | None


_DEFAULT_PATTERNS: tuple[str, ...] = (
    MANIFEST_FILENAME,             # eval3r_prediction.json — preferred
    "geometry/pred_mesh.ply",      # PredictionWriter default mesh
    "geometry/pred_points.ply",    # PredictionWriter default point cloud
    "{scene_id}_mesh.ply",
    "{scene_id}.ply",
    "mesh.ply",
    "pred_mesh.ply",
    "pred.ply",
    "points.ply",
    "{scene_id}_points.ply",
    "pred_points.ply",
)

_POINT_HINTS = ("points", "pcd", "pointcloud")


def _classify(rel: str) -> Literal["manifest", "mesh_file", "point_cloud_file"]:
    if rel.endswith(MANIFEST_FILENAME):
        return "manifest"
    base = rel.rsplit("/", 1)[-1].lower()
    if any(h in base for h in _POINT_HINTS):
        return "point_cloud_file"
    return "mesh_file"


@dataclass
class PredictionLocator:
    """Resolve scene id → prediction directory or geometry file.

    Custom patterns are *prepended* to the default list so user paths win.
    Each pattern may use ``{scene_id}`` and is resolved relative to the
    per-scene directory (controlled by ``scene_dir``).
    """

    preds_root: Path
    scene_dir: str = "{scene_id}"
    extra_patterns: tuple[str, ...] = ()
    flat_layout: bool = False
    """If True, also try patterns directly under ``preds_root`` without the
    per-scene directory (e.g. ``preds_root/scene0707_00.ply``)."""

    @property
    def patterns(self) -> tuple[str, ...]:
        return tuple(self.extra_patterns) + _DEFAULT_PATTERNS

    def __post_init__(self) -> None:
        self.preds_root = Path(self.preds_root)

    def resolve(self, scene_id: str) -> ResolvedPrediction | None:
        scene_root = self.preds_root / self.scene_dir.format(scene_id=scene_id)

        for pat in self.patterns:
            rel = pat.format(scene_id=scene_id)
            for base in (scene_root, *( (self.preds_root,) if self.flat_layout else () )):
                cand = base / rel
                if cand.exists():
                    return self._make(scene_id, cand)
        # Manifest dir without trailing filename, e.g. preds_root is itself the
        # prediction dir for a single scene — rare, but cheap to support.
        if (scene_root / MANIFEST_FILENAME).exists():
            return self._make(scene_id, scene_root / MANIFEST_FILENAME)
        return None

    def _make(self, scene_id: str, path: Path) -> ResolvedPrediction:
        kind = _classify(str(path))
        reader: PredictionReader | None = None
        resolved_path = path
        if kind == "manifest":
            reader = PredictionReader(path.parent, verify_hashes=False)
            resolved_path = path.parent
        return ResolvedPrediction(
            scene_id=scene_id,
            kind=kind,
            path=resolved_path,
            reader=reader,
        )


def find_predictions(
    locator: PredictionLocator, scene_ids: list[str]
) -> dict[str, ResolvedPrediction | None]:
    return {sid: locator.resolve(sid) for sid in scene_ids}
