"""Dataset-adapter interface (``.agent/datasets.md``).

An adapter turns dataset-native files into eval3r evaluation objects: it discovers
scenes for a split, resolves ground truth and predictions per scene, declares its
capabilities, and states whether a split is locally evaluable. Adapters must not
implement geometry algorithms — mesh sampling, nearest-neighbor search, and metric
formulas belong to backends/metrics.

This slice (task 008) exercises the interface with a generic point-cloud adapter and
a benchmark loop; real dataset adapters (DTU, ScanNet, ...) arrive in tasks 009+.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from eval3r.core.schema import (
    DatasetCapabilities,
    GroundTruthSpec,
    LocalEvaluationSpec,
    Reconstruction,
    SceneData,
)

if TYPE_CHECKING:  # avoid import cycles; only needed for typing
    from eval3r.core.manifest import PredictionManifest
    from eval3r.core.protocol import EvalProtocol


@runtime_checkable
class DatasetAdapter(Protocol):
    """Structural interface every dataset adapter satisfies."""

    name: str
    capabilities: DatasetCapabilities

    def iter_scenes(self, split: str) -> list[str]: ...

    def load_scene(self, scene_id: str) -> SceneData: ...

    def load_ground_truth(self, scene_id: str, protocol: EvalProtocol) -> GroundTruthSpec: ...

    def resolve_prediction(
        self,
        pred_root: Path,
        scene_id: str,
        manifest: PredictionManifest | None,
    ) -> Reconstruction: ...

    def gt_fingerprint(self, scene_id: str, protocol: EvalProtocol) -> str | None: ...

    def local_evaluation(self, split: str, protocol: EvalProtocol) -> LocalEvaluationSpec: ...


def file_fingerprint(path: Path) -> str | None:
    """Strong content hash of a (small) GT file, or ``None`` if it is absent.

    Suitable for the tiny fixtures used in CI. Large real-dataset GT should use a
    documented size+mtime+partial-hash policy instead (``.agent/reproducibility.md``).
    """
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def gt_geometry(scene: SceneData) -> tuple[Path, str]:
    """Return the ``(path, kind)`` of a scene's ground-truth geometry.

    ``kind`` is ``"mesh"`` or ``"pointcloud"``. Raises when neither is present so the
    benchmark loop can record a structured, per-scene failure.
    """
    from eval3r.core.errors import DatasetError

    if scene.gt_mesh is not None:
        return Path(scene.gt_mesh), "mesh"
    if scene.gt_pointcloud is not None:
        return Path(scene.gt_pointcloud), "pointcloud"
    raise DatasetError(
        f"scene '{scene.scene_id}' has no ground-truth mesh or point cloud to evaluate against."
    )


def prediction_kind(recon: Reconstruction) -> str:
    """Map a prediction modality to a geometry kind for the geometry benchmark path."""
    from eval3r.core.errors import DatasetError

    if recon.modality in ("mesh", "pointcloud"):
        return recon.modality
    raise DatasetError(
        f"prediction modality '{recon.modality}' is not a geometry input for this benchmark "
        f"(expected 'mesh' or 'pointcloud'). Path: {recon.path}."
    )


def read_split_file(path: Path) -> list[str]:
    """Read a split file: one scene id per line, ignoring blanks and ``#`` comments."""
    lines: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines
