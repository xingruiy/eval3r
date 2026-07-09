"""Generic point-cloud/mesh directory adapter (``custom``).

A minimal, honest adapter for a user-supplied dataset root laid out as::

    <root>/splits/<split>.txt      one scene id per line (blank / '#' lines ignored)
    <root>/gt/<scene_id>.ply       ground-truth geometry for each scene

Predictions live under a separate ``pred_root`` and are resolved from a manifest
when present, else inferred as ``<pred_root>/<scene_id>.ply``. Ground truth is
user-supplied, so provenance/independence/density are recorded as ``unknown`` and
the fidelity is native — the adapter does not pretend the GT is an
independent measurement (``.agent/datasets.md`` common-mistakes rules).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from eval3r.core.errors import DatasetError
from eval3r.core.schema import (
    DatasetCapabilities,
    GroundTruthSpec,
    LocalEvaluationSpec,
    Reconstruction,
    SceneData,
)
from eval3r.datasets.base import file_fingerprint, read_split_file

if TYPE_CHECKING:
    from eval3r.core.manifest import PredictionManifest
    from eval3r.core.protocol import EvalProtocol


class CustomAdapter:
    """Adapter over a ``splits/`` + ``gt/`` directory root."""

    name = "custom"

    def __init__(self, root: Path, *, gt_modality: str = "pointcloud") -> None:
        self.root = Path(root)
        self.gt_modality = gt_modality
        self.capabilities = DatasetCapabilities(
            dense_geometry=True,
            independent_gt=False,
            supports_full_scene_geometry=True,
            official_local_eval=False,
            official_local_eval_method="none",
            notes=["User-supplied geometry; GT provenance is unknown / native."],
        )

    @classmethod
    def factory(cls, root: Path | None) -> CustomAdapter:
        if root is None:
            raise DatasetError(
                "the 'custom' dataset adapter requires a dataset root (pass --root). "
                "Expected layout: <root>/splits/<split>.txt and <root>/gt/<scene_id>.ply."
            )
        return cls(Path(root))

    # --- discovery -------------------------------------------------------------

    def _split_file(self, split: str) -> Path:
        return self.root / "splits" / f"{split}.txt"

    def iter_scenes(self, split: str) -> list[str]:
        split_file = self._split_file(split)
        if not split_file.is_file():
            raise DatasetError(
                f"split '{split}' not found for the custom dataset at {self.root}: "
                f"expected a scene-id list at {split_file}."
            )
        scenes = read_split_file(split_file)
        if not scenes:
            raise DatasetError(f"split file {split_file} lists no scenes.")
        return scenes

    # --- ground truth ----------------------------------------------------------

    def _gt_path(self, scene_id: str) -> Path:
        return self.root / "gt" / f"{scene_id}.ply"

    def load_scene(self, scene_id: str) -> SceneData:
        gt_path = self._gt_path(scene_id)
        if not gt_path.is_file():
            raise DatasetError(
                f"ground-truth file for scene '{scene_id}' is missing: expected {gt_path}."
            )
        gt_spec = GroundTruthSpec(
            modality="pointcloud",
            provenance="unknown",
            independence="unknown",
            density="unknown",
            path=gt_path,
            fingerprint=file_fingerprint(gt_path),
        )
        return SceneData(
            scene_id=scene_id,
            dataset=self.name,
            gt_pointcloud=gt_path,
            ground_truth=gt_spec,
            capabilities=self.capabilities,
        )

    def load_ground_truth(self, scene_id: str, protocol: EvalProtocol) -> GroundTruthSpec:
        return self.load_scene(scene_id).ground_truth

    def gt_fingerprint(self, scene_id: str, protocol: EvalProtocol) -> str | None:
        return file_fingerprint(self._gt_path(scene_id))

    # --- predictions -----------------------------------------------------------

    def resolve_prediction(
        self,
        pred_root: Path,
        scene_id: str,
        manifest: PredictionManifest | None,
    ) -> Reconstruction:
        pred_root = Path(pred_root)
        modality = "pointcloud"
        rel: Path | None = None
        if manifest is not None and scene_id in manifest.scenes:
            entry = manifest.scenes[scene_id]
            modality = manifest.prediction_modality
            if entry.mesh is not None:
                rel, modality = entry.mesh, "mesh"
            elif entry.pointcloud is not None:
                rel, modality = entry.pointcloud, "pointcloud"
        if rel is None:
            rel = Path(f"{scene_id}.ply")  # inferred simple layout

        path = rel if rel.is_absolute() else pred_root / rel
        if not path.is_file():
            raise DatasetError(
                f"prediction for scene '{scene_id}' not found at {path}. "
                f"Provide a manifest entry or place the file at <pred_root>/{scene_id}.ply."
            )
        return Reconstruction(
            path=path,
            modality=modality,  # type: ignore[arg-type]
            coordinate_frame="world",
            scale="metric",
        )

    # --- local evaluation ------------------------------------------------------

    def local_evaluation(self, split: str, protocol: EvalProtocol) -> LocalEvaluationSpec:
        return LocalEvaluationSpec(
            status="supported",
            reason="User-supplied local GT and predictions.",
            public_gt_available=True,
        )
