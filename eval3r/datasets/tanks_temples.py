"""Tanks and Temples adapter (``tanks_temples``).

Tanks and Temples ground truth is an **independent laser-scanned point cloud in
metres** (``.agent/datasets.md`` Tanks and Temples rules). Only the public **training**
split is locally evaluable; intermediate and advanced splits are server-only (GT is
withheld) and this adapter refuses them before any computation.

Per-scene dataset layout (one directory per scene under the dataset root)::

    <root>/<Scene>/<Scene>.ply                   laser-scan GT point cloud (metres)
    <root>/<Scene>/<Scene>.json                  crop volume (SelectionPolygonVolume)
    <root>/<Scene>/<Scene>_trans.txt             4x4 alignment transform
    <root>/<Scene>/<Scene>_COLMAP_SfM.log        reference SfM trajectory (.log)
    <root>/<Scene>/<Scene>_mapping_reference.txt image->trajectory mapping

Local official evaluation is delegated to the official Tanks and Temples Python
toolbox via the ``tnt_official`` backend (``official_eval`` kind); this adapter does
not reimplement precision/recall/F-score, ICP registration, cropping, or the per-scene
distance thresholds — those come from the official backend (``.agent/backends.md``).
The five per-scene artifacts are resolved here and jointly fingerprinted.
"""

from __future__ import annotations

import hashlib
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

# Official benchmark scene lists (fixed by the benchmark, not thresholds). Only the
# training split has public GT; intermediate/advanced GT is withheld (server-only).
TRAINING_SCENES: tuple[str, ...] = (
    "Barn",
    "Caterpillar",
    "Church",
    "Courthouse",
    "Ignatius",
    "Meetingroom",
    "Truck",
)
INTERMEDIATE_SCENES: tuple[str, ...] = (
    "Family",
    "Francis",
    "Horse",
    "Lighthouse",
    "M60",
    "Panther",
    "Playground",
    "Train",
)
ADVANCED_SCENES: tuple[str, ...] = (
    "Auditorium",
    "Ballroom",
    "Courtroom",
    "Museum",
    "Palace",
    "Temple",
)

_SERVER_ONLY_SPLITS = {"intermediate", "advanced"}


class TanksAndTemplesAdapter:
    """Adapter over a Tanks and Temples dataset root (point-cloud geometry)."""

    name = "tanks_temples"
    native_unit = "m"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.capabilities = DatasetCapabilities(
            dense_geometry=True,
            independent_gt=True,
            # Headline capability is the public training split, evaluated by wrapping
            # the official toolbox. Per-split server-only status is enforced by
            # local_evaluation(split); see notes.
            official_local_eval=True,
            official_local_eval_method="official_script_wrapper",
            supports_full_scene_geometry=True,
            supports_object_centric_geometry=True,
            notes=[
                "GT is an independent laser-scanned point cloud in metres.",
                "Only the public training split is locally evaluable; intermediate and "
                "advanced splits are server-only (GT withheld) and are refused.",
                "Official precision/recall/F-score, ICP, cropping, and per-scene distance "
                "thresholds are delegated to the official toolbox (tnt_official backend).",
            ],
        )

    @classmethod
    def factory(cls, root: Path | None) -> TanksAndTemplesAdapter:
        if root is None:
            raise DatasetError(
                "the 'tanks_temples' dataset adapter requires the dataset root (pass --root). "
                "Expected <root>/<Scene>/<Scene>.ply + <Scene>.json + <Scene>_trans.txt + "
                "<Scene>_COLMAP_SfM.log per scene."
            )
        return cls(Path(root))

    # --- discovery -------------------------------------------------------------

    def _official_scene_list(self, split: str) -> tuple[str, ...]:
        if split == "training":
            return TRAINING_SCENES
        if split == "intermediate":
            return INTERMEDIATE_SCENES
        if split == "advanced":
            return ADVANCED_SCENES
        raise DatasetError(
            f"unknown Tanks and Temples split '{split}'. Known splits: training (public GT), "
            f"intermediate, advanced (both server-only)."
        )

    def iter_scenes(self, split: str) -> list[str]:
        # An explicit split file overrides the official list (e.g. to evaluate a subset).
        split_file = self.root / f"{split}.txt"
        if split_file.is_file():
            scenes = read_split_file(split_file)
            if not scenes:
                raise DatasetError(f"Tanks and Temples split file {split_file} lists no scenes.")
            return scenes

        official = self._official_scene_list(split)
        if split in _SERVER_ONLY_SPLITS:
            # These are never evaluated locally, but list them for reporting/coverage.
            return list(official)
        present = [s for s in official if self._scene_dir(s).is_dir()]
        if not present:
            raise DatasetError(
                f"no Tanks and Temples training scenes found under {self.root}. Expected scene "
                f"directories such as {self.root / 'Barn'} with Barn.ply / Barn.json / "
                f"Barn_trans.txt / Barn_COLMAP_SfM.log."
            )
        return present

    # --- per-scene artifacts ---------------------------------------------------

    def _scene_dir(self, scene_id: str) -> Path:
        return self.root / scene_id

    def _gt_pointcloud_path(self, scene_id: str) -> Path:
        return self._scene_dir(scene_id) / f"{scene_id}.ply"

    def _crop_path(self, scene_id: str) -> Path:
        return self._scene_dir(scene_id) / f"{scene_id}.json"

    def _alignment_path(self, scene_id: str) -> Path:
        return self._scene_dir(scene_id) / f"{scene_id}_trans.txt"

    def _trajectory_log_path(self, scene_id: str) -> Path:
        return self._scene_dir(scene_id) / f"{scene_id}_COLMAP_SfM.log"

    def official_artifacts(self, scene_id: str) -> dict[str, Path]:
        """Resolve and validate the five per-scene artifacts the official toolbox needs.

        Returns ``dataset_dir`` (the scene directory the toolbox reads GT/crop/trans/
        mapping from by naming convention), plus the explicit ``gt_pointcloud``,
        ``crop``, ``alignment`` and ``trajectory_log`` paths. Any missing artifact is
        an explicit error naming the file and the scene.
        """
        scene_dir = self._scene_dir(scene_id)
        if not scene_dir.is_dir():
            raise DatasetError(
                f"Tanks and Temples scene '{scene_id}' directory is missing: expected {scene_dir}."
            )
        artifacts = {
            "gt_pointcloud": self._gt_pointcloud_path(scene_id),
            "crop": self._crop_path(scene_id),
            "alignment": self._alignment_path(scene_id),
            "trajectory_log": self._trajectory_log_path(scene_id),
        }
        for kind, path in artifacts.items():
            if not path.is_file():
                raise DatasetError(
                    f"Tanks and Temples scene '{scene_id}' is missing its {kind} file: "
                    f"expected {path}. The official evaluation requires the GT point cloud, "
                    f"crop volume, alignment transform, and .log trajectory."
                )
        return {"dataset_dir": scene_dir, **artifacts}

    def load_scene(self, scene_id: str) -> SceneData:
        art = self.official_artifacts(scene_id)
        return SceneData(
            scene_id=scene_id,
            dataset=self.name,
            gt_pointcloud=art["gt_pointcloud"],
            gt_trajectory=art["trajectory_log"],
            masks={"crop": art["crop"], "alignment": art["alignment"]},
            ground_truth=self.load_ground_truth(scene_id, protocol=None),  # type: ignore[arg-type]
            capabilities=self.capabilities,
            metadata={
                "native_unit": self.native_unit,
                "dataset_dir": str(art["dataset_dir"]),
                "crop_path": str(art["crop"]),
                "alignment_path": str(art["alignment"]),
                "trajectory_log_path": str(art["trajectory_log"]),
                "source_pose_format": "tanks_temples_log",
            },
        )

    def load_ground_truth(self, scene_id: str, protocol: EvalProtocol) -> GroundTruthSpec:
        gt_path = self._gt_pointcloud_path(scene_id)
        return GroundTruthSpec(
            modality="pointcloud",
            provenance="laser_scan",
            independence="independent",
            density="dense_surface",
            path=gt_path,
            fingerprint=file_fingerprint(gt_path),
            unit=self.native_unit,
            source_pose_format="tanks_temples_log",
            notes=[
                "Official toolbox handles crop volume, alignment, ICP, and per-scene threshold."
            ],
        )

    def gt_fingerprint(self, scene_id: str, protocol: EvalProtocol) -> str | None:
        """Joint fingerprint over GT point cloud + crop + alignment transform.

        All three influence the official score, so a change in any of them changes the
        fingerprint (missing files are omitted, never silently treated as equal).
        """
        parts = [
            file_fingerprint(self._gt_pointcloud_path(scene_id)),
            file_fingerprint(self._crop_path(scene_id)),
            file_fingerprint(self._alignment_path(scene_id)),
        ]
        present = [p for p in parts if p is not None]
        if not present:
            return None
        digest = hashlib.sha256("|".join(present).encode()).hexdigest()
        return f"sha256:{digest}"

    # --- predictions -----------------------------------------------------------

    def resolve_prediction(
        self,
        pred_root: Path,
        scene_id: str,
        manifest: PredictionManifest | None,
    ) -> Reconstruction:
        pred_root = Path(pred_root)
        rel: Path | None = None
        if manifest is not None and scene_id in manifest.scenes:
            entry = manifest.scenes[scene_id]
            if entry.pointcloud is not None:
                rel = entry.pointcloud
            elif entry.mesh is not None:
                rel = entry.mesh
        if rel is not None:
            path = rel if rel.is_absolute() else pred_root / rel
        else:
            path = pred_root / f"{scene_id}.ply"

        if not path.is_file():
            raise DatasetError(
                f"Tanks and Temples prediction for scene '{scene_id}' not found under "
                f"{pred_root}. Expected {pred_root / f'{scene_id}.ply'} or a manifest entry."
            )
        return Reconstruction(
            path=path,
            modality="pointcloud",
            coordinate_frame="world",
            source_pose_format="tanks_temples_log",
            scale="metric",
            unit=self.native_unit,
            metadata={"scene": scene_id},
        )

    # --- local evaluation ------------------------------------------------------

    def local_evaluation(self, split: str, protocol: EvalProtocol) -> LocalEvaluationSpec:
        if split in _SERVER_ONLY_SPLITS:
            return LocalEvaluationSpec(
                status="server_only",
                reason=(
                    f"Tanks and Temples '{split}' GT is withheld; submit predictions to the "
                    f"official benchmark server. Only the training split is locally evaluable."
                ),
                public_gt_available=False,
                official_server_required=True,
            )
        return LocalEvaluationSpec(
            status="supported",
            reason="Tanks and Temples training GT is public; evaluated via the official toolbox.",
            public_gt_available=True,
        )
