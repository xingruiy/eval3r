"""ETH3D adapter (``eth3d``, high-res DSLR multi-view benchmark).

ETH3D ground truth is an **independent laser scan in metres**, released for the
public training split as occlusion-aware evaluation clouds (``dslr_scan_eval``:
``scan_alignment.mlp`` referencing per-scan PLYs with their poses). The test split
GT is withheld (server-only) and this adapter refuses it before any computation.
Camera data is COLMAP text format and is parsed through pycolmap (``camera``
backend); non-pinhole camera models are recorded, never silently approximated.

Per-scene dataset layout (one directory per scene under the dataset root, as
unpacked from the official ``<scene>_dslr_undistorted`` and
``<scene>_dslr_scan_eval`` archives)::

    <root>/<scene>/dslr_scan_eval/scan_alignment.mlp       GT scan poses (MeshLab project)
    <root>/<scene>/dslr_scan_eval/scan*.ply                 laser scans referenced by the .mlp
    <root>/<scene>/dslr_calibration_undistorted/cameras.txt COLMAP text cameras
    <root>/<scene>/dslr_calibration_undistorted/images.txt  COLMAP text image poses
    <root>/<scene>/dslr_calibration_undistorted/points3D.txt

Local official evaluation is delegated to the official ETH3D multi-view-evaluation
tool via the ``eth3d_official`` backend (``official_eval`` kind); this adapter does
not reimplement the voxel-normalized accuracy/completeness/F1 or the beam-based
free-space classification — the official binary is invoked on the prediction PLY
and ``scan_alignment.mlp`` directly. This adapter covers the high-res DSLR
multi-view benchmark; the low-res multi-camera variant is a different benchmark
and is intentionally not part of this adapter.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
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
    from eval3r.backends.camera_pycolmap import ColmapCameraSet
    from eval3r.core.manifest import PredictionManifest
    from eval3r.core.protocol import EvalProtocol

# Official high-res DSLR multi-view scene lists (fixed by the benchmark). Only the
# training split has public GT; test-split GT is withheld (server-only).
TRAINING_SCENES: tuple[str, ...] = (
    "courtyard",
    "delivery_area",
    "electro",
    "facade",
    "kicker",
    "meadow",
    "office",
    "pipes",
    "playground",
    "relief",
    "relief_2",
    "terrace",
    "terrains",
)
TEST_SCENES: tuple[str, ...] = (
    "botanical_garden",
    "boulders",
    "bridge",
    "door",
    "exhibition_hall",
    "lecture_room",
    "living_room",
    "lounge",
    "observatory",
    "old_computer",
    "statue",
    "terrace_2",
)

_SERVER_ONLY_SPLITS = {"test"}

_SCAN_EVAL_DIR = "dslr_scan_eval"
_SCAN_MLP_NAME = "scan_alignment.mlp"
_CALIBRATION_DIR = "dslr_calibration_undistorted"


def parse_scan_mlp(mlp_path: Path) -> list[str]:
    """Return the scan PLY filenames referenced by a ``scan_alignment.mlp``.

    Parses the MeshLab project XML the same way the official tool does (MLMesh
    ``filename`` attributes under MeshGroup). Malformed or empty projects are
    explicit errors naming the file.
    """
    try:
        root = ET.parse(mlp_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise DatasetError(
            f"could not parse the ETH3D scan alignment MeshLab project {mlp_path}: {exc}"
        ) from exc
    filenames = [
        mesh.get("filename")
        for group in root.iter("MeshGroup")
        for mesh in group.iter("MLMesh")
    ]
    scans = [name for name in filenames if name]
    if not scans:
        raise DatasetError(
            f"the ETH3D scan alignment project {mlp_path} references no scan meshes "
            f"(no MLMesh filename entries); the official evaluation needs at least one scan."
        )
    return scans


class Eth3dAdapter:
    """Adapter over an ETH3D high-res DSLR multi-view dataset root."""

    name = "eth3d"
    native_unit = "m"
    variant = "training_public_gt"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.capabilities = DatasetCapabilities(
            dense_geometry=True,
            independent_gt=True,
            # Headline capability is the public training split, evaluated by wrapping
            # the official multi-view-evaluation binary. Per-split server-only status
            # is enforced by local_evaluation(split); see notes.
            official_local_eval=True,
            official_local_eval_method="official_script_wrapper",
            supports_full_scene_geometry=True,
            notes=[
                "GT is an independent laser scan in metres, released as occlusion-aware "
                "evaluation clouds (dslr_scan_eval).",
                "Only the training split is locally evaluable; the test split is "
                "server-only (GT withheld) and is refused.",
                "Accuracy/completeness/F1 at the official tolerances are delegated to the "
                "official multi-view-evaluation tool (eth3d_official backend).",
                "Cameras are COLMAP text parsed via pycolmap; non-pinhole models are "
                "recorded, never approximated.",
                "This adapter covers the high-res DSLR multi-view benchmark only; the "
                "low-res multi-camera variant is a different benchmark.",
            ],
        )

    @classmethod
    def factory(cls, root: Path | None) -> Eth3dAdapter:
        if root is None:
            raise DatasetError(
                "the 'eth3d' dataset adapter requires the dataset root (pass --root). Expected "
                f"<root>/<scene>/{_SCAN_EVAL_DIR}/{_SCAN_MLP_NAME} plus "
                f"<root>/<scene>/{_CALIBRATION_DIR}/ COLMAP text files per scene."
            )
        return cls(Path(root))

    # --- discovery -------------------------------------------------------------

    def _official_scene_list(self, split: str) -> tuple[str, ...]:
        if split == "training":
            return TRAINING_SCENES
        if split == "test":
            return TEST_SCENES
        raise DatasetError(
            f"unknown ETH3D split '{split}'. Known splits: training (public GT), "
            f"test (server-only)."
        )

    def iter_scenes(self, split: str) -> list[str]:
        # An explicit split file overrides the official list (e.g. to evaluate a subset).
        split_file = self.root / f"{split}.txt"
        if split_file.is_file():
            scenes = read_split_file(split_file)
            if not scenes:
                raise DatasetError(f"ETH3D split file {split_file} lists no scenes.")
            return scenes

        official = self._official_scene_list(split)
        if split in _SERVER_ONLY_SPLITS:
            # Never evaluated locally, but listed for reporting/coverage.
            return list(official)
        present = [s for s in official if self._scene_dir(s).is_dir()]
        if not present:
            raise DatasetError(
                f"no ETH3D training scenes found under {self.root}. Expected scene directories "
                f"such as {self.root / 'courtyard'} containing "
                f"{_SCAN_EVAL_DIR}/{_SCAN_MLP_NAME}."
            )
        return present

    # --- per-scene artifacts ---------------------------------------------------

    def _scene_dir(self, scene_id: str) -> Path:
        return self.root / scene_id

    def _scan_mlp_path(self, scene_id: str) -> Path:
        return self._scene_dir(scene_id) / _SCAN_EVAL_DIR / _SCAN_MLP_NAME

    def _calibration_dir(self, scene_id: str) -> Path:
        return self._scene_dir(scene_id) / _CALIBRATION_DIR

    def official_artifacts(self, scene_id: str) -> dict[str, object]:
        """Resolve and validate the inputs the official evaluation tool needs.

        Returns the ``scan_mlp`` path plus the resolved ``scan_paths`` it references.
        Any missing file is an explicit error naming the file and the scene.
        """
        scene_dir = self._scene_dir(scene_id)
        if not scene_dir.is_dir():
            raise DatasetError(
                f"ETH3D scene '{scene_id}' directory is missing: expected {scene_dir}."
            )
        mlp = self._scan_mlp_path(scene_id)
        if not mlp.is_file():
            raise DatasetError(
                f"ETH3D scene '{scene_id}' is missing its ground-truth scan alignment: "
                f"expected {mlp}. The official evaluation requires the "
                f"{_SCAN_EVAL_DIR}/{_SCAN_MLP_NAME} MeshLab project and the scan PLYs it "
                f"references (from the <scene>_dslr_scan_eval archive)."
            )
        scan_paths: list[Path] = []
        for filename in parse_scan_mlp(mlp):
            scan = Path(filename) if filename.startswith("/") else mlp.parent / filename
            if not scan.is_file():
                raise DatasetError(
                    f"ETH3D scene '{scene_id}' scan alignment {mlp} references scan "
                    f"'{filename}' but it does not exist at {scan}."
                )
            scan_paths.append(scan)
        return {"scan_mlp": mlp, "scan_paths": scan_paths}

    def load_cameras(self, scene_id: str) -> ColmapCameraSet:
        """COLMAP text cameras/poses for a scene, normalized via the pycolmap backend."""
        from eval3r.backends.camera_pycolmap import PycolmapCameraBackend

        calib = self._calibration_dir(scene_id)
        if not calib.is_dir():
            raise DatasetError(
                f"ETH3D scene '{scene_id}' has no COLMAP calibration directory: expected "
                f"{calib} (from the <scene>_dslr_undistorted archive)."
            )
        return PycolmapCameraBackend().load_cameras(calib)

    def load_scene(self, scene_id: str) -> SceneData:
        art = self.official_artifacts(scene_id)
        scan_mlp = art["scan_mlp"]
        scan_paths = art["scan_paths"]
        assert isinstance(scan_mlp, Path) and isinstance(scan_paths, list)

        metadata: dict[str, object] = {
            "native_unit": self.native_unit,
            "scan_mlp_path": str(scan_mlp),
            "scan_paths": [str(p) for p in scan_paths],
            "source_pose_format": "world_to_cam_colmap",
        }
        camera_paths: list[Path] | None = None
        calib = self._calibration_dir(scene_id)
        if calib.is_dir():
            cameras = self.load_cameras(scene_id)
            camera_paths = [calib]
            metadata["camera_models"] = {
                str(cid): model for cid, model in sorted(cameras.camera_models.items())
            }
            metadata["n_images"] = len(cameras.images)
            if cameras.non_pinhole_models:
                # Record the limitation explicitly instead of approximating (task 013).
                metadata["camera_model_limitations"] = list(cameras.notes)
        else:
            metadata["camera_model_limitations"] = [
                f"COLMAP calibration directory {calib} is absent; camera models unknown."
            ]

        return SceneData(
            scene_id=scene_id,
            dataset=self.name,
            variant=self.variant,
            camera_paths=camera_paths,
            # The GT is the set of laser scans referenced by scan_alignment.mlp; the
            # .mlp is the single canonical entry point the official tool consumes.
            gt_pointcloud=scan_mlp,
            ground_truth=self.load_ground_truth(scene_id, protocol=None),  # type: ignore[arg-type]
            capabilities=self.capabilities,
            metadata=metadata,
        )

    def load_ground_truth(self, scene_id: str, protocol: EvalProtocol) -> GroundTruthSpec:
        mlp = self._scan_mlp_path(scene_id)
        return GroundTruthSpec(
            modality="pointcloud",
            provenance="laser_scan",
            independence="independent",
            density="dense_surface",
            path=mlp,
            fingerprint=self.gt_fingerprint(scene_id, protocol),
            unit=self.native_unit,
            source_pose_format="world_to_cam_colmap",
            notes=[
                "GT is the occlusion-aware dslr_scan_eval laser-scan release "
                "(scan_alignment.mlp + referenced scan PLYs).",
                "The official tool derives observed free space from the scan positions; "
                "unobserved prediction regions are excluded from accuracy by the tool.",
            ],
        )

    def gt_fingerprint(self, scene_id: str, protocol: EvalProtocol) -> str | None:
        """Joint fingerprint over scan_alignment.mlp and every scan PLY it references.

        The scan poses (in the .mlp) and the scan geometry both determine the official
        score, so a change in either changes the fingerprint.
        """
        mlp = self._scan_mlp_path(scene_id)
        if not mlp.is_file():
            return None
        parts = [file_fingerprint(mlp)]
        try:
            art = self.official_artifacts(scene_id)
        except DatasetError:
            return None
        scan_paths = art["scan_paths"]
        assert isinstance(scan_paths, list)
        parts.extend(file_fingerprint(p) for p in scan_paths)
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
                f"ETH3D prediction for scene '{scene_id}' not found under {pred_root}. "
                f"Expected {pred_root / f'{scene_id}.ply'} or a manifest entry. Predictions "
                f"must already be in the ETH3D ground-truth (COLMAP) frame in metres; the "
                f"official tool applies no alignment."
            )
        return Reconstruction(
            path=path,
            modality="pointcloud",
            coordinate_frame="world",
            source_pose_format="world_to_cam_colmap",
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
                    "ETH3D test-split GT is withheld; submit reconstructions to the official "
                    "benchmark server at https://www.eth3d.net. Only the training split is "
                    "locally evaluable."
                ),
                public_gt_available=False,
                official_server_required=True,
            )
        return LocalEvaluationSpec(
            status="supported",
            reason=(
                "ETH3D training GT (dslr_scan_eval) is public; evaluated via the official "
                "multi-view-evaluation tool."
            ),
            public_gt_available=True,
        )
