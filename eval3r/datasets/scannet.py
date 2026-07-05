"""ScanNet adapter (``scannet``).

ScanNet ground truth is a **BundleFusion-reconstructed mesh** (``sceneXXXX_XX_vh_clean_2.ply``)
— reconstruction-derived, dense-surface geometry, **not** an independent laser scan
(``.agent/datasets.md`` ScanNet rules). This adapter resolves scenes from the official
train/val/test split files, reads the exported SensReader layout, records the native
metre unit and 16-bit-millimetre depth unit (``depth_unit = 0.001``), and exposes the
GT camera trajectory (camera-to-world, OpenCV axes) used for evaluation-time visibility
culling on the test split.

Exported scene layout (``<root>/scans/<scene>/``)::

    color/<f>.jpg                RGB frames
    depth/<f>.png                16-bit millimetre depth
    pose/<f>.txt                 4x4 camera-to-world (OpenCV axes)
    intrinsic_depth.txt          4x4 depth intrinsics
    <scene>_vh_clean_2.ply       reconstruction-derived GT mesh
    <scene>.txt                  scene metadata (depthWidth/Height, ...)

Single-layer vs double-layer is a **protocol/variant** decision, never chosen here:
the adapter records the convention the protocol declares (``metadata.layer_convention``)
and resolves the GT mesh from the scene directory. Visibility culling is never enabled
or disabled by the adapter — the protocol's masking decides, and the culling itself runs
in the ``visibility`` backend (``.agent/backends.md``). ScanNet has no official geometry
server benchmark; these protocols are ``eval3r_native`` and must not be reported as
official ScanNet numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from eval3r.backends.visibility_render import CameraTrajectory, trajectory_fingerprint
from eval3r.core.errors import DatasetError
from eval3r.core.schema import (
    DatasetCapabilities,
    GroundTruthSpec,
    LocalEvaluationSpec,
    Reconstruction,
    SceneData,
)
from eval3r.core.types import SourcePoseFormat
from eval3r.datasets.base import file_fingerprint, read_split_file

if TYPE_CHECKING:
    from eval3r.core.manifest import PredictionManifest
    from eval3r.core.protocol import EvalProtocol

_GT_MESH_SUFFIX = "_vh_clean_2.ply"


class ScanNetAdapter:
    """Adapter over a ScanNet v2 dataset root (mesh geometry + GT trajectory)."""

    name = "scannet"
    native_unit = "m"
    depth_unit = 0.001  # 16-bit millimetre depth
    source_pose_format: SourcePoseFormat = "cam_to_world_opencv"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.capabilities = DatasetCapabilities(
            dense_geometry=True,
            independent_gt=False,  # BundleFusion-reconstructed mesh, not laser scan
            depth_metric=True,
            pose_metric=True,
            official_local_eval=False,  # eval3r_native; no official ScanNet geometry benchmark
            official_local_eval_method="none",
            supports_full_scene_geometry=True,
            requires_external_renderer=True,  # only for the test visibility-culling protocols
            notes=[
                "GT is a BundleFusion-reconstructed mesh (reconstruction-derived, dense surface).",
                "Depth is 16-bit millimetres; depth_unit = 0.001. Poses are cam-to-world OpenCV.",
                "Visibility culling (test single-/double-layer) renders prediction depth from the "
                "GT trajectory and TSDF-trims it; val needs no renderer.",
                "eval3r_native geometry protocol — not an official ScanNet benchmark number.",
            ],
        )

    @classmethod
    def factory(cls, root: Path | None) -> ScanNetAdapter:
        if root is None:
            raise DatasetError(
                "the 'scannet' dataset adapter requires the ScanNet root (pass --root). "
                "Expected <root>/scans/<scene>/<scene>_vh_clean_2.ply and split files."
            )
        return cls(Path(root))

    # --- discovery -------------------------------------------------------------

    def _split_file(self, split: str) -> Path:
        candidates = [self.root / f"{split}.txt", self.root / "splits" / f"{split}.txt"]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise DatasetError(
            f"ScanNet split '{split}' not found: looked for {candidates[0]} and {candidates[1]}."
        )

    def iter_scenes(self, split: str) -> list[str]:
        scenes = read_split_file(self._split_file(split))
        if not scenes:
            raise DatasetError(f"ScanNet split file {self._split_file(split)} lists no scenes.")
        return scenes

    # --- paths -----------------------------------------------------------------

    def _scene_dir(self, scene_id: str) -> Path:
        return self.root / "scans" / scene_id

    def _gt_mesh_path(self, scene_id: str) -> Path:
        return self._scene_dir(scene_id) / f"{scene_id}{_GT_MESH_SUFFIX}"

    def _intrinsic_depth_path(self, scene_id: str) -> Path:
        scene_dir = self._scene_dir(scene_id)
        candidates = (
            scene_dir / "intrinsic_depth.txt",
            scene_dir / "intrinsic" / "intrinsic_depth.txt",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    @staticmethod
    def _layer_convention(protocol: EvalProtocol | None) -> str:
        """Record whether the protocol declares single-layer or double-layer geometry."""
        if protocol is None:
            return "unspecified"
        variant = (protocol.dataset.variant or "").lower()
        if "double" in variant:
            return "double_layer"
        if "single" in variant:
            return "single_layer"
        return "unspecified"

    # --- ground truth ----------------------------------------------------------

    def load_scene(self, scene_id: str) -> SceneData:
        gt_path = self._gt_mesh_path(scene_id)
        if not gt_path.is_file():
            raise DatasetError(
                f"ScanNet GT mesh for '{scene_id}' is missing: expected {gt_path}. Ensure the "
                f"scene is exported with its {_GT_MESH_SUFFIX} reconstruction."
            )
        scene_dir = self._scene_dir(scene_id)
        pose_dir = scene_dir / "pose"
        depth_dir = scene_dir / "depth"
        return SceneData(
            scene_id=scene_id,
            dataset=self.name,
            gt_mesh=gt_path,
            depth_paths=sorted(depth_dir.glob("*.png")) if depth_dir.is_dir() else None,
            camera_paths=sorted(pose_dir.glob("*.txt")) if pose_dir.is_dir() else None,
            ground_truth=self.load_ground_truth(scene_id, protocol=None),  # type: ignore[arg-type]
            capabilities=self.capabilities,
            metadata={
                "native_unit": self.native_unit,
                "depth_unit": self.depth_unit,
                "source_pose_format": self.source_pose_format,
                "gt_mesh": str(gt_path),
                "has_trajectory": pose_dir.is_dir(),
                "visibility_source": "gt_trajectory",
            },
        )

    def load_ground_truth(self, scene_id: str, protocol: EvalProtocol) -> GroundTruthSpec:
        gt_path = self._gt_mesh_path(scene_id)
        return GroundTruthSpec(
            modality="mesh",
            provenance="reconstructed",
            independence="reconstruction_derived",
            density="dense_surface",
            path=gt_path,
            fingerprint=file_fingerprint(gt_path),
            unit=self.native_unit,
            source_pose_format=self.source_pose_format,
            notes=[
                "BundleFusion-reconstructed mesh; not an independent laser scan.",
                f"layer_convention={self._layer_convention(protocol)}",
            ],
        )

    def gt_fingerprint(self, scene_id: str, protocol: EvalProtocol) -> str | None:
        return file_fingerprint(self._gt_mesh_path(scene_id))

    # --- trajectory (for visibility culling) -----------------------------------

    def load_trajectory(self, scene_id: str) -> CameraTrajectory:
        """Load the GT depth-camera trajectory (cam-to-world OpenCV) + intrinsics.

        Used by the ``visibility`` backend to render the observed region. Non-finite
        pose files (ScanNet marks lost tracking with inf) are dropped and the count is
        recorded in the trajectory fingerprint via the surviving poses.
        """
        scene_dir = self._scene_dir(scene_id)
        pose_dir = scene_dir / "pose"
        if not pose_dir.is_dir():
            raise DatasetError(
                f"ScanNet scene '{scene_id}' has no pose/ directory at {pose_dir}; the visibility-"
                f"culling protocol needs the GT camera trajectory."
            )
        pose_files = sorted(pose_dir.glob("*.txt"), key=lambda p: int(p.stem))
        poses = []
        for pf in pose_files:
            T = np.loadtxt(pf)
            if T.shape == (4, 4) and np.all(np.isfinite(T)):
                poses.append(T)
        if not poses:
            raise DatasetError(
                f"ScanNet scene '{scene_id}' has no finite camera poses in {pose_dir}; cannot "
                f"render the observed region for visibility culling."
            )
        poses_arr = np.stack(poses, axis=0)

        intr_path = self._intrinsic_depth_path(scene_id)
        if not intr_path.is_file():
            raise DatasetError(
                f"ScanNet scene '{scene_id}' has no depth intrinsics at {intr_path}."
            )
        K = np.loadtxt(intr_path)[:3, :3]
        width, height = self._depth_dims(scene_id)
        return CameraTrajectory(
            poses=poses_arr,
            intrinsics=K,
            width=width,
            height=height,
            fingerprint=trajectory_fingerprint(poses_arr, K, width, height),
        )

    def _depth_dims(self, scene_id: str) -> tuple[int, int]:
        """Depth (width, height) from the scene metadata, or a depth PNG as fallback."""
        meta = self._scene_dir(scene_id) / f"{scene_id}.txt"
        if meta.is_file():
            fields: dict[str, str] = {}
            for line in meta.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    key, _, value = line.partition("=")
                    fields[key.strip()] = value.strip()
            if "depthWidth" in fields and "depthHeight" in fields:
                return int(fields["depthWidth"]), int(fields["depthHeight"])
        depth_dir = self._scene_dir(scene_id) / "depth"
        pngs = sorted(depth_dir.glob("*.png")) if depth_dir.is_dir() else []
        if pngs:
            import imageio.v2 as imageio

            arr = imageio.imread(pngs[0])
            return int(arr.shape[1]), int(arr.shape[0])
        raise DatasetError(
            f"ScanNet scene '{scene_id}': cannot determine depth image size (no {meta} with "
            f"depthWidth/depthHeight and no depth PNGs)."
        )

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
            rel = entry.mesh or entry.pointcloud
        if rel is not None:
            path = rel if rel.is_absolute() else pred_root / rel
        else:
            path = pred_root / f"{scene_id}.ply"
        if not path.is_file():
            raise DatasetError(
                f"ScanNet prediction for scene '{scene_id}' not found under {pred_root}. Expected "
                f"<pred_root>/{scene_id}.ply (a reconstructed mesh) or a manifest entry for it."
            )
        return Reconstruction(
            path=path,
            modality="mesh",
            coordinate_frame="world",
            source_pose_format=self.source_pose_format,
            scale="metric",
            unit=self.native_unit,
            metadata={"scene": scene_id},
        )

    # --- local evaluation ------------------------------------------------------

    def local_evaluation(self, split: str, protocol: EvalProtocol) -> LocalEvaluationSpec:
        return LocalEvaluationSpec(
            status="supported",
            reason=(
                "ScanNet GT is a locally-available reconstructed mesh; geometry is evaluable "
                "with the eval3r_native protocol. Results are not official ScanNet numbers."
            ),
            public_gt_available=True,
        )
