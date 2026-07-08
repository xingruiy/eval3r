"""Neural-RGBD adapter (``neural_rgbd``) — geometry only.

Neural-RGBD is a standard indoor neural-surface / SLAM reconstruction benchmark. The
community reports geometry metrics (accuracy / completeness / chamfer / precision / recall /
F-score) against the **released culled** ground-truth mesh. This adapter evaluates a
prediction **mesh** against those released meshes; depth and pose evaluation are deferred
(``.agent/tasks/020-neural-rgbd-geometry.md``).

Official mesh layout (``<root>/<scene>/``, i.e. the dataset's ``nrgbd_meshes/official`` dir)::

    gt_mesh.ply          exact (uncropped/source) synthetic GT mesh
    gt_mesh_culled.ply   GT mesh culled to the observed region (the headline variant)
    neural_rgbd.ply      the Neural-RGBD method's own reconstruction (an example prediction)
    gt_trajectory_tum.txt optional GT camera trajectory for trajectory-first alignment

**Culled vs source is a protocol/variant decision, never chosen here silently.** The protocol
declares ``dataset.variant`` (``*culled*`` or ``*source*``/``*uncropped*``) and the adapter
resolves the matching GT mesh; an unknown/absent variant is an explicit error naming both
protocols. Because geometry evaluation compares two meshes directly in the shared world frame,
the OpenGL-vs-OpenCV camera-pose convention does **not** affect the score (no camera is used);
it is recorded as a note and only matters for the deferred depth/pose work.

These scenes are synthetic with exact artist-mesh GT, so the GT is honestly ``independent`` /
``synthetic_exact`` (unlike ScanNet's reconstruction-derived mesh). The geometry protocols are
``eval3r_native`` community conventions (Neural-RGBD has no official evaluation server), so
results must not be presented as official Neural-RGBD numbers.
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
from eval3r.core.types import GTIndependence, GTProvenance, SourcePoseFormat
from eval3r.datasets.base import file_fingerprint, read_split_file

if TYPE_CHECKING:
    from eval3r.core.manifest import PredictionManifest
    from eval3r.core.protocol import EvalProtocol

_MESH_FILES = {"culled": "gt_mesh_culled.ply", "source": "gt_mesh.ply"}

# Scene names whose released GT is a real capture (reconstruction-derived) rather than exact
# synthetic geometry. The public Neural-RGBD release we support ships only synthetic scenes
# with exact artist meshes, so this is empty today; the classifier keeps the synthetic/real
# distinction the datasets doc requires (``.agent/datasets.md`` Neural-RGBD rules) so a future
# real-scene bundle records ``reconstructed`` / ``reconstruction_derived`` honestly.
_KNOWN_REAL_SCENES: frozenset[str] = frozenset()


class NeuralRGBDAdapter:
    """Adapter over a Neural-RGBD official-mesh root (mesh geometry only)."""

    name = "neural_rgbd"
    native_unit = "m"
    depth_unit = 0.001  # depth tree is 16-bit millimetres; recorded for the future depth task
    # Recorded for provenance. Neural-RGBD poses are natively OpenGL; the reference driver
    # declares OpenCV. Geometry is mesh-to-mesh, so this does not affect the score — see the
    # module docstring. Only the deferred depth/pose work must resolve the convention.
    source_pose_format: SourcePoseFormat = "cam_to_world_opengl"

    def __init__(self, root: Path, *, mesh_variant: str = "culled") -> None:
        self.root = Path(root)
        self.mesh_variant = self._check_variant(mesh_variant)
        self.capabilities = DatasetCapabilities(
            dense_geometry=True,
            independent_gt=True,  # exact synthetic artist-mesh GT (not reconstruction-derived)
            depth_metric=True,  # available in principle; not implemented in this slice
            pose_metric=True,  # available in principle; not implemented in this slice
            official_local_eval=False,  # eval3r_native; no official Neural-RGBD benchmark
            official_local_eval_method="none",
            supports_full_scene_geometry=True,
            notes=[
                "GT is exact synthetic geometry; the released culled mesh crops it to the "
                "observed region. Culled vs source (uncropped) are distinct protocol variants.",
                "Geometry is mesh-to-mesh; the native OpenGL pose convention does not affect "
                "the score and is only relevant to the deferred depth/pose evaluation.",
                "eval3r_native community F-score convention — not an official Neural-RGBD "
                "benchmark number.",
            ],
        )

    @classmethod
    def factory(cls, root: Path | None) -> NeuralRGBDAdapter:
        if root is None:
            raise DatasetError(
                "the 'neural_rgbd' dataset adapter requires the official mesh root (pass "
                "--root). Expected <root>/<scene>/gt_mesh_culled.ply and <root>/<scene>/"
                "gt_mesh.ply (the Neural-RGBD 'nrgbd_meshes/official' directory)."
            )
        return cls(Path(root))

    # --- protocol binding ------------------------------------------------------

    @staticmethod
    def _check_variant(variant: str) -> str:
        if variant not in _MESH_FILES:
            raise DatasetError(
                f"Neural-RGBD mesh variant '{variant}' is not recognized; expected one of "
                f"{sorted(_MESH_FILES)} (culled -> gt_mesh_culled.ply, source -> gt_mesh.ply)."
            )
        return variant

    @staticmethod
    def _mesh_variant(protocol: EvalProtocol | None) -> str:
        """Map the protocol's declared variant to a mesh variant; never choose silently."""
        variant = (protocol.dataset.variant or "") if protocol is not None else ""
        low = variant.lower()
        if "culled" in low:
            return "culled"
        if "source" in low or "uncropped" in low:
            return "source"
        raise DatasetError(
            "Neural-RGBD geometry needs an explicit culled/source GT mesh variant, but the "
            f"protocol declared dataset.variant={variant!r}. Use protocol "
            "'neural_rgbd_geometry_culled' (gt_mesh_culled.ply) or "
            "'neural_rgbd_geometry_source' (gt_mesh.ply)."
        )

    def bind_protocol(self, protocol: EvalProtocol) -> None:
        """Select the GT-mesh variant the protocol declares before scenes are loaded.

        The geometry benchmark loop calls this opt-in hook so that ``load_scene`` (which the
        interface does not hand a protocol) resolves the correct culled/source mesh. The
        culled/source *decision* stays here in the adapter.
        """
        self.mesh_variant = self._mesh_variant(protocol)

    # --- discovery -------------------------------------------------------------

    def _scene_dir(self, scene_id: str) -> Path:
        return self.root / scene_id

    def _gt_mesh_path(self, scene_id: str, variant: str | None = None) -> Path:
        return self._scene_dir(scene_id) / _MESH_FILES[variant or self.mesh_variant]

    def _gt_trajectory_path(self, scene_id: str) -> Path | None:
        path = self._scene_dir(scene_id) / "gt_trajectory_tum.txt"
        return path if path.is_file() else None

    def _split_file(self, split: str) -> Path | None:
        for candidate in (self.root / f"{split}.txt", self.root / "splits" / f"{split}.txt"):
            if candidate.is_file():
                return candidate
        return None

    def iter_scenes(self, split: str) -> list[str]:
        split_file = self._split_file(split)
        if split_file is not None:
            scenes = read_split_file(split_file)
            if not scenes:
                raise DatasetError(f"Neural-RGBD split file {split_file} lists no scenes.")
            return scenes
        # No split file: 'all' enumerates every scene directory that has a GT mesh.
        if split != "all":
            raise DatasetError(
                f"Neural-RGBD split '{split}' not found: looked for {self.root / f'{split}.txt'} "
                f"and {self.root / 'splits' / f'{split}.txt'}. Provide a split file, or use "
                f"split 'all' to evaluate every scene directory under {self.root}."
            )
        scenes = sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and (p / _MESH_FILES["source"]).is_file()
        )
        if not scenes:
            raise DatasetError(
                f"Neural-RGBD root {self.root} has no scene directories with a "
                f"{_MESH_FILES['source']}; expected <root>/<scene>/gt_mesh.ply."
            )
        return scenes

    # --- ground truth ----------------------------------------------------------

    def _provenance(self, scene_id: str) -> tuple[GTProvenance, GTIndependence]:
        if scene_id in _KNOWN_REAL_SCENES:
            return "reconstructed", "reconstruction_derived"
        return "synthetic_exact", "independent"

    def load_scene(self, scene_id: str) -> SceneData:
        gt_path = self._gt_mesh_path(scene_id)
        if not gt_path.is_file():
            raise DatasetError(
                f"Neural-RGBD GT mesh for scene '{scene_id}' (variant '{self.mesh_variant}') is "
                f"missing: expected {gt_path}. Ensure the official mesh root has "
                f"<scene>/{_MESH_FILES[self.mesh_variant]}."
            )
        return SceneData(
            scene_id=scene_id,
            dataset=self.name,
            variant=self.mesh_variant,
            gt_mesh=gt_path,
            gt_trajectory=self._gt_trajectory_path(scene_id),
            ground_truth=self.load_ground_truth(scene_id, protocol=None),  # type: ignore[arg-type]
            capabilities=self.capabilities,
            metadata={
                "native_unit": self.native_unit,
                "source_pose_format": self.source_pose_format,
                "mesh_variant": self.mesh_variant,
                "gt_mesh": str(gt_path),
                "gt_trajectory": str(self._gt_trajectory_path(scene_id))
                if self._gt_trajectory_path(scene_id) is not None
                else None,
                "scene_kind": "real" if scene_id in _KNOWN_REAL_SCENES else "synthetic",
            },
        )

    def load_ground_truth(self, scene_id: str, protocol: EvalProtocol) -> GroundTruthSpec:
        variant = self._mesh_variant(protocol) if protocol is not None else self.mesh_variant
        gt_path = self._gt_mesh_path(scene_id, variant)
        provenance, independence = self._provenance(scene_id)
        return GroundTruthSpec(
            modality="mesh",
            provenance=provenance,
            independence=independence,
            density="dense_surface",
            path=gt_path,
            fingerprint=file_fingerprint(gt_path),
            unit=self.native_unit,
            source_pose_format=self.source_pose_format,
            notes=[
                f"mesh_variant={variant}",
                "Exact synthetic GT mesh; culled variant is cropped to the observed region."
                if independence == "independent"
                else "Reconstruction-derived GT mesh for a real Neural-RGBD capture.",
                "Native OpenGL pose convention; irrelevant to mesh-to-mesh geometry scoring.",
            ],
        )

    def gt_fingerprint(self, scene_id: str, protocol: EvalProtocol) -> str | None:
        variant = self._mesh_variant(protocol) if protocol is not None else self.mesh_variant
        return file_fingerprint(self._gt_mesh_path(scene_id, variant))

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
                f"Neural-RGBD prediction for scene '{scene_id}' not found under {pred_root}. "
                f"Expected <pred_root>/{scene_id}.ply (a reconstructed mesh) or a manifest entry."
            )
        return Reconstruction(
            path=path,
            modality="mesh",
            coordinate_frame="world",
            source_pose_format=self.source_pose_format,
            scale="metric",
            unit=self.native_unit,
            metadata={"scene": scene_id, "mesh_variant": self.mesh_variant},
        )

    # --- local evaluation ------------------------------------------------------

    def local_evaluation(self, split: str, protocol: EvalProtocol) -> LocalEvaluationSpec:
        return LocalEvaluationSpec(
            status="supported",
            reason=(
                "Neural-RGBD GT is a locally-available released mesh (culled or source); "
                "geometry is evaluable with the eval3r_native protocol. Results are not "
                "official Neural-RGBD numbers."
            ),
            public_gt_available=True,
        )
