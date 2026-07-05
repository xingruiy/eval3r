"""DTU adapter (``dtu``).

DTU ground truth is a **laser-scanned point cloud in millimetres** — an independent,
dense-surface reference, not a mesh (``.agent/datasets.md`` DTU rules). This adapter
resolves scans and their GT / ObsMask / Plane files, records the native millimetre
unit (normalized to metres by the pipeline's normalize stage), and resolves method
prediction filenames including the DTU light-condition suffix (``<method>XXX_l3.ply``).

Dataset root layout (SampleSet / MVS Data)::

    <root>/Points/stl/stl<NNN>_total.ply    laser-scan GT point cloud (mm)
    <root>/ObsMask/ObsMask<N>_10.mat        observability mask (object volume)
    <root>/ObsMask/Plane<N>.mat             background-plane cull (may be missing)
    <root>/splits/<split>.txt               scan ids, one per line

This slice does **not** run the official-like ObsMask/Plane evaluation — that is
task 010. Until then ``official_local_eval`` is ``False`` and DTU point-cloud runs
are labelled eval3r-native, though the GT provenance is honestly ``laser_scan`` /
``independent``. Missing Plane files are recorded, never silently ignored.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

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

# DTU predictions are named e.g. ``furu001_l3.ply`` — <method><scan:03d>_l<light>.ply.
_PRED_RE = re.compile(r"^(?P<method>.+?)(?P<scan>\d{3})_l(?P<light>\d+)\.ply$")


class DTUAdapter:
    """Adapter over a DTU MVS dataset root (point-cloud geometry)."""

    name = "dtu"
    native_unit = "mm"

    def __init__(self, root: Path, *, method: str | None = None, light: int = 3) -> None:
        self.root = Path(root)
        self.method = method
        self.light = light
        self.capabilities = DatasetCapabilities(
            dense_geometry=True,
            independent_gt=True,
            official_local_eval=True,
            official_local_eval_method="validated_official_port",
            supports_full_scene_geometry=False,
            supports_object_centric_geometry=True,
            notes=[
                "GT is a laser-scanned point cloud in millimetres (independent, dense surface).",
                "Official-like ObsMask/Plane evaluation via the validated dtu_eval Python port "
                "(dtu_official_like_pointcloud); eval3r-native protocols skip ObsMask/Plane.",
            ],
        )

    @classmethod
    def factory(cls, root: Path | None) -> DTUAdapter:
        if root is None:
            raise DatasetError(
                "the 'dtu' dataset adapter requires the DTU dataset root (pass --root). "
                "Expected <root>/Points/stl/stl<NNN>_total.ply and <root>/ObsMask/."
            )
        return cls(Path(root))

    # --- discovery -------------------------------------------------------------

    def _scan_int(self, scene_id: str) -> int:
        try:
            return int(str(scene_id).lstrip("scan").lstrip("_") or scene_id)
        except ValueError as exc:
            raise DatasetError(
                f"DTU scene id '{scene_id}' is not a scan number (e.g. '1' or 'scan1')."
            ) from exc

    def iter_scenes(self, split: str) -> list[str]:
        split_file = self.root / "splits" / f"{split}.txt"
        if not split_file.is_file():
            raise DatasetError(
                f"DTU split '{split}' not found: expected a scan-id list at {split_file}."
            )
        scenes = read_split_file(split_file)
        if not scenes:
            raise DatasetError(f"DTU split file {split_file} lists no scans.")
        return scenes

    # --- ground truth / masks --------------------------------------------------

    def _gt_path(self, scan: int) -> Path:
        # Real DTU releases store GT under groundtruth/; the SampleSet uses Points/stl/.
        name = f"stl{scan:03d}_total.ply"
        candidates = [self.root / "groundtruth" / name, self.root / "Points" / "stl" / name]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]  # primary path for the (missing-file) error message

    def _obs_mask_path(self, scan: int) -> Path:
        return self.root / "ObsMask" / f"ObsMask{scan}_10.mat"

    def _plane_path(self, scan: int) -> Path:
        return self.root / "ObsMask" / f"Plane{scan}.mat"

    def load_scene(self, scene_id: str) -> SceneData:
        scan = self._scan_int(scene_id)
        gt_path = self._gt_path(scan)
        if not gt_path.is_file():
            raise DatasetError(
                f"DTU GT point cloud for scan {scan} is missing: expected {gt_path}."
            )
        obs_mask = self._obs_mask_path(scan)
        plane = self._plane_path(scan)
        plane_available = plane.is_file()

        masks: dict[str, Path] = {}
        if obs_mask.is_file():
            masks["obs_mask"] = obs_mask
        if plane_available:
            masks["plane"] = plane

        return SceneData(
            scene_id=scene_id,
            dataset=self.name,
            gt_pointcloud=gt_path,
            masks=masks,
            ground_truth=self.load_ground_truth(scene_id, protocol=None),  # type: ignore[arg-type]
            capabilities=self.capabilities,
            metadata={
                "scan": scan,
                "native_unit": self.native_unit,
                "obs_mask_available": obs_mask.is_file(),
                "plane_available": plane_available,
                # Recorded explicitly so official-like fidelity (task 010) cannot be
                # claimed for a scan whose Plane file is absent.
                "plane_path": str(plane) if plane_available else None,
            },
        )

    def load_ground_truth(self, scene_id: str, protocol: EvalProtocol) -> GroundTruthSpec:
        scan = self._scan_int(scene_id)
        gt_path = self._gt_path(scan)
        return GroundTruthSpec(
            modality="pointcloud",
            provenance="laser_scan",
            independence="independent",
            density="dense_surface",
            path=gt_path,
            fingerprint=file_fingerprint(gt_path),
            unit=self.native_unit,
        )

    def gt_fingerprint(self, scene_id: str, protocol: EvalProtocol) -> str | None:
        scan = self._scan_int(scene_id)
        parts = [file_fingerprint(self._gt_path(scan))]
        parts.append(file_fingerprint(self._obs_mask_path(scan)))
        parts.append(file_fingerprint(self._plane_path(scan)))
        present = [p for p in parts if p is not None]
        if not present:
            return None
        # Joint fingerprint over GT + ObsMask + Plane (missing files omitted).
        import hashlib

        digest = hashlib.sha256("|".join(present).encode()).hexdigest()
        return f"sha256:{digest}"

    # --- visibility (ObsMask / Plane) ------------------------------------------

    def load_visibility_data(self, scene_id: str, protocol: EvalProtocol) -> dict[str, Any]:
        """Load ObsMask (observability volume) and Plane (ground cull) for a scan.

        Returns ``obs_mask`` / ``bb`` / ``res`` / ``plane`` for the official-like
        evaluator. A missing ObsMask fails explicitly (it is required); a missing
        Plane is returned as ``None`` and recorded, so the official protocol's failure
        policy decides what happens rather than silently skipping the cull.
        """
        from scipy.io import loadmat

        scan = self._scan_int(scene_id)
        obs_path = self._obs_mask_path(scan)
        if not obs_path.is_file():
            raise DatasetError(
                f"DTU ObsMask for scan {scan} is missing: expected {obs_path}. The official-like "
                f"protocol requires ObsMask; use an eval3r-native protocol if it is unavailable."
            )
        mat = loadmat(str(obs_path))
        for key in ("ObsMask", "BB", "Res"):
            if key not in mat:
                raise DatasetError(f"DTU ObsMask file {obs_path} has no '{key}' variable.")

        plane_path = self._plane_path(scan)
        plane = None
        if plane_path.is_file():
            plane_mat = loadmat(str(plane_path))
            if "P" not in plane_mat:
                raise DatasetError(f"DTU Plane file {plane_path} has no 'P' variable.")
            plane = plane_mat["P"].astype(float).reshape(4)

        return {
            "obs_mask": mat["ObsMask"],
            "bb": mat["BB"],
            "res": float(np.asarray(mat["Res"]).reshape(-1)[0]),
            "plane": plane,
            "plane_available": plane is not None,
            "obs_mask_path": str(obs_path),
            "plane_path": str(plane_path) if plane is not None else None,
        }

    # --- predictions -----------------------------------------------------------

    def resolve_prediction(
        self,
        pred_root: Path,
        scene_id: str,
        manifest: PredictionManifest | None,
    ) -> Reconstruction:
        pred_root = Path(pred_root)
        scan = self._scan_int(scene_id)

        rel: Path | None = None
        if manifest is not None and scene_id in manifest.scenes:
            entry = manifest.scenes[scene_id]
            if entry.pointcloud is not None:
                rel = entry.pointcloud
        path = None
        if rel is not None:
            path = rel if rel.is_absolute() else pred_root / rel
        else:
            path = self._infer_prediction(pred_root, scan)

        if path is None or not path.is_file():
            raise DatasetError(
                f"DTU prediction for scan {scan} not found under {pred_root}. Expected a file "
                f"like <method>{scan:03d}_l{self.light}.ply (the l{self.light} light-condition "
                f"suffix must not be dropped), or a manifest entry for scene '{scene_id}'."
            )

        light = _PRED_RE.match(path.name)
        return Reconstruction(
            path=path,
            modality="pointcloud",
            coordinate_frame="world",
            scale="metric",
            unit=self.native_unit,  # DTU predictions are in the mm model frame
            metadata={
                "scan": scan,
                "light_condition": int(light.group("light")) if light else self.light,
                "method": light.group("method") if light else self.method,
            },
        )

    def _infer_prediction(self, pred_root: Path, scan: int) -> Path | None:
        """Find ``<method><scan:03d>_l<light>.ply`` honouring the light suffix."""
        if self.method is not None:
            candidate = pred_root / f"{self.method}{scan:03d}_l{self.light}.ply"
            return candidate if candidate.is_file() else None
        # No method pinned: match any method with the right scan + light suffix.
        matches = []
        for path in sorted(pred_root.glob(f"*{scan:03d}_l*.ply")):
            m = _PRED_RE.match(path.name)
            if m and int(m.group("scan")) == scan:
                matches.append(path)
        if not matches:
            return None
        # Prefer the configured light condition when several exist.
        for path in matches:
            m = _PRED_RE.match(path.name)
            if m and int(m.group("light")) == self.light:
                return path
        return matches[0]

    # --- local evaluation ------------------------------------------------------

    def local_evaluation(self, split: str, protocol: EvalProtocol) -> LocalEvaluationSpec:
        return LocalEvaluationSpec(
            status="supported",
            reason="DTU laser-scan GT is public; point-cloud geometry is locally evaluable.",
            public_gt_available=True,
        )
