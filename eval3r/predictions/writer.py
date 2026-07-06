"""``PredictionWriter``: export predictions into the official eval3r-native layout.

Any reconstruction method can hand its outputs (files, or arrays where eval3r owns
a writer) to :class:`PredictionWriter` and get back a self-contained prediction
directory: ``manifest.yaml`` plus ``<scene_id>/`` directories with canonical
filenames. The manifest is the unchanged task-002 :class:`PredictionManifest`
schema; layout metadata (layout name, eval3r version, per-file sha256
fingerprints) lives in the designated open-ended ``metadata`` dicts.

eval3r builds no geometry here: meshes, depth files, and camera files are
copy-only. Array inputs exist only where eval3r owns a plain writer — point
clouds (PLY via the pointcloud backend), pointmaps and confidence (``.npy``),
and trajectories ((N, 8) ``[t x y z qx qy qz qw]`` rows to TUM text).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import yaml
from pydantic import ValidationError

from eval3r.backends.pointcloud_plyfile import PlyfilePointCloudBackend
from eval3r.core.errors import PredictionLayoutError
from eval3r.core.manifest import (
    ConfidenceManifestSpec,
    PredictionManifest,
    ScenePredictionEntry,
    UsesGTSpec,
)
from eval3r.core.schema import DatasetVariant
from eval3r.core.types import (
    IntrinsicsSource,
    PredictionModality,
    ScaleType,
    SourcePoseFormat,
)
from eval3r.predictions.layout import (
    ARRAY_FIELD_EXTENSIONS,
    DIR_FIELD_NAMES,
    LAYOUT_NAME,
    canonical_relpath,
    field_fingerprint,
    required_field,
)

SceneInput = Path | str | np.ndarray | None


def write_tum_trajectory(rows: np.ndarray, path: Path) -> None:
    """Write an (N, 8) array of ``[timestamp x y z qx qy qz qw]`` rows as TUM text."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 8 or rows.shape[0] == 0:
        raise PredictionLayoutError(
            f"a trajectory array must have shape (N, 8) with rows "
            f"[timestamp x y z qx qy qz qw]; got shape {rows.shape}."
        )
    if not np.isfinite(rows).all():
        raise PredictionLayoutError(
            "a trajectory array must be finite; it contains NaN or Inf values."
        )
    if not (np.diff(rows[:, 0]) > 0).all():
        raise PredictionLayoutError(
            "trajectory timestamps (column 0) must be strictly increasing; TUM "
            "association assumes a time-sorted trajectory."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(f"{value:.9f}" for value in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class PredictionWriter:
    """Write one method's predictions into the eval3r-native prediction layout.

    Usable as a context manager: ``__exit__`` calls :meth:`finalize` when the
    block completed without an exception (a half-written directory is never
    silently given a manifest).
    """

    def __init__(
        self,
        root: Path | str,
        *,
        method: str,
        dataset: str | DatasetVariant,
        prediction_modality: PredictionModality,
        scale: ScaleType,
        coordinate_frame: str,
        variant: str | None = None,
        split: str | None = None,
        source_pose_format: SourcePoseFormat = "unknown",
        intrinsics_source: IntrinsicsSource = "unknown",
        unit: str = "m",
        depth_unit: float | None = None,
        uses_gt: UsesGTSpec | dict[str, bool] | None = None,
        confidence: ConfidenceManifestSpec | dict[str, Any] | None = None,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.yaml"
        if manifest_path.exists():
            raise PredictionLayoutError(
                f"prediction root {self.root} already contains a manifest.yaml — "
                f"PredictionWriter never overwrites an existing export. Remove the "
                f"directory or choose a fresh one."
            )
        # Validates the modality defines a canonical layout (fail before any IO).
        self._required_field = required_field(prediction_modality)

        if isinstance(dataset, DatasetVariant):
            if variant is not None or split is not None:
                raise PredictionLayoutError(
                    "pass either a DatasetVariant or dataset name + variant/split "
                    "strings, not both — which variant/split wins would be ambiguous."
                )
            self._dataset = dataset
        else:
            self._dataset = DatasetVariant(dataset=dataset, variant=variant, split=split)

        self._uses_gt = (
            uses_gt if isinstance(uses_gt, UsesGTSpec) else UsesGTSpec.model_validate(uses_gt or {})
        )
        self._confidence = (
            confidence
            if isinstance(confidence, ConfidenceManifestSpec)
            else ConfidenceManifestSpec.model_validate(confidence or {})
        )
        self._method = method
        self._version = version
        self._modality: PredictionModality = prediction_modality
        self._scale: ScaleType = scale
        self._coordinate_frame = coordinate_frame
        self._source_pose_format: SourcePoseFormat = source_pose_format
        self._intrinsics_source: IntrinsicsSource = intrinsics_source
        self._unit = unit
        self._depth_unit = depth_unit
        self._metadata = dict(metadata or {})
        self._scenes: dict[str, ScenePredictionEntry] = {}
        self._finalized = False
        self._pointcloud_backend = PlyfilePointCloudBackend()

    # -- per-field writers ------------------------------------------------------

    def _copy_file(self, scene_id: str, field: str, source: Path) -> Path:
        if not source.is_file():
            raise PredictionLayoutError(
                f"scene '{scene_id}' field '{field}': source file does not exist: "
                f"{source}."
            )
        rel = canonical_relpath(scene_id, field, source.suffix)
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return rel

    def _copy_dir(self, scene_id: str, field: str, source: Path) -> Path:
        if not source.is_dir():
            raise PredictionLayoutError(
                f"scene '{scene_id}' field '{field}': source directory does not "
                f"exist: {source}."
            )
        if not any(p.is_file() for p in source.rglob("*")):
            raise PredictionLayoutError(
                f"scene '{scene_id}' field '{field}': source directory {source} "
                f"contains no files."
            )
        rel = canonical_relpath(scene_id, field, "")
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        return rel

    def _write_array(self, scene_id: str, field: str, array: np.ndarray) -> Path:
        extension = ARRAY_FIELD_EXTENSIONS.get(field)
        if extension is None:
            raise PredictionLayoutError(
                f"scene '{scene_id}' field '{field}': eval3r owns no writer for this "
                f"field, so it is copy-only — pass a source file path, not an array. "
                f"Array inputs exist for {sorted(ARRAY_FIELD_EXTENSIONS)} only "
                f"(eval3r builds no mesh/depth/camera data)."
            )
        rel = canonical_relpath(scene_id, field, extension)
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if field == "pointcloud":
            points = np.asarray(array, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
                raise PredictionLayoutError(
                    f"scene '{scene_id}' field 'pointcloud': expected a non-empty "
                    f"(N, 3) array, got shape {points.shape}."
                )
            if not np.isfinite(points).all():
                raise PredictionLayoutError(
                    f"scene '{scene_id}' field 'pointcloud': the array contains NaN "
                    f"or Inf values; remove invalid points before export."
                )
            self._pointcloud_backend.save_pointcloud(points, target)
        elif field == "pointmap":
            pointmap = np.asarray(array, dtype=np.float32)
            if pointmap.ndim not in (2, 3) or pointmap.shape[-1] != 3 or pointmap.size == 0:
                raise PredictionLayoutError(
                    f"scene '{scene_id}' field 'pointmap': expected a non-empty "
                    f"(N, 3) or (H, W, 3) array (NaN marks invalid points), got "
                    f"shape {pointmap.shape}."
                )
            np.save(target, pointmap)
        elif field == "trajectory":
            write_tum_trajectory(np.asarray(array), target)
        else:  # confidence
            confidence = np.asarray(array, dtype=np.float32)
            if confidence.size == 0 or not np.isfinite(confidence).all():
                raise PredictionLayoutError(
                    f"scene '{scene_id}' field 'confidence': expected a non-empty "
                    f"finite float array, got shape {confidence.shape}."
                )
            np.save(target, confidence)
        return rel

    def _place(self, scene_id: str, field: str, value: Path | str | np.ndarray) -> Path:
        if isinstance(value, np.ndarray):
            return self._write_array(scene_id, field, value)
        if field in DIR_FIELD_NAMES:
            return self._copy_dir(scene_id, field, Path(value))
        return self._copy_file(scene_id, field, Path(value))

    # -- public API -------------------------------------------------------------

    def add_scene(
        self,
        scene_id: str,
        *,
        mesh: Path | str | None = None,
        pointcloud: SceneInput = None,
        pointmap: SceneInput = None,
        depth: Path | str | None = None,
        depth_dir: Path | str | None = None,
        trajectory: SceneInput = None,
        camera_file: Path | str | None = None,
        confidence: SceneInput = None,
        confidence_dir: Path | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ScenePredictionEntry:
        """Copy/write one scene's files into the layout and record its entry."""
        if self._finalized:
            raise PredictionLayoutError(
                f"cannot add scene '{scene_id}': this PredictionWriter is already "
                f"finalized ({self.root / 'manifest.yaml'} was written)."
            )
        if not scene_id or scene_id in (".", "..") or "/" in scene_id or "\\" in scene_id:
            raise PredictionLayoutError(
                f"scene id {scene_id!r} is not usable as a directory name; scene ids "
                f"must be plain names without path separators."
            )
        if scene_id in self._scenes:
            raise PredictionLayoutError(
                f"scene '{scene_id}' was already added to this prediction directory; "
                f"each scene must be added exactly once."
            )
        scene_dir = self.root / scene_id
        if scene_dir.exists():
            raise PredictionLayoutError(
                f"scene directory {scene_dir} already exists on disk (stale output "
                f"from a previous export?); remove it before writing."
            )

        inputs: dict[str, Path | str | np.ndarray | None] = {
            "mesh": mesh,
            "pointcloud": pointcloud,
            "pointmap": pointmap,
            "depth": depth,
            "depth_dir": depth_dir,
            "trajectory": trajectory,
            "camera_file": camera_file,
            "confidence": confidence,
            "confidence_dir": confidence_dir,
        }
        if inputs[self._required_field] is None:
            raise PredictionLayoutError(
                f"scene '{scene_id}': declared prediction_modality "
                f"'{self._modality}' requires the '{self._required_field}' entry, "
                f"but none was given."
            )
        if (confidence is not None or confidence_dir is not None) and not (
            self._confidence.present
        ):
            raise PredictionLayoutError(
                f"scene '{scene_id}' provides confidence files, but the manifest "
                f"declares confidence.present = false. Declare "
                f"confidence={{'present': True, ...}} on the writer so the result "
                f"can be interpreted."
            )

        entry_fields: dict[str, Any] = {"metadata": dict(metadata or {})}
        fingerprints: dict[str, str] = {}
        try:
            for field, value in inputs.items():
                if value is None:
                    continue
                rel = self._place(scene_id, field, value)
                entry_fields[field] = rel
                fingerprint = field_fingerprint(field, self.root / rel)
                if fingerprint is None:  # pragma: no cover - we just wrote the file
                    raise PredictionLayoutError(
                        f"scene '{scene_id}' field '{field}': written file "
                        f"{self.root / rel} vanished before fingerprinting."
                    )
                fingerprints[field] = fingerprint
        except Exception:
            # add_scene is atomic per scene: we refused a pre-existing scene
            # directory above, so anything under it was written by this call and
            # is removed so the scene can be re-added after the caller fixes the
            # input.
            shutil.rmtree(scene_dir, ignore_errors=True)
            raise
        entry_fields["metadata"]["fingerprints"] = fingerprints

        entry = ScenePredictionEntry.model_validate(entry_fields)
        self._scenes[scene_id] = entry
        return entry

    def finalize(self) -> PredictionManifest:
        """Validate the whole manifest and write ``manifest.yaml``."""
        if self._finalized:
            raise PredictionLayoutError(
                f"this PredictionWriter is already finalized "
                f"({self.root / 'manifest.yaml'} was written); finalize() must be "
                f"called exactly once."
            )
        if not self._scenes:
            raise PredictionLayoutError(
                f"no scenes were added to {self.root}; a prediction directory "
                f"without scenes is not a valid export."
            )
        from eval3r import __version__

        metadata = dict(self._metadata)
        metadata["layout"] = LAYOUT_NAME
        metadata["eval3r_version"] = __version__
        try:
            manifest = PredictionManifest.model_validate(
                {
                    "method": self._method,
                    "version": self._version,
                    "dataset": self._dataset,
                    "prediction_modality": self._modality,
                    "coordinate_frame": self._coordinate_frame,
                    "source_pose_format": self._source_pose_format,
                    "scale": self._scale,
                    "unit": self._unit,
                    "depth_unit": self._depth_unit,
                    "uses_gt": self._uses_gt,
                    "intrinsics_source": self._intrinsics_source,
                    "confidence": self._confidence,
                    "scenes": self._scenes,
                    "metadata": metadata,
                }
            )
        except ValidationError as exc:
            raise PredictionLayoutError(
                f"the assembled prediction manifest for {self.root} failed schema "
                f"validation:\n{exc}"
            ) from exc

        manifest_path = self.root / "manifest.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        self._finalized = True
        return manifest

    def __enter__(self) -> PredictionWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Only finalize a clean export; on error the directory stays manifest-less
        # so it can never be mistaken for a complete prediction export.
        if exc_type is None:
            self.finalize()
