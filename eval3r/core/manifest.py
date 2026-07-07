"""Prediction manifest models, transcribed from ``.agent/schema.md``.

The manifest describes a method's predictions for a dataset: modality, coordinate
conventions, whether GT was used, confidence metadata, and the per-scene file
entries. Resolution against a concrete dataset layout is task 008; this module
owns only the schema.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from eval3r.core.schema import DatasetVariant, E3RModel
from eval3r.core.types import (
    IntrinsicsSource,
    NormalizedConvention,
    PredictionModality,
    ScaleType,
    SourcePoseFormat,
    WorldAxes,
)


class UsesGTSpec(E3RModel):
    pose: bool = False
    intrinsics: bool = False
    scale: bool = False


class ConfidenceManifestSpec(E3RModel):
    present: bool = False
    native_threshold: float | None = None
    self_filtered: bool = False


class ScenePredictionEntry(E3RModel):
    mesh: Path | None = None
    pointcloud: Path | None = None
    pointmap: Path | None = None
    depth: Path | None = None
    depth_dir: Path | None = None
    trajectory: Path | None = None
    camera_file: Path | None = None
    confidence: Path | None = None
    confidence_dir: Path | None = None
    metadata: dict = {}


class PredictionManifest(E3RModel):
    method: str
    version: str | None = None
    dataset: DatasetVariant
    prediction_modality: PredictionModality
    coordinate_frame: str
    source_pose_format: SourcePoseFormat = "unknown"
    normalized_convention: NormalizedConvention = "cam_to_world_opencv_meters"
    world_frame: WorldAxes = "opencv"
    scale: ScaleType
    unit: str = "m"
    depth_unit: float | None = None
    uses_gt: UsesGTSpec = Field(default_factory=UsesGTSpec)
    intrinsics_source: IntrinsicsSource = "unknown"
    confidence: ConfidenceManifestSpec = Field(default_factory=ConfidenceManifestSpec)
    scenes: dict[str, ScenePredictionEntry]
    metadata: dict = {}
