"""Shared Pydantic spec models, transcribed field-for-field from ``.agent/schema.md``.

This module holds the reusable specification models referenced by protocols,
manifests, and results: dataset capabilities/variants, ground-truth and
local-evaluation specs, reconstruction/scene descriptors, and the alignment,
sampling, masking, confidence, metric, aggregation, failure, and reporting specs.

``.agent/schema.md`` is the source of truth. Any field change here must update the
doc, the models, the serialization tests, and the sample fixtures together
(CLAUDE.md schema rules).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from eval3r.core.types import (
    GroundTruthModality,
    GTDensity,
    GTIndependence,
    GTProvenance,
    IntrinsicsSource,
    LocalEvaluationStatus,
    NormalizedConvention,
    OfficialLocalEvalMethod,
    PredictionModality,
    ScaleType,
    SourcePoseFormat,
)


class E3RModel(BaseModel):
    """Base for every eval3r schema model.

    ``extra="forbid"`` makes unexpected fields an explicit validation error rather
    than a silently ignored typo. Each model carries an explicit ``metadata`` (or
    ``parameters``) dict for open-ended, non-schema data, so forbidding stray
    top-level keys costs no flexibility.
    """

    model_config = ConfigDict(extra="forbid")


# --- dataset capabilities and variants -----------------------------------------


class DatasetCapabilities(E3RModel):
    dense_geometry: bool = False
    independent_gt: bool = False
    depth_metric: bool = False
    pose_metric: bool = False
    official_local_eval: bool = False
    official_local_eval_method: OfficialLocalEvalMethod = "none"
    server_only_eval: bool = False
    requires_external_renderer: bool = False
    requires_paid_assets: bool = False
    supports_full_scene_geometry: bool = False
    supports_object_centric_geometry: bool = False
    supports_sparse_lidar_geometry: bool = False
    notes: list[str] = []


class DatasetVariant(E3RModel):
    dataset: str
    variant: str | None = None
    split: str | None = None
    version: str | None = None
    fingerprint: str | None = None
    notes: list[str] = []


# --- ground truth and local evaluation -----------------------------------------


class GroundTruthSpec(E3RModel):
    modality: GroundTruthModality
    provenance: GTProvenance
    independence: GTIndependence
    density: GTDensity
    path: Path | None = None
    fingerprint: str | None = None
    unit: str | None = "m"
    source_pose_format: SourcePoseFormat | None = None
    normalized_convention: NormalizedConvention | None = "cam_to_world_opencv_meters"
    local_evaluation_status: LocalEvaluationStatus = "supported"
    server_url: str | None = None
    notes: list[str] = []


class LocalEvaluationSpec(E3RModel):
    status: LocalEvaluationStatus
    reason: str | None = None
    public_gt_available: bool = True
    external_assets_required: list[str] = []
    official_server_required: bool = False


# --- reconstruction and scene data ---------------------------------------------


class Reconstruction(E3RModel):
    path: Path | None = None
    modality: PredictionModality
    coordinate_frame: str
    source_pose_format: SourcePoseFormat = "unknown"
    normalized_convention: NormalizedConvention = "cam_to_world_opencv_meters"
    scale: ScaleType
    unit: str = "m"
    depth_unit: float | None = None
    cameras: Path | None = None
    intrinsics_source: IntrinsicsSource = "unknown"
    confidence_paths: list[Path] | None = None
    native_confidence_threshold: float | None = None
    metadata: dict = {}


class SceneData(E3RModel):
    scene_id: str
    dataset: str
    variant: str | None = None
    rgb_paths: list[Path] | None = None
    depth_paths: list[Path] | None = None
    camera_paths: list[Path] | None = None
    gt_mesh: Path | None = None
    gt_pointcloud: Path | None = None
    gt_depth_paths: list[Path] | None = None
    gt_trajectory: Path | None = None
    visibility_source: Path | None = None
    masks: dict[str, Path] = {}
    ground_truth: GroundTruthSpec
    capabilities: DatasetCapabilities | None = None
    metadata: dict = {}


# --- alignment -----------------------------------------------------------------


class AlignmentSpec(E3RModel):
    mode: Literal[
        "none",
        "se3",
        "sim3",
        "icp",
        "trajectory_se3",
        "trajectory_sim3",
        "scale_median",
        "scale_least_squares",
        "scale_affine",
    ] = "none"
    estimate_on: Literal[
        "none",
        "trajectory",
        "pointcloud",
        "depth",
        "manual",
    ] = "none"
    solver: Literal[
        "none",
        "umeyama",
        "ransac_umeyama",
        "icp",
        "evo",
        "official_backend",
    ] = "none"
    granularity: Literal["global", "per_scene", "per_sequence", "per_frame"] = "per_scene"
    allow_override: bool = False
    parameters: dict = {}


# --- sampling ------------------------------------------------------------------


class SamplingSideSpec(E3RModel):
    method: Literal[
        "none",
        "surface_area",
        "uniform_points",
        "voxel_downsample",
        "random_points",
        "all_points",
    ] = "all_points"
    n_points: int | None = None
    seed: int | Literal["derive"] | None = "derive"
    voxel_size: float | None = None
    return_normals: bool = False


class SamplingSpec(E3RModel):
    pred: SamplingSideSpec
    gt: SamplingSideSpec


# --- masking and culling -------------------------------------------------------


class CullingSpec(E3RModel):
    method: Literal[
        "none",
        "scene_bounds",
        "visibility_mask",
        "gt_visibility",
        "obs_mask",
        "dataset_official_mask",
        "object_mask",
        "valid_depth",
        "custom",
    ] = "none"
    tolerance: float | None = None
    source: Literal["gt", "pred", "dataset", "custom", "none"] = "none"
    path: Path | None = None
    parameters: dict = {}


class MaskingSpec(E3RModel):
    pred_culling: CullingSpec = Field(default_factory=CullingSpec)
    gt_culling: CullingSpec = Field(default_factory=CullingSpec)
    valid_region: CullingSpec = Field(default_factory=CullingSpec)
    ignore_invalid_depth: bool = True
    invalid_depth_values: list[float] = []


# --- confidence ----------------------------------------------------------------


class ConfidenceSpec(E3RModel):
    policy: Literal[
        "none",
        "method_default",
        "threshold",
        "percentile",
        "top_k_fraction",
    ] = "none"
    threshold: float | None = None
    percentile: float | None = None
    top_k_fraction: float | None = None
    allow_override: bool = False


# --- metrics and aggregation ---------------------------------------------------


class MetricSpec(E3RModel):
    name: str
    threshold: float | None = None
    statistic: Literal["mean", "median", "rmse", "percentile"] | None = None
    percentile: float | None = None
    reduction: Literal["sum", "mean", "none"] | None = None
    clamp: float | None = None
    aggregation: (
        Literal[
            "per_scene_then_mean",
            "global",
            "per_frame_then_scene_then_mean",
        ]
        | None
    ) = None
    parameters: dict = {}


class AggregationSpec(E3RModel):
    per_frame: bool = False
    per_scene: bool = True
    mean: bool = True
    median: bool = True
    weighted_mean: bool = False
    weights: Literal["none", "n_points", "n_pixels", "scene_weight"] = "none"


# --- failure policy and reporting ----------------------------------------------


class FailurePolicySpec(E3RModel):
    policy: Literal["abort", "skip_and_flag", "score_worst"] = "abort"
    worst_values: dict[str, float] = {}


class ReportingSpec(E3RModel):
    save_protocol_copy: bool = True
    save_manifest_copy: bool = True
    save_environment: bool = True
    save_alignment_transforms: bool = True
    save_colored_errors: bool = False
    save_distance_histogram: bool = False
    formats: list[Literal["json", "csv", "markdown", "latex", "html"]] = ["json", "csv"]
