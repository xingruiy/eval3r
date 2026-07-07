"""Shared enumerations for eval3r, transcribed from ``.agent/schema.md``.

These are ``Literal`` type aliases rather than ``enum.Enum`` so that YAML/JSON
values map directly to the string literals the schema documents, and Pydantic
validates them without a conversion layer. ``.agent/schema.md`` is the source of
truth; keep these lists in exact sync with it.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

PredictionModality: TypeAlias = Literal[
    "mesh",
    "pointcloud",
    "pointmap",
    "single_depth",
    "depth_sequence",
    "camera_trajectory",
    "colmap_reconstruction",
]

GroundTruthModality: TypeAlias = Literal[
    "mesh",
    "pointcloud",
    "depth",
    "depth_sequence",
    "trajectory",
    "lidar",
    "server_only",
]

GTProvenance: TypeAlias = Literal[
    "laser_scan",
    "synthetic_exact",
    "reconstructed",
    "lidar_sparse",
    "sensor_depth",
    "server_only",
    "unknown",
]

GTIndependence: TypeAlias = Literal[
    "independent",
    "reconstruction_derived",
    "sensor_derived",
    "server_only",
    "unknown",
]

GTDensity: TypeAlias = Literal[
    "dense_surface",
    "sparse_lidar",
    "object_pointcloud",
    "depth_image",
    "trajectory_only",
    "server_only",
    "unknown",
]

SourcePoseFormat: TypeAlias = Literal[
    "cam_to_world_opencv",
    "cam_to_world_opengl",
    "world_to_cam_opencv",
    "world_to_cam_opengl",
    "world_to_cam_colmap",
    "world_to_cam_mvsnet",
    "tanks_temples_log",
    "co3d_frame_annotations",
    "kitti360_cam0_to_world",
    "tum",
    "unknown",
]

NormalizedConvention: TypeAlias = Literal[
    "cam_to_world_opencv_meters",
]

# World-frame (handedness) convention of geometry vertices — mesh / point-cloud /
# pointmap. Distinct from a camera-pose convention: it labels the world axes the
# geometry lives in, not a per-camera pose. eval3r's internal world frame is
# ``opencv``; an ``opengl`` prediction is flipped to it before geometry metrics.
WorldAxes: TypeAlias = Literal[
    "opencv",
    "opengl",
]

ScaleType: TypeAlias = Literal[
    "metric",
    "relative",
    "unknown",
]

Fidelity: TypeAlias = Literal[
    "official",
    "official_like",
    "eval3r_native",
    "server_only",
]

LocalEvaluationStatus: TypeAlias = Literal[
    "supported",
    "server_only",
    "missing_public_gt",
    "requires_external_assets",
    "requires_external_renderer",
    "unsupported",
]

OfficialLocalEvalMethod: TypeAlias = Literal[
    "none",
    "official_script_wrapper",
    "validated_official_port",
    "matlab_official_script",
    "server_only",
]

IntrinsicsSource: TypeAlias = Literal[
    "predicted",
    "gt",
    "dataset_default",
    "unknown",
]
