"""Pydantic model for the eval3r prediction manifest."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from eval3r._version import FORMAT_VERSION, __version__

MANIFEST_FILENAME = "eval3r_prediction.json"


class Unit(str, Enum):
    M = "m"
    CM = "cm"
    MM = "mm"
    UNSPECIFIED = "unspecified"


class CoordinateSystem(str, Enum):
    COLMAP = "colmap"
    OPENGL = "opengl"
    OPENCV = "opencv"
    UNSPECIFIED = "unspecified"


class PoseConvention(str, Enum):
    T_WC = "T_wc"
    T_CW = "T_cw"
    UNSPECIFIED = "unspecified"


class Artifact(BaseModel):
    """A file referenced by the manifest, with its sha256 hash and a count."""

    path: str
    format: str
    sha256: str
    count: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class GeometrySection(BaseModel):
    mesh: Artifact | None = None
    point_cloud: Artifact | None = None


class TrajectorySection(BaseModel):
    tum: Artifact | None = None
    kitti: Artifact | None = None


class CameraSection(BaseModel):
    intrinsics: Artifact | None = None
    poses: Artifact | None = None


class Manifest(BaseModel):
    eval3r_version: str = __version__
    format_version: str = FORMAT_VERSION
    scene_id: str
    dataset: str
    method: str
    unit: Unit = Unit.UNSPECIFIED
    coordinate_system: CoordinateSystem = CoordinateSystem.UNSPECIFIED
    pose_convention: PoseConvention = PoseConvention.UNSPECIFIED
    geometry: GeometrySection = Field(default_factory=GeometrySection)
    trajectory: TrajectorySection = Field(default_factory=TrajectorySection)
    cameras: CameraSection = Field(default_factory=CameraSection)
    metadata: dict[str, Any] = Field(default_factory=dict)
