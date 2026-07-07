"""Normalize stage: convert loaded geometry to the internal metre convention.

eval3r evaluates in metres internally (``.agent/datasets.md``). A dataset stored in
millimetres (DTU) or centimetres must be scaled before alignment and metrics, and the
source unit must be recorded, never silently converted. This stage applies a single
uniform unit scale; coordinate-frame/pose normalization for camera datasets is added
by the adapters that need it (tasks 011+).
"""

from __future__ import annotations

import numpy as np

from eval3r.core.errors import DatasetError
from eval3r.core.pose_convention import (
    INTERNAL_WORLD_AXES,
    PoseConventionTransform,
)
from eval3r.core.types import WorldAxes
from eval3r.pipeline.stages.load import LoadedGeometry

_WORLD_TRANSFORM = PoseConventionTransform()

# Length units eval3r knows how to convert to metres. Keys are lower-cased.
_UNIT_TO_METERS: dict[str, float] = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
}


def unit_to_meters(unit: str | None) -> float:
    """Return the scale that converts ``unit`` to metres (``None`` -> 1.0)."""
    if unit is None:
        return 1.0
    key = unit.strip().lower()
    if key not in _UNIT_TO_METERS:
        raise DatasetError(
            f"unknown length unit '{unit}'. Known units: {', '.join(sorted(_UNIT_TO_METERS))}. "
            f"Record the dataset's native unit explicitly so it can be normalized to metres."
        )
    return _UNIT_TO_METERS[key]


def scale_matrix(scale: float) -> np.ndarray:
    """4x4 uniform-scale transform (reused by :meth:`LoadedGeometry.transformed`)."""
    matrix = np.eye(4)
    matrix[0, 0] = matrix[1, 1] = matrix[2, 2] = scale
    return matrix


def normalize_to_meters(geometry: LoadedGeometry, unit: str | None) -> LoadedGeometry:
    """Scale ``geometry`` from ``unit`` to metres (a no-op when already metric)."""
    scale = unit_to_meters(unit)
    if scale == 1.0:
        return geometry
    return geometry.transformed(scale_matrix(scale))


def normalize_world_frame(
    geometry: LoadedGeometry, world_frame: WorldAxes
) -> LoadedGeometry:
    """Rotate ``geometry`` from its declared world frame to eval3r's internal frame.

    A no-op when ``world_frame`` is already the internal (``opencv``) frame; otherwise a
    single global involution ``F`` (a proper 180 degree rotation about X, never a
    reflection) maps the vertices into the internal frame. This is the geometry
    counterpart of the pose "convert" stage: it is applied to a *prediction* built in an
    OpenGL world frame before alignment/masking/sampling/metrics, so a declared
    world-frame mismatch is fixed deterministically rather than left to alignment. GT is
    already internal (adapters normalize it), so only the prediction side is passed here.
    """
    if world_frame == INTERNAL_WORLD_AXES:
        return geometry
    transform = _WORLD_TRANSFORM.world_transform(world_frame, INTERNAL_WORLD_AXES)
    return geometry.transformed(transform)
