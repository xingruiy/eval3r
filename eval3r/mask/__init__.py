"""Occlusion-mask generation utilities.

Two volumetric generators — :func:`from_depth` (sensor depth) and
:func:`from_rendered` (rendered GT mesh) — both return the same
:class:`OcclusionMask` consumed by :mod:`eval3r.mask.occlusion`.
"""

from __future__ import annotations

from eval3r.mask.base import CropToGT, GeometryMask
from eval3r.mask.crop import CropVolume, crop_points_inside, load_crop_volume_json
from eval3r.mask.generate import from_depth, from_rendered
from eval3r.mask.occlusion import (
    OcclusionMask,
    filter_visible_points,
    load_occlusion_mask,
    save_occlusion_mask,
)

__all__ = [
    "from_depth",
    "from_rendered",
    "GeometryMask",
    "CropToGT",
    "CropVolume",
    "crop_points_inside",
    "load_crop_volume_json",
    "OcclusionMask",
    "filter_visible_points",
    "load_occlusion_mask",
    "save_occlusion_mask",
]
