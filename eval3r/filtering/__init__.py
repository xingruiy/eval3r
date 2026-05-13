"""Prediction-point filtering utilities.

Three filter implementations — :class:`BBoxFilter`, :class:`OcclusionFilter`,
:class:`PolygonFilter` — all inherit from :class:`BaseFilter` and expose a
common ``filter_points`` interface consumed by geometry evaluation.
"""

from __future__ import annotations

from eval3r.filtering.base import BaseFilter
from eval3r.filtering.bbox import BBoxFilter
from eval3r.filtering.generate import from_depth, from_rendered
from eval3r.filtering.occlusion import (
    OcclusionFilter,
    filter_visible_points,
    load_occlusion_mask,
    save_occlusion_mask,
)
from eval3r.filtering.polygon import PolygonFilter, crop_points_inside, load_crop_volume_json

__all__ = [
    "from_depth",
    "from_rendered",
    "BaseFilter",
    "BBoxFilter",
    "OcclusionFilter",
    "PolygonFilter",
    "crop_points_inside",
    "load_crop_volume_json",
    "filter_visible_points",
    "load_occlusion_mask",
    "save_occlusion_mask",
]
