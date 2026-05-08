"""Occlusion-mask generation utilities.

Two volumetric generators — :func:`from_depth` (sensor depth) and
:func:`from_rendered` (rendered GT mesh) — both return the same
:class:`OcclusionMask` consumed by :mod:`eval3r.metrics.occlusion`.
"""

from __future__ import annotations

from eval3r.masks.generate import from_depth, from_rendered
from eval3r.metrics.occlusion import (
    OcclusionMask,
    load_occlusion_mask,
    save_occlusion_mask,
)

__all__ = [
    "from_depth",
    "from_rendered",
    "OcclusionMask",
    "load_occlusion_mask",
    "save_occlusion_mask",
]
