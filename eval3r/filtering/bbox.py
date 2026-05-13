"""Axis-aligned bounding-box filter using GT mesh bounds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval3r.filtering.base import BaseFilter


@dataclass
class BBoxFilter(BaseFilter):
    """Axis-aligned GT-bounds crop, expanded by ``margin``."""

    bbox_min: np.ndarray
    bbox_max: np.ndarray
    margin: float = 0.0
    source: str = "gt_bbox"

    def filter_points(self, points: np.ndarray) -> tuple[np.ndarray, int, int]:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
        n_total = len(pts)
        lo = np.asarray(self.bbox_min, dtype=np.float64) - float(self.margin)
        hi = np.asarray(self.bbox_max, dtype=np.float64) + float(self.margin)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("bbox_min and bbox_max must have shape (3,)")
        keep = np.all((pts >= lo) & (pts <= hi), axis=1)
        n_kept = int(keep.sum())
        return pts[keep], n_kept, n_total


__all__ = ["BBoxFilter"]
