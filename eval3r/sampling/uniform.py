"""Uniform (vertex-subset / random-subset) sampler."""
from __future__ import annotations

import numpy as np

from eval3r.io.geometry import MeshData, PointCloudData
from eval3r.sampling.base import PointSampler
from eval3r.utils.errors import EmptyGeometryError
from eval3r.utils.typing import Points


class UniformSampler(PointSampler):
    """Uniform random-subset sampler.

    For :class:`~eval3r.io.geometry.MeshData`, draws from mesh *vertices*.
    For :class:`~eval3r.io.geometry.PointCloudData` and raw arrays, draws
    a random subset without replacement.
    """

    def sample(
        self,
        geom: MeshData | PointCloudData | np.ndarray,
        n: int,
        *,
        seed: int = 42,
    ) -> Points:
        rng = np.random.default_rng(seed)
        if isinstance(geom, MeshData):
            pts = geom.vertices
        elif isinstance(geom, PointCloudData):
            pts = geom.points
        else:
            pts = np.asarray(geom, dtype=np.float64)
        if len(pts) == 0:
            raise EmptyGeometryError("Cannot sample from an empty point set")
        if n >= len(pts):
            return pts.astype(np.float64)
        idx = rng.choice(len(pts), size=n, replace=False)
        return pts[idx].astype(np.float64)


__all__ = ["UniformSampler"]
