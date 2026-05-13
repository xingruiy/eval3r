"""Deterministic point sampling from meshes and point clouds."""
from __future__ import annotations

from typing import Literal

import numpy as np

from eval3r.io.geometry import MeshData, PointCloudData
from eval3r.sampling.base import PointSampler
from eval3r.sampling.importance import ImportanceSampler
from eval3r.sampling.uniform import UniformSampler
from eval3r.utils.typing import Points

SampleMethod = Literal["area", "vertex", "uniform"]

_IMPORTANCE = ImportanceSampler()
_UNIFORM = UniformSampler()


def sample_points(
    geom: MeshData | PointCloudData | np.ndarray,
    n: int,
    *,
    method: SampleMethod = "area",
    seed: int = 42,
) -> Points:
    """Return ``n`` deterministic samples from ``geom``.

    ``method="area"`` and ``method="uniform"`` use area-weighted barycentric
    sampling for meshes and uniform random subset for point clouds/arrays.
    ``method="vertex"`` draws from mesh vertices directly.
    """
    if method in ("area", "uniform"):
        return _IMPORTANCE.sample(geom, n, seed=seed)
    if method == "vertex":
        return _UNIFORM.sample(geom, n, seed=seed)
    raise ValueError(f"Unknown sample method: {method!r}")


__all__ = [
    "SampleMethod",
    "sample_points",
    "PointSampler",
    "UniformSampler",
    "ImportanceSampler",
]
