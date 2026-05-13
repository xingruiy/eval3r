"""Abstract base class for deterministic point samplers."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from eval3r.io.geometry import MeshData, PointCloudData
from eval3r.utils.typing import Points


class PointSampler(ABC):
    @abstractmethod
    def sample(
        self,
        geom: MeshData | PointCloudData | np.ndarray,
        n: int,
        *,
        seed: int = 42,
    ) -> Points: ...


__all__ = ["PointSampler"]
