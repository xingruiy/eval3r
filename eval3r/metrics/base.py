"""Abstract base classes for eval3r metrics."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from eval3r.utils.typing import Points


class GeometryMetric(ABC):
    name: str = ""

    @abstractmethod
    def __call__(self, pred: Points, gt: Points) -> float | tuple[float, ...]: ...


class DepthMetric(ABC):
    @abstractmethod
    def __call__(
        self,
        pred: NDArray[np.floating],
        gt: NDArray[np.floating],
        mask: NDArray[np.bool_] | None = None,
    ) -> float: ...


__all__ = ["GeometryMetric", "DepthMetric"]
