"""Abstract base class for prediction-point filters."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseFilter(ABC):
    """Abstract base class for prediction-point filters used by geometry evaluation."""

    @abstractmethod
    def filter_points(self, points: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Return ``(kept_points, n_kept, n_total)`` for ``points``."""


__all__ = ["BaseFilter"]
