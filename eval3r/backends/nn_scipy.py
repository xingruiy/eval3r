"""scipy ``cKDTree`` nearest-neighbor backend (default).

Exact (non-approximate) nearest-neighbor distances from each query point to the
nearest reference point. Empty or malformed inputs fail explicitly *before* the
backend call (``.agent/backends.md`` NN rules).
"""

from __future__ import annotations

import numpy as np
import scipy
from scipy.spatial import cKDTree

from eval3r.core.errors import InvalidGeometryError
from eval3r.core.registry import BackendInfo


def _validate_points(points: np.ndarray, *, role: str) -> np.ndarray:
    arr = np.asarray(points)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise InvalidGeometryError(
            f"{role} points must be an (N, 3) array; got shape {arr.shape}."
        )
    if arr.shape[0] == 0:
        raise InvalidGeometryError(f"{role} points are empty; cannot run nearest-neighbor search.")
    if not np.isfinite(arr).all():
        raise InvalidGeometryError(f"{role} points contain non-finite (NaN/Inf) values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


class ScipyNNBackend:
    """Nearest-neighbor distances via :class:`scipy.spatial.cKDTree`."""

    name = "scipy"

    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            kind="nearest_neighbor",
            name=self.name,
            library="scipy",
            version=scipy.__version__,
            approximate=False,
        )

    def nearest_distances(
        self,
        query_points: np.ndarray,
        reference_points: np.ndarray,
    ) -> np.ndarray:
        """Distance from each query point to its nearest reference point."""
        query = _validate_points(query_points, role="query")
        reference = _validate_points(reference_points, role="reference")
        tree = cKDTree(reference)
        distances, _ = tree.query(query, k=1)
        return np.asarray(distances, dtype=np.float64)
