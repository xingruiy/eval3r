"""Point-to-point ICP using SciPy cKDTree."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from eval3r.alignment.base import AlignResult, umeyama
from eval3r.utils.typing import Points


class ICPAligner:
    """Iterative Closest Point aligner.

    Args:
        estimate_scale: When True, solves Sim(3) (scale + rotation + translation).
            When False, solves SE(3) (rotation + translation only).
        max_iters: Maximum number of ICP iterations.
        tol: Convergence tolerance on RMSE change.
    """

    def __init__(
        self,
        *,
        estimate_scale: bool = False,
        max_iters: int = 50,
        tol: float = 1e-6,
    ) -> None:
        self.estimate_scale = estimate_scale
        self.max_iters = max_iters
        self.tol = tol

    def align(
        self,
        source: Points,
        target: Points,
        *,
        init: AlignResult | None = None,
    ) -> AlignResult:
        """Iterate nearest-neighbour correspondences and Umeyama until convergence."""
        src = np.asarray(source, dtype=np.float64)
        tgt = np.asarray(target, dtype=np.float64)
        if src.ndim != 2 or src.shape[1] != 3 or tgt.ndim != 2 or tgt.shape[1] != 3:
            raise ValueError("icp: source and target must be (N, 3) and (M, 3) arrays")
        tree = cKDTree(tgt)

        if init is not None:
            R = init.rotation.copy()
            t = init.translation.copy()
            s = init.scale
        else:
            # Centroid-translation init — ICP from identity fails on large offsets.
            R = np.eye(3)
            s = 1.0
            t = tgt.mean(axis=0) - src.mean(axis=0)

        prev_rmse = np.inf
        mode = "sim3" if self.estimate_scale else "se3"
        for _ in range(self.max_iters):
            warped = (s * src @ R.T) + t
            dists, idx = tree.query(warped, k=1)
            result = umeyama(src, tgt[idx], mode=mode)
            R, t, s = result.rotation, result.translation, result.scale
            rmse = float(np.sqrt(np.mean(dists**2)))
            if abs(prev_rmse - rmse) < self.tol:
                break
            prev_rmse = rmse
        return AlignResult(
            scale=s,
            rotation=R,
            translation=t,
            mode=mode,
        )
