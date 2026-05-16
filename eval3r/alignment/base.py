"""Closed-form Umeyama alignment for scale / SE(3) / Sim(3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from eval3r.utils.errors import AlignmentError
from eval3r.utils.typing import Points

AlignMode = Literal["scale", "se3", "sim3"]


class IdentityAligner:
    """No-op aligner — returns an identity AlignResult."""

    def align(self, source: Points, target: Points) -> AlignResult:
        return AlignResult(
            scale=1.0,
            rotation=np.eye(3),
            translation=np.zeros(3),
            mode="none",
        )


@dataclass
class AlignResult:
    """Result of estimating ``aligned = scale * R @ source.T + t.T``."""

    scale: float
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    mode: str
    matched_pred_idx: np.ndarray | None = None
    matched_gt_idx: np.ndarray | None = None
    pred_poses: np.ndarray | None = None
    gt_poses: np.ndarray | None = None
    pred_convention: str = "unspecified"
    gt_convention: str = "unspecified"

    def transform(self, points: Points) -> Points:
        pts = np.asarray(points, dtype=np.float64)
        return (self.scale * pts @ self.rotation.T) + self.translation


def umeyama(source: Points, target: Points, *, mode: AlignMode) -> AlignResult:
    """Estimate the alignment that maps ``source`` onto ``target``.

    Both arrays must have shape (N, 3) with N >= 3 and matching N.
    """
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if src.shape != tgt.shape or src.ndim != 2 or src.shape[1] != 3:
        raise AlignmentError(
            f"umeyama: shapes must match and be (N, 3); got {src.shape} and {tgt.shape}"
        )
    if src.shape[0] < 3:
        raise AlignmentError(f"umeyama: need >= 3 correspondences, got {src.shape[0]}")

    mu_s = src.mean(axis=0)
    mu_t = tgt.mean(axis=0)
    src_c = src - mu_s
    tgt_c = tgt - mu_t

    var_s = float((src_c**2).sum() / src.shape[0])
    cov = (tgt_c.T @ src_c) / src.shape[0]

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    if mode == "se3":
        scale = 1.0
    elif mode == "sim3":
        if var_s <= 0:
            raise AlignmentError("umeyama: source has zero variance, cannot estimate scale")
        scale = float((D * np.diag(S)).sum() / var_s)
    elif mode == "scale":
        # scale-only: keep R = I; scale is the ratio of stds along principal axes.
        if var_s <= 0:
            raise AlignmentError("scale alignment requires non-degenerate source")
        var_t = float((tgt_c**2).sum() / tgt.shape[0])
        scale = float(np.sqrt(max(var_t, 0.0) / var_s))
        R = np.eye(3)
    else:
        raise AlignmentError(f"umeyama: unknown mode '{mode}'")

    t = mu_t - scale * R @ mu_s
    return AlignResult(scale=scale, rotation=R, translation=t, mode=mode)
