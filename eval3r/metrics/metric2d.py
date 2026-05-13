"""Depth metrics: AbsRel, SqRel, RMSE, RMSE log, threshold accuracy (δ)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eval3r.metrics.base import DepthMetric
from eval3r.utils.errors import Eval3rError


class EmptyDepthError(Eval3rError, ValueError):
    """Depth input has no valid pixels."""


def _valid_mask(
    pred: NDArray[np.floating],
    gt: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    """Boolean mask of pixels where both pred and gt are > 0 and finite."""
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred and gt must have the same shape, got {pred.shape} and {gt.shape}"
        )
    if mask is not None and mask.shape != pred.shape:
        raise ValueError(
            f"mask must have the same shape as pred/gt, got {mask.shape} and {pred.shape}"
        )
    m = np.isfinite(pred) & (pred > 0) & np.isfinite(gt) & (gt > 0)
    if mask is not None:
        m = m & mask
    if not np.any(m):
        raise EmptyDepthError("No valid pixels in depth input")
    return m


class AbsRel(DepthMetric):
    """Mean absolute relative error: mean(|pred - gt| / gt)."""

    def __call__(
        self,
        pred: NDArray[np.floating],
        gt: NDArray[np.floating],
        mask: NDArray[np.bool_] | None = None,
    ) -> float:
        m = _valid_mask(pred, gt, mask)
        return float(np.mean(np.abs(pred[m] - gt[m]) / gt[m]))


class SqRel(DepthMetric):
    """Mean squared relative error: mean((pred - gt)² / gt)."""

    def __call__(
        self,
        pred: NDArray[np.floating],
        gt: NDArray[np.floating],
        mask: NDArray[np.bool_] | None = None,
    ) -> float:
        m = _valid_mask(pred, gt, mask)
        return float(np.mean((pred[m] - gt[m]) ** 2 / gt[m]))


class RMSE(DepthMetric):
    """Root mean squared error: sqrt(mean((pred - gt)²))."""

    def __call__(
        self,
        pred: NDArray[np.floating],
        gt: NDArray[np.floating],
        mask: NDArray[np.bool_] | None = None,
    ) -> float:
        m = _valid_mask(pred, gt, mask)
        return float(np.sqrt(np.mean((pred[m] - gt[m]) ** 2)))


class RMSELog(DepthMetric):
    """RMSE in log space: sqrt(mean((log(pred) - log(gt))²))."""

    def __call__(
        self,
        pred: NDArray[np.floating],
        gt: NDArray[np.floating],
        mask: NDArray[np.bool_] | None = None,
    ) -> float:
        m = _valid_mask(pred, gt, mask)
        return float(np.sqrt(np.mean((np.log(pred[m]) - np.log(gt[m])) ** 2)))


class DeltaAccuracy(DepthMetric):
    """Fraction of pixels where max(pred/gt, gt/pred) < ``threshold``."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def __call__(
        self,
        pred: NDArray[np.floating],
        gt: NDArray[np.floating],
        mask: NDArray[np.bool_] | None = None,
    ) -> float:
        m = _valid_mask(pred, gt, mask)
        ratio = np.maximum(pred[m] / gt[m], gt[m] / pred[m])
        return float(np.mean(ratio < self.threshold))


@dataclass
class DepthEvalResult:
    abs_rel: float
    sq_rel: float
    rmse: float
    rmse_log: float
    delta1: float  # δ < 1.25
    delta2: float  # δ < 1.25²
    delta3: float  # δ < 1.25³
    valid_pixels: int
    total_pixels: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "abs_rel": self.abs_rel,
            "sq_rel": self.sq_rel,
            "rmse": self.rmse,
            "rmse_log": self.rmse_log,
            "delta1": self.delta1,
            "delta2": self.delta2,
            "delta3": self.delta3,
            "valid_pixels": self.valid_pixels,
            "total_pixels": self.total_pixels,
        }


def depth_metrics(
    pred: NDArray[np.floating],
    gt: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
) -> DepthEvalResult:
    """Compute all standard depth metrics at once."""
    m = _valid_mask(pred, gt, mask)
    n_valid = int(np.sum(m))
    pred_v = pred[m]
    gt_v = gt[m]
    diff = pred_v - gt_v
    ratio = np.maximum(pred_v / gt_v, gt_v / pred_v)

    return DepthEvalResult(
        abs_rel=float(np.mean(np.abs(diff) / gt_v)),
        sq_rel=float(np.mean((diff**2) / gt_v)),
        rmse=float(np.sqrt(np.mean(diff**2))),
        rmse_log=float(np.sqrt(np.mean((np.log(pred_v) - np.log(gt_v)) ** 2))),
        delta1=float(np.mean(ratio < 1.25)),
        delta2=float(np.mean(ratio < 1.25**2)),
        delta3=float(np.mean(ratio < 1.25**3)),
        valid_pixels=n_valid,
        total_pixels=int(pred.size),
    )
