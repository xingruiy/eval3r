"""Geometry metrics: chamfer (4 variants), accuracy, completeness, precision/recall/F-score."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.spatial import cKDTree

from eval3r.alignment import AlignMode, align
from eval3r.filtering.base import BaseFilter
from eval3r.io.geometry import MeshData, PointCloudData
from eval3r.metrics.base import GeometryMetric
from eval3r.sampling import SampleMethod, sample_points
from eval3r.utils.errors import EmptyGeometryError
from eval3r.utils.typing import Points, Poses

ChamferVariant = Literal[
    "l1_mean_bidirectional",
    "l1_sum_bidirectional",
    "l2_squared",
    "l2_unsquared",
]


def _nn_dists(a: Points, b: Points) -> np.ndarray:
    """Euclidean nearest-neighbour distance from each row of a to its nearest in b."""
    if len(a) == 0 or len(b) == 0:
        raise EmptyGeometryError("Cannot compute nearest-neighbour distances with empty inputs")
    tree = cKDTree(b)
    d, _ = tree.query(a, k=1)
    return d


class ChamferDistance(GeometryMetric):
    """Bidirectional Chamfer distance with decomposed configuration.

    - ``bidirectional``: include both pred→gt and gt→pred directions
    - ``squared``: square the NN distances before averaging (like l2_squared)
    - ``reduction``: ``"mean"`` halves the bidirectional sum (l1_mean); ``"sum"`` does not

    Default (``bidirectional=True, squared=False, reduction="mean"``) matches the
    former ``l1_mean_bidirectional`` variant: ``0.5 * (mean(d_pg) + mean(d_gp))``.
    """

    name = "chamfer"

    def __init__(
        self,
        *,
        bidirectional: bool = True,
        squared: bool = False,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> None:
        self.bidirectional = bidirectional
        self.squared = squared
        self.reduction = reduction

    def __call__(
        self,
        pred: Points,
        gt: Points,
        *,
        d_pg: np.ndarray | None = None,
        d_gp: np.ndarray | None = None,
    ) -> float:
        if d_pg is None:
            d_pg = _nn_dists(pred, gt)
        if not self.bidirectional:
            return float((d_pg ** 2).mean() if self.squared else d_pg.mean())
        if d_gp is None:
            d_gp = _nn_dists(gt, pred)
        val = (
            (d_pg ** 2).mean() + (d_gp ** 2).mean()
            if self.squared
            else d_pg.mean() + d_gp.mean()
        )
        return float(val / 2.0 if self.reduction == "mean" else val)


class Accuracy(GeometryMetric):
    """Mean nearest-neighbour distance from pred to gt."""

    name = "accuracy"

    def __call__(
        self,
        pred: Points,
        gt: Points,
        *,
        d_pg: np.ndarray | None = None,
        d_gp: np.ndarray | None = None,
    ) -> float:
        if d_pg is None:
            d_pg = _nn_dists(pred, gt)
        return float(d_pg.mean())


class Completeness(GeometryMetric):
    """Mean nearest-neighbour distance from gt to pred."""

    name = "completeness"

    def __call__(
        self,
        pred: Points,
        gt: Points,
        *,
        d_pg: np.ndarray | None = None,
        d_gp: np.ndarray | None = None,
    ) -> float:
        if d_gp is None:
            d_gp = _nn_dists(gt, pred)
        return float(d_gp.mean())


class Precision(GeometryMetric):
    """Fraction of pred points within ``threshold`` of their nearest gt point."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.name = f"precision@{threshold}"

    def __call__(
        self,
        pred: Points,
        gt: Points,
        *,
        d_pg: np.ndarray | None = None,
        d_gp: np.ndarray | None = None,
    ) -> float:
        if d_pg is None:
            d_pg = _nn_dists(pred, gt)
        return float((d_pg < self.threshold).mean())


class Recall(GeometryMetric):
    """Fraction of gt points within ``threshold`` of their nearest pred point."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.name = f"recall@{threshold}"

    def __call__(
        self,
        pred: Points,
        gt: Points,
        *,
        d_pg: np.ndarray | None = None,
        d_gp: np.ndarray | None = None,
    ) -> float:
        if d_gp is None:
            d_gp = _nn_dists(gt, pred)
        return float((d_gp < self.threshold).mean())


class FScore(GeometryMetric):
    """F-score at ``threshold``, returning ``(f, precision, recall)``.

    When precision and recall are both zero (no point within threshold in
    either direction) the F-score is mathematically undefined; this returns
    ``float('nan')`` instead of silently reporting ``0.0`` so pathological
    evaluations are distinguishable from legitimately-bad ones.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.name = f"fscore@{threshold}"

    def __call__(
        self,
        pred: Points,
        gt: Points,
        *,
        d_pg: np.ndarray | None = None,
        d_gp: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        if d_pg is None:
            d_pg = _nn_dists(pred, gt)
        if d_gp is None:
            d_gp = _nn_dists(gt, pred)
        p = float((d_pg < self.threshold).mean())
        r = float((d_gp < self.threshold).mean())
        f = 2 * p * r / (p + r) if (p + r) > 0 else float("nan")
        return float(f), float(p), float(r)


@dataclass
class EvalResult3D:
    chamfer: float
    chamfer_variant: ChamferVariant
    accuracy: float
    completeness: float
    fscore: dict[float, dict[str, float]] = field(default_factory=dict)
    samples: int = 0
    seed: int = 0
    sample_method: SampleMethod = "area"
    align_mode: AlignMode = "none"
    align_scale: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)
    masked: bool = False
    visible_points: int = 0
    total_pred_points: int = 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "chamfer": self.chamfer,
            "chamfer_variant": self.chamfer_variant,
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "fscore": {
                f"{thr}": {"f": v["f"], "precision": v["precision"], "recall": v["recall"]}
                for thr, v in sorted(self.fscore.items())
            },
            "samples": self.samples,
            "seed": self.seed,
            "sample_method": self.sample_method,
            "align_mode": self.align_mode,
            "align_scale": self.align_scale,
        }
        if self.masked:
            d["masked"] = True
            d["visible_points"] = self.visible_points
            d["total_pred_points"] = self.total_pred_points
        d.update(self.extra)
        return d


def evaluate_geometry(
    pred: Points | PointCloudData | MeshData,
    gt: Points | PointCloudData | MeshData,
    *,
    samples: int = 200_000,
    seed: int = 42,
    sample_method: SampleMethod = "area",
    align_mode: AlignMode = "none",
    thresholds: list[float] | tuple[float, ...] = (0.05,),
    chamfer_variant: ChamferVariant = "l1_mean_bidirectional",
    debug_plot_path: str | None = None,
    pred_poses: Poses | None = None,
    gt_poses: Poses | None = None,
    pred_convention: str = "unspecified",
    gt_convention: str = "unspecified",
    pred_timestamps: np.ndarray | None = None,
    gt_timestamps: np.ndarray | None = None,
    pred_mask: BaseFilter | None = None,
) -> EvalResult3D:
    """Sample → align → compute chamfer / accuracy / completeness / F-score.

    When *debug_plot_path* is set, a 3D scatter plot of the aligned
    point clouds is saved to that path.

    Trajectory-based alignment (``traj_se3`` / ``traj_sim3``) requires
    *pred_poses* and *gt_poses* as ``(T, 4, 4)`` arrays with their
    respective conventions.

    If *pred_mask* is provided, it filters the aligned prediction points
    after alignment and before metric computation. The mask is interpreted
    in the post-alignment frame; GT points are not filtered.
    """

    # Step 1 - Sample points
    pred_pts = sample_points(pred, samples, method=sample_method, seed=seed)
    gt_pts = sample_points(gt, samples, method=sample_method, seed=seed + 1)

    # Step 2 - (Optional) Align points
    al = align(
        pred_pts, gt_pts, mode=align_mode,
        pred_poses=pred_poses, gt_poses=gt_poses,
        pred_convention=pred_convention, gt_convention=gt_convention,
        pred_timestamps=pred_timestamps, gt_timestamps=gt_timestamps,
    )
    pred_aligned = al.transform(pred_pts) if align_mode != "none" else pred_pts

    n_visible = len(pred_aligned)
    total_pred_points = len(pred_aligned)

    # Step 3 - (Optional) Filter points
    if pred_mask is not None:
        pred_aligned, n_visible, total_pred_points = pred_mask.filter_points(pred_aligned)

    if debug_plot_path is not None:
        from eval3r.utils.debug_plot import save_debug_plot

        save_debug_plot(
            gt_pts,
            pred_aligned,
            debug_plot_path,
            align_mode,
            al.scale,
            rotation=al.rotation,
            translation=al.translation,
            pred_poses=pred_poses,
            gt_poses=gt_poses,
            pred_convention=pred_convention,
            gt_convention=gt_convention,
            matched_pred_idx=al.matched_pred_idx,
            matched_gt_idx=al.matched_gt_idx,
        )

    # Step 4 - Compute metrics (NN distances computed once, reused across all metrics)
    d_pg = _nn_dists(pred_aligned, gt_pts)
    d_gp = _nn_dists(gt_pts, pred_aligned)

    acc = float(d_pg.mean())
    comp = float(d_gp.mean())

    if chamfer_variant == "l1_mean_bidirectional":
        cd = float(0.5 * (d_pg.mean() + d_gp.mean()))
    elif chamfer_variant == "l1_sum_bidirectional":
        cd = float(d_pg.mean() + d_gp.mean())
    elif chamfer_variant == "l2_squared":
        cd = float((d_pg**2).mean() + (d_gp**2).mean())
    elif chamfer_variant == "l2_unsquared":
        cd = float(d_pg.mean() + d_gp.mean())
    else:
        raise ValueError(f"Unknown chamfer variant: {chamfer_variant!r}")

    fdict: dict[float, dict[str, float]] = {}
    for thr in thresholds:
        p = float((d_pg < thr).mean())
        r = float((d_gp < thr).mean())
        # F-score is undefined when both p and r are zero; emit NaN so callers
        # can distinguish "no points within threshold" from a true 0.0 score.
        f = 2 * p * r / (p + r) if (p + r) > 0 else float("nan")
        fdict[float(thr)] = {"f": f, "precision": p, "recall": r}

    return EvalResult3D(
        chamfer=cd,
        chamfer_variant=chamfer_variant,
        accuracy=acc,
        completeness=comp,
        fscore=fdict,
        samples=samples,
        seed=seed,
        sample_method=sample_method,
        align_mode=align_mode,
        align_scale=al.scale,
        masked=pred_mask is not None,
        visible_points=n_visible,
        total_pred_points=total_pred_points,
    )
