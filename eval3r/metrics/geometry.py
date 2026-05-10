"""Geometry metrics: chamfer (4 variants), accuracy, completeness, precision/recall/F-score."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.spatial import cKDTree

from typing import TYPE_CHECKING

from eval3r.align import AlignMode, align
from eval3r.io.geometry import MeshData, PointCloudData
from eval3r.metrics.sampling import SampleMethod, sample_points
from eval3r.utils.errors import EmptyGeometryError
from eval3r.utils.typing import Points, Poses

if TYPE_CHECKING:
    from eval3r.metrics.occlusion import OcclusionMask

ChamferVariant = Literal[
    "l1_mean_bidirectional",
    "l1_sum_bidirectional",
    "l2_squared",
    "l2_unsquared",
]
MaskMode = Literal["pred", "gt", "both"]


def _nn_dists(a: Points, b: Points) -> np.ndarray:
    """Euclidean nearest-neighbour distance from each row of a to its nearest in b."""
    if len(a) == 0 or len(b) == 0:
        raise EmptyGeometryError("Cannot compute nearest-neighbour distances with empty inputs")
    tree = cKDTree(b)
    d, _ = tree.query(a, k=1)
    return d


def chamfer_distance(
    pred: Points,
    gt: Points,
    *,
    variant: ChamferVariant = "l1_mean_bidirectional",
) -> float:
    """Bidirectional Chamfer with explicit variant.

    - l1_mean_bidirectional: 0.5 * (mean(|p-q|) + mean(|q-p|))
    - l1_sum_bidirectional:        mean(|p-q|) + mean(|q-p|)
    - l2_squared:           mean(|p-q|^2) + mean(|q-p|^2)
    - l2_unsquared:         mean(|p-q|)   + mean(|q-p|)   (= l1_sum_bidirectional)
    """
    d_pg = _nn_dists(pred, gt)
    d_gp = _nn_dists(gt, pred)
    if variant == "l1_mean_bidirectional":
        return float(0.5 * (d_pg.mean() + d_gp.mean()))
    if variant == "l1_sum_bidirectional":
        return float(d_pg.mean() + d_gp.mean())
    if variant == "l2_squared":
        return float((d_pg**2).mean() + (d_gp**2).mean())
    if variant == "l2_unsquared":
        return float(d_pg.mean() + d_gp.mean())
    raise ValueError(f"Unknown chamfer variant: {variant!r}")


def accuracy(pred: Points, gt: Points) -> float:
    """Mean nearest-neighbour distance from pred to gt."""
    return float(_nn_dists(pred, gt).mean())


def completeness(pred: Points, gt: Points) -> float:
    """Mean nearest-neighbour distance from gt to pred."""
    return float(_nn_dists(gt, pred).mean())


def precision_at(pred: Points, gt: Points, threshold: float) -> float:
    return float((_nn_dists(pred, gt) < threshold).mean())


def recall_at(pred: Points, gt: Points, threshold: float) -> float:
    return float((_nn_dists(gt, pred) < threshold).mean())


def fscore_at(pred: Points, gt: Points, threshold: float) -> tuple[float, float, float]:
    p = precision_at(pred, gt, threshold)
    r = recall_at(pred, gt, threshold)
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return float(f), float(p), float(r)


@dataclass
class GeometryEvalResult:
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
    pred_mask: OcclusionMask | None = None,
    gt_mask: OcclusionMask | None = None,
) -> GeometryEvalResult:
    """Sample → align → compute chamfer / accuracy / completeness / F-score.

    When *debug_plot_path* is set, a 3D scatter plot of the aligned
    point clouds is saved to that path.

    Trajectory-based alignment (``traj_se3`` / ``traj_sim3``) requires
    *pred_poses* and *gt_poses* as ``(T, 4, 4)`` arrays with their
    respective conventions.
    """
    pred_pts = sample_points(pred, samples, method=sample_method, seed=seed)
    gt_pts = sample_points(gt, samples, method=sample_method, seed=seed + 1)

    al = align(
        pred_pts, gt_pts, mode=align_mode,
        pred_poses=pred_poses, gt_poses=gt_poses,
        pred_convention=pred_convention, gt_convention=gt_convention,
        pred_timestamps=pred_timestamps, gt_timestamps=gt_timestamps,
    )
    pred_aligned = al.transform(pred_pts) if align_mode != "none" else pred_pts

    if debug_plot_path is not None:
        from eval3r.utils.debug_plot import save_debug_plot

        save_debug_plot(
            pred_pts,
            gt_pts,
            pred_aligned,
            debug_plot_path,
            align_mode,
            al.scale,
            pred_poses=pred_poses,
            gt_poses=gt_poses,
            pred_convention=pred_convention,
            gt_convention=gt_convention,
            matched_pred_idx=al.matched_pred_idx,
            matched_gt_idx=al.matched_gt_idx,
        )

    # --- occlusion mask filtering ---
    n_visible = len(pred_aligned)
    pred_aligned_full = pred_aligned
    gt_eval = gt_pts
    if pred_mask is not None:
        from eval3r.metrics.occlusion import filter_visible_points

        pred_aligned, n_visible, _ = filter_visible_points(pred_aligned, pred_mask)
    if gt_mask is not None:
        from eval3r.metrics.occlusion import filter_visible_points

        gt_eval, _, _ = filter_visible_points(gt_pts, gt_mask)

    d_pg = _nn_dists(pred_aligned, gt_eval)
    d_gp = _nn_dists(gt_eval, pred_aligned)

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
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        fdict[float(thr)] = {"f": f, "precision": p, "recall": r}

    return GeometryEvalResult(
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
        masked=pred_mask is not None or gt_mask is not None,
        visible_points=n_visible,
        total_pred_points=len(pred_aligned_full),
    )


class Evaluator:
    """Thin facade for library users.

    Example::

        ev = Evaluator(samples=100_000, seed=42, align_mode="none")
        result = ev.evaluate(pred_points, gt_points)
    """

    def __init__(
        self,
        *,
        samples: int = 200_000,
        seed: int = 42,
        sample_method: SampleMethod = "area",
        align_mode: AlignMode = "none",
        thresholds: list[float] | tuple[float, ...] = (0.05,),
        chamfer_variant: ChamferVariant = "l1_mean_bidirectional",
    ) -> None:
        self.samples = samples
        self.seed = seed
        self.sample_method = sample_method
        self.align_mode = align_mode
        self.thresholds = list(thresholds)
        self.chamfer_variant = chamfer_variant

    def evaluate(
        self,
        pred: Points | PointCloudData | MeshData,
        gt: Points | PointCloudData | MeshData,
        *,
        pred_mask: OcclusionMask | None = None,
    ) -> GeometryEvalResult:
        return evaluate_geometry(
            pred,
            gt,
            samples=self.samples,
            seed=self.seed,
            sample_method=self.sample_method,
            align_mode=self.align_mode,
            thresholds=self.thresholds,
            chamfer_variant=self.chamfer_variant,
            pred_mask=pred_mask,
        )
