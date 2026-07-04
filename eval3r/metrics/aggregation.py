"""Cross-scene aggregation primitives (``.agent/metrics.md`` "Aggregation").

Aggregation must be explicit. F-score in particular must declare either
``per_scene_then_mean`` (mean of per-scene F-scores) or ``global`` (pool precision/
recall counts across scenes, then one F-score). Both are provided here; the caller
picks based on ``MetricSpec.aggregation``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval3r.core.errors import MetricError
from eval3r.metrics.geometry import fscore


def mean(values: list[float]) -> float:
    if not values:
        raise MetricError("cannot take the mean of zero scene values.")
    return float(np.mean(values))


def median(values: list[float]) -> float:
    if not values:
        raise MetricError("cannot take the median of zero scene values.")
    return float(np.median(values))


def weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        raise MetricError("cannot take the weighted mean of zero scene values.")
    if len(values) != len(weights):
        raise MetricError(
            f"weighted mean needs one weight per value; got {len(values)} values, "
            f"{len(weights)} weights."
        )
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() == 0:
        raise MetricError("weighted mean has zero total weight.")
    return float(np.average(np.asarray(values, dtype=np.float64), weights=w))


def per_scene_then_mean(per_scene_values: list[float]) -> float:
    """Mean of per-scene metric values (the usual dataset-benchmark aggregation)."""
    return mean(per_scene_values)


@dataclass(frozen=True)
class PrecisionRecallCounts:
    """Per-scene counts needed to pool a *global* precision/recall/F-score."""

    n_pred_within: int
    n_pred_total: int
    n_gt_within: int
    n_gt_total: int


def global_fscore(counts: list[PrecisionRecallCounts]) -> tuple[float, float, float]:
    """Pool counts across scenes into one (precision, recall, fscore).

    This differs from ``per_scene_then_mean`` because it weights by point counts,
    not by scene.
    """
    if not counts:
        raise MetricError("cannot compute a global F-score over zero scenes.")
    pred_within = sum(c.n_pred_within for c in counts)
    pred_total = sum(c.n_pred_total for c in counts)
    gt_within = sum(c.n_gt_within for c in counts)
    gt_total = sum(c.n_gt_total for c in counts)
    if pred_total == 0 or gt_total == 0:
        raise MetricError("global F-score has zero pred or gt points across all scenes.")
    precision = pred_within / pred_total
    recall = gt_within / gt_total
    return precision, recall, fscore(precision, recall)


def reduce_scenes(
    per_scene_values: list[float],
    *,
    mean_: bool = True,
    median_: bool = False,
    weights: list[float] | None = None,
) -> dict[str, float]:
    """Return the requested reductions across scenes as a ``{stat: value}`` dict."""
    out: dict[str, float] = {}
    if mean_:
        out["mean"] = mean(per_scene_values)
    if median_:
        out["median"] = median(per_scene_values)
    if weights is not None:
        out["weighted_mean"] = weighted_mean(per_scene_values, weights)
    return out
