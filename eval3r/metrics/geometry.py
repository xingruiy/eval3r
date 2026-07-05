"""Point-set geometry metrics (``.agent/metrics.md`` "Geometry metrics").

Operates on two point sets *after* the pipeline has applied any alignment, masking,
culling, and sampling. Every choice (statistic, Chamfer reduction, threshold, clamp)
comes from the :class:`MetricSpec` — there are no hidden defaults here. Nearest-
neighbor distances go through the ``nearest_neighbor`` backend (task 004); no KD-tree
code lives in this module.

Diagnostics that need pipeline context (e.g. ``culled_fraction``) are not computed
here; they are injected by the runner. The set of names this module computes is
:data:`GEOMETRY_METRIC_NAMES`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval3r.core.errors import InvalidGeometryError, MetricError
from eval3r.core.registry import NNBackend
from eval3r.core.result import MetricResult
from eval3r.core.schema import MetricSpec

# Metrics computable from two point sets alone.
GEOMETRY_METRIC_NAMES = frozenset(
    {"accuracy", "completeness", "chamfer", "precision", "recall", "fscore", "coverage"}
)

# Metrics whose value is a distance (unit metres) vs a unitless fraction.
_DISTANCE_METRICS = frozenset({"accuracy", "completeness", "chamfer"})


def clean_points(points: np.ndarray, *, role: str) -> tuple[np.ndarray, int]:
    """Validate shape, drop non-finite rows, and return (clean_points, n_original).

    Empty input (or input that is empty after filtering) is a structured failure so
    the runner can apply the protocol's failure policy.
    """
    arr = np.asarray(points)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise InvalidGeometryError(f"{role} points must be (N, 3); got shape {arr.shape}.")
    n_original = arr.shape[0]
    if n_original == 0:
        raise InvalidGeometryError(f"{role} points are empty before metric computation.")
    finite = np.isfinite(arr).all(axis=1)
    clean = np.ascontiguousarray(arr[finite], dtype=np.float64)
    if clean.shape[0] == 0:
        raise InvalidGeometryError(
            f"{role} points are all non-finite (NaN/Inf); nothing left to evaluate."
        )
    return clean, n_original


@dataclass(frozen=True)
class DirectionalDistances:
    """Nearest-neighbor distances in both directions plus point-count diagnostics.

    ``pred_points`` / ``gt_points`` are the *cleaned* (finite-only) point arrays the
    distances were computed on, row-aligned with ``pred_to_gt`` / ``gt_to_pred`` —
    kept so debug outputs (error-colored clouds, task 016) can color exactly the
    points that were scored, without recomputing anything.
    """

    pred_to_gt: np.ndarray
    gt_to_pred: np.ndarray
    n_points_pred: int
    n_points_gt: int
    valid_fraction: float
    pred_points: np.ndarray | None = None
    gt_points: np.ndarray | None = None


def compute_directional_distances(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    nn_backend: NNBackend,
) -> DirectionalDistances:
    """pred→gt and gt→pred nearest distances via the NN backend."""
    pred, n_pred_orig = clean_points(pred_points, role="pred")
    gt, n_gt_orig = clean_points(gt_points, role="gt")
    pred_to_gt = nn_backend.nearest_distances(pred, gt)
    gt_to_pred = nn_backend.nearest_distances(gt, pred)
    kept = pred.shape[0] + gt.shape[0]
    total = n_pred_orig + n_gt_orig
    return DirectionalDistances(
        pred_to_gt=pred_to_gt,
        gt_to_pred=gt_to_pred,
        n_points_pred=pred.shape[0],
        n_points_gt=gt.shape[0],
        valid_fraction=kept / total,
        pred_points=pred,
        gt_points=gt,
    )


def _statistic(
    values: np.ndarray,
    statistic: str | None,
    percentile: float | None,
    clamp: float | None,
    *,
    metric: str,
) -> float:
    if statistic is None:
        raise MetricError(f"metric '{metric}' requires an explicit statistic in its MetricSpec.")
    if clamp is not None:
        values = np.minimum(values, clamp)
    if statistic == "mean":
        return float(values.mean())
    if statistic == "median":
        return float(np.median(values))
    if statistic == "rmse":
        return float(np.sqrt(np.mean(values**2)))
    if statistic == "percentile":
        if percentile is None:
            raise MetricError(f"metric '{metric}' statistic 'percentile' requires 'percentile'.")
        return float(np.percentile(values, percentile))
    raise MetricError(f"metric '{metric}' has unsupported statistic '{statistic}'.")


def fraction_within(distances: np.ndarray, threshold: float) -> float:
    """Fraction of distances strictly below ``threshold`` (precision/recall/coverage)."""
    return float(np.count_nonzero(distances < threshold) / distances.size)


def fscore(precision: float, recall: float) -> float:
    """Harmonic mean; 0 (not NaN) when precision + recall == 0."""
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return float(2.0 * precision * recall / denom)


def _require_threshold(spec: MetricSpec) -> float:
    if spec.threshold is None:
        raise MetricError(f"metric '{spec.name}' requires an explicit threshold in its MetricSpec.")
    return spec.threshold


def _metric_value(spec: MetricSpec, dist: DirectionalDistances) -> float:
    name = spec.name
    if name == "accuracy":
        return _statistic(dist.pred_to_gt, spec.statistic, spec.percentile, spec.clamp, metric=name)
    if name == "completeness":
        return _statistic(dist.gt_to_pred, spec.statistic, spec.percentile, spec.clamp, metric=name)
    if name == "chamfer":
        acc = _statistic(dist.pred_to_gt, spec.statistic, spec.percentile, spec.clamp, metric=name)
        comp = _statistic(dist.gt_to_pred, spec.statistic, spec.percentile, spec.clamp, metric=name)
        total = acc + comp
        if spec.reduction == "mean":
            return total / 2.0
        if spec.reduction in ("sum", None):
            # Chamfer reduction must be explicit in real protocols; sum is the raw sum.
            if spec.reduction is None:
                raise MetricError("metric 'chamfer' requires an explicit reduction (sum|mean).")
            return total
        raise MetricError(f"metric 'chamfer' has unsupported reduction '{spec.reduction}'.")
    if name == "precision":
        return fraction_within(dist.pred_to_gt, _require_threshold(spec))
    if name in ("recall", "coverage"):
        return fraction_within(dist.gt_to_pred, _require_threshold(spec))
    if name == "fscore":
        tau = _require_threshold(spec)
        p = fraction_within(dist.pred_to_gt, tau)
        r = fraction_within(dist.gt_to_pred, tau)
        return fscore(p, r)
    raise MetricError(
        f"metric '{name}' is not a point-set geometry metric "
        f"(known: {', '.join(sorted(GEOMETRY_METRIC_NAMES))})."
    )


def evaluate_geometry_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    specs: list[MetricSpec],
    *,
    protocol: str,
    protocol_hash: str,
    nn_backend: NNBackend,
    backend_name: str | None = None,
    scene_id: str | None = None,
    distances: DirectionalDistances | None = None,
) -> list[MetricResult]:
    """Compute the geometry metrics in ``specs`` for one (pred, gt) pair.

    ``specs`` may only contain names in :data:`GEOMETRY_METRIC_NAMES`; any other name
    raises :class:`MetricError` so nothing is silently skipped. ``distances`` may
    carry precomputed :func:`compute_directional_distances` output (the runner
    computes it once up front when debug outputs are requested) — the same values,
    never a different computation.
    """
    dist = distances if distances is not None else compute_directional_distances(
        pred_points, gt_points, nn_backend
    )
    results: list[MetricResult] = []
    for spec in specs:
        value = _metric_value(spec, dist)
        results.append(
            MetricResult(
                name=spec.name,
                value=value,
                unit="m" if spec.name in _DISTANCE_METRICS else None,
                threshold=spec.threshold,
                statistic=spec.statistic,
                reduction=spec.reduction,
                scene_id=scene_id,
                protocol=protocol,
                protocol_hash=protocol_hash,
                backend=backend_name or nn_backend.name,
                n_points_pred=dist.n_points_pred,
                n_points_gt=dist.n_points_gt,
                valid_fraction=dist.valid_fraction,
            )
        )
    return results
