"""Aggregate per-scene benchmark outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from eval3r.benchmark.core import SceneOutcome


# Distance metrics: lower is better. Default penalty for a missing scene.
_DISTANCE_METRICS: frozenset[str] = frozenset({"chamfer", "accuracy", "completeness"})


def is_distance_metric(name: str) -> bool:
    """``True`` for chamfer/accuracy/completeness; ``False`` for f@/precision@/recall@."""
    return name in _DISTANCE_METRICS


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "median": float(np.median(arr)) if len(arr) else float("nan"),
        "std": float(arr.std()) if len(arr) else float("nan"),
        "n": int(len(arr)),
    }


def collect_values(
    outcomes: Iterable["SceneOutcome"],
    *,
    thresholds: Iterable[float] = (),
) -> dict[str, list[float]]:
    """Return per-metric value lists from ``status == "ok"`` outcomes.

    Threshold-keyed metrics for any threshold listed in ``thresholds`` are
    pre-seeded as empty lists so the caller can pad them even when zero
    scenes succeeded.
    """
    chamfer: list[float] = []
    accuracy: list[float] = []
    completeness: list[float] = []
    fscore: dict[float, dict[str, list[float]]] = {
        float(t): {"f": [], "precision": [], "recall": []} for t in thresholds
    }

    for o in outcomes:
        if o.status != "ok" or o.result is None:
            continue
        chamfer.append(o.result.chamfer)
        accuracy.append(o.result.accuracy)
        completeness.append(o.result.completeness)
        for thr, v in o.result.fscore.items():
            d = fscore.setdefault(float(thr), {"f": [], "precision": [], "recall": []})
            d["f"].append(v["f"])
            d["precision"].append(v["precision"])
            d["recall"].append(v["recall"])

    out: dict[str, list[float]] = {
        "chamfer": chamfer,
        "accuracy": accuracy,
        "completeness": completeness,
    }
    for thr in sorted(fscore):
        out[f"f@{thr}"] = fscore[thr]["f"]
        out[f"precision@{thr}"] = fscore[thr]["precision"]
        out[f"recall@{thr}"] = fscore[thr]["recall"]
    return out


def aggregate(
    outcomes: list["SceneOutcome"],
    *,
    thresholds: Iterable[float] = (),
) -> dict[str, dict[str, float]]:
    """Mean / median / std / n over **successful** outcomes only."""
    values = collect_values(outcomes, thresholds=thresholds)
    return {name: _stats(vs) for name, vs in values.items()}


def aggregate_all(
    outcomes: list["SceneOutcome"],
    *,
    n_total: int,
    thresholds: Iterable[float] = (),
    distance_default: float = 1.0,
    fscore_default: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Mean / median / std / n over **all** scenes; missing get a default.

    - distance metrics (chamfer / accuracy / completeness) → ``distance_default``
    - f-score / precision / recall → ``fscore_default``

    Use this when you want a single number per metric across the whole split
    that penalises missing or failed scenes instead of silently dropping them.
    """
    values = collect_values(outcomes, thresholds=thresholds)
    padded: dict[str, list[float]] = {}
    for name, vs in values.items():
        default = distance_default if is_distance_metric(name) else fscore_default
        n_missing = max(0, n_total - len(vs))
        padded[name] = list(vs) + [float(default)] * n_missing
    return {name: _stats(vs) for name, vs in padded.items()}
