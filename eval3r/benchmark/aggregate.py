"""Aggregate per-scene benchmark outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from eval3r.benchmark.core import SceneOutcome


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "median": float(np.median(arr)) if len(arr) else float("nan"),
        "std": float(arr.std()) if len(arr) else float("nan"),
        "n": int(len(arr)),
    }


def aggregate(outcomes: list["SceneOutcome"]) -> dict[str, dict[str, float]]:
    """Compute mean / median / std / n per metric over successful outcomes."""
    chamfer: list[float] = []
    accuracy: list[float] = []
    completeness: list[float] = []
    fscore: dict[float, dict[str, list[float]]] = {}

    for o in outcomes:
        if o.status != "ok" or o.result is None:
            continue
        chamfer.append(o.result.chamfer)
        accuracy.append(o.result.accuracy)
        completeness.append(o.result.completeness)
        for thr, v in o.result.fscore.items():
            d = fscore.setdefault(thr, {"f": [], "precision": [], "recall": []})
            d["f"].append(v["f"])
            d["precision"].append(v["precision"])
            d["recall"].append(v["recall"])

    summary: dict[str, dict[str, float]] = {
        "chamfer": _stats(chamfer),
        "accuracy": _stats(accuracy),
        "completeness": _stats(completeness),
    }
    for thr in sorted(fscore):
        summary[f"f@{thr}"] = _stats(fscore[thr]["f"])
        summary[f"precision@{thr}"] = _stats(fscore[thr]["precision"])
        summary[f"recall@{thr}"] = _stats(fscore[thr]["recall"])
    return summary
