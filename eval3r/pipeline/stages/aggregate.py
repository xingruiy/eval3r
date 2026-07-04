"""Aggregate stage: reduce per-scene metric results into benchmark aggregates.

Aggregation is explicit. For this slice every metric is reduced by
``per_scene_then_mean`` (the mean of its per-scene values, i.e. F-score aggregation
``per_scene_then_mean``); pooled ``global`` F-score is deferred until the geometry
metric records per-scene precision/recall counts. Only scenes that produced a value
contribute, and the runner separately records how many scenes were expected vs
evaluated so partial coverage stays visible (``.agent/metrics.md`` aggregation rules).
"""

from __future__ import annotations

from collections import defaultdict

from eval3r.core.result import MetricResult
from eval3r.core.schema import MetricSpec
from eval3r.metrics.aggregation import mean


def aggregate_scene_metrics(
    per_scene: list[MetricResult],
    specs: list[MetricSpec],
) -> dict[str, float | None]:
    """Return ``{metric_name: per_scene_mean}`` (``None`` when no scene produced it)."""
    values: dict[str, list[float]] = defaultdict(list)
    for result in per_scene:
        if result.value is not None:
            values[result.name].append(result.value)

    out: dict[str, float | None] = {}
    for spec in specs:
        vals = values.get(spec.name, [])
        out[spec.name] = mean(vals) if vals else None
    return out
