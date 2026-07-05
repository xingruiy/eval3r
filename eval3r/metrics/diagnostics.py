"""Diagnostic metrics and coverage statistics.

Diagnostics describe *how* a scene was evaluated rather than geometric agreement:
``culled_fraction`` (share of prediction removed by masking/visibility culling) and
``valid_fraction`` (share of points that survived NaN/Inf and masking). They need
pipeline context the point-set metric layer does not have, so the runner computes
their values and this module turns them into :class:`MetricResult` records. Splitting
specs here keeps :mod:`eval3r.metrics.geometry` free of pipeline concerns and makes an
unknown metric name fail explicitly instead of being silently dropped.
"""

from __future__ import annotations

from eval3r.core.errors import MetricError
from eval3r.core.result import MetricResult
from eval3r.core.schema import MetricSpec
from eval3r.metrics.geometry import GEOMETRY_METRIC_NAMES

# Metrics injected by the runner from pipeline context (not computed from two point sets).
DIAGNOSTIC_METRIC_NAMES = frozenset({"culled_fraction", "valid_fraction"})


def partition_specs(specs: list[MetricSpec]) -> tuple[list[MetricSpec], list[MetricSpec]]:
    """Split ``specs`` into (geometry, diagnostic), rejecting any unknown metric name.

    An unrecognised name is a protocol error surfaced immediately, so nothing is
    silently skipped (CLAUDE.md error rules).
    """
    geometry: list[MetricSpec] = []
    diagnostic: list[MetricSpec] = []
    for spec in specs:
        if spec.name in GEOMETRY_METRIC_NAMES:
            geometry.append(spec)
        elif spec.name in DIAGNOSTIC_METRIC_NAMES:
            diagnostic.append(spec)
        else:
            raise MetricError(
                f"metric '{spec.name}' is neither a geometry metric "
                f"({', '.join(sorted(GEOMETRY_METRIC_NAMES))}) nor a diagnostic "
                f"({', '.join(sorted(DIAGNOSTIC_METRIC_NAMES))}). Fix the protocol's metric list."
            )
    return geometry, diagnostic


def build_diagnostic_metrics(
    diagnostic_specs: list[MetricSpec],
    values: dict[str, float | None],
    *,
    scene_id: str,
    protocol: str,
    protocol_hash: str,
    backend: str | None = None,
    metadata: dict[str, object] | None = None,
) -> list[MetricResult]:
    """Build :class:`MetricResult` records for the requested diagnostic specs.

    ``values`` must provide a value for every requested diagnostic name; a missing
    entry is a programming error and fails explicitly rather than emitting ``None``.
    """
    results: list[MetricResult] = []
    for spec in diagnostic_specs:
        if spec.name not in values:
            raise MetricError(
                f"diagnostic metric '{spec.name}' was requested but the runner supplied no "
                f"value for scene '{scene_id}'."
            )
        results.append(
            MetricResult(
                name=spec.name,
                value=values[spec.name],
                unit=None,
                scene_id=scene_id,
                protocol=protocol,
                protocol_hash=protocol_hash,
                backend=backend,
                metadata=dict(metadata) if metadata else {},
            )
        )
    return results
