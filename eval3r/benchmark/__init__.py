"""eval3r benchmark — run a method's predictions against a dataset split."""

from __future__ import annotations

from eval3r.benchmark.aggregate import aggregate, aggregate_all
from eval3r.benchmark.core import (
    BenchmarkConfig,
    BenchmarkResult,
    SceneOutcome,
    SceneStatus,
    run_benchmark,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "SceneOutcome",
    "SceneStatus",
    "run_benchmark",
    "aggregate",
    "aggregate_all",
]
