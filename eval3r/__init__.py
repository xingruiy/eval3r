"""eval3r — toolkit for saving, evaluating, and visualizing 3D reconstruction predictions."""

from eval3r._version import __version__
from eval3r.pipeline import EvalConfig, Pipeline, PipelineResult
from eval3r.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    SceneOutcome,
    run_benchmark,
)
from eval3r.datasets import (
    DatasetAdapter,
    ScanNetAdapter,
    get_dataset,
    list_datasets,
)
from eval3r.manifest.discovery import PredictionLocator
from eval3r.manifest.reader import PredictionReader
from eval3r.manifest.writer import PredictionWriter
from eval3r.metrics.metric3d import evaluate_geometry

__all__ = [
    "__version__",
    "EvalConfig",
    "Pipeline",
    "PipelineResult",
    "PredictionWriter",
    "PredictionReader",
    "PredictionLocator",
    "evaluate_geometry",
    "DatasetAdapter",
    "ScanNetAdapter",
    "get_dataset",
    "list_datasets",
    "BenchmarkConfig",
    "BenchmarkResult",
    "SceneOutcome",
    "run_benchmark",
]
