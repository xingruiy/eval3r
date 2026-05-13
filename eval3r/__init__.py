"""eval3r — toolkit for saving, evaluating, and visualizing 3D reconstruction predictions."""

from eval3r._version import __version__
from eval3r.benchmark import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
    DTUBenchmark,
    DTUBenchmarkConfig,
    ETH3DBenchmark,
    ETH3DBenchmarkConfig,
    GenericBenchmark,
    GenericBenchmarkConfig,
    ReplicaBenchmark,
    ReplicaBenchmarkConfig,
    SceneOutcome,
    ScanNetBenchmark,
    ScanNetBenchmarkConfig,
    TanksTemplesBenchmark,
    TanksTemplesBenchmarkConfig,
    TumRGBDBenchmark,
    TumRGBDBenchmarkConfig,
)
from eval3r.manifest.discovery import PredictionLocator
from eval3r.manifest.reader import PredictionReader
from eval3r.manifest.writer import PredictionWriter
from eval3r.pipeline import EvalConfig, Pipeline, PipelineResult

__all__ = [
    "__version__",
    # Pipeline
    "EvalConfig",
    "Pipeline",
    "PipelineResult",
    # Manifest
    "PredictionWriter",
    "PredictionReader",
    "PredictionLocator",
    # Benchmark base
    "BaseBenchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "SceneOutcome",
    # Dataset benchmarks
    "DTUBenchmark",
    "DTUBenchmarkConfig",
    "ETH3DBenchmark",
    "ETH3DBenchmarkConfig",
    "GenericBenchmark",
    "GenericBenchmarkConfig",
    "ReplicaBenchmark",
    "ReplicaBenchmarkConfig",
    "ScanNetBenchmark",
    "ScanNetBenchmarkConfig",
    "TanksTemplesBenchmark",
    "TanksTemplesBenchmarkConfig",
    "TumRGBDBenchmark",
    "TumRGBDBenchmarkConfig",
]
