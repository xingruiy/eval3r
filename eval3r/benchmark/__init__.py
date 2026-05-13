"""eval3r benchmark — dataset benchmark classes and shared infrastructure."""

from __future__ import annotations

from eval3r.benchmark.base import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
    SceneOutcome,
    SceneStatus,
    aggregate,
    aggregate_all,
    collect_values,
)
from eval3r.benchmark.dtu import DTUBenchmark, DTUBenchmarkConfig
from eval3r.benchmark.eth3d import ETH3DBenchmark, ETH3DBenchmarkConfig
from eval3r.benchmark.generic import GenericBenchmark, GenericBenchmarkConfig
from eval3r.benchmark.replica import ReplicaBenchmark, ReplicaBenchmarkConfig
from eval3r.benchmark.scannet import ScanNetBenchmark, ScanNetBenchmarkConfig
from eval3r.benchmark.tanks_temples import TanksTemplesBenchmark, TanksTemplesBenchmarkConfig
from eval3r.benchmark.tum_rgbd import TumRGBDBenchmark, TumRGBDBenchmarkConfig

__all__ = [
    # Base
    "BaseBenchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "SceneOutcome",
    "SceneStatus",
    "aggregate",
    "aggregate_all",
    "collect_values",
    # Datasets
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
