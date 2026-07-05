"""eval3r: dataset-aware 3D reconstruction evaluation under explicit protocols.

Evaluate meshes, point clouds, depth predictions, and trajectories under explicit,
dataset-aware protocols.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.3.0"

from eval3r.protocols import load_protocol


def evaluate_geometry(*args: Any, **kwargs: Any) -> Any:
    """Evaluate one predicted mesh/point cloud against ground-truth geometry.

    Thin re-export of :func:`eval3r.api.evaluate_geometry`, imported lazily so that
    ``import eval3r`` stays cheap and free of import cycles (the API module pulls in
    the pipeline, backends, and protocols).
    """
    from eval3r.api import evaluate_geometry as _impl

    return _impl(*args, **kwargs)


def evaluate_depth(*args: Any, **kwargs: Any) -> Any:
    """Evaluate a predicted depth map/sequence against ground-truth depth.

    Thin re-export of :func:`eval3r.api.evaluate_depth` (imported lazily to keep
    ``import eval3r`` cheap and cycle-free).
    """
    from eval3r.api import evaluate_depth as _impl

    return _impl(*args, **kwargs)


def evaluate_pose(*args: Any, **kwargs: Any) -> Any:
    """Evaluate a predicted trajectory against a ground-truth trajectory.

    Thin re-export of :func:`eval3r.api.evaluate_pose` (imported lazily to keep
    ``import eval3r`` cheap and cycle-free).
    """
    from eval3r.api import evaluate_pose as _impl

    return _impl(*args, **kwargs)


def run_benchmark(*args: Any, **kwargs: Any) -> Any:
    """Evaluate a method's predictions across a dataset split under a named protocol.

    Thin re-export of :func:`eval3r.api.run_benchmark` (imported lazily to keep
    ``import eval3r`` cheap and cycle-free).
    """
    from eval3r.api import run_benchmark as _impl

    return _impl(*args, **kwargs)


def diff_runs(*args: Any, **kwargs: Any) -> Any:
    """Compare two run result directories (strict on protocol hashes by default).

    Thin re-export of :func:`eval3r.reports.diff.diff_runs` (imported lazily to
    keep ``import eval3r`` cheap and cycle-free).
    """
    from eval3r.reports.diff import diff_runs as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "load_protocol",
    "evaluate_geometry",
    "evaluate_depth",
    "evaluate_pose",
    "run_benchmark",
    "diff_runs",
    "__version__",
]
