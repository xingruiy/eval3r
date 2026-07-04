"""eval3r: dataset-aware 3D reconstruction evaluation under explicit protocols.

Evaluate meshes, point clouds, depth predictions, and trajectories under explicit,
dataset-aware protocols.

The remaining public API (``diff_runs``) is wired up in later task slices.
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


def run_benchmark(*args: Any, **kwargs: Any) -> Any:
    """Evaluate a method's predictions across a dataset split under a named protocol.

    Thin re-export of :func:`eval3r.api.run_benchmark` (imported lazily to keep
    ``import eval3r`` cheap and cycle-free).
    """
    from eval3r.api import run_benchmark as _impl

    return _impl(*args, **kwargs)


__all__ = ["load_protocol", "evaluate_geometry", "run_benchmark", "__version__"]
