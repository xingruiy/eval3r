"""Metric stage: compute the protocol's geometry metrics for one scene.

Thin wrapper over :func:`eval3r.metrics.geometry.evaluate_geometry_metrics`. The
metric layer owns every formula and reduction choice; this stage only supplies the
already-sampled point sets, the nearest-neighbor backend, and the scene identity.
"""

from __future__ import annotations

import numpy as np

from eval3r.core.registry import NNBackend
from eval3r.core.result import MetricResult
from eval3r.core.schema import MetricSpec
from eval3r.metrics.geometry import DirectionalDistances, evaluate_geometry_metrics


def compute_scene_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    specs: list[MetricSpec],
    *,
    protocol: str,
    protocol_hash: str,
    nn_backend: NNBackend,
    scene_id: str,
    distances: DirectionalDistances | None = None,
) -> list[MetricResult]:
    """Compute ``specs`` for one (pred, gt) point-set pair.

    ``distances`` optionally carries precomputed directional distances (the runner
    computes them once up front when the protocol requests debug outputs).
    """
    return evaluate_geometry_metrics(
        pred_points,
        gt_points,
        specs,
        protocol=protocol,
        protocol_hash=protocol_hash,
        nn_backend=nn_backend,
        backend_name=nn_backend.name,
        scene_id=scene_id,
        distances=distances,
    )
