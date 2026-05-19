"""Pipeline behaviour when filters reduce predictions to zero points."""

from __future__ import annotations

import math

import numpy as np

from eval3r.alignment.base import IdentityAligner
from eval3r.filtering.bbox import BBoxFilter
from eval3r.metrics.metric3d import Accuracy, ChamferDistance, FScore
from eval3r.pipeline import EvalConfig, Pipeline
from eval3r.sampling.uniform import UniformSampler


def _grid(n: int = 6) -> np.ndarray:
    g = np.linspace(-1, 1, n)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def test_pipeline_returns_nan_when_filter_drops_all_points() -> None:
    """A bbox filter that excludes every prediction point must not raise.

    The pipeline should report n_visible=0 and emit NaN values for every metric
    (and a NaN-triple for F-score) instead of propagating ``EmptyGeometryError``.
    """
    pts = _grid()
    pipe = Pipeline(
        sampler=UniformSampler(),
        aligner=IdentityAligner(),
        filters=[
            BBoxFilter(
                bbox_min=np.array([100.0, 100.0, 100.0]),
                bbox_max=np.array([101.0, 101.0, 101.0]),
                margin=0.0,
            )
        ],
        metrics=[ChamferDistance(), Accuracy(), FScore(0.05)],
        config=EvalConfig(samples=256, seed=0),
    )

    result = pipe.evaluate(pts, pts)

    assert result.n_visible == 0
    assert result.n_total > 0
    assert math.isnan(result.values["chamfer"])
    assert math.isnan(result.values["accuracy"])
    f, p, r = result.values["fscore@0.05"]
    assert math.isnan(f)
    assert math.isnan(p)
    assert math.isnan(r)
