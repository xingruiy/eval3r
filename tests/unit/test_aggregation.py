"""Task 005 aggregation tests: reductions and F-score aggregation order."""

from __future__ import annotations

import pytest

from eval3r.core.errors import MetricError
from eval3r.metrics.aggregation import (
    PrecisionRecallCounts,
    global_fscore,
    mean,
    median,
    per_scene_then_mean,
    reduce_scenes,
    weighted_mean,
)


def test_mean_median_weighted() -> None:
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert median([1.0, 2.0, 9.0]) == pytest.approx(2.0)
    assert weighted_mean([1.0, 3.0], [1.0, 3.0]) == pytest.approx((1 + 9) / 4)


def test_empty_reductions_raise() -> None:
    with pytest.raises(MetricError):
        mean([])
    with pytest.raises(MetricError):
        median([])
    with pytest.raises(MetricError):
        weighted_mean([], [])


def test_weighted_mean_length_mismatch_raises() -> None:
    with pytest.raises(MetricError):
        weighted_mean([1.0, 2.0], [1.0])


def test_per_scene_then_mean() -> None:
    assert per_scene_then_mean([1.0, 0.0]) == pytest.approx(0.5)


def test_fscore_aggregation_order_matters() -> None:
    # Scene A perfect on 1 point; Scene B all-miss on 100 points.
    counts = [
        PrecisionRecallCounts(n_pred_within=1, n_pred_total=1, n_gt_within=1, n_gt_total=1),
        PrecisionRecallCounts(n_pred_within=0, n_pred_total=100, n_gt_within=0, n_gt_total=100),
    ]
    per_scene_fscores = [1.0, 0.0]
    per_scene = per_scene_then_mean(per_scene_fscores)
    _, _, glob = global_fscore(counts)
    assert per_scene == pytest.approx(0.5)
    # global weights by points: precision = recall = 1/101 -> tiny F, != 0.5
    assert glob == pytest.approx(2 * (1 / 101) * (1 / 101) / (2 / 101))
    assert glob != pytest.approx(per_scene)


def test_global_fscore_zero_points_raises() -> None:
    with pytest.raises(MetricError):
        global_fscore([])


def test_reduce_scenes_selects_requested_stats() -> None:
    out = reduce_scenes([1.0, 3.0], mean_=True, median_=True, weights=[1.0, 3.0])
    assert out["mean"] == pytest.approx(2.0)
    assert out["median"] == pytest.approx(2.0)
    assert out["weighted_mean"] == pytest.approx(2.5)
