"""Task 005 geometry-metric tests with analytically derived expectations.

Every expected value is computed by hand in the test, never copied from a run.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval3r.backends import ScipyNNBackend
from eval3r.core.errors import InvalidGeometryError, MetricError
from eval3r.core.schema import MetricSpec
from eval3r.metrics.geometry import (
    compute_directional_distances,
    evaluate_geometry_metrics,
    fraction_within,
    fscore,
)

NN = ScipyNNBackend()
PROTO = "test_geometry"
PHASH = "sha256:00"


def _eval(pred, gt, specs):
    return evaluate_geometry_metrics(
        np.asarray(pred, dtype=float),
        np.asarray(gt, dtype=float),
        specs,
        protocol=PROTO,
        protocol_hash=PHASH,
        nn_backend=NN,
    )


def _one(pred, gt, spec):
    return _eval(pred, gt, [spec])[0].value


# --- identical clouds ----------------------------------------------------------


def test_identical_clouds_zero_distance_and_perfect_scores() -> None:
    pts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    assert _one(pts, pts, MetricSpec(name="accuracy", statistic="mean")) == 0.0
    assert _one(pts, pts, MetricSpec(name="completeness", statistic="mean")) == 0.0
    assert _one(pts, pts, MetricSpec(name="precision", threshold=0.05)) == 1.0
    assert _one(pts, pts, MetricSpec(name="recall", threshold=0.05)) == 1.0
    assert _one(pts, pts, MetricSpec(name="fscore", threshold=0.05)) == 1.0


# --- known distances -----------------------------------------------------------


def test_accuracy_mean_and_median_differ() -> None:
    # pred->gt distances are exactly [1, 2, 9]
    gt = [[0, 0, 0]]
    pred = [[1, 0, 0], [2, 0, 0], [9, 0, 0]]
    assert _one(pred, gt, MetricSpec(name="accuracy", statistic="mean")) == pytest.approx(4.0)
    assert _one(pred, gt, MetricSpec(name="accuracy", statistic="median")) == pytest.approx(2.0)


def test_chamfer_sum_vs_mean() -> None:
    gt = [[0, 0, 0]]
    pred = [[1, 0, 0], [2, 0, 0], [3, 0, 0]]
    # accuracy(mean) = mean([1,2,3]) = 2 ; completeness(mean) = dist(gt->nearest pred)=1
    s = _one(pred, gt, MetricSpec(name="chamfer", statistic="mean", reduction="sum"))
    m = _one(pred, gt, MetricSpec(name="chamfer", statistic="mean", reduction="mean"))
    assert s == pytest.approx(3.0)
    assert m == pytest.approx(1.5)


def test_chamfer_requires_explicit_reduction() -> None:
    with pytest.raises(MetricError):
        _one([[1, 0, 0]], [[0, 0, 0]], MetricSpec(name="chamfer", statistic="mean"))


def test_accuracy_percentile_statistic() -> None:
    gt = [[0, 0, 0]]
    pred = [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]  # distances 1..4
    # 50th percentile of [1,2,3,4] via linear interpolation = 2.5
    val = _one(pred, gt, MetricSpec(name="accuracy", statistic="percentile", percentile=50))
    assert val == pytest.approx(2.5)


def test_clamped_accuracy() -> None:
    gt = [[0, 0, 0]]
    pred = [[1, 0, 0], [2, 0, 0], [10, 0, 0]]  # distances [1,2,10]; clamp 3 -> [1,2,3]
    val = _one(pred, gt, MetricSpec(name="accuracy", statistic="mean", clamp=3.0))
    assert val == pytest.approx(2.0)


# --- threshold metrics ---------------------------------------------------------


def test_precision_recall_partial() -> None:
    gt = [[0, 0, 0]]
    # one pred within tau=0.5 (dist 0.1), one outside (dist 2.0)
    pred = [[0.1, 0, 0], [2.0, 0, 0]]
    assert _one(pred, gt, MetricSpec(name="precision", threshold=0.5)) == pytest.approx(0.5)
    # recall: gt->nearest pred dist = 0.1 < 0.5 -> 1.0
    assert _one(pred, gt, MetricSpec(name="recall", threshold=0.5)) == pytest.approx(1.0)


def test_fscore_zero_when_no_matches() -> None:
    gt = [[0, 0, 0]]
    pred = [[10, 0, 0]]
    # precision = recall = 0 -> fscore 0 (not NaN)
    val = _one(pred, gt, MetricSpec(name="fscore", threshold=0.05))
    assert val == 0.0


def test_fscore_formula() -> None:
    assert fscore(1.0, 1.0) == 1.0
    assert fscore(0.0, 0.0) == 0.0
    assert fscore(0.5, 1.0) == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_fraction_within_is_strict_less_than() -> None:
    d = np.array([0.04, 0.05, 0.2])
    # strictly < 0.05: only 0.04 qualifies; the boundary 0.05 is excluded.
    assert fraction_within(d, 0.05) == pytest.approx(1 / 3)


# --- validation and failures ---------------------------------------------------


def test_empty_pred_raises() -> None:
    with pytest.raises(InvalidGeometryError):
        _one(np.zeros((0, 3)), [[0, 0, 0]], MetricSpec(name="accuracy", statistic="mean"))


def test_wrong_shape_raises() -> None:
    with pytest.raises(InvalidGeometryError):
        _one([[0, 0]], [[0, 0, 0]], MetricSpec(name="accuracy", statistic="mean"))


def test_nan_rows_filtered_and_valid_fraction_recorded() -> None:
    gt = [[0, 0, 0]]
    pred = np.array([[1.0, 0, 0], [np.nan, 0, 0], [3.0, 0, 0]])
    results = _eval(pred, gt, [MetricSpec(name="accuracy", statistic="mean")])
    r = results[0]
    # NaN row dropped -> distances [1,3] mean = 2
    assert r.value == pytest.approx(2.0)
    assert r.n_points_pred == 2
    # valid fraction over all input points: (2 pred kept + 1 gt kept) / (3 + 1)
    assert r.valid_fraction == pytest.approx(3 / 4)


def test_unknown_metric_name_raises() -> None:
    with pytest.raises(MetricError):
        _one([[0, 0, 0]], [[0, 0, 0]], MetricSpec(name="not_a_metric"))


def test_missing_statistic_raises() -> None:
    with pytest.raises(MetricError):
        _one([[1, 0, 0]], [[0, 0, 0]], MetricSpec(name="accuracy"))


def test_missing_threshold_raises() -> None:
    with pytest.raises(MetricError):
        _one([[1, 0, 0]], [[0, 0, 0]], MetricSpec(name="precision"))


# --- result metadata -----------------------------------------------------------


def test_result_metadata_units_and_counts() -> None:
    gt = [[0, 0, 0], [1, 1, 1]]
    pred = [[0, 0, 0], [1, 1, 1]]
    specs = [
        MetricSpec(name="accuracy", statistic="mean"),
        MetricSpec(name="fscore", threshold=0.05),
    ]
    results = _eval(pred, gt, specs)
    by_name = {r.name: r for r in results}
    assert by_name["accuracy"].unit == "m"
    assert by_name["fscore"].unit is None
    assert by_name["accuracy"].protocol_hash == PHASH
    assert by_name["accuracy"].n_points_gt == 2


def test_distances_diagnostics() -> None:
    d = compute_directional_distances(
        np.array([[0.0, 0, 0]]), np.array([[0.0, 0, 0]]), NN
    )
    assert d.valid_fraction == 1.0
    assert d.n_points_pred == 1
    assert d.n_points_gt == 1
