from __future__ import annotations

import numpy as np
import pytest

from eval3r.metrics.geometry import (
    accuracy,
    chamfer_distance,
    completeness,
    evaluate_geometry,
    fscore_at,
    precision_at,
    recall_at,
)
from eval3r.utils.errors import EmptyGeometryError


def _grid(n: int = 10) -> np.ndarray:
    g = np.linspace(-1, 1, n)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def test_chamfer_identical_is_zero() -> None:
    pts = _grid()
    for variant in (
        "l1_mean_bidirectional",
        "l1_sum_bidirectional",
        "l2_squared",
        "l2_unsquared",
    ):
        assert chamfer_distance(pts, pts, variant=variant) == pytest.approx(0.0, abs=1e-12)


def test_accuracy_completeness_translated() -> None:
    pts = _grid()
    shifted = pts + np.array([0.05, 0.0, 0.0])
    # Each shifted point's nearest neighbour in pts is offset by [0.05, 0, 0].
    assert accuracy(shifted, pts) == pytest.approx(0.05, abs=1e-6)
    assert completeness(shifted, pts) == pytest.approx(0.05, abs=1e-6)


def test_fscore_perfect() -> None:
    pts = _grid()
    f, p, r = fscore_at(pts, pts, threshold=0.01)
    assert (f, p, r) == (1.0, 1.0, 1.0)


def test_fscore_outliers_reduce() -> None:
    pts = _grid()
    rng = np.random.default_rng(0)
    outliers = rng.uniform(low=10, high=20, size=(50, 3))
    polluted = np.concatenate([pts, outliers], axis=0)
    f_clean, _, _ = fscore_at(pts, pts, threshold=0.05)
    f_polluted, _, _ = fscore_at(polluted, pts, threshold=0.05)
    assert f_polluted < f_clean


def test_precision_recall_threshold_consistency() -> None:
    pts = _grid()
    shifted = pts + np.array([0.04, 0.0, 0.0])
    p_low = precision_at(shifted, pts, 0.01)
    p_high = precision_at(shifted, pts, 0.05)
    assert p_low == pytest.approx(0.0)
    assert p_high == pytest.approx(1.0)
    r_low = recall_at(shifted, pts, 0.01)
    r_high = recall_at(shifted, pts, 0.05)
    assert r_low == pytest.approx(0.0)
    assert r_high == pytest.approx(1.0)


def test_chamfer_rejects_empty() -> None:
    with pytest.raises(EmptyGeometryError):
        chamfer_distance(np.zeros((0, 3)), np.zeros((1, 3)))


def test_evaluate_geometry_with_align_se3() -> None:
    pts = _grid()
    pred = pts + np.array([1.0, -2.0, 0.5])
    result_none = evaluate_geometry(
        pts, pred, samples=4096, seed=0, align_mode="none", thresholds=[0.05]
    )
    result_se3 = evaluate_geometry(
        pts, pred, samples=4096, seed=0, align_mode="se3", thresholds=[0.05]
    )
    assert result_none.chamfer > 1.0
    # SE(3) recovers the offset; residual reflects independent sampling, not misalignment.
    assert result_se3.chamfer < 0.05
    assert result_se3.fscore[0.05]["f"] > 0.95


def test_evaluate_geometry_with_align_sim3() -> None:
    pts = _grid()
    scaled = pts * 1.05  # ICP from identity recovers small scale changes only.
    result_none = evaluate_geometry(
        pts, scaled, samples=4096, seed=0, align_mode="none", thresholds=[0.05]
    )
    result_sim3 = evaluate_geometry(
        pts, scaled, samples=4096, seed=0, align_mode="sim3", thresholds=[0.05]
    )
    assert result_sim3.chamfer < result_none.chamfer
    # alignment maps pred -> gt, so recovered scale ≈ 1.05.
    assert result_sim3.align_scale == pytest.approx(1.05, rel=0.05)


def test_duplicate_points_do_not_break() -> None:
    pts = _grid()
    dup = np.concatenate([pts, pts], axis=0)
    assert chamfer_distance(dup, pts) == pytest.approx(0.0, abs=1e-12)


def test_sampling_is_deterministic() -> None:
    from eval3r.metrics.sampling import sample_points

    pts = _grid()
    a = sample_points(pts, 1000, method="uniform", seed=42)
    b = sample_points(pts, 1000, method="uniform", seed=42)
    assert np.array_equal(a, b)
