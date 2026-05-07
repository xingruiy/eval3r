from __future__ import annotations

import numpy as np
import pytest

from eval3r.metrics.depth import (
    DepthEvalResult,
    abs_rel,
    delta_accuracy,
    depth_metrics,
    rmse,
    rmse_log,
    sq_rel,
)
from eval3r.utils.errors import Eval3rError


def _depth(h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w), 2.0, dtype=np.float32)


def test_identical_is_perfect() -> None:
    d = _depth()
    r = depth_metrics(d, d)
    assert r.abs_rel == pytest.approx(0.0, abs=1e-12)
    assert r.sq_rel == pytest.approx(0.0, abs=1e-12)
    assert r.rmse == pytest.approx(0.0, abs=1e-12)
    assert r.rmse_log == pytest.approx(0.0, abs=1e-12)
    assert r.delta1 == pytest.approx(1.0)
    assert r.delta2 == pytest.approx(1.0)
    assert r.delta3 == pytest.approx(1.0)


def test_rmse_with_constant_offset() -> None:
    pred = _depth()
    gt = np.full_like(pred, 3.0)
    r = depth_metrics(pred, gt)
    assert r.rmse == pytest.approx(1.0, abs=1e-6)
    assert r.abs_rel == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_delta_zero_for_large_error() -> None:
    pred = _depth()
    gt = np.full_like(pred, 100.0)
    # ratio = max(2/100, 100/2) = 50, well above 1.25
    assert delta_accuracy(pred, gt, 1.25) == pytest.approx(0.0)


def test_delta_one_for_small_error() -> None:
    pred = _depth()
    gt = np.full_like(pred, 2.01)
    # ratio = max(2/2.01, 2.01/2) ≈ 1.005 < 1.25
    assert delta_accuracy(pred, gt, 1.25) == pytest.approx(1.0)


def test_mask_excludes_pixels() -> None:
    pred = _depth()
    gt = _depth()
    mask = np.zeros_like(pred, dtype=bool)
    mask[0, 0] = True  # only one valid pixel
    r = depth_metrics(pred, gt, mask)
    assert r.valid_pixels == 1
    assert r.total_pixels == 64 * 64


def test_zero_gt_pixels_are_excluded() -> None:
    pred = _depth()
    gt = _depth()
    gt[0, 0] = 0.0
    r = depth_metrics(pred, gt)
    assert r.valid_pixels == 64 * 64 - 1
    assert r.abs_rel == pytest.approx(0.0, abs=1e-12)


def test_negative_gt_pixels_are_excluded() -> None:
    pred = _depth()
    gt = _depth()
    gt[0, :] = -1.0
    r = depth_metrics(pred, gt)
    assert r.valid_pixels == 64 * 63


def test_empty_input_raises() -> None:
    pred = _depth()
    gt = np.zeros_like(pred)  # all pixels invalid (zero)
    with pytest.raises(Eval3rError):
        depth_metrics(pred, gt)


def test_abs_rel_sq_rel_individual() -> None:
    pred = np.full((8, 8), 3.0, dtype=np.float32)
    gt = np.full((8, 8), 2.0, dtype=np.float32)
    # For each pixel: |3-2|/2 = 0.5, (3-2)²/2 = 0.5
    assert abs_rel(pred, gt) == pytest.approx(0.5)
    assert sq_rel(pred, gt) == pytest.approx(0.5)


def test_rmse_log_with_ratio() -> None:
    pred = np.full((8, 8), 4.0, dtype=np.float32)
    gt = np.full((8, 8), 2.0, dtype=np.float32)
    expected = np.sqrt((np.log(4.0) - np.log(2.0)) ** 2)
    assert rmse_log(pred, gt) == pytest.approx(expected, abs=1e-6)


def test_to_dict() -> None:
    d = _depth()
    r = depth_metrics(d, d)
    out = r.to_dict()
    assert isinstance(out, dict)
    assert set(out.keys()) == {
        "abs_rel", "sq_rel", "rmse", "rmse_log",
        "delta1", "delta2", "delta3",
        "valid_pixels", "total_pixels",
    }


def test_delta2_delta3_stricter() -> None:
    pred = np.full((16, 16), 2.0, dtype=np.float32)
    gt = np.full((16, 16), 1.9, dtype=np.float32)
    d1 = delta_accuracy(pred, gt, 1.25)
    d2 = delta_accuracy(pred, gt, 1.25 ** 2)
    d3 = delta_accuracy(pred, gt, 1.25 ** 3)
    assert d1 == pytest.approx(1.0)
    assert d2 == pytest.approx(1.0)
    assert d3 == pytest.approx(1.0)
    assert d1 <= d2 <= d3  # larger thresholds are more forgiving


def test_finite_check_excludes_nan_and_inf() -> None:
    pred = _depth(4, 4)
    gt = _depth(4, 4)
    pred[0, 0] = np.nan
    pred[0, 1] = np.inf
    gt[0, 2] = -np.inf
    r = depth_metrics(pred, gt)
    assert r.valid_pixels == 13  # 16 - 3


def test_delta_with_threshold() -> None:
    d = _depth(8, 8)
    # ratio is exactly 1.0, so threshold 1.0 is strict (1.0 < 1.0 = false)
    assert delta_accuracy(d, d, 1.0) == pytest.approx(0.0)
    assert delta_accuracy(d, d, 1.01) == pytest.approx(1.0)


def test_shape_mismatch_pred_gt_raises() -> None:
    pred = _depth(8, 8)
    gt = _depth(8, 7)
    with pytest.raises(ValueError, match="pred and gt must have the same shape"):
        depth_metrics(pred, gt)


def test_shape_mismatch_mask_raises() -> None:
    pred = _depth(8, 8)
    gt = _depth(8, 8)
    mask = np.ones((8, 7), dtype=bool)
    with pytest.raises(ValueError, match="mask must have the same shape"):
        depth_metrics(pred, gt, mask)
