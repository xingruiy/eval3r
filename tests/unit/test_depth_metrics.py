"""Task 014 unit tests: depth metric formulas, masking, and scale alignment.

All expectations are analytic: predictions constructed as exact multiples/affine
transforms of ground truth so every formula value is hand-derivable.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval3r.core.errors import InvalidDepthError, MetricError
from eval3r.core.schema import MaskingSpec, MetricSpec
from eval3r.metrics.depth import (
    DEPTH_METRIC_NAMES,
    DepthScaleAlignment,
    compute_depth_validity,
    estimate_depth_scale,
    evaluate_depth_metrics,
    validate_depth_pair,
)

MASKING = MaskingSpec.model_validate({"invalid_depth_values": [0]})


def _identity_alignment() -> DepthScaleAlignment:
    return DepthScaleAlignment(
        mode="none", granularity="per_frame", scale=1.0, shift=0.0, n_pixels_used=0
    )


def _evaluate(pred: np.ndarray, gt: np.ndarray, specs: list[MetricSpec]) -> dict[str, float]:
    pred, gt = validate_depth_pair(pred, gt, frame="f0")
    validity = compute_depth_validity(pred, gt, MASKING, frame="f0")
    results = evaluate_depth_metrics(
        pred, gt, validity, specs,
        alignment=_identity_alignment(),
        protocol="test", protocol_hash="sha256:test",
        scene_id="s", frame_id="f0",
    )
    return {r.name: r.value for r in results}


ALL_SPECS = [
    MetricSpec(name="absrel"),
    MetricSpec(name="sqrel"),
    MetricSpec(name="rmse"),
    MetricSpec(name="rmse_log"),
    MetricSpec(name="silog"),
    MetricSpec(name="delta_1", threshold=1.25),
    MetricSpec(name="delta_2", threshold=1.5625),
    MetricSpec(name="delta_3", threshold=1.953125),
]


# --- formulas --------------------------------------------------------------------


def test_identical_depths_are_perfect() -> None:
    gt = np.array([[1.0, 2.0], [3.0, 4.0]])
    values = _evaluate(gt.copy(), gt, ALL_SPECS)
    assert values["absrel"] == 0.0
    assert values["sqrel"] == 0.0
    assert values["rmse"] == 0.0
    assert values["rmse_log"] == 0.0
    assert values["silog"] == 0.0
    assert values["delta_1"] == 1.0
    assert values["delta_2"] == 1.0
    assert values["delta_3"] == 1.0


def test_constant_ratio_prediction_analytic_values() -> None:
    # pred = 1.1 * gt: absrel = 0.1 exactly; sqrel = 0.01 * mean(gt);
    # rmse = 0.1 * sqrt(mean(gt^2)); rmse_log = ln(1.1); ratio 1.1 < 1.25.
    gt = np.array([[1.0, 2.0], [4.0, 5.0]])
    values = _evaluate(1.1 * gt, gt, ALL_SPECS)
    assert values["absrel"] == pytest.approx(0.1)
    assert values["sqrel"] == pytest.approx(0.01 * gt.mean())
    assert values["rmse"] == pytest.approx(0.1 * np.sqrt((gt**2).mean()))
    assert values["rmse_log"] == pytest.approx(np.log(1.1))
    assert values["delta_1"] == 1.0


def test_silog_is_zero_for_any_constant_scale() -> None:
    # A global scale error shifts all log-diffs equally: scale-invariant error is 0.
    gt = np.array([[1.0, 2.0], [4.0, 8.0]])
    values = _evaluate(3.7 * gt, gt, [MetricSpec(name="silog"), MetricSpec(name="rmse_log")])
    assert values["silog"] == pytest.approx(0.0, abs=1e-12)
    assert values["rmse_log"] == pytest.approx(np.log(3.7))


def test_silog_hand_computed() -> None:
    gt = np.array([[1.0, 1.0]])
    pred = np.array([[np.e, 1.0]])  # d = [1, 0] -> mean(d^2)=0.5, mean(d)^2=0.25
    values = _evaluate(pred, gt, [MetricSpec(name="silog")])
    assert values["silog"] == pytest.approx(np.sqrt(0.25))


def test_delta_thresholds_are_strict_and_symmetric() -> None:
    # ratio uses max(pred/gt, gt/pred): under- and over-prediction count equally,
    # and a ratio exactly at the threshold is NOT within (strict <).
    gt = np.array([[1.0, 1.0, 1.0, 1.0]])
    pred = np.array([[1.2, 1.0 / 1.2, 1.25, 2.0]])
    values = _evaluate(pred, gt, [MetricSpec(name="delta_1", threshold=1.25)])
    assert values["delta_1"] == pytest.approx(2 / 4)


def test_delta_requires_explicit_threshold() -> None:
    gt = np.ones((2, 2))
    with pytest.raises(MetricError, match="delta_1.*explicit threshold"):
        _evaluate(gt, gt, [MetricSpec(name="delta_1")])


def test_unknown_metric_name_fails_loudly() -> None:
    gt = np.ones((2, 2))
    with pytest.raises(MetricError, match="not a depth metric"):
        _evaluate(gt, gt, [MetricSpec(name="chamfer")])
    assert "chamfer" not in DEPTH_METRIC_NAMES


def test_log_metrics_reject_nonpositive_aligned_prediction() -> None:
    # Masking keeps pred > 0, but a negative alignment shift can reintroduce
    # non-positive values; log metrics must fail loudly, not warn or clamp.
    gt = np.array([[1.0, 2.0]])
    pred = np.array([[-0.5, 2.0]])
    pred_v, gt_v = validate_depth_pair(pred, gt, frame="f0")
    validity = compute_depth_validity(
        np.array([[0.5, 2.0]]), gt_v, MASKING, frame="f0"
    )
    with pytest.raises(MetricError, match="non-positive"):
        evaluate_depth_metrics(
            pred_v, gt_v, validity, [MetricSpec(name="rmse_log")],
            alignment=_identity_alignment(),
            protocol="test", protocol_hash="sha256:test",
        )


# --- validation and masking --------------------------------------------------------


def test_shape_mismatch_fails_explicitly() -> None:
    with pytest.raises(InvalidDepthError, match="shapes differ.*frame 'f9'"):
        validate_depth_pair(np.ones((2, 3)), np.ones((3, 2)), frame="f9")


def test_non_2d_input_fails() -> None:
    with pytest.raises(InvalidDepthError, match="must be a 2D"):
        validate_depth_pair(np.ones((2, 2, 3)), np.ones((2, 2, 3)), frame="f0")


def test_invalid_value_sentinels_and_nonfinite_are_masked() -> None:
    gt = np.array([[1.0, 0.0], [np.nan, 65535.0]])
    pred = np.array([[1.0, 1.0], [1.0, np.inf]])
    masking = MaskingSpec.model_validate({"invalid_depth_values": [0, 65535]})
    validity = compute_depth_validity(pred, gt, masking, frame="f0")
    assert validity.n_pixels_valid == 1
    assert validity.mask[0, 0]
    assert validity.valid_fraction == pytest.approx(0.25)
    breakdown = validity.as_metadata()
    # The 0 sentinel is counted; the 65535 pixel was already masked by pred=Inf.
    assert breakdown["n_gt_invalid_value"] == 1
    assert breakdown["n_gt_nonfinite"] == 1
    assert breakdown["n_pred_nonfinite"] == 1


def test_nonpositive_gt_and_pred_masked_and_counted() -> None:
    gt = np.array([[1.0, -2.0], [3.0, 4.0]])
    pred = np.array([[1.0, 1.0], [-0.1, 4.0]])
    validity = compute_depth_validity(pred, gt, MaskingSpec(), frame="f0")
    assert validity.n_pixels_valid == 2
    assert validity.as_metadata()["n_gt_nonpositive"] == 1
    assert validity.as_metadata()["n_pred_nonpositive"] == 1


def test_ignore_invalid_depth_false_keeps_nonpositive() -> None:
    # A protocol may declare that zero/negative depths are meaningful; then only
    # listed sentinels and non-finite values are masked.
    gt = np.array([[1.0, -2.0]])
    pred = np.array([[1.0, -2.0]])
    masking = MaskingSpec.model_validate({"ignore_invalid_depth": False})
    validity = compute_depth_validity(pred, gt, masking, frame="f0")
    assert validity.n_pixels_valid == 2


def test_empty_mask_fails_with_breakdown() -> None:
    gt = np.zeros((2, 2))
    pred = np.ones((2, 2))
    with pytest.raises(InvalidDepthError, match="no valid pixels remain.*breakdown"):
        compute_depth_validity(pred, gt, MASKING, frame="f0")


def test_external_mask_intersects_and_shape_checked() -> None:
    gt = np.ones((2, 2))
    pred = np.ones((2, 2))
    extra = np.array([[True, False], [False, False]])
    validity = compute_depth_validity(pred, gt, MASKING, frame="f0", extra_mask=extra)
    assert validity.n_pixels_valid == 1
    with pytest.raises(InvalidDepthError, match="mask shape"):
        compute_depth_validity(pred, gt, MASKING, frame="f0", extra_mask=np.ones((3, 3), bool))


# --- scale alignment ----------------------------------------------------------------


def test_median_scale_recovers_known_factor() -> None:
    gt = np.array([1.0, 2.0, 3.0, 4.0])
    pred = 2.0 * gt
    scale, shift = estimate_depth_scale(pred, gt, "scale_median", context="t")
    assert scale == pytest.approx(0.5)
    assert shift == 0.0


def test_least_squares_scale_recovers_known_factor() -> None:
    gt = np.array([1.0, 2.0, 3.0, 4.0])
    pred = 2.0 * gt
    scale, shift = estimate_depth_scale(pred, gt, "scale_least_squares", context="t")
    assert scale == pytest.approx(0.5)
    assert shift == 0.0


def test_affine_recovers_known_scale_and_shift() -> None:
    gt = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    pred = 2.0 * gt + 3.0  # gt = 0.5 * pred - 1.5
    scale, shift = estimate_depth_scale(pred, gt, "scale_affine", context="t")
    assert scale == pytest.approx(0.5)
    assert shift == pytest.approx(-1.5)


def test_median_scale_zeroes_error_end_to_end() -> None:
    # pred = 2 x gt, then median alignment: aligned prediction equals gt exactly.
    gt = np.array([[1.0, 2.0], [3.0, 4.0]])
    pred = 2.0 * gt
    scale, shift = estimate_depth_scale(pred.ravel(), gt.ravel(), "scale_median", context="t")
    values = _evaluate(pred * scale + shift, gt, ALL_SPECS)
    assert values["absrel"] == 0.0
    assert values["rmse"] == 0.0
    assert values["delta_1"] == 1.0


def test_affine_degenerate_constant_prediction_fails() -> None:
    with pytest.raises(InvalidDepthError, match="constant over all valid pixels"):
        estimate_depth_scale(np.ones(5), np.arange(5.0) + 1, "scale_affine", context="t")


def test_least_squares_zero_prediction_fails() -> None:
    with pytest.raises(InvalidDepthError, match="zero over all valid pixels"):
        estimate_depth_scale(np.zeros(5), np.ones(5), "scale_least_squares", context="t")


def test_unknown_alignment_mode_fails() -> None:
    with pytest.raises(MetricError, match="not a depth scale-alignment mode"):
        estimate_depth_scale(np.ones(3), np.ones(3), "icp", context="t")


# --- result metadata -----------------------------------------------------------------


def test_alignment_mode_and_granularity_in_every_result_metadata() -> None:
    gt = np.array([[1.0, 2.0]])
    pred, gt = validate_depth_pair(1.1 * gt, gt, frame="f0")
    validity = compute_depth_validity(pred, gt, MASKING, frame="f0")
    align = DepthScaleAlignment(
        mode="scale_median", granularity="per_frame",
        scale=0.5, shift=0.0, n_pixels_used=2, frame_id="f0",
    )
    results = evaluate_depth_metrics(
        pred, gt, validity, ALL_SPECS,
        alignment=align, protocol="test", protocol_hash="sha256:test",
        scene_id="s", frame_id="f0",
    )
    assert len(results) == len(ALL_SPECS)
    for r in results:
        assert r.metadata["alignment_mode"] == "scale_median"
        assert r.metadata["alignment_granularity"] == "per_frame"
        assert r.metadata["alignment_scale"] == 0.5
        assert r.n_pixels_valid == 2
        assert r.valid_fraction == 1.0
        assert "mask_breakdown" in r.metadata
    units = {r.name: r.unit for r in results}
    assert units["rmse"] == "m"
    assert units["sqrel"] == "m"
    assert units["absrel"] is None
    assert units["delta_1"] is None
