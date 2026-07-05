"""Task 015 unit tests: evo trajectory backend + pose metric layer.

Synthetic TUM trajectories with analytically known transforms: identity → zero
ATE; a known Sim3 scale is recovered exactly; a constant translation offset under
``align='none'`` gives ATE == the offset norm; a constant per-frame extra
rotation gives RPE rotation == that angle. evo is a required base dependency, so
nothing here uses importorskip.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from eval3r.backends.trajectory_evo import EvoTrajectoryBackend
from eval3r.core.errors import (
    AlignmentError,
    InvalidTrajectoryError,
    MetricError,
)
from eval3r.core.schema import AlignmentSpec, MetricSpec
from eval3r.metrics.pose import (
    POSE_ALIGNMENT_MODES,
    alignment_scale_error,
    check_pose_alignment_spec,
    pose_metric_result,
    require_rpe_parameters,
)

BACKEND = EvoTrajectoryBackend()
ASSOCIATION = {"associate_max_diff": 0.01, "offset": 0.0}


def write_tum(
    path: Path,
    positions: np.ndarray,
    *,
    timestamps: np.ndarray | None = None,
    quats_xyzw: np.ndarray | None = None,
) -> Path:
    """Write a TUM trajectory file (timestamp x y z qx qy qz qw)."""
    n = len(positions)
    ts = timestamps if timestamps is not None else np.arange(n, dtype=float) * 0.1
    quats = (
        quats_xyzw
        if quats_xyzw is not None
        else np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))  # identity, xyzw
    )
    lines = [
        f"{ts[i]:.6f} {positions[i][0]:.9f} {positions[i][1]:.9f} "
        f"{positions[i][2]:.9f} {quats[i][0]:.9f} {quats[i][1]:.9f} "
        f"{quats[i][2]:.9f} {quats[i][3]:.9f}"
        for i in range(n)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def helix_positions(n: int = 20) -> np.ndarray:
    t = np.arange(n, dtype=float) * 0.1
    return np.stack([np.cos(t), np.sin(t), 0.1 * t], axis=1)


# --- backend: ATE ------------------------------------------------------------------


def test_identity_trajectories_give_zero_ate(tmp_path: Path) -> None:
    pos = helix_positions()
    gt = write_tum(tmp_path / "gt.txt", pos)
    pred = write_tum(tmp_path / "pred.txt", pos)
    out = BACKEND.evaluate_ate(pred, gt, "none", ASSOCIATION)
    assert out["stats"]["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert out["n_associated"] == 20
    assert out["n_dropped_pred"] == 0 and out["n_dropped_gt"] == 0
    assert out["alignment"]["mode"] == "none"
    assert out["alignment"]["scale"] == 1.0
    assert out["unit"] == "m"


def test_sim3_recovers_known_scale_and_zeroes_ate(tmp_path: Path) -> None:
    pos = helix_positions()
    gt = write_tum(tmp_path / "gt.txt", pos)
    pred = write_tum(tmp_path / "pred.txt", 2.0 * pos + np.array([1.0, 2.0, 3.0]))
    out = BACKEND.evaluate_ate(pred, gt, "trajectory_sim3", ASSOCIATION)
    assert out["alignment"]["scale"] == pytest.approx(0.5, abs=1e-9)
    assert out["stats"]["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_se3_does_not_correct_scale(tmp_path: Path) -> None:
    pos = helix_positions()
    gt = write_tum(tmp_path / "gt.txt", pos)
    pred = write_tum(tmp_path / "pred.txt", 2.0 * pos)
    out = BACKEND.evaluate_ate(pred, gt, "trajectory_se3", ASSOCIATION)
    assert out["alignment"]["scale"] == 1.0
    assert out["stats"]["rmse"] > 0.1  # residual scale error remains


def test_unaligned_constant_offset_is_exact_ate(tmp_path: Path) -> None:
    pos = helix_positions()
    gt = write_tum(tmp_path / "gt.txt", pos)
    pred = write_tum(tmp_path / "pred.txt", pos + np.array([3.0, 0.0, 4.0]))
    out = BACKEND.evaluate_ate(pred, gt, "none", ASSOCIATION)
    assert out["stats"]["rmse"] == pytest.approx(5.0, abs=1e-9)  # |(3,0,4)| = 5


# --- backend: RPE ------------------------------------------------------------------


def test_rpe_rotation_measures_per_frame_rotation_exactly(tmp_path: Path) -> None:
    # pred orientation at frame i is rotated about z by i*theta relative to gt's
    # identity orientations, so every consecutive relative pose differs by exactly
    # theta. Rigid alignment cannot change relative rotations.
    theta_deg = 10.0
    n = 10
    pos = helix_positions(n)
    half = np.deg2rad(theta_deg * np.arange(n)) / 2.0
    quats = np.stack(
        [np.zeros(n), np.zeros(n), np.sin(half), np.cos(half)], axis=1
    )  # xyzw about z
    gt = write_tum(tmp_path / "gt.txt", pos)
    pred = write_tum(tmp_path / "pred.txt", pos, quats_xyzw=quats)
    out = BACKEND.evaluate_rpe(
        pred, gt, "none", ASSOCIATION,
        pose_relation="rotation_angle_deg", delta=1, delta_unit="frames",
    )
    assert out["stats"]["rmse"] == pytest.approx(theta_deg, abs=1e-6)
    assert out["unit"] == "deg"
    assert out["delta"] == 1.0 and out["delta_unit"] == "frames"


def test_rpe_translation_zero_for_rigidly_transformed_trajectory(tmp_path: Path) -> None:
    pos = helix_positions()
    gt = write_tum(tmp_path / "gt.txt", pos)
    pred = write_tum(tmp_path / "pred.txt", pos + np.array([1.0, 2.0, 3.0]))
    out = BACKEND.evaluate_rpe(
        pred, gt, "none", ASSOCIATION,
        pose_relation="translation_part", delta=1, delta_unit="frames",
    )
    # A constant offset with identity orientations preserves relative translations.
    assert out["stats"]["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert out["unit"] == "m"


def test_rpe_rejects_unknown_pose_relation_and_delta_unit(tmp_path: Path) -> None:
    pos = helix_positions(5)
    gt = write_tum(tmp_path / "gt.txt", pos)
    pred = write_tum(tmp_path / "pred.txt", pos)
    with pytest.raises(MetricError, match="pose relation"):
        BACKEND.evaluate_rpe(
            pred, gt, "none", ASSOCIATION,
            pose_relation="full_transformation", delta=1, delta_unit="frames",
        )
    with pytest.raises(MetricError, match="delta_unit"):
        BACKEND.evaluate_rpe(
            pred, gt, "none", ASSOCIATION,
            pose_relation="translation_part", delta=1, delta_unit="fortnights",
        )


# --- backend: association and failure paths -----------------------------------------


def test_association_within_max_diff_and_dropped_counts(tmp_path: Path) -> None:
    pos = helix_positions()
    gt = write_tum(tmp_path / "gt.txt", pos)
    # Timestamps offset by 5 ms (< 10 ms tolerance), and the last 3 frames missing.
    ts = np.arange(20, dtype=float) * 0.1 + 0.005
    pred = write_tum(tmp_path / "pred.txt", pos[:17], timestamps=ts[:17])
    out = BACKEND.evaluate_ate(pred, gt, "none", ASSOCIATION)
    assert out["n_pred_poses"] == 17
    assert out["n_gt_poses"] == 20
    assert out["n_associated"] == 17
    assert out["n_dropped_pred"] == 0
    assert out["n_dropped_gt"] == 3
    assert out["association"] == {
        "policy": "nearest_timestamp",
        "associate_max_diff": 0.01,
        "offset": 0.0,
    }


def test_no_timestamp_overlap_fails_naming_tolerance(tmp_path: Path) -> None:
    pos = helix_positions(5)
    gt = write_tum(tmp_path / "gt.txt", pos)
    pred = write_tum(
        tmp_path / "pred.txt", pos, timestamps=np.arange(5, dtype=float) + 100.0
    )
    with pytest.raises(InvalidTrajectoryError) as exc:
        BACKEND.evaluate_ate(pred, gt, "none", ASSOCIATION)
    msg = str(exc.value)
    assert "associate_max_diff=0.01" in msg
    assert str(pred) in msg and str(gt) in msg


def test_missing_max_diff_is_never_defaulted(tmp_path: Path) -> None:
    pos = helix_positions(5)
    gt = write_tum(tmp_path / "gt.txt", pos)
    pred = write_tum(tmp_path / "pred.txt", pos)
    with pytest.raises(InvalidTrajectoryError, match="associate_max_diff"):
        BACKEND.evaluate_ate(pred, gt, "none", {})


def test_malformed_tum_file_fails_naming_path(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("this is not a trajectory\n1 2 3\n")
    with pytest.raises(InvalidTrajectoryError) as exc:
        BACKEND.load_trajectory(bad)
    assert str(bad) in str(exc.value)
    assert "TUM" in str(exc.value)


def test_unknown_alignment_mode_rejected(tmp_path: Path) -> None:
    pos = helix_positions(5)
    gt = write_tum(tmp_path / "gt.txt", pos)
    pred = write_tum(tmp_path / "pred.txt", pos)
    with pytest.raises(AlignmentError, match="sim3"):
        BACKEND.evaluate_ate(pred, gt, "icp", ASSOCIATION)


def test_backend_info_records_evo_version() -> None:
    info = BACKEND.backend_info()
    assert info.kind == "trajectory"
    assert info.name == "evo" and info.library == "evo"
    assert info.version and not info.version.startswith("v")


# --- metric layer -------------------------------------------------------------------


def _ate_result(tmp_path: Path, mode: str = "trajectory_sim3") -> dict:
    pos = helix_positions()
    gt = write_tum(tmp_path / "gt.txt", pos)
    pred = write_tum(tmp_path / "pred.txt", 2.0 * pos)
    return BACKEND.evaluate_ate(pred, gt, mode, ASSOCIATION)


def test_pose_metric_result_carries_alignment_and_association(tmp_path: Path) -> None:
    out = _ate_result(tmp_path)
    result = pose_metric_result(
        MetricSpec(name="ate", statistic="rmse"), out,
        protocol="single_pose", protocol_hash="sha256:x",
        backend_name="evo", scene_id="scene",
    )
    assert result.value == pytest.approx(0.0, abs=1e-9)
    assert result.unit == "m"
    assert result.statistic == "rmse"
    assert result.n_points_pred == 20 and result.n_points_gt == 20
    assert result.metadata["alignment_mode"] == "trajectory_sim3"
    assert result.metadata["alignment_scale"] == pytest.approx(0.5, abs=1e-9)
    assert result.metadata["n_associated"] == 20
    assert result.metadata["association"]["associate_max_diff"] == 0.01
    assert result.metadata["statistics"]["rmse"] == result.value


def test_alignment_scale_error_metric_is_abs_log_scale(tmp_path: Path) -> None:
    out = _ate_result(tmp_path)
    result = pose_metric_result(
        MetricSpec(name="alignment_scale_error"), out,
        protocol="single_pose", protocol_hash="sha256:x",
        backend_name="evo", scene_id="scene",
    )
    assert result.value == pytest.approx(math.log(2.0), abs=1e-9)  # |ln 0.5|
    assert result.unit is None and result.statistic is None


def test_alignment_scale_error_formula_and_domain() -> None:
    assert alignment_scale_error(1.0) == 0.0
    assert alignment_scale_error(2.0) == pytest.approx(math.log(2.0))
    assert alignment_scale_error(0.5) == pytest.approx(math.log(2.0))
    with pytest.raises(MetricError, match="positive"):
        alignment_scale_error(0.0)
    with pytest.raises(MetricError, match="positive"):
        alignment_scale_error(float("nan"))


def test_statistic_is_never_defaulted(tmp_path: Path) -> None:
    out = _ate_result(tmp_path)
    with pytest.raises(MetricError, match="statistic"):
        pose_metric_result(
            MetricSpec(name="ate"), out,
            protocol="single_pose", protocol_hash="sha256:x",
            backend_name="evo", scene_id="scene",
        )


def test_unsupported_pose_metric_name_rejected(tmp_path: Path) -> None:
    out = _ate_result(tmp_path)
    with pytest.raises(MetricError, match="unsupported pose metric"):
        pose_metric_result(
            MetricSpec(name="absrel"), out,
            protocol="single_pose", protocol_hash="sha256:x",
            backend_name="evo", scene_id="scene",
        )


def test_rpe_parameters_are_required_and_explicit() -> None:
    spec = MetricSpec(
        name="rpe_translation", statistic="rmse",
        parameters={"delta": 1, "delta_unit": "frames"},
    )
    assert require_rpe_parameters(spec) == (1.0, "frames", False)
    with pytest.raises(MetricError, match="never defaulted"):
        require_rpe_parameters(MetricSpec(name="rpe_translation", statistic="rmse"))


def test_check_pose_alignment_spec_rejects_non_trajectory_modes() -> None:
    check_pose_alignment_spec(
        AlignmentSpec(
            mode="trajectory_sim3", estimate_on="trajectory", solver="evo",
            granularity="per_scene",
        )
    )
    for mode in ("icp", "scale_median", "se3"):
        with pytest.raises(AlignmentError):
            check_pose_alignment_spec(
                AlignmentSpec(mode=mode, estimate_on="trajectory", granularity="per_scene")  # type: ignore[arg-type]
            )
    with pytest.raises(AlignmentError, match="per_scene"):
        check_pose_alignment_spec(
            AlignmentSpec(
                mode="trajectory_se3", estimate_on="trajectory",
                granularity="per_frame",
            )
        )
    assert POSE_ALIGNMENT_MODES == ("none", "trajectory_se3", "trajectory_sim3")
