"""Integration tests: convention transforms wired into the pose / geometry runners.

Exercises the auto-pickup-and-transform flow end to end through the public API: a
prediction declared in a different convention is transformed to the protocol's internal
convention *before* metrics, so a same-trajectory / same-cloud prediction scores near
zero once the convention is declared and large when it is not.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from eval3r.api import evaluate_geometry, evaluate_pose
from eval3r.core.errors import SceneEvaluationError
from eval3r.core.pose_convention import F, PoseConvention, PoseConventionTransform
from eval3r.predictions.writer import write_tum_trajectory


def _same_cameras(tmp_path, dst_convention: PoseConvention, seed: int = 1):
    """Write a GT (c2w OpenCV) and a prediction of the *same* cameras in another convention."""
    T = PoseConventionTransform()
    n = 12
    mats = np.repeat(np.eye(4)[None], n, axis=0)
    mats[:, :3, :3] = Rotation.random(n, random_state=seed).as_matrix()
    mats[:, :3, 3] = np.random.default_rng(seed).normal(size=(n, 3)) * 2.0
    ts = np.arange(n, dtype=float) + 1.0

    gt = tmp_path / "gt.txt"
    pred = tmp_path / "pred.txt"
    write_tum_trajectory(T.matrices_to_tum_rows(ts, mats), gt)
    write_tum_trajectory(T.matrices_to_tum_rows(ts, T.from_internal(mats, dst_convention)), pred)
    return pred, gt


def test_direction_mismatch_fixed_by_declaring_convention(tmp_path) -> None:
    # Prediction stored world-to-camera (Tcw); the direction flip moves the camera
    # centres, so ATE-translation discriminates it.
    pred, gt = _same_cameras(tmp_path, PoseConvention("opencv", "world_to_cam"))

    no_flag = evaluate_pose(pred, gt, align="none")
    flagged = evaluate_pose(pred, gt, align="none", pred_pose_format="world_to_cam_opencv")

    assert no_flag.metrics["ate"] > 1e-2
    assert flagged.metrics["ate"] < 1e-6
    conv = flagged.metadata["convention"]
    assert conv["pred_pose_format"] == "world_to_cam_opencv"
    assert conv["pred_transformed"] is True
    assert conv["target"] == "opencv/cam_to_world"


def test_axis_mismatch_fixed_and_seen_by_rotation_metric(tmp_path) -> None:
    # An OpenCV<->OpenGL axis flip preserves camera centres (ATE-translation is blind
    # to it) but corrupts orientation, so RPE-rotation is the discriminator.
    pred, gt = _same_cameras(tmp_path, PoseConvention("opengl", "cam_to_world"))

    no_flag = evaluate_pose(pred, gt, align="none")
    flagged = evaluate_pose(pred, gt, align="none", pred_pose_format="cam_to_world_opengl")

    assert no_flag.metrics["rpe_rotation"] > 1.0
    assert flagged.metrics["rpe_rotation"] < 1e-4
    assert flagged.metadata["convention"]["pred_transformed"] is True


def test_passthrough_when_convention_matches_target(tmp_path) -> None:
    pred, gt = _same_cameras(tmp_path, PoseConvention("opencv", "cam_to_world"))
    run = evaluate_pose(pred, gt, align="none", pred_pose_format="cam_to_world_opencv")
    assert run.metrics["ate"] < 1e-6
    assert run.metadata["convention"]["pred_transformed"] is False


def test_convert_stage_fails_for_unmapped_format(tmp_path) -> None:
    pred, gt = _same_cameras(tmp_path, PoseConvention("opencv", "cam_to_world"))
    with pytest.raises(SceneEvaluationError) as exc:
        evaluate_pose(pred, gt, align="none", pred_pose_format="co3d_frame_annotations")
    assert exc.value.stage == "convert"
    assert "co3d" in exc.value.reason.lower()


# --- geometry world-frame ------------------------------------------------------


def _write_ply(path, points: np.ndarray) -> None:
    from eval3r.backends.pointcloud_plyfile import PlyfilePointCloudBackend

    PlyfilePointCloudBackend().save_pointcloud(np.asarray(points, dtype=np.float64), path)


def test_geometry_world_frame_flip_scores_near_zero(tmp_path) -> None:
    rng = np.random.default_rng(0)
    gt_pts = rng.normal(size=(500, 3))
    # The prediction is the same cloud expressed in an OpenGL world frame (flipped by F).
    pred_pts = gt_pts @ F[:3, :3].T
    gt = tmp_path / "gt.ply"
    pred = tmp_path / "pred.ply"
    _write_ply(gt, gt_pts)
    _write_ply(pred, pred_pts)

    # Default (opencv, passthrough): the flipped cloud is far from GT.
    passthrough = evaluate_geometry(pred, gt, input_type="pointcloud", gt_type="pointcloud")
    assert passthrough.metrics["chamfer"] > 1e-2

    # Declaring the OpenGL world frame rotates it back and chamfer collapses to ~0.
    flipped = evaluate_geometry(
        pred, gt, input_type="pointcloud", gt_type="pointcloud", pred_world_frame="opengl"
    )
    assert flipped.metrics["chamfer"] < 1e-9
