"""Task 018 unit tests: align-stage dispatch (ICP + trajectory-first paths).

Covers the new solver semantics: ``mode: icp`` refusal, required-parameter
enforcement (``max_correspondence_distance`` / ``associate_max_diff`` are never
defaulted), metric-scale Sim3 alignment, analytic trajectory-first propagation
via the real evo backend, and the visualization capture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval3r.backends.registration_open3d import Open3dRegistrationBackend
from eval3r.backends.trajectory_evo import EvoTrajectoryBackend
from eval3r.core.errors import AlignmentError
from eval3r.core.schema import AlignmentSpec
from eval3r.pipeline.stages.align import (
    align_geometry,
    capture_alignment_vis,
)
from eval3r.pipeline.stages.load import LoadedGeometry

REGISTRATION = Open3dRegistrationBackend()
TRAJECTORY = EvoTrajectoryBackend()


def _cloud(points: np.ndarray) -> LoadedGeometry:
    return LoadedGeometry(kind="pointcloud", path=Path("mem.ply"), points=points)


def _rotation_z(degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _write_tum(path: Path, positions: np.ndarray) -> None:
    lines = [
        f"{i * 0.1:.3f} {p[0]} {p[1]} {p[2]} 0 0 0 1" for i, p in enumerate(positions)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- refusals and required parameters -------------------------------------------


def test_mode_icp_is_refused_scale_handling_must_be_explicit() -> None:
    pts = np.random.default_rng(0).random((10, 3))
    with pytest.raises(AlignmentError, match="transform class"):
        align_geometry(
            _cloud(pts), _cloud(pts), AlignmentSpec(mode="icp"),
            scene_id="s", metric_scale=True,
        )


def test_icp_solver_requires_explicit_max_correspondence_distance() -> None:
    pts = np.random.default_rng(0).random((10, 3))
    spec = AlignmentSpec(mode="se3", solver="icp", estimate_on="pointcloud")
    with pytest.raises(AlignmentError, match="max_correspondence_distance"):
        align_geometry(
            _cloud(pts), _cloud(pts), spec,
            scene_id="s", metric_scale=True, registration_backend=REGISTRATION,
        )


def test_icp_solver_rejects_estimate_on_trajectory() -> None:
    pts = np.random.default_rng(0).random((10, 3))
    spec = AlignmentSpec(
        mode="se3", solver="icp", estimate_on="trajectory",
        parameters={"max_correspondence_distance": 0.1},
    )
    with pytest.raises(AlignmentError, match="estimates on the geometries"):
        align_geometry(
            _cloud(pts), _cloud(pts), spec,
            scene_id="s", metric_scale=True, registration_backend=REGISTRATION,
        )


def test_sim3_icp_runs_on_metric_scale_when_selected() -> None:
    pts = np.random.default_rng(0).random((10, 3))
    spec = AlignmentSpec(
        mode="sim3", solver="icp", estimate_on="pointcloud",
        parameters={"max_correspondence_distance": 0.1},
    )
    _aligned, result = align_geometry(
        _cloud(pts), _cloud(pts), spec,
        scene_id="s", metric_scale=True, registration_backend=REGISTRATION,
    )
    assert result.mode == "sim3"
    assert result.scale == pytest.approx(1.0)


def test_sim3_trajectory_runs_on_metric_scale_when_selected(tmp_path: Path) -> None:
    pts = np.random.default_rng(0).random((10, 3))
    _write_tum(tmp_path / "p.txt", pts)
    _write_tum(tmp_path / "g.txt", pts)
    spec = AlignmentSpec(
        mode="sim3", solver="umeyama", estimate_on="trajectory",
        parameters={"associate_max_diff": 0.01},
    )
    _aligned, result = align_geometry(
        _cloud(pts), _cloud(pts), spec,
        scene_id="s", metric_scale=True, trajectory_backend=TRAJECTORY,
        pred_trajectory=tmp_path / "p.txt", gt_trajectory=tmp_path / "g.txt",
    )
    assert result.mode == "sim3"
    assert result.scale == pytest.approx(1.0)


def test_trajectory_mode_requires_both_trajectories() -> None:
    pts = np.random.default_rng(0).random((10, 3))
    spec = AlignmentSpec(
        mode="se3", solver="umeyama", estimate_on="trajectory",
        parameters={"associate_max_diff": 0.01},
    )
    with pytest.raises(AlignmentError, match="ground-truth trajectory is missing"):
        align_geometry(
            _cloud(pts), _cloud(pts), spec,
            scene_id="s", metric_scale=False, trajectory_backend=TRAJECTORY,
            pred_trajectory=Path("pred.txt"), gt_trajectory=None,
        )


def test_trajectory_mode_requires_explicit_associate_max_diff(tmp_path: Path) -> None:
    pts = np.random.default_rng(0).random((10, 3))
    spec = AlignmentSpec(mode="se3", solver="umeyama", estimate_on="trajectory")
    with pytest.raises(AlignmentError, match="associate_max_diff"):
        align_geometry(
            _cloud(pts), _cloud(pts), spec,
            scene_id="s", metric_scale=False, trajectory_backend=TRAJECTORY,
            pred_trajectory=tmp_path / "p.txt", gt_trajectory=tmp_path / "g.txt",
        )


def test_missing_backend_is_reported_as_wiring_bug() -> None:
    pts = np.random.default_rng(0).random((10, 3))
    spec = AlignmentSpec(
        mode="se3", solver="icp", estimate_on="pointcloud",
        parameters={"max_correspondence_distance": 0.1},
    )
    with pytest.raises(AlignmentError, match="wiring bug"):
        align_geometry(_cloud(pts), _cloud(pts), spec, scene_id="s", metric_scale=True)


def test_corresponded_umeyama_rejects_unequal_counts_pointing_at_alternatives() -> None:
    rng = np.random.default_rng(0)
    spec = AlignmentSpec(mode="se3", solver="umeyama", estimate_on="pointcloud")
    with pytest.raises(AlignmentError, match="solver 'icp'"):
        align_geometry(
            _cloud(rng.random((10, 3))), _cloud(rng.random((12, 3))), spec,
            scene_id="s", metric_scale=True,
        )


# --- solver paths -----------------------------------------------------------------


def test_icp_path_recovers_known_sim3_and_records_provenance() -> None:
    rng = np.random.default_rng(1)
    src = rng.random((400, 3)) * 2.0
    rot = _rotation_z(15.0)
    dst = 0.5 * src @ rot.T + np.array([0.3, -0.2, 0.1])
    spec = AlignmentSpec(
        mode="sim3", solver="icp", estimate_on="pointcloud",
        parameters={"max_correspondence_distance": 0.5},
    )
    aligned, result = align_geometry(
        _cloud(src), _cloud(dst), spec,
        scene_id="scene1", metric_scale=True, registration_backend=REGISTRATION,
    )
    assert result.solver == "icp"
    assert result.scale == pytest.approx(0.5, abs=1e-6)
    assert result.fitness == pytest.approx(1.0)
    np.testing.assert_allclose(aligned.points, dst, atol=1e-8)
    # Provenance: the coarse init and subsample policy are recorded.
    assert result.parameters["init"]["method"] == "centroid+rms_radius_scale"
    assert result.parameters["subsample_seed"] == 0
    assert result.parameters["n_points_pred_total"] == 400


def test_trajectory_path_propagates_transform_to_geometry(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    rot = _rotation_z(30.0)
    t = np.array([2.0, -1.0, 0.5])

    pos_pred = np.cumsum(rng.normal(scale=0.1, size=(30, 3)), axis=0)
    _write_tum(tmp_path / "pred.txt", pos_pred)
    _write_tum(tmp_path / "gt.txt", pos_pred @ rot.T + t)

    cloud = rng.random((100, 3))
    spec = AlignmentSpec(
        mode="se3", solver="umeyama", estimate_on="trajectory",
        parameters={"associate_max_diff": 0.01},
    )
    aligned, result = align_geometry(
        _cloud(cloud), _cloud(cloud @ rot.T + t), spec,
        scene_id="scene1", metric_scale=True, trajectory_backend=TRAJECTORY,
        pred_trajectory=tmp_path / "pred.txt", gt_trajectory=tmp_path / "gt.txt",
    )
    assert result.estimate_on == "trajectory"
    assert result.scale == pytest.approx(1.0)
    assert result.n_correspondences == 30
    matrix = np.asarray(result.matrix)
    np.testing.assert_allclose(matrix[:3, :3], rot, atol=1e-9)
    np.testing.assert_allclose(matrix[:3, 3], t, atol=1e-9)
    np.testing.assert_allclose(aligned.points, cloud @ rot.T + t, atol=1e-9)
    # Association accounting is recorded for the writer.
    assert result.parameters["n_associated"] == 30
    assert result.parameters["association"]["associate_max_diff"] == 0.01


def test_trajectory_path_missing_file_error_names_the_file(tmp_path: Path) -> None:
    pts = np.random.default_rng(0).random((10, 3))
    _write_tum(tmp_path / "pred.txt", pts)
    spec = AlignmentSpec(
        mode="se3", solver="umeyama", estimate_on="trajectory",
        parameters={"associate_max_diff": 0.01},
    )
    with pytest.raises(Exception, match="gt.txt"):
        align_geometry(
            _cloud(pts), _cloud(pts), spec,
            scene_id="s", metric_scale=False, trajectory_backend=TRAJECTORY,
            pred_trajectory=tmp_path / "pred.txt", gt_trajectory=tmp_path / "gt.txt",
        )


# --- visualization capture ---------------------------------------------------------


def test_capture_alignment_vis_is_point_for_point_comparable() -> None:
    rng = np.random.default_rng(3)
    src = rng.random((200, 3))
    rot = _rotation_z(10.0)
    dst = src @ rot.T + np.array([1.0, 0.0, 0.0])
    spec = AlignmentSpec(
        mode="se3", solver="icp", estimate_on="pointcloud",
        parameters={"max_correspondence_distance": 0.5},
    )
    _aligned, result = align_geometry(
        _cloud(src), _cloud(dst), spec,
        scene_id="scene1", metric_scale=True, registration_backend=REGISTRATION,
    )
    vis = capture_alignment_vis(_cloud(src), _cloud(dst), result)
    assert vis.scene_id == "scene1"
    assert vis.pred_before.shape == vis.pred_after.shape == (200, 3)
    matrix = np.asarray(result.matrix)
    np.testing.assert_allclose(
        vis.pred_after, vis.pred_before @ matrix[:3, :3].T + matrix[:3, 3]
    )
    assert vis.alignment["mode"] == "se3"


def test_capture_alignment_vis_subsamples_deterministically() -> None:
    rng = np.random.default_rng(4)
    src = rng.random((5000, 3))
    spec_result = align_geometry(
        _cloud(src), _cloud(src), AlignmentSpec(mode="none"),
        scene_id="s", metric_scale=True,
    )[1]
    a = capture_alignment_vis(_cloud(src), _cloud(src), spec_result, max_points=100)
    b = capture_alignment_vis(_cloud(src), _cloud(src), spec_result, max_points=100)
    assert a.pred_before.shape == (100, 3)
    np.testing.assert_array_equal(a.pred_before, b.pred_before)
