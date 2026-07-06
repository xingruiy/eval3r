"""Task 018 unit tests: Open3D point-to-point ICP registration backend.

Closest-point ICP is one of the two alignment estimation paths (the other is
trajectory-first Umeyama) — there is deliberately no feature-based global
registration. Recovery tests use analytically constructed Sim3/SE3 pairs so the
expected transform is known exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval3r.backends.registration_open3d import Open3dRegistrationBackend, matrix_scale
from eval3r.core.errors import AlignmentError
from eval3r.core.registry import default_registry
from eval3r.pipeline.stages.align import centroid_scale_init


def _rotation_z(degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _sim3_pair(
    n: int = 500, *, scale: float = 1.0, degrees: float = 15.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """(source, target, R, s, t) with target = s * R @ source + t exactly."""
    rng = np.random.default_rng(seed)
    src = rng.random((n, 3)) * 2.0
    rot = _rotation_z(degrees)
    t = np.array([0.3, -0.2, 0.1])
    dst = scale * src @ rot.T + t
    return src, dst, rot, scale, t


def test_registered_in_backend_registry() -> None:
    assert default_registry().available("registration") == ["open3d"]
    info = Open3dRegistrationBackend().backend_info()
    assert info.kind == "registration"
    assert info.library == "open3d"


def test_icp_recovers_known_sim3() -> None:
    src, dst, _rot, scale, _t = _sim3_pair(scale=0.5)
    init, _ = centroid_scale_init(src, dst, with_scale=True)
    out = Open3dRegistrationBackend().icp(
        src, dst, init_transform=init,
        max_correspondence_distance=0.5, max_iterations=100, with_scaling=True,
    )
    assert out.scale == pytest.approx(scale, abs=1e-6)
    matrix = np.asarray(out.matrix)
    moved = src @ matrix[:3, :3].T + matrix[:3, 3]
    np.testing.assert_allclose(moved, dst, atol=1e-9)
    assert out.fitness == pytest.approx(1.0)
    assert out.n_correspondences == src.shape[0]


def test_icp_recovers_known_se3_with_unit_scale() -> None:
    src, dst, _rot, _scale, _t = _sim3_pair(scale=1.0, degrees=20.0)
    init, _ = centroid_scale_init(src, dst, with_scale=False)
    out = Open3dRegistrationBackend().icp(
        src, dst, init_transform=init,
        max_correspondence_distance=0.5, max_iterations=100, with_scaling=False,
    )
    assert out.scale == 1.0  # rigid ICP: no scale is estimated
    assert matrix_scale(np.asarray(out.matrix)) == pytest.approx(1.0, abs=1e-9)
    matrix = np.asarray(out.matrix)
    moved = src @ matrix[:3, :3].T + matrix[:3, 3]
    np.testing.assert_allclose(moved, dst, atol=1e-9)


def test_icp_is_deterministic() -> None:
    src, dst, *_ = _sim3_pair(scale=0.8)
    init, _ = centroid_scale_init(src, dst, with_scale=True)
    backend = Open3dRegistrationBackend()
    kwargs = dict(
        init_transform=init, max_correspondence_distance=0.5,
        max_iterations=100, with_scaling=True,
    )
    a = backend.icp(src, dst, **kwargs)
    b = backend.icp(src, dst, **kwargs)
    # No RANSAC/seeded randomness; Open3D's multithreaded reductions are only
    # reproducible to floating-point summation order (~1e-15).
    np.testing.assert_allclose(np.asarray(a.matrix), np.asarray(b.matrix), atol=1e-12)
    assert a.fitness == pytest.approx(b.fitness, abs=1e-12)
    assert a.inlier_rmse == pytest.approx(b.inlier_rmse, abs=1e-12)


def test_icp_records_every_shaping_parameter() -> None:
    src, dst, *_ = _sim3_pair()
    init, _ = centroid_scale_init(src, dst, with_scale=True)
    out = Open3dRegistrationBackend().icp(
        src, dst, init_transform=init,
        max_correspondence_distance=0.25, max_iterations=42, with_scaling=True,
    )
    params = out.parameters
    assert params["estimation"] == "point_to_point"
    assert params["with_scaling"] is True
    assert params["max_correspondence_distance"] == 0.25
    assert params["max_iterations"] == 42
    assert params["init_transform"] == init.tolist()
    assert params["n_source_points"] == src.shape[0]
    assert params["n_target_points"] == dst.shape[0]


def test_empty_input_fails_explicitly() -> None:
    with pytest.raises(AlignmentError, match="non-empty"):
        Open3dRegistrationBackend().icp(
            np.zeros((0, 3)), np.ones((5, 3)), init_transform=np.eye(4),
            max_correspondence_distance=0.1, max_iterations=10,
        )


def test_nonfinite_input_fails_explicitly() -> None:
    bad = np.array([[0.0, 0.0, np.nan], [1.0, 1.0, 1.0]])
    with pytest.raises(AlignmentError, match="NaN/Inf"):
        Open3dRegistrationBackend().icp(
            bad, np.ones((5, 3)), init_transform=np.eye(4),
            max_correspondence_distance=0.1, max_iterations=10,
        )


def test_nonpositive_max_correspondence_distance_fails() -> None:
    src, dst, *_ = _sim3_pair(n=10)
    with pytest.raises(AlignmentError, match="max_correspondence_distance"):
        Open3dRegistrationBackend().icp(
            src, dst, init_transform=np.eye(4),
            max_correspondence_distance=0.0, max_iterations=10,
        )


def test_bad_init_shape_fails() -> None:
    src, dst, *_ = _sim3_pair(n=10)
    with pytest.raises(AlignmentError, match="4x4"):
        Open3dRegistrationBackend().icp(
            src, dst, init_transform=np.eye(3),
            max_correspondence_distance=0.1, max_iterations=10,
        )


def test_no_correspondences_fails_with_guidance() -> None:
    # Two clusters far apart with a tiny correspondence distance and identity init:
    # ICP finds no pairs, which must be an explicit error, not a silent identity.
    src = np.random.default_rng(0).random((50, 3))
    dst = src + 100.0
    with pytest.raises(AlignmentError, match="no correspondences"):
        Open3dRegistrationBackend().icp(
            src, dst, init_transform=np.eye(4),
            max_correspondence_distance=1e-3, max_iterations=10,
        )


def test_matrix_scale_rejects_degenerate_transform() -> None:
    reflected = np.diag([-1.0, 1.0, 1.0, 1.0])
    with pytest.raises(AlignmentError, match="degenerate"):
        matrix_scale(reflected)
