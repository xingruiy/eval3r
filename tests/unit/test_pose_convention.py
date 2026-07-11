"""Unit tests for the pose/coordinate convention transformer and source mapping.

Covers the geometric core (``F`` involution, axis-only centre preservation, direction
== inverse, all 8 axes x direction round-trips), the TUM path, the world-frame path,
the always-on truthfulness validators, and the ``SourcePoseFormat`` mapping (including
the deliberately-unmapped formats that must raise).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from eval3r.core.errors import PoseConventionError
from eval3r.core.pose_convention import (
    INTERNAL_POSE_CONVENTION,
    F,
    PoseConvention,
    PoseConventionTransform,
    read_tum_rows,
)
from eval3r.datasets.conventions import (
    convention_for,
    normalized_convention_target,
)
from eval3r.predictions.writer import write_tum_trajectory

_ALL_CONVENTIONS = [
    PoseConvention(axes=a, direction=d)
    for a in ("opencv", "opengl")
    for d in ("cam_to_world", "world_to_cam")
]


def _random_c2w_opencv(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mats = np.repeat(np.eye(4)[None], n, axis=0)
    mats[:, :3, :3] = Rotation.random(n, random_state=seed).as_matrix()
    mats[:, :3, 3] = rng.normal(size=(n, 3)) * 2.0
    return mats


# --- geometric core ------------------------------------------------------------


def test_F_is_an_involutive_proper_rotation() -> None:
    assert np.allclose(F @ F, np.eye(4))
    assert np.isclose(np.linalg.det(F[:3, :3]), 1.0)  # proper rotation, not reflection


def test_axis_only_change_preserves_camera_centre() -> None:
    T = PoseConventionTransform()
    mats = _random_c2w_opencv(6, seed=1)
    out = T.convert(
        mats,
        PoseConvention("opencv", "cam_to_world"),
        PoseConvention("opengl", "cam_to_world"),
    )
    # Right-multiplying a c2w pose by F rotates the camera axes but never moves the
    # camera centre (the translation column).
    assert np.allclose(out[:, :3, 3], mats[:, :3, 3])


def test_direction_change_equals_matrix_inverse() -> None:
    T = PoseConventionTransform()
    mats = _random_c2w_opencv(5, seed=2)
    w2c = T.convert(
        mats,
        PoseConvention("opencv", "cam_to_world"),
        PoseConvention("opencv", "world_to_cam"),
    )
    for c, w in zip(mats, w2c, strict=True):
        assert np.allclose(np.linalg.inv(c), w, atol=1e-9)


@pytest.mark.parametrize("src", _ALL_CONVENTIONS)
@pytest.mark.parametrize("dst", _ALL_CONVENTIONS)
def test_all_pairs_round_trip(src: PoseConvention, dst: PoseConvention) -> None:
    T = PoseConventionTransform()
    canonical = _random_c2w_opencv(7, seed=3)
    src_mats = T.from_internal(canonical, src)
    out = T.convert(src_mats, src, dst)
    back = T.convert(out, dst, src)
    assert np.allclose(back, src_mats, atol=1e-9)


def test_colmap_w2c_to_c2w_is_the_inverse() -> None:
    T = PoseConventionTransform()
    canonical = _random_c2w_opencv(4, seed=4)
    colmap = convention_for("world_to_cam_colmap")
    assert colmap == PoseConvention("opencv", "world_to_cam")
    w2c = T.from_internal(canonical, colmap)
    c2w = T.convert(w2c, colmap, INTERNAL_POSE_CONVENTION)
    assert np.allclose(c2w, canonical, atol=1e-9)


def test_single_matrix_returns_single_matrix() -> None:
    T = PoseConventionTransform()
    m = _random_c2w_opencv(1, seed=5)[0]
    out = T.convert(
        m,
        PoseConvention("opencv", "cam_to_world"),
        PoseConvention("opengl", "cam_to_world"),
    )
    assert out.shape == (4, 4)


# --- TUM path ------------------------------------------------------------------


def test_tum_round_trip_preserves_timestamps_and_poses() -> None:
    T = PoseConventionTransform()
    mats = _random_c2w_opencv(8, seed=6)
    ts = np.arange(8, dtype=float) + 0.5
    rows = T.matrices_to_tum_rows(ts, mats)
    assert rows.shape == (8, 8)
    assert np.allclose(rows[:, 0], ts)
    back = T.tum_rows_to_matrices(rows)
    assert np.allclose(back, mats, atol=1e-9)
    # quaternions are unit
    assert np.allclose(np.linalg.norm(rows[:, 4:8], axis=1), 1.0)


def test_convert_tum_rows_matches_matrix_path() -> None:
    T = PoseConventionTransform()
    canonical = _random_c2w_opencv(6, seed=7)
    ts = np.linspace(0.0, 1.0, 6)
    ogl = PoseConvention("opengl", "cam_to_world")
    ocv = PoseConvention("opencv", "cam_to_world")
    ogl_rows = T.matrices_to_tum_rows(ts, T.from_internal(canonical, ogl))
    conv_rows = T.convert_tum_rows(ogl_rows, ogl, ocv)
    assert np.allclose(conv_rows[:, 0], ts)  # timestamps preserved
    assert np.allclose(T.tum_rows_to_matrices(conv_rows), canonical, atol=1e-9)


def test_read_tum_rows_round_trips_a_written_file(tmp_path) -> None:
    T = PoseConventionTransform()
    mats = _random_c2w_opencv(5, seed=8)
    ts = np.arange(5, dtype=float) + 1.0
    rows = T.matrices_to_tum_rows(ts, mats)
    path = tmp_path / "traj.txt"
    write_tum_trajectory(rows, path)
    read = read_tum_rows(path)
    assert np.allclose(read, rows, atol=1e-6)


# --- world-frame path ----------------------------------------------------------


def test_world_transform_identity_and_flip() -> None:
    T = PoseConventionTransform()
    assert np.allclose(T.world_transform("opencv", "opencv"), np.eye(4))
    assert np.allclose(T.world_transform("opengl", "opencv"), F)


def test_world_transform_flips_a_point_cloud() -> None:
    T = PoseConventionTransform()
    m = T.world_transform("opengl", "opencv")
    pts = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    flipped = pts @ m[:3, :3].T + m[:3, 3]
    assert np.allclose(flipped, pts * np.array([1.0, -1.0, -1.0]))


# --- validation guards ---------------------------------------------------------


def _bad(matrix_edit) -> np.ndarray:
    m = np.eye(4)
    matrix_edit(m)
    return m


@pytest.mark.parametrize(
    "edit, needle",
    [
        (lambda m: m.__setitem__((0, 0), -1.0), "determinant"),  # reflection
        (lambda m: m.__setitem__((0, 0), 2.0), "orthonormal"),  # scaled row
        (lambda m: m.__setitem__((0, 3), np.nan), "NaN"),  # non-finite
        (lambda m: m.__setitem__((3, 0), 0.5), "homogeneous"),  # broken bottom row
    ],
)
def test_convert_rejects_invalid_input(edit, needle: str) -> None:
    T = PoseConventionTransform()
    with pytest.raises(PoseConventionError) as exc:
        T.convert(
            _bad(edit),
            PoseConvention("opencv", "cam_to_world"),
            PoseConvention("opengl", "cam_to_world"),
        )
    assert needle.lower() in str(exc.value).lower()


def test_convert_accepts_text_precision_rotation_noise() -> None:
    """Real exported poses (ScanNet SensReader text, float32) carry ~1e-6..1e-5
    orthonormality error; the acceptance gate (ORTHO_TOL=1e-4) must pass them
    while still rejecting genuinely corrupt rotations (task 028 real-data run)."""
    rng = np.random.default_rng(28)
    m = np.eye(4)
    # perturb the rotation block at just above the old 1e-6 gate
    m[:3, :3] += rng.normal(scale=2e-6, size=(3, 3))
    T = PoseConventionTransform()
    out = T.convert(
        m,
        PoseConvention("opencv", "cam_to_world"),
        PoseConvention("opengl", "cam_to_world"),
    )
    assert out.shape == (4, 4)

    corrupt = np.eye(4)
    corrupt[0, 1] = 5e-3  # shear well past ORTHO_TOL: not a rotation
    with pytest.raises(PoseConventionError) as exc:
        T.convert(
            corrupt,
            PoseConvention("opencv", "cam_to_world"),
            PoseConvention("opengl", "cam_to_world"),
        )
    assert "orthonormal" in str(exc.value).lower()


# --- source-format mapping -----------------------------------------------------


@pytest.mark.parametrize(
    "fmt, expected",
    [
        ("cam_to_world_opencv", PoseConvention("opencv", "cam_to_world")),
        ("cam_to_world_opengl", PoseConvention("opengl", "cam_to_world")),
        ("world_to_cam_opencv", PoseConvention("opencv", "world_to_cam")),
        ("world_to_cam_opengl", PoseConvention("opengl", "world_to_cam")),
        ("world_to_cam_colmap", PoseConvention("opencv", "world_to_cam")),
        ("world_to_cam_mvsnet", PoseConvention("opencv", "world_to_cam")),
        ("kitti360_cam0_to_world", PoseConvention("opencv", "cam_to_world")),
        ("tum", PoseConvention("opencv", "cam_to_world")),
    ],
)
def test_convention_for_maps_known_formats(fmt: str, expected: PoseConvention) -> None:
    assert convention_for(fmt) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("fmt", ["unknown", "co3d_frame_annotations", "tanks_temples_log"])
def test_convention_for_raises_for_unverified_formats(fmt: str) -> None:
    with pytest.raises(PoseConventionError):
        convention_for(fmt)  # type: ignore[arg-type]


def test_normalized_convention_target_is_internal() -> None:
    assert normalized_convention_target("cam_to_world_opencv_meters") == INTERNAL_POSE_CONVENTION
