"""Task 013 unit tests: pycolmap camera backend (COLMAP text parsing + normalization)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval3r.backends.camera_pycolmap import PycolmapCameraBackend
from eval3r.core.errors import DatasetError
from eval3r.core.registry import default_registry

CALIB = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "eth3d_tiny" / "dataset_root" / "courtyard"
    / "dslr_calibration_undistorted"
)


def _cameras():
    return PycolmapCameraBackend().load_cameras(CALIB)


def test_registered_in_backend_registry() -> None:
    assert "pycolmap" in default_registry().available("camera")


def test_backend_info_reports_pycolmap_version() -> None:
    info = PycolmapCameraBackend().backend_info()
    assert info.kind == "camera" and info.name == "pycolmap"
    assert info.library == "pycolmap"
    assert info.approximate is False


def test_load_cameras_parses_models_and_images() -> None:
    cams = _cameras()
    assert cams.camera_models == {1: "PINHOLE", 2: "THIN_PRISM_FISHEYE"}
    assert [im.name for im in cams.images] == [
        "dslr_images_undistorted/DSC_0001.JPG",
        "dslr_images_undistorted/DSC_0002.JPG",
    ]
    assert cams.source_pose_format == "world_to_cam_colmap"
    assert cams.normalized_convention == "cam_to_world_opencv_meters"


def test_world_to_cam_colmap_normalizes_to_hand_computed_cam_to_world() -> None:
    # Fixture image 1: q = (w=1/sqrt(2), x=0, y=1/sqrt(2), z=0), t = (1, 2, 3).
    # R_w2c = [[0,0,1],[0,1,0],[-1,0,0]]; C = -R^T t = (3, -2, -1).
    cams = _cameras()
    pose = next(im for im in cams.images if im.image_id == 1)
    expected_rotation = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(pose.cam_to_world[:3, :3], expected_rotation, atol=1e-12)
    np.testing.assert_allclose(pose.cam_to_world[:3, 3], [3.0, -2.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(pose.cam_to_world[3], [0.0, 0.0, 0.0, 1.0], atol=0)


def test_identity_pose_with_translation() -> None:
    # Fixture image 2: identity rotation, t = (0.5, 0, 0) -> C = (-0.5, 0, 0).
    cams = _cameras()
    pose = next(im for im in cams.images if im.image_id == 2)
    np.testing.assert_allclose(pose.cam_to_world[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(pose.cam_to_world[:3, 3], [-0.5, 0.0, 0.0], atol=1e-12)


def test_pinhole_intrinsics_for_pinhole_camera() -> None:
    cams = _cameras()
    K = PycolmapCameraBackend().pinhole_intrinsics(cams.cameras[1])
    np.testing.assert_allclose(K, [[500, 0, 320], [0, 500, 240], [0, 0, 1]])


def test_non_pinhole_model_refused_not_approximated() -> None:
    cams = _cameras()
    backend = PycolmapCameraBackend()
    with pytest.raises(DatasetError, match="THIN_PRISM_FISHEYE.*not a simple pinhole"):
        backend.pinhole_intrinsics(cams.cameras[2])
    # ...and the limitation is recorded on the camera set itself.
    assert any("THIN_PRISM_FISHEYE" in note for note in cams.notes)
    assert cams.non_pinhole_models == {2: "THIN_PRISM_FISHEYE"}


def test_missing_model_directory_errors() -> None:
    with pytest.raises(DatasetError, match="COLMAP model directory does not exist"):
        PycolmapCameraBackend().load_cameras(CALIB / "nope")
