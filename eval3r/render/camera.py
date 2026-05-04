"""Camera utilities for rendering: default orbit pose + convention conversion.

Renderers (pyrender, OpenGL) consume an OpenGL camera-to-world pose.
The user's manifest may store either ``T_wc`` (camera-to-world) or
``T_cw`` (world-to-camera) and the camera frame may be OpenCV (X right,
Y down, Z forward) instead of OpenGL (X right, Y up, Z back).

Use :func:`to_pyrender_pose` to convert; never apply silent flips elsewhere.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

CameraFrame = Literal["opengl", "opencv"]
PoseFrame = Literal["T_wc", "T_cw"]


# OpenCV → OpenGL camera-frame change-of-basis: flip Y and Z.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def to_pyrender_pose(
    pose: np.ndarray,
    *,
    pose_convention: PoseFrame,
    camera_frame: CameraFrame,
) -> np.ndarray:
    """Convert a 4x4 pose into the OpenGL camera-to-world form pyrender expects."""
    P = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    if pose_convention == "T_cw":
        P = np.linalg.inv(P)
    if camera_frame == "opencv":
        P = P @ _CV_TO_GL
    return P


def default_orbit_pose(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    *,
    distance_factor: float = 2.5,
    azimuth_deg: float = 35.0,
    elevation_deg: float = 25.0,
) -> np.ndarray:
    """A simple OpenGL camera-to-world pose looking at the bbox centre."""
    center = 0.5 * (bbox_min + bbox_max)
    radius = float(np.linalg.norm(bbox_max - bbox_min)) * 0.5
    if radius == 0:
        radius = 1.0
    distance = radius * distance_factor

    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    eye = center + distance * np.array(
        [np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)],
        dtype=np.float64,
    )

    forward = center - eye
    forward /= np.linalg.norm(forward) + 1e-12
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    pose = np.eye(4)
    # OpenGL camera looks down -Z, so columns are [right, up, -forward].
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def default_intrinsics(image_size: tuple[int, int], fov_y_deg: float = 45.0) -> np.ndarray:
    """Build a simple K matrix from a vertical FOV."""
    w, h = image_size
    f = 0.5 * h / np.tan(np.deg2rad(fov_y_deg) * 0.5)
    K = np.array(
        [[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )
    return K
