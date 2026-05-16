"""Pyrender-based renderer (mesh, point cloud, side-by-side compare).

Pyrender is loaded lazily via :func:`eval3r.utils.optional_import` so the base
install does not depend on it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from eval3r.io.geometry import MeshData, PointCloudData
from eval3r.render.camera import default_intrinsics, default_orbit_pose
from eval3r.utils.optional import optional_import
from eval3r.utils.typing import PathLike


def _enable_headless() -> None:
    if "PYOPENGL_PLATFORM" not in os.environ:
        os.environ["PYOPENGL_PLATFORM"] = "egl"


def _build_scene(
    pyrender,
    *,
    geom: MeshData | PointCloudData,
    camera_pose: np.ndarray,
    intrinsics: np.ndarray | None,
    image_size: tuple[int, int],
):
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.4, 0.4, 0.4])
    if isinstance(geom, MeshData):
        import trimesh

        kwargs: dict = {
            "vertices": geom.vertices,
            "faces": geom.faces,
            "process": False,
        }
        if geom.vertex_colors is not None:
            kwargs["vertex_colors"] = geom.vertex_colors
        tm = trimesh.Trimesh(**kwargs)
        scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False))
    else:
        colors = (
            geom.colors.astype(np.float32) / 255.0
            if geom.colors is not None
            else np.tile([0.4, 0.5, 0.9], (len(geom.points), 1))
        )
        scene.add(pyrender.Mesh.from_points(geom.points, colors=colors))

    if intrinsics is None:
        intrinsics = default_intrinsics(image_size)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=1000.0)
    scene.add(cam, pose=camera_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)
    return scene


def _render_scene(
    pyrender,
    scene,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Render a scene and return ``(color, depth)``.

    ``depth`` is the per-pixel distance along the camera's viewing direction,
    in the same units as the geometry, with ``0`` for pixels that miss the
    scene. Pyrender produces depth in the OpenGL camera frame.
    """
    w, h = image_size
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    try:
        color, depth = renderer.render(scene)
    finally:
        renderer.delete()
    return color, depth


def _save_image(image: np.ndarray, path: PathLike) -> Path:
    imageio = optional_import("imageio.v3", extra="render")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out, image)
    return out


def render_geometry(
    geom: MeshData | PointCloudData,
    *,
    out_path: PathLike,
    image_size: tuple[int, int] = (640, 480),
    camera_pose: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    headless: bool = True,
) -> Path:
    if headless:
        _enable_headless()
    pyrender = optional_import("pyrender", extra="render")
    pts = geom.vertices if isinstance(geom, MeshData) else geom.points
    if camera_pose is None:
        camera_pose = default_orbit_pose(pts.min(0), pts.max(0))
    scene = _build_scene(
        pyrender,
        geom=geom,
        camera_pose=camera_pose,
        intrinsics=intrinsics,
        image_size=image_size,
    )
    img, _ = _render_scene(pyrender, scene, image_size)
    return _save_image(img, out_path)


def render_compare(
    pred: MeshData | PointCloudData,
    gt: MeshData | PointCloudData,
    *,
    out_path: PathLike,
    image_size: tuple[int, int] = (640, 480),
    camera_pose: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    headless: bool = True,
) -> Path:
    if headless:
        _enable_headless()
    pyrender = optional_import("pyrender", extra="render")
    pred_pts = pred.vertices if isinstance(pred, MeshData) else pred.points
    gt_pts = gt.vertices if isinstance(gt, MeshData) else gt.points
    bbox_min = np.minimum(pred_pts.min(0), gt_pts.min(0))
    bbox_max = np.maximum(pred_pts.max(0), gt_pts.max(0))
    if camera_pose is None:
        camera_pose = default_orbit_pose(bbox_min, bbox_max)

    panels = []
    for g in (pred, gt):
        scene = _build_scene(
            pyrender,
            geom=g,
            camera_pose=camera_pose,
            intrinsics=intrinsics,
            image_size=image_size,
        )
        color, _ = _render_scene(pyrender, scene, image_size)
        panels.append(color)
    side = np.concatenate(panels, axis=1)
    return _save_image(side, out_path)
