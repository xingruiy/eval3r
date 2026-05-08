"""Volumetric occlusion mask generation via voxel-centric TSDF carving.

Two pluggable methods produce the same :class:`OcclusionMask` format:

- :func:`from_depth`    — back-project sensor depth using poses + intrinsics.
- :func:`from_rendered` — render a GT mesh per camera, use the rendered depth.

Both share a two-stage core:

1. **Bbox** — the union of camera frustum corners (image bounds × `[near, max_depth]`)
   in world space, padded by ``margin``. Inspired by NeuralRecon's
   ``tools/tsdf_fusion/generate_gt.py``: no depth I/O, constant per-frame cost,
   robust against depth holes.
2. **Carve** — project every voxel centre through ``T_cw`` and ``K`` to look up
   the observed depth at the projected pixel. The voxel is **visible** iff it
   sits in the camera frustum **and** its camera-frame Z is
   ``≤ depth_at_pixel + truncation`` (free space + thin TSDF band past surface).
   Voxels behind the observed surface stay occluded.

The consumer (:func:`eval3r.metrics.occlusion.filter_visible_points`) samples
the mask with trilinear interpolation, so visible voxels are dilated by one
cell by default; pass ``dilation=0`` for a crisp mask.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from eval3r.io.geometry import MeshData
from eval3r.metrics.occlusion import OcclusionMask
from eval3r.render.camera import CameraFrame, PoseFrame, to_pyrender_pose
from eval3r.utils.optional import optional_import
from eval3r.utils.typing import PathLike


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _points_to_mask(
    points: np.ndarray,
    *,
    voxel_size: float,
    margin: float,
    bbox: tuple[np.ndarray, np.ndarray] | None = None,
    dilation: int = 1,
    source: str = "<generated>",
) -> OcclusionMask:
    """Voxelize world-space points into an :class:`OcclusionMask`.

    Kept as a private helper exercised by tests; the public API
    (:func:`from_depth`, :func:`from_rendered`) does not call it.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if len(points) == 0:
        raise ValueError("Cannot build occlusion mask from zero points")
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")
    if dilation < 0:
        raise ValueError(f"dilation must be ≥ 0, got {dilation}")

    if bbox is None:
        bbox_min = points.min(axis=0) - margin
        bbox_max = points.max(axis=0) + margin
    else:
        bbox_min = np.asarray(bbox[0], dtype=np.float64).reshape(3)
        bbox_max = np.asarray(bbox[1], dtype=np.float64).reshape(3)

    extent = bbox_max - bbox_min
    if (extent <= 0).any():
        raise ValueError(
            f"bbox has non-positive extent {extent}; check input points or margin"
        )

    dims = np.maximum(np.ceil(extent / voxel_size).astype(int), 1)

    rel = (points - bbox_min) / voxel_size
    idx = np.floor(rel + 0.5).astype(np.int64)
    inside = (
        (idx[:, 0] >= 0) & (idx[:, 0] < dims[0])
        & (idx[:, 1] >= 0) & (idx[:, 1] < dims[1])
        & (idx[:, 2] >= 0) & (idx[:, 2] < dims[2])
    )
    idx = idx[inside]

    visible = np.zeros(tuple(dims), dtype=bool)
    if len(idx) > 0:
        visible[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    if dilation > 0:
        visible = binary_dilation(visible, iterations=dilation)

    grid = np.where(visible, 0.0, 1.0)
    return OcclusionMask(
        grid=grid,
        T_mask_scene=_build_T_mask_scene(bbox_min, voxel_size),
        source=source,
    )


def _build_T_mask_scene(bbox_min: np.ndarray, voxel_size: float) -> np.ndarray:
    """Return the 4×4 affine that maps world coords to voxel-grid coords."""
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = T[1, 1] = T[2, 2] = 1.0 / voxel_size
    T[:3, 3] = -np.asarray(bbox_min, dtype=np.float64) / voxel_size
    return T


def _select_frames(
    n_total: int,
    frames: Sequence[int] | str | None,
    frames_file: PathLike | None,
    frame_stride: int,
    max_frames: int | None,
) -> np.ndarray:
    """Resolve a frame-selection request to an integer index array."""
    if frames_file is not None:
        text = Path(frames_file).read_text()
        idx_list = [int(line.strip()) for line in text.splitlines() if line.strip()]
    elif isinstance(frames, str):
        idx_list = [int(s.strip()) for s in frames.split(",") if s.strip()]
    elif frames is not None:
        idx_list = [int(i) for i in frames]
    else:
        stride = max(1, int(frame_stride))
        idx_list = list(range(0, n_total, stride))

    if max_frames is not None and max_frames > 0 and len(idx_list) > max_frames:
        idx_list = idx_list[:max_frames]
    if not idx_list:
        raise ValueError("Frame selection produced zero frames")

    arr = np.asarray(idx_list, dtype=np.int64)
    bad = arr[(arr < 0) | (arr >= n_total)]
    if len(bad) > 0:
        raise ValueError(f"frame index {int(bad[0])} out of range [0, {n_total})")
    return arr


def _to_T_wc(pose: np.ndarray, pose_convention: PoseFrame) -> np.ndarray:
    """Return camera-to-world in the user's camera frame."""
    P = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    if pose_convention == "T_cw":
        return np.linalg.inv(P)
    if pose_convention == "T_wc":
        return P
    raise ValueError(f"unknown pose_convention: {pose_convention!r}")


def _to_T_cw(pose: np.ndarray, pose_convention: PoseFrame) -> np.ndarray:
    """Return world-to-camera in the user's camera frame."""
    P = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    if pose_convention == "T_cw":
        return P
    if pose_convention == "T_wc":
        return np.linalg.inv(P)
    raise ValueError(f"unknown pose_convention: {pose_convention!r}")


def _backproject(
    depth: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    *,
    camera_frame: CameraFrame,
    depth_scale: float = 1.0,
    depth_max: float | None = None,
) -> np.ndarray:
    """Back-project a single depth map to world-space points (M, 3)."""
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D (H, W), got {depth.shape}")
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    T_wc = np.asarray(T_wc, dtype=np.float64).reshape(4, 4)

    z = depth.astype(np.float64)
    if depth_scale != 1.0:
        z = z / float(depth_scale)
    valid = np.isfinite(z) & (z > 0)
    if depth_max is not None:
        valid &= z <= float(depth_max)

    vs, us = np.nonzero(valid)
    if len(vs) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    z_v = z[vs, us]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (us - cx) * z_v / fx
    y_cam = (vs - cy) * z_v / fy

    if camera_frame == "opencv":
        Z_cam = z_v
    elif camera_frame == "opengl":
        y_cam = -y_cam
        Z_cam = -z_v
    else:
        raise ValueError(f"unknown camera_frame: {camera_frame!r}")

    pts_cam = np.column_stack([x_cam, y_cam, Z_cam, np.ones_like(z_v)])
    pts_world = (T_wc @ pts_cam.T).T[:, :3]
    return np.ascontiguousarray(pts_world)


def _frustum_corners_world(
    K: np.ndarray,
    T_wc: np.ndarray,
    image_size: tuple[int, int],
    *,
    near: float,
    far: float,
    camera_frame: CameraFrame,
) -> np.ndarray:
    """Return the 8 frustum-corner world points for one camera (8, 3)."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    T_wc = np.asarray(T_wc, dtype=np.float64).reshape(4, 4)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    W, H = image_size
    corners = []
    for u in (0.0, float(W)):
        for v in (0.0, float(H)):
            for z in (float(near), float(far)):
                xc = (u - cx) * z / fx
                yc = (v - cy) * z / fy
                if camera_frame == "opencv":
                    corners.append((xc, yc, z))
                elif camera_frame == "opengl":
                    corners.append((xc, -yc, -z))
                else:
                    raise ValueError(f"unknown camera_frame: {camera_frame!r}")
    pts_cam = np.asarray(corners, dtype=np.float64)
    pts_h = np.column_stack([pts_cam, np.ones(len(pts_cam))])
    return (T_wc @ pts_h.T).T[:, :3]


def _carve_visible_grid(
    *,
    idx: np.ndarray,
    depth_maps: Sequence[np.ndarray],
    Ks: list[np.ndarray],
    poses: np.ndarray,
    pose_convention: PoseFrame,
    camera_frame: CameraFrame,
    bbox_min: np.ndarray,
    dims: np.ndarray,
    voxel_size: float,
    near: float,
    max_depth: float,
    truncation: float,
    depth_scale: float,
    depth_max: float | None,
    chunk_size: int = 1_000_000,
) -> np.ndarray:
    """Project every voxel centre through each selected camera and TSDF-carve.

    Returns a boolean grid of shape ``dims`` where ``True`` = visible
    (free space + thin band past the observed surface).
    """
    visible = np.zeros(tuple(dims), dtype=bool)
    total = int(np.prod(dims))
    if total == 0:
        return visible

    T_cw_list = [_to_T_cw(poses[i], pose_convention) for i in idx]
    K_list = list(Ks)

    # Pre-cast depth maps once so the inner loop is cheap.
    depth_cache: dict[int, np.ndarray] = {}

    def get_depth(frame_idx: int) -> np.ndarray:
        cached = depth_cache.get(frame_idx)
        if cached is not None:
            return cached
        d = np.asarray(depth_maps[frame_idx], dtype=np.float64)
        if depth_scale != 1.0:
            d = d / float(depth_scale)
        depth_cache[frame_idx] = d
        return d

    flip_yz = camera_frame == "opengl"

    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        flat = np.arange(chunk_start, chunk_end)
        i_idx, j_idx, k_idx = np.unravel_index(flat, tuple(dims))
        centers = (
            bbox_min[None, :]
            + np.stack([i_idx, j_idx, k_idx], axis=1).astype(np.float64) * voxel_size
        )
        centers_h = np.column_stack([centers, np.ones(len(flat), dtype=np.float64)])

        chunk_visible = np.zeros(len(flat), dtype=bool)
        for n, frame_idx in enumerate(idx):
            T_cw = T_cw_list[n]
            K_i = K_list[n]
            depth_i = get_depth(int(frame_idx))
            H, W = depth_i.shape

            cam = (T_cw @ centers_h.T).T[:, :3]
            if flip_yz:
                cam = cam * np.array([1.0, -1.0, -1.0])
            z = cam[:, 2]
            in_z = (z >= near) & (z <= max_depth)
            z_safe = np.where(in_z, z, 1.0)
            u = K_i[0, 0] * cam[:, 0] / z_safe + K_i[0, 2]
            v = K_i[1, 1] * cam[:, 1] / z_safe + K_i[1, 2]
            in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)

            ui = np.clip(np.floor(u).astype(np.int64), 0, W - 1)
            vi = np.clip(np.floor(v).astype(np.int64), 0, H - 1)
            d_pix = depth_i[vi, ui]
            valid_d = np.isfinite(d_pix) & (d_pix > 0)
            if depth_max is not None:
                valid_d &= d_pix <= float(depth_max)

            in_front = z <= d_pix + truncation

            chunk_visible |= in_z & in_image & valid_d & in_front

        visible.flat[chunk_start:chunk_end] = chunk_visible

    return visible


def _resolve_intrinsics_per_frame(
    intrinsics: np.ndarray, n_frames: int
) -> list[np.ndarray]:
    """Return a length-``n_frames`` list of (3, 3) intrinsics."""
    K = np.asarray(intrinsics, dtype=np.float64)
    if K.ndim == 2:
        if K.shape != (3, 3):
            raise ValueError(f"intrinsics must be (3, 3), got {K.shape}")
        return [K] * n_frames
    if K.ndim == 3 and K.shape[0] == n_frames and K.shape[1:] == (3, 3):
        return [K[i] for i in range(n_frames)]
    raise ValueError(f"intrinsics must be (3, 3) or (T, 3, 3), got {K.shape}")


# ---------------------------------------------------------------------------
# Public: from_depth
# ---------------------------------------------------------------------------


def from_depth(
    depth_maps: Sequence[np.ndarray],
    poses: np.ndarray,
    intrinsics: np.ndarray,
    *,
    voxel_size: float,
    margin: float = 0.05,
    pose_convention: PoseFrame = "T_cw",
    camera_frame: CameraFrame = "opencv",
    depth_scale: float = 1.0,
    depth_max: float | None = None,
    max_depth: float = 5.0,
    near: float = 0.05,
    truncation: float | None = None,
    frames: Sequence[int] | str | None = None,
    frames_file: PathLike | None = None,
    frame_stride: int = 10,
    max_frames: int | None = None,
    dilation: int = 1,
) -> OcclusionMask:
    """Build a volumetric occlusion mask from per-frame sensor depth.

    Two-stage TSDF-style carving:

    1. Bbox = union of camera frustum corners across selected frames, padded
       by ``margin``. ``near`` and ``max_depth`` set the frustum near / far
       planes used for the corner construction.
    2. Each voxel centre is projected through every selected camera. The
       voxel is visible if it lands inside the frustum and its camera-frame
       Z is ``≤ depth_at_pixel + truncation``. ``truncation`` defaults to
       ``4 * voxel_size``.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (T, 4, 4), got {poses.shape}")
    if len(depth_maps) != len(poses):
        raise ValueError(
            f"depth_maps ({len(depth_maps)}) and poses ({len(poses)}) "
            f"have different lengths"
        )
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")
    if not (0 < near < max_depth):
        raise ValueError(
            f"need 0 < near ({near}) < max_depth ({max_depth})"
        )

    Ks_full = _resolve_intrinsics_per_frame(intrinsics, len(poses))

    idx = _select_frames(
        len(depth_maps), frames, frames_file, frame_stride, max_frames
    )
    Ks = [Ks_full[i] for i in idx]

    # Stage 1: bbox from frustum corners.
    corner_chunks = []
    for n, i in enumerate(idx):
        T_wc = _to_T_wc(poses[i], pose_convention)
        H, W = np.asarray(depth_maps[i]).shape
        corner_chunks.append(
            _frustum_corners_world(
                Ks[n],
                T_wc,
                image_size=(W, H),
                near=near,
                far=max_depth,
                camera_frame=camera_frame,
            )
        )
    pts = np.concatenate(corner_chunks, axis=0)
    bbox_min = pts.min(0) - margin
    bbox_max = pts.max(0) + margin
    extent = bbox_max - bbox_min
    if (extent <= 0).any():
        raise ValueError(f"bbox has non-positive extent {extent}")
    dims = np.maximum(np.ceil(extent / voxel_size).astype(int), 1)

    # Stage 2: voxel-centric TSDF carve.
    trunc = float(truncation) if truncation is not None else 4.0 * float(voxel_size)
    visible = _carve_visible_grid(
        idx=idx,
        depth_maps=depth_maps,
        Ks=Ks,
        poses=poses,
        pose_convention=pose_convention,
        camera_frame=camera_frame,
        bbox_min=bbox_min,
        dims=dims,
        voxel_size=voxel_size,
        near=near,
        max_depth=max_depth,
        truncation=trunc,
        depth_scale=depth_scale,
        depth_max=depth_max,
    )

    if dilation > 0:
        visible = binary_dilation(visible, iterations=dilation)
    grid = np.where(visible, 0.0, 1.0)

    return OcclusionMask(
        grid=grid,
        T_mask_scene=_build_T_mask_scene(bbox_min, voxel_size),
        source="<from_depth>",
    )


# ---------------------------------------------------------------------------
# Public: from_rendered
# ---------------------------------------------------------------------------


def from_rendered(
    geom: MeshData,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    *,
    voxel_size: float,
    margin: float = 0.05,
    pose_convention: PoseFrame = "T_cw",
    camera_frame: CameraFrame = "opencv",
    max_depth: float = 5.0,
    near: float = 0.05,
    truncation: float | None = None,
    frames: Sequence[int] | str | None = None,
    frames_file: PathLike | None = None,
    frame_stride: int = 10,
    max_frames: int | None = None,
    dilation: int = 1,
    headless: bool = True,
) -> OcclusionMask:
    """Build a volumetric mask by rendering ``geom`` per pose, then TSDF-carving.

    Rendered depth is clean and complete — no holes, no sensor noise. Useful
    when the GT mesh is reliable but the dataset doesn't ship sensor depth.
    Requires the ``[render]`` extra (pyrender).
    """
    if not isinstance(geom, MeshData):
        raise TypeError(
            f"from_rendered requires MeshData, got {type(geom).__name__}"
        )
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (T, 4, 4), got {poses.shape}")
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")
    if not (0 < near < max_depth):
        raise ValueError(f"need 0 < near ({near}) < max_depth ({max_depth})")

    Ks_full = _resolve_intrinsics_per_frame(intrinsics, len(poses))

    from eval3r.render.pyrender_backend import (  # noqa: PLC0415
        _build_scene,
        _enable_headless,
        _render_scene,
    )

    if headless:
        _enable_headless()
    pyrender = optional_import("pyrender", extra="render")

    idx = _select_frames(len(poses), frames, frames_file, frame_stride, max_frames)
    Ks = [Ks_full[i] for i in idx]

    # Render depth for each selected frame. Pyrender depth lives in the
    # OpenGL camera frame relative to ``pose_gl``; we feed (pose_gl, "opengl")
    # into the carving stage so projection and lookup agree.
    rendered_depths: dict[int, np.ndarray] = {}
    pose_gl_list: list[np.ndarray] = []
    W, H = image_size
    for n, i in enumerate(idx):
        pose_gl = to_pyrender_pose(
            poses[i],
            pose_convention=pose_convention,
            camera_frame=camera_frame,
        )
        pose_gl_list.append(pose_gl)
        scene = _build_scene(
            pyrender,
            geom=geom,
            camera_pose=pose_gl,
            intrinsics=Ks[n],
            image_size=image_size,
        )
        _, depth = _render_scene(pyrender, scene, image_size)
        rendered_depths[int(i)] = np.asarray(depth, dtype=np.float64)

    # Build a length-len(poses) sequence so depth_maps[i] indexing works.
    depth_seq: list[np.ndarray] = [
        rendered_depths.get(j, np.zeros((H, W), dtype=np.float64))
        for j in range(len(poses))
    ]

    # Synthetic poses array in OpenGL camera_frame that pairs with rendered depth.
    poses_for_carve = poses.copy()
    for n, i in enumerate(idx):
        # Pyrender's pose_gl is camera-to-world (T_wc) in OpenGL.
        # _carve_visible_grid expects "raw" pose + (pose_convention, camera_frame).
        # Easiest: pre-store the OpenGL T_wc, declare convention "T_wc"+frame "opengl".
        poses_for_carve[i] = pose_gl_list[n]

    # Stage 1: bbox from frustum corners (using pyrender's OpenGL poses).
    corner_chunks = []
    for n, i in enumerate(idx):
        corner_chunks.append(
            _frustum_corners_world(
                Ks[n],
                pose_gl_list[n],
                image_size=(W, H),
                near=near,
                far=max_depth,
                camera_frame="opengl",
            )
        )
    pts = np.concatenate(corner_chunks, axis=0)
    bbox_min = pts.min(0) - margin
    bbox_max = pts.max(0) + margin
    extent = bbox_max - bbox_min
    if (extent <= 0).any():
        raise ValueError(f"bbox has non-positive extent {extent}")
    dims = np.maximum(np.ceil(extent / voxel_size).astype(int), 1)

    # Stage 2: voxel-centric TSDF carve in the OpenGL frame (matching pyrender).
    trunc = float(truncation) if truncation is not None else 4.0 * float(voxel_size)
    visible = _carve_visible_grid(
        idx=idx,
        depth_maps=depth_seq,
        Ks=Ks,
        poses=poses_for_carve,
        pose_convention="T_wc",
        camera_frame="opengl",
        bbox_min=bbox_min,
        dims=dims,
        voxel_size=voxel_size,
        near=near,
        max_depth=max_depth,
        truncation=trunc,
        depth_scale=1.0,
        depth_max=None,
    )

    if dilation > 0:
        visible = binary_dilation(visible, iterations=dilation)
    grid = np.where(visible, 0.0, 1.0)

    return OcclusionMask(
        grid=grid,
        T_mask_scene=_build_T_mask_scene(bbox_min, voxel_size),
        source="<from_rendered>",
    )
