"""Visibility-culling backend (``visibility`` kind): render + TSDF trim.

Implements the community ScanNet single-/double-layer visibility culling
(NeuralRecon/TransformerFusion style): render the *prediction's* depth from the
ground-truth camera trajectory, TSDF-integrate those rendered depths to recover the
observed region, and trim the prediction to vertices near that region before scoring.

This is the CLAUDE.md "evaluation-time visibility culling exception": no new scene
geometry is produced — only the caller's prediction is masked — and every run records
the renderer + TSDF backend, voxel size, trajectory fingerprint, and culled fraction.

Convention: ScanNet exports **camera-to-world, OpenCV-style** poses; pyrender uses an
OpenGL camera (looking down -Z). The camera pose handed to pyrender is therefore
``cam2world @ diag(1, -1, -1, 1)``. Open3D's TSDF integrate takes the OpenCV
world-to-camera extrinsic (``inv(cam2world)``).

`pyrender` needs a headless GL context; this module selects EGL via
``PYOPENGL_PLATFORM`` when it is unset, before importing pyrender.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from eval3r.core.errors import CullingError
from eval3r.core.registry import BackendInfo

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import pyrender  # noqa: E402  (import after PYOPENGL_PLATFORM is set)

# OpenCV camera axes -> OpenGL camera axes (flip Y and Z).
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class CameraTrajectory:
    """A GT camera trajectory used to render the observed region.

    ``poses`` are ``(N, 4, 4)`` camera-to-world OpenCV matrices; ``intrinsics`` is the
    ``(3, 3)`` pinhole matrix for the depth camera at ``(width, height)``.
    """

    poses: np.ndarray
    intrinsics: np.ndarray
    width: int
    height: int
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        self.poses = np.asarray(self.poses, dtype=np.float64)
        self.intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        if self.poses.ndim != 3 or self.poses.shape[1:] != (4, 4):
            raise CullingError(
                f"visibility trajectory poses must be (N, 4, 4); got {self.poses.shape}."
            )
        if self.intrinsics.shape != (3, 3):
            raise CullingError(
                f"visibility trajectory intrinsics must be (3, 3); got {self.intrinsics.shape}."
            )


@dataclass
class VisibilityCullResult:
    """Outcome of trimming a mesh to the observed region."""

    trimmed_mesh: Any
    kept_mask: np.ndarray
    culled_fraction: float
    n_vertices: int
    n_kept: int
    metadata: dict[str, Any] = field(default_factory=dict)


def trajectory_fingerprint(
    poses: np.ndarray, intrinsics: np.ndarray, width: int, height: int
) -> str:
    """Stable content hash of a trajectory (poses + intrinsics + image size)."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.asarray(poses, dtype=np.float64)).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(intrinsics, dtype=np.float64)).tobytes())
    digest.update(np.asarray([width, height], dtype=np.int64).tobytes())
    return f"sha256:{digest.hexdigest()}"


class RenderTsdfVisibilityCull:
    """Render prediction depth from the GT trajectory, TSDF-trim to the observed region."""

    name = "render_tsdf"

    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            kind="visibility",
            name=self.name,
            library="pyrender+open3d",
            version=f"pyrender {pyrender.__version__}; open3d {o3d.__version__}",
            approximate=True,
        )

    def cull(
        self,
        mesh: Any,
        trajectory: CameraTrajectory,
        *,
        tolerance: float,
        voxel_length: float = 0.04,
        sdf_trunc_factor: float = 3.0,
        depth_trunc: float = 6.0,
        znear: float = 0.05,
        pose_stride: int = 1,
    ) -> VisibilityCullResult:
        """Trim ``mesh`` (a trimesh mesh) to the region observed by ``trajectory``.

        ``tolerance`` (metres) is the keep radius: a vertex survives when it lies within
        ``tolerance`` of an observed TSDF voxel. All parameters are supplied by the
        protocol / caller — nothing about thresholds or voxel sizing is chosen silently.
        """
        import trimesh

        if not isinstance(mesh, trimesh.Trimesh):
            raise CullingError(
                f"visibility culling needs a trimesh mesh; got {type(mesh).__name__}."
            )
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        n_vertices = int(verts.shape[0])
        if n_vertices == 0:
            raise CullingError("visibility culling received a mesh with no vertices.")

        poses = trajectory.poses[:: max(pose_stride, 1)]
        finite = np.array([bool(np.all(np.isfinite(T))) for T in poses], dtype=bool)
        poses = poses[finite]
        if poses.shape[0] == 0:
            raise CullingError(
                "visibility trajectory has no finite camera poses after filtering; cannot "
                "render the observed region. Check the scene's pose files."
            )

        W, H = int(trajectory.width), int(trajectory.height)
        K = trajectory.intrinsics
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # --- render depth from every GT pose ---
        scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        camera = pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=cx, cy=cy, znear=znear, zfar=depth_trunc
        )
        cam_node = scene.add(camera, pose=np.eye(4))
        renderer = pyrender.OffscreenRenderer(W, H)

        sdf_trunc = sdf_trunc_factor * voxel_length
        tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        empty_color = o3d.geometry.Image(np.zeros((H, W, 3), np.uint8))

        n_integrated = 0
        try:
            for T in poses:
                scene.set_pose(cam_node, T @ _CV_TO_GL)
                depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
                if not np.any(depth > 0):
                    continue
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    empty_color,
                    o3d.geometry.Image(np.ascontiguousarray(depth, dtype=np.float32)),
                    depth_scale=1.0,
                    depth_trunc=depth_trunc,
                    convert_rgb_to_intensity=False,
                )
                tsdf.integrate(rgbd, intr, np.linalg.inv(T))
                n_integrated += 1
        finally:
            renderer.delete()

        if n_integrated == 0:
            raise CullingError(
                "visibility culling rendered no non-empty depth from the trajectory; the "
                "prediction may not overlap the GT camera frustums (check pose convention)."
            )

        observed = np.asarray(tsdf.extract_voxel_point_cloud().points, dtype=np.float64)
        if observed.shape[0] == 0:
            raise CullingError("visibility TSDF produced no observed voxels to trim against.")

        # --- keep vertices within `tolerance` of the observed region ---
        tree = cKDTree(observed)
        dists, _ = tree.query(verts, k=1)
        keep_mask = dists <= float(tolerance)
        n_kept = int(keep_mask.sum())
        if n_kept == 0:
            raise CullingError(
                "visibility culling removed every prediction vertex (nothing within "
                f"{tolerance} m of the observed region); the prediction and GT trajectory "
                "may be misaligned."
            )

        trimmed = _trim_faces(mesh, keep_mask)
        culled_fraction = 1.0 - n_kept / n_vertices
        metadata = {
            "method": "render_tsdf_trim",
            "renderer": f"pyrender {pyrender.__version__} (EGL)",
            "tsdf_backend": f"open3d {o3d.__version__} ScalableTSDFVolume",
            "voxel_length": voxel_length,
            "sdf_trunc": sdf_trunc,
            "keep_tolerance": float(tolerance),
            "depth_trunc": depth_trunc,
            "pose_stride": int(pose_stride),
            "n_poses_total": int(trajectory.poses.shape[0]),
            "n_poses_used": int(poses.shape[0]),
            "n_poses_integrated": n_integrated,
            "n_observed_voxels": int(observed.shape[0]),
            "n_vertices": n_vertices,
            "n_kept": n_kept,
            "culled_fraction": culled_fraction,
            "trajectory_fingerprint": trajectory.fingerprint,
        }
        return VisibilityCullResult(
            trimmed_mesh=trimmed,
            kept_mask=keep_mask,
            culled_fraction=culled_fraction,
            n_vertices=n_vertices,
            n_kept=n_kept,
            metadata=metadata,
        )


def _trim_faces(mesh: Any, keep_mask: np.ndarray) -> Any:
    """Return a copy of ``mesh`` keeping only faces whose vertices are all kept."""
    faces = np.asarray(mesh.faces)
    face_keep = keep_mask[faces].all(axis=1)
    trimmed = mesh.copy()
    trimmed.update_faces(face_keep)
    trimmed.remove_unreferenced_vertices()
    return trimmed
