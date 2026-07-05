"""Task 011 visibility-culling backend tests (render + TSDF trim).

pyrender needs a headless GL context (EGL/OSMesa). Like the DTU MATLAB path, these
tests skip cleanly with an explicit reason when no GL context is available (e.g. CI
without EGL), rather than failing.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from eval3r.backends.visibility_render import (
    CameraTrajectory,
    RenderTsdfVisibilityCull,
    trajectory_fingerprint,
)
from eval3r.core.errors import CullingError


def _gl_available() -> tuple[bool, str]:
    try:
        import pyrender

        r = pyrender.OffscreenRenderer(16, 16)
        r.delete()
        return True, ""
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"no headless GL context for pyrender: {exc}"


_GL_OK, _GL_REASON = _gl_available()
pytestmark = pytest.mark.skipif(not _GL_OK, reason=_GL_REASON)

# 128x96 depth camera looking straight down (cam-to-world OpenCV) at a slab at z=0.
_W, _H = 128, 96
_K = np.array([[80.0, 0.0, 64.0], [0.0, 80.0, 48.0], [0.0, 0.0, 1.0]])
_R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def _trajectory(n: int = 4) -> CameraTrajectory:
    poses = []
    for x, y in [(-0.3, -0.3), (0.3, -0.3), (0.0, 0.3), (0.4, 0.4)][:n]:
        T = np.eye(4)
        T[:3, :3] = _R
        T[:3, 3] = [x, y, 1.5]
        poses.append(T)
    arr = np.stack(poses)
    return CameraTrajectory(arr, _K, _W, _H, fingerprint=trajectory_fingerprint(arr, _K, _W, _H))


def _pred_with_far_box() -> trimesh.Trimesh:
    slab = trimesh.creation.box(extents=(2.0, 2.0, 0.05))
    far = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    far.apply_translation([5.0, 5.0, 0.0])
    return trimesh.util.concatenate([slab, far])


def test_cull_removes_unobserved_geometry() -> None:
    backend = RenderTsdfVisibilityCull()
    result = backend.cull(_pred_with_far_box(), _trajectory(), tolerance=0.05, voxel_length=0.04)
    # The far box (unobserved) is culled; the observed slab is kept.
    assert result.culled_fraction > 0.2
    assert result.n_kept < result.n_vertices
    far_verts = np.asarray(result.trimmed_mesh.vertices)
    assert far_verts.shape[0] > 0
    # nothing near (5,5,0) survives
    assert not np.any(np.linalg.norm(far_verts - np.array([5.0, 5.0, 0.0]), axis=1) < 1.0)


def test_cull_metadata_records_provenance() -> None:
    backend = RenderTsdfVisibilityCull()
    result = backend.cull(_pred_with_far_box(), _trajectory(), tolerance=0.05, voxel_length=0.04)
    meta = result.metadata
    assert meta["method"] == "render_tsdf_trim"
    assert "pyrender" in meta["renderer"]
    assert meta["voxel_length"] == 0.04
    assert meta["keep_tolerance"] == 0.05
    assert meta["n_poses_used"] == 4
    assert meta["n_observed_voxels"] > 0
    assert meta["trajectory_fingerprint"] is not None


def test_cull_is_deterministic() -> None:
    backend = RenderTsdfVisibilityCull()
    mesh = _pred_with_far_box()
    a = backend.cull(mesh, _trajectory(), tolerance=0.05, voxel_length=0.04)
    b = backend.cull(mesh, _trajectory(), tolerance=0.05, voxel_length=0.04)
    assert a.culled_fraction == b.culled_fraction
    assert np.array_equal(a.kept_mask, b.kept_mask)


def test_no_finite_poses_fails_explicitly() -> None:
    backend = RenderTsdfVisibilityCull()
    bad = np.stack([np.full((4, 4), np.inf)])
    traj = CameraTrajectory(bad, _K, _W, _H)
    with pytest.raises(CullingError) as exc:
        backend.cull(_pred_with_far_box(), traj, tolerance=0.05)
    assert "no finite camera poses" in str(exc.value)


def test_backend_info_names_both_libraries() -> None:
    info = RenderTsdfVisibilityCull().backend_info()
    assert info.kind == "visibility"
    assert "pyrender" in info.version and "open3d" in info.version
