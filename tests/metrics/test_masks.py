"""Tests for eval3r.mask (volumetric TSDF carving) and the e3r mask CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from eval3r.cli.main import app
from eval3r.io.geometry import MeshData, save_mesh_ply
from eval3r.mask.generate import (
    _backproject,
    _frustum_corners_world,
    _points_to_mask,
    _select_frames,
    _to_T_cw,
    _to_T_wc,
    from_depth,
)
from eval3r.mask.occlusion import (
    filter_visible_points,
    load_occlusion_mask,
    save_occlusion_mask,
)

runner = CliRunner(env={"NO_COLOR": "1"})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _icosphere(n_subdiv: int = 2, radius: float = 0.5) -> MeshData:
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=n_subdiv, radius=radius)
    return MeshData(
        vertices=np.asarray(sphere.vertices, dtype=np.float64),
        faces=np.asarray(sphere.faces, dtype=np.int64),
    )


def _identity_intrinsics(w: int, h: int, focal: float) -> np.ndarray:
    return np.array(
        [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# unit: helpers
# ---------------------------------------------------------------------------


def test_to_T_wc_inverts_T_cw() -> None:
    R = np.eye(4)
    R[0, 3] = 1.0
    T_wc = _to_T_wc(R, "T_cw")
    np.testing.assert_allclose(T_wc[0, 3], -1.0)


def test_to_T_cw_passes_through() -> None:
    R = np.eye(4)
    R[0, 3] = 0.7
    T_cw = _to_T_cw(R, "T_cw")
    np.testing.assert_allclose(T_cw, R)


def test_to_T_cw_inverts_T_wc() -> None:
    R = np.eye(4)
    R[1, 3] = 0.4
    T_cw = _to_T_cw(R, "T_wc")
    np.testing.assert_allclose(T_cw[1, 3], -0.4)


def test_frustum_corners_world_opencv_identity() -> None:
    K = _identity_intrinsics(32, 32, 100.0)
    corners = _frustum_corners_world(
        K, np.eye(4), image_size=(32, 32),
        near=0.05, far=1.0, camera_frame="opencv",
    )
    assert corners.shape == (8, 3)
    # Near corners cluster near origin; far corners at z=1.
    far = corners[corners[:, 2] > 0.5]
    near = corners[corners[:, 2] <= 0.5]
    assert len(far) == 4 and len(near) == 4
    np.testing.assert_allclose(far[:, 2], 1.0, atol=1e-9)
    # x extent at z=1 is (W-cx)*z/fx = (32-16)*1/100 = 0.16.
    assert np.allclose(np.abs(far[:, 0]).max(), 0.16, atol=1e-9)


# ---------------------------------------------------------------------------
# unit: _select_frames
# ---------------------------------------------------------------------------


def test_select_frames_default_stride() -> None:
    out = _select_frames(10, None, None, frame_stride=2, max_frames=None)
    np.testing.assert_array_equal(out, [0, 2, 4, 6, 8])


def test_select_frames_string_list() -> None:
    out = _select_frames(10, "0, 5, 9", None, frame_stride=1, max_frames=None)
    np.testing.assert_array_equal(out, [0, 5, 9])


def test_select_frames_max_frames_caps() -> None:
    out = _select_frames(100, None, None, frame_stride=1, max_frames=3)
    np.testing.assert_array_equal(out, [0, 1, 2])


def test_select_frames_file(tmp_path: Path) -> None:
    p = tmp_path / "frames.txt"
    p.write_text("1\n3\n7\n")
    out = _select_frames(10, None, p, frame_stride=1, max_frames=None)
    np.testing.assert_array_equal(out, [1, 3, 7])


def test_select_frames_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="out of range"):
        _select_frames(5, "0, 10", None, 1, None)


def test_select_frames_stride_10_picks_two_of_eleven() -> None:
    out = _select_frames(11, None, None, frame_stride=10, max_frames=None)
    np.testing.assert_array_equal(out, [0, 10])


# ---------------------------------------------------------------------------
# unit: _backproject (kept for reuse / debugging)
# ---------------------------------------------------------------------------


def test_backproject_opencv_identity_pose() -> None:
    H, W = 64, 64
    focal = 100.0
    K = _identity_intrinsics(W, H, focal)
    depth = np.ones((H, W), dtype=np.float32)
    pts = _backproject(depth, K, np.eye(4), camera_frame="opencv")
    assert pts.shape == (H * W, 3)
    centre = pts[(H // 2) * W + (W // 2)]
    np.testing.assert_allclose(centre, [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(pts[:, 2], 1.0, atol=1e-6)


def test_backproject_invalid_depth_excluded() -> None:
    K = _identity_intrinsics(8, 8, 50.0)
    depth = np.zeros((8, 8), dtype=np.float32)
    depth[3, 3] = 1.0
    pts = _backproject(depth, K, np.eye(4), camera_frame="opencv")
    assert pts.shape == (1, 3)


# ---------------------------------------------------------------------------
# unit: TSDF carving via from_depth
# ---------------------------------------------------------------------------


def test_from_depth_carves_free_space_and_occludes_behind_surface() -> None:
    """1m wall at z=1: voxels in front are visible, behind are occluded."""
    H, W = 32, 32
    focal = 200.0
    K = _identity_intrinsics(W, H, focal)
    depth = np.full((H, W), 1.0, dtype=np.float32)
    poses = np.eye(4)[None, :, :]  # T_cw = I → camera at origin looking +Z.

    mask = from_depth(
        [depth],
        poses,
        K,
        voxel_size=0.05,
        margin=0.05,
        pose_convention="T_cw",
        camera_frame="opencv",
        max_depth=2.0,
        near=0.05,
        truncation=0.05,
        frame_stride=1,
        dilation=0,
    )

    free_space = np.array([[0.0, 0.0, 0.97]])  # just in front of the wall.
    kept, _, _ = filter_visible_points(free_space, mask)
    assert len(kept) == 1, "voxel near the observed surface must be visible"

    near_band = np.array([[0.0, 0.0, 1.02]])  # within truncation past surface.
    kept, _, _ = filter_visible_points(near_band, mask)
    assert len(kept) == 1, "voxel just past surface within truncation must be visible"

    behind_wall = np.array([[0.0, 0.0, 1.5]])  # well past surface + truncation.
    with pytest.raises(ValueError):
        filter_visible_points(behind_wall, mask)


def test_from_depth_rejects_outside_frustum() -> None:
    H, W = 32, 32
    focal = 200.0
    K = _identity_intrinsics(W, H, focal)
    depth = np.full((H, W), 1.0, dtype=np.float32)
    poses = np.eye(4)[None, :, :]
    mask = from_depth(
        [depth], poses, K,
        voxel_size=0.05, margin=0.5,
        pose_convention="T_cw", camera_frame="opencv",
        max_depth=2.0, frame_stride=1, dilation=0,
    )
    # (2, 0, 1) sits at u = 200*2/1 + 16 = 416, way outside W=32.
    far_off_axis = np.array([[2.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        filter_visible_points(far_off_axis, mask)


def test_from_depth_bbox_tracks_observed_samples_plus_margin() -> None:
    """Depth-derived bbox is tight around observed samples plus margin."""
    H, W = 32, 32
    focal = 200.0
    K = _identity_intrinsics(W, H, focal)
    depth = np.full((H, W), 1.0, dtype=np.float32)
    poses = np.eye(4)[None, :, :]
    voxel_size = 0.05
    margin = 0.05
    mask = from_depth(
        [depth], poses, K,
        voxel_size=voxel_size, margin=margin,
        pose_convention="T_cw", camera_frame="opencv",
        max_depth=2.0, near=0.05, truncation=0.05,
        frame_stride=1, dilation=0,
    )

    bbox_min = -mask.T_mask_scene[:3, 3] / np.diag(mask.T_mask_scene)[:3]
    x_samples = (np.arange(W) - W / 2) / focal
    y_samples = (np.arange(H) - H / 2) / focal
    expected_min = np.array([x_samples.min(), y_samples.min(), 1.0]) - margin
    np.testing.assert_allclose(bbox_min, expected_min, atol=1e-9)

    expected_max = np.array([x_samples.max(), y_samples.max(), 1.0]) + margin
    expected_extent = expected_max - expected_min
    expected_dims = np.maximum(np.ceil(expected_extent / voxel_size).astype(int), 1)
    assert mask.grid.shape == tuple(expected_dims)


def test_from_depth_ignores_far_pixels_when_building_bbox() -> None:
    H, W = 8, 8
    K = _identity_intrinsics(W, H, 50.0)
    depth = np.full((H, W), 4.0, dtype=np.float32)
    depth[3, 3] = 1.0
    poses = np.eye(4)[None, :, :]

    mask = from_depth(
        [depth], poses, K,
        voxel_size=0.05, margin=0.05,
        pose_convention="T_cw", camera_frame="opencv",
        max_depth=2.0, frame_stride=1, dilation=0,
    )

    bbox_min = -mask.T_mask_scene[:3, 3] / np.diag(mask.T_mask_scene)[:3]
    expected_point = np.array([(3 - W / 2) / 50.0, (3 - H / 2) / 50.0, 1.0])
    np.testing.assert_allclose(bbox_min, expected_point - 0.05, atol=1e-9)


def test_from_depth_all_invalid_bbox_pixels_raise() -> None:
    K = _identity_intrinsics(8, 8, 50.0)
    depth = np.full((8, 8), 4.0, dtype=np.float32)
    poses = np.eye(4)[None, :, :]

    with pytest.raises(ValueError, match="depth-derived bbox"):
        from_depth(
            [depth], poses, K,
            voxel_size=0.05, margin=0.05,
            pose_convention="T_cw", camera_frame="opencv",
            max_depth=2.0, frame_stride=1, dilation=0,
        )


def _carve_visible_grid_reference(
    *,
    idx,
    depth_maps,
    Ks,
    poses,
    pose_convention,
    camera_frame,
    bbox_min,
    dims,
    voxel_size,
    near,
    max_depth,
    truncation,
    depth_scale,
    depth_max,
    chunk_size=1_000_000,
):
    """Pre-acceleration reference: chunks-outside, frames-inside, full grid."""
    visible = np.zeros(tuple(int(d) for d in dims), dtype=bool)
    total = int(np.prod(dims))
    if total == 0:
        return visible

    T_cw_list = [_to_T_cw(poses[i], pose_convention) for i in idx]

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
            K_i = Ks[n]
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


def test_carve_visible_grid_matches_reference_implementation() -> None:
    """The accelerated carve produces the same visible grid as the pre-AABB reference."""
    from eval3r.mask.generate import _carve_visible_grid

    rng = np.random.default_rng(0)
    H, W = 32, 32
    focal = 40.0
    K = _identity_intrinsics(W, H, focal)

    # 4 cameras placed around a plane at z=1, looking down +Z (OpenCV).
    poses = []
    depth_maps = []
    for dx, dy in [(0.0, 0.0), (0.3, -0.2), (-0.4, 0.1), (0.1, 0.4)]:
        T_wc = np.eye(4)
        T_wc[:3, 3] = [dx, dy, 0.0]
        # T_cw = inv(T_wc): camera looking down +Z from offset.
        poses.append(np.linalg.inv(T_wc))
        # Synthetic depth: plane at z=1 plus mild jitter.
        depth = np.full((H, W), 1.0, dtype=np.float64)
        depth += 0.02 * rng.standard_normal((H, W))
        depth_maps.append(depth.astype(np.float32))
    poses_arr = np.stack(poses, axis=0)

    voxel_size = 0.05
    near, max_depth, truncation = 0.05, 2.0, 0.05
    bbox_min = np.array([-0.7, -0.7, -0.05])
    bbox_max = np.array([0.7, 0.7, 1.10])
    dims = np.maximum(np.ceil((bbox_max - bbox_min) / voxel_size).astype(int), 1)

    idx = np.arange(len(poses_arr), dtype=np.int64)
    Ks = [K for _ in idx]

    new = _carve_visible_grid(
        idx=idx,
        depth_maps=depth_maps,
        Ks=Ks,
        poses=poses_arr,
        pose_convention="T_cw",
        camera_frame="opencv",
        bbox_min=bbox_min,
        dims=dims,
        voxel_size=voxel_size,
        near=near,
        max_depth=max_depth,
        truncation=truncation,
        depth_scale=1.0,
        depth_max=None,
    )
    ref = _carve_visible_grid_reference(
        idx=idx,
        depth_maps=depth_maps,
        Ks=Ks,
        poses=poses_arr,
        pose_convention="T_cw",
        camera_frame="opencv",
        bbox_min=bbox_min,
        dims=dims,
        voxel_size=voxel_size,
        near=near,
        max_depth=max_depth,
        truncation=truncation,
        depth_scale=1.0,
        depth_max=None,
    )
    np.testing.assert_array_equal(new, ref)


def test_from_depth_pose_count_mismatch_raises() -> None:
    K = _identity_intrinsics(8, 8, 50.0)
    depth = np.ones((8, 8))
    poses = np.tile(np.eye(4), (2, 1, 1))
    with pytest.raises(ValueError, match="different lengths"):
        from_depth([depth], poses, K, voxel_size=0.05, frame_stride=1)


def test_from_depth_invalid_near_max_depth_raises() -> None:
    K = _identity_intrinsics(8, 8, 50.0)
    depth = np.ones((8, 8))
    poses = np.eye(4)[None, :, :]
    with pytest.raises(ValueError, match="near"):
        from_depth(
            [depth], poses, K,
            voxel_size=0.05, near=2.0, max_depth=1.0, frame_stride=1,
        )


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_gen_manual_depth_with_pattern(tmp_path: Path) -> None:
    """Manual depth ingestion via str.format-style pattern + frame stride 1."""
    imageio = pytest.importorskip("imageio.v3")

    H, W = 16, 16
    focal = 100.0
    K = _identity_intrinsics(W, H, focal)

    depth_dir = tmp_path / "depth"
    poses_dir = tmp_path / "poses"
    depth_dir.mkdir()
    poses_dir.mkdir()
    K_path = tmp_path / "K.txt"
    np.savetxt(K_path, K)

    # Three frames of a 1m plane at z=1, identity world pose.
    depth = np.full((H, W), 1.0, dtype=np.float32)
    for i in range(3):
        imageio.imwrite(depth_dir / f"{i:06d}.png", (depth * 1000).astype(np.uint16))
        np.savetxt(poses_dir / f"{i:06d}.txt", np.eye(4))

    out_dir = tmp_path / "mask"
    result = runner.invoke(
        app,
        [
            "mask", "gen",
            "--depth-path", str(depth_dir),
            "--depth-pattern", "{frame:06d}.png",
            "--poses-path", str(poses_dir),
            "--poses-pattern", "{frame:06d}.txt",
            "--intrinsics-path", str(K_path),
            "--depth-scale", "1000",
            "--pose-convention", "T_cw",
            "--camera-frame", "opencv",
            "--max-depth", "2.0",
            "--voxel-size", "0.05",
            "--margin", "0.05",
            "--truncation", "0.05",
            "--frame-stride", "1",
            "--dilation", "0",
            "--out-dir", str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    mask = load_occlusion_mask(
        out_dir / "occlusion_mask.npy", out_dir / "T_mask_scene.txt"
    )
    surface = np.array([[0.0, 0.0, 1.0]])
    kept, _, _ = filter_visible_points(surface, mask)
    assert len(kept) == 1


def test_cli_gen_inspect_round_trip(tmp_path: Path) -> None:
    """Generate a tiny mask via CLI, then inspect prints sane metadata."""
    imageio = pytest.importorskip("imageio.v3")

    H, W = 16, 16
    K = _identity_intrinsics(W, H, 100.0)
    depth_dir = tmp_path / "depth"
    poses_dir = tmp_path / "poses"
    depth_dir.mkdir()
    poses_dir.mkdir()
    K_path = tmp_path / "K.txt"
    np.savetxt(K_path, K)
    imageio.imwrite(depth_dir / "0.png", (np.ones((H, W)) * 1000).astype(np.uint16))
    np.savetxt(poses_dir / "0.txt", np.eye(4))

    out_dir = tmp_path / "mask"
    gen = runner.invoke(
        app,
        [
            "mask", "gen",
            "--depth-path", str(depth_dir),
            "--poses-path", str(poses_dir),
            "--intrinsics-path", str(K_path),
            "--depth-scale", "1000",
            "--max-depth", "2.0",
            "--voxel-size", "0.1",
            "--frame-stride", "1",
            "--out-dir", str(out_dir),
        ],
    )
    assert gen.exit_code == 0, gen.stdout

    insp = runner.invoke(
        app,
        [
            "mask", "inspect",
            "--mask", str(out_dir / "occlusion_mask.npy"),
            "--t-mask-scene", str(out_dir / "T_mask_scene.txt"),
            "--json",
        ],
    )
    assert insp.exit_code == 0, insp.stdout
    payload = json.loads(insp.stdout)
    assert payload["shape"][0] > 0
    assert 0 < payload["visible_fraction"] <= 1.0
    assert payload["voxel_size"] == pytest.approx(0.1, rel=1e-3)


def test_cli_gen_depth_and_mesh_paths_are_mutually_exclusive(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "mask", "gen",
            "--depth-path", str(tmp_path),
            "--mesh-path", str(tmp_path / "m.ply"),
            "--out-dir", str(tmp_path / "out"),
        ],
        env={"NO_COLOR": "1", "TERM": "dumb"},
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


def test_cli_gen_no_inputs_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["mask", "gen", "--out-dir", str(tmp_path / "out")],
        env={"NO_COLOR": "1", "TERM": "dumb"},
    )
    assert result.exit_code != 0
    assert "--preset" in result.output and "--depth-path" in result.output


def test_cli_gen_help_uses_single_max_depth_option() -> None:
    result = runner.invoke(
        app,
        ["mask", "gen", "--help"],
        env={"NO_COLOR": "1", "TERM": "dumb"},
    )
    assert result.exit_code == 0, result.output
    assert "--max-depth" in result.output
    assert "--depth-max" not in result.output
    assert "Default: 3.5" in result.output
    assert "bounds" in result.output
    assert "carving" in result.output
    assert "discard" in result.output.lower()


def test_cli_gen_max_depth_requires_at_least_one_valid_bbox_pixel(tmp_path: Path) -> None:
    imageio = pytest.importorskip("imageio.v3")

    H, W = 8, 8
    K = _identity_intrinsics(W, H, 50.0)
    depth_dir = tmp_path / "depth"
    poses_dir = tmp_path / "poses"
    depth_dir.mkdir()
    poses_dir.mkdir()
    np.savetxt(tmp_path / "K.txt", K)

    imageio.imwrite(depth_dir / "0.png", np.full((H, W), 4000, dtype=np.uint16))
    np.savetxt(poses_dir / "0.txt", np.eye(4))

    out_dir = tmp_path / "mask"
    result = runner.invoke(
        app,
        [
            "mask", "gen",
            "--depth-path", str(depth_dir),
            "--poses-path", str(poses_dir),
            "--intrinsics-path", str(tmp_path / "K.txt"),
            "--depth-scale", "1000",
            "--voxel-size", "0.5",
            "--frame-stride", "1",
            "--dilation", "0",
            "--out-dir", str(out_dir),
        ],
        env={"NO_COLOR": "1", "TERM": "dumb"},
    )
    assert result.exit_code != 0
    assert "depth-derived bbox" in result.output


def test_cli_gen_preset_requires_root_and_scene(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "mask", "gen",
            "--preset", "replica",
            "--out-dir", str(tmp_path / "out"),
        ],
        env={"NO_COLOR": "1", "TERM": "dumb"},
    )
    assert result.exit_code != 0
    assert "--root" in result.output and "--scene" in result.output


def test_cli_gen_depth_count_mismatch(tmp_path: Path) -> None:
    imageio = pytest.importorskip("imageio.v3")
    depth_dir = tmp_path / "depth"
    poses_dir = tmp_path / "poses"
    depth_dir.mkdir()
    poses_dir.mkdir()
    np.savetxt(tmp_path / "K.txt", np.eye(3) * 100)

    imageio.imwrite(depth_dir / "0.png", np.ones((4, 4), dtype=np.uint16))
    np.savetxt(poses_dir / "0.txt", np.eye(4))
    np.savetxt(poses_dir / "1.txt", np.eye(4))

    result = runner.invoke(
        app,
        [
            "mask", "gen",
            "--depth-path", str(depth_dir),
            "--poses-path", str(poses_dir),
            "--intrinsics-path", str(tmp_path / "K.txt"),
            "--out-dir", str(tmp_path / "mask"),
            "--frame-stride", "1",
        ],
        env={"NO_COLOR": "1", "TERM": "dumb"},
    )
    assert result.exit_code != 0
    assert "counts must match" in result.output.lower()


def test_cli_mask_visualize_exports_overview_and_point_cloud(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib.pyplot")

    grid = np.ones((12, 10, 8), dtype=bool)
    grid[3:9, 3:7, 2:6] = False
    mask_path = tmp_path / "occlusion_mask.npy"
    np.save(mask_path, grid)
    np.savetxt(tmp_path / "T_mask_scene.txt", np.eye(4))

    out_dir = tmp_path / "viz"
    result = runner.invoke(
        app,
        [
            "mask",
            "visualize",
            "--mask",
            str(mask_path),
            "--out-dir",
            str(out_dir),
            "--value",
            "visible",
            "--max-dim",
            "8",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "visible_overview.png").exists()
    assert (out_dir / "visible_surface.ply").exists()
    assert "coordinate_space : scene/world" in result.output


# ---------------------------------------------------------------------------
# from_rendered (skipped if pyrender unavailable)
# ---------------------------------------------------------------------------


def test_from_rendered_carves_observed_surface_volume() -> None:
    pytest.importorskip("pyrender")
    from eval3r.mask.generate import from_rendered

    sphere = _icosphere(n_subdiv=3, radius=0.5)

    # 4 cameras orbiting the sphere at distance 2, looking at origin (OpenGL).
    poses = []
    for theta in np.linspace(0, 2 * np.pi, 4, endpoint=False):
        eye = np.array([2 * np.cos(theta), 0.0, 2 * np.sin(theta)])
        forward = -eye / np.linalg.norm(eye)
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        T_wc = np.eye(4)
        T_wc[:3, 0] = right
        T_wc[:3, 1] = up
        T_wc[:3, 2] = -forward
        T_wc[:3, 3] = eye
        poses.append(T_wc)
    poses_arr = np.stack(poses, axis=0)

    K = _identity_intrinsics(64, 64, 80.0)

    mask = from_rendered(
        sphere,
        poses_arr,
        K,
        image_size=(64, 64),
        voxel_size=0.06,
        margin=0.1,
        pose_convention="T_wc",
        camera_frame="opengl",
        max_depth=4.0,
        near=0.1,
        truncation=0.06,
        frame_stride=1,
        dilation=0,
        headless=True,
    )

    # The depth-derived bbox is tight around observed geometry; use a surface
    # point rather than camera-to-surface free space.
    surface = np.array([[0.5, 0.0, 0.0]])
    kept, _, _ = filter_visible_points(surface, mask)
    assert len(kept) == 1

    # A point well outside any frustum (far above the orbit plane) → occluded.
    far = np.array([[0.0, 5.0, 0.0]])
    with pytest.raises(ValueError):
        filter_visible_points(far, mask)


# ---------------------------------------------------------------------------
# save / load round-trip on the still-public helper
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    pts = np.array([[0.1, 0.2, 0.3], [-0.1, 0.0, 0.5]])
    mask = _points_to_mask(pts, voxel_size=0.05, margin=0.1, dilation=0)
    save_occlusion_mask(mask, tmp_path)
    loaded = load_occlusion_mask(
        tmp_path / "occlusion_mask.npy", tmp_path / "T_mask_scene.txt"
    )
    np.testing.assert_array_equal(loaded.grid, mask.grid)
    np.testing.assert_allclose(loaded.T_mask_scene, mask.T_mask_scene)
