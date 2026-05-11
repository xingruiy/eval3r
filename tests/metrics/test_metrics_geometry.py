from __future__ import annotations

import numpy as np
import pytest

from eval3r.metric.geometry import (
    accuracy,
    chamfer_distance,
    completeness,
    evaluate_geometry,
    fscore_at,
    precision_at,
    recall_at,
)
from eval3r.mask.base import CropToGT
from eval3r.mask.crop import CropVolume
from eval3r.mask.occlusion import (
    OcclusionMask,
    filter_visible_points,
    load_occlusion_mask,
)
from eval3r.utils.errors import EmptyGeometryError


def _grid(n: int = 10) -> np.ndarray:
    g = np.linspace(-1, 1, n)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def test_chamfer_identical_is_zero() -> None:
    pts = _grid()
    for variant in (
        "l1_mean_bidirectional",
        "l1_sum_bidirectional",
        "l2_squared",
        "l2_unsquared",
    ):
        assert chamfer_distance(pts, pts, variant=variant) == pytest.approx(0.0, abs=1e-12)


def test_accuracy_completeness_translated() -> None:
    pts = _grid()
    shifted = pts + np.array([0.05, 0.0, 0.0])
    # Each shifted point's nearest neighbour in pts is offset by [0.05, 0, 0].
    assert accuracy(shifted, pts) == pytest.approx(0.05, abs=1e-6)
    assert completeness(shifted, pts) == pytest.approx(0.05, abs=1e-6)


def test_fscore_perfect() -> None:
    pts = _grid()
    f, p, r = fscore_at(pts, pts, threshold=0.01)
    assert (f, p, r) == (1.0, 1.0, 1.0)


def test_fscore_outliers_reduce() -> None:
    pts = _grid()
    rng = np.random.default_rng(0)
    outliers = rng.uniform(low=10, high=20, size=(50, 3))
    polluted = np.concatenate([pts, outliers], axis=0)
    f_clean, _, _ = fscore_at(pts, pts, threshold=0.05)
    f_polluted, _, _ = fscore_at(polluted, pts, threshold=0.05)
    assert f_polluted < f_clean


def test_precision_recall_threshold_consistency() -> None:
    pts = _grid()
    shifted = pts + np.array([0.04, 0.0, 0.0])
    p_low = precision_at(shifted, pts, 0.01)
    p_high = precision_at(shifted, pts, 0.05)
    assert p_low == pytest.approx(0.0)
    assert p_high == pytest.approx(1.0)
    r_low = recall_at(shifted, pts, 0.01)
    r_high = recall_at(shifted, pts, 0.05)
    assert r_low == pytest.approx(0.0)
    assert r_high == pytest.approx(1.0)


def test_chamfer_rejects_empty() -> None:
    with pytest.raises(EmptyGeometryError):
        chamfer_distance(np.zeros((0, 3)), np.zeros((1, 3)))


def test_evaluate_geometry_with_align_se3() -> None:
    pts = _grid()
    pred = pts + np.array([1.0, -2.0, 0.5])
    result_none = evaluate_geometry(
        pts, pred, samples=4096, seed=0, align_mode="none", thresholds=[0.05]
    )
    result_se3 = evaluate_geometry(
        pts, pred, samples=4096, seed=0, align_mode="se3", thresholds=[0.05]
    )
    assert result_none.chamfer > 1.0
    # SE(3) recovers the offset; residual reflects independent sampling, not misalignment.
    assert result_se3.chamfer < 0.05
    assert result_se3.fscore[0.05]["f"] > 0.95


def test_evaluate_geometry_with_align_sim3() -> None:
    pts = _grid()
    scaled = pts * 1.05  # ICP from identity recovers small scale changes only.
    result_none = evaluate_geometry(
        pts, scaled, samples=4096, seed=0, align_mode="none", thresholds=[0.05]
    )
    result_sim3 = evaluate_geometry(
        pts, scaled, samples=4096, seed=0, align_mode="sim3", thresholds=[0.05]
    )
    assert result_sim3.chamfer < result_none.chamfer
    # alignment maps pred -> gt, so recovered scale ≈ 1.05.
    assert result_sim3.align_scale == pytest.approx(1.05, rel=0.05)


def test_duplicate_points_do_not_break() -> None:
    pts = _grid()
    dup = np.concatenate([pts, pts], axis=0)
    assert chamfer_distance(dup, pts) == pytest.approx(0.0, abs=1e-12)


def test_sampling_is_deterministic() -> None:
    from eval3r.metric.sampling import sample_points

    pts = _grid()
    a = sample_points(pts, 1000, method="uniform", seed=42)
    b = sample_points(pts, 1000, method="uniform", seed=42)
    assert np.array_equal(a, b)


def test_sampling_without_replacement_when_possible() -> None:
    from eval3r.metric.sampling import sample_points

    pts = np.arange(30, dtype=np.float64).reshape(10, 3)
    sampled = sample_points(pts, 10, method="uniform", seed=42)

    assert np.unique(sampled, axis=0).shape[0] == 10


def test_sampling_returns_all_points_when_n_exceeds_total() -> None:
    from eval3r.metric.sampling import sample_points

    pts = np.arange(15, dtype=np.float64).reshape(5, 3)
    sampled = sample_points(pts, 12, method="uniform", seed=42)

    assert sampled.shape == (5, 3)
    assert np.array_equal(sampled, pts)


def test_mesh_vertex_sampling_without_replacement_when_possible() -> None:
    from eval3r.io.geometry import MeshData
    from eval3r.metric.sampling import sample_points

    vertices = np.arange(30, dtype=np.float64).reshape(10, 3)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = MeshData(vertices=vertices, faces=faces)

    sampled = sample_points(mesh, 10, method="vertex", seed=42)

    assert np.unique(sampled, axis=0).shape[0] == 10


def test_mesh_vertex_sampling_returns_all_vertices_when_n_exceeds_total() -> None:
    from eval3r.io.geometry import MeshData
    from eval3r.metric.sampling import sample_points

    vertices = np.arange(15, dtype=np.float64).reshape(5, 3)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = MeshData(vertices=vertices, faces=faces)

    sampled = sample_points(mesh, 12, method="vertex", seed=42)

    assert sampled.shape == (5, 3)
    assert np.array_equal(sampled, vertices)


# ---------------------------------------------------------------------------
# Occlusion mask tests
# ---------------------------------------------------------------------------


def test_occlusion_mask_visible_center() -> None:
    """Point at the centre voxel (only visible one) is kept."""
    grid = np.ones((3, 3, 3), dtype=np.float64)
    grid[1, 1, 1] = 0.0
    mask = OcclusionMask(grid=grid, T_mask_scene=np.eye(4))
    pts = np.array([[1.0, 1.0, 1.0]])
    vis, n_vis, n_tot = filter_visible_points(pts, mask)
    assert n_vis == 1
    assert len(vis) == 1
    assert n_tot == 1


def test_occlusion_mask_all_occluded_raises() -> None:
    """When every point is occluded an explicit error is raised."""
    grid = np.ones((3, 3, 3), dtype=np.float64)
    mask = OcclusionMask(grid=grid, T_mask_scene=np.eye(4))
    pts = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="All points in the evaluated set were marked occluded"):
        filter_visible_points(pts, mask)


def test_occlusion_mask_mixed() -> None:
    """Mixed visible/occluded points are correctly filtered."""
    grid = np.ones((3, 3, 3), dtype=np.float64)
    grid[1, 1, 1] = 0.0
    mask = OcclusionMask(grid=grid, T_mask_scene=np.eye(4))
    pts = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    vis, n_vis, n_tot = filter_visible_points(pts, mask)
    assert n_vis == 1
    assert len(vis) == 1
    assert n_tot == 2


def test_occlusion_mask_out_of_bounds() -> None:
    """Points outside the grid are treated as occluded."""
    grid = np.ones((3, 3, 3), dtype=np.float64)
    grid[1, 1, 1] = 0.0
    mask = OcclusionMask(grid=grid, T_mask_scene=np.eye(4))
    pts = np.array([[10.0, 10.0, 10.0]])
    with pytest.raises(ValueError, match="including out-of-bounds treated as occluded"):
        filter_visible_points(pts, mask)


def test_occlusion_mask_with_transform() -> None:
    """Non-trivial T_mask_scene transform maps world coords correctly."""
    grid = np.ones((5, 5, 5), dtype=np.float64)
    grid[2, 2, 2] = 0.0  # visible at voxel index [2,2,2]
    # world [2.5, 2.5, 2.5] → 0.4*2.5 + 1.0 = 2.0
    w2g = np.array([
        [0.4, 0.0, 0.0, 1.0],
        [0.0, 0.4, 0.0, 1.0],
        [0.0, 0.0, 0.4, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    mask = OcclusionMask(grid=grid, T_mask_scene=w2g)
    pts = np.array([[2.5, 2.5, 2.5]])
    vis, n_vis, _ = filter_visible_points(pts, mask)
    assert n_vis == 1


def test_occlusion_mask_load_roundtrip(tmp_path) -> None:
    """load_occlusion_mask round-trips through disk correctly."""
    grid = np.ones((3, 3, 3), dtype=np.float64)
    grid[1, 1, 1] = 0.0
    w2g = np.eye(4)
    mask_path = tmp_path / "mask.npy"
    w2g_path = tmp_path / "w2g.txt"
    np.save(mask_path, grid)
    np.savetxt(w2g_path, w2g)
    loaded = load_occlusion_mask(mask_path, w2g_path)
    assert np.allclose(loaded.grid, grid)
    assert np.allclose(loaded.T_mask_scene, w2g)


def test_occlusion_mask_filter_points_contract() -> None:
    grid = np.array([[[0.0]], [[1.0]]], dtype=np.float64)
    mask = OcclusionMask(grid=grid, T_mask_scene=np.eye(4))
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    kept, n_kept, n_total = mask.filter_points(pts)
    np.testing.assert_array_equal(kept, pts[:1])
    assert n_kept == 1
    assert n_total == 2


def test_crop_to_gt_filter_points_contract() -> None:
    mask = CropToGT(
        bbox_min=np.array([0.0, 0.0, 0.0]),
        bbox_max=np.array([1.0, 1.0, 1.0]),
        margin=0.0,
    )
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.1, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    kept, n_kept, n_total = mask.filter_points(pts)
    np.testing.assert_array_equal(kept, pts[:2])
    assert n_kept == 2
    assert n_total == 3


def test_occlusion_mask_no_mask_backward_compat() -> None:
    """pred_mask=None produces the same result as the unmasked path."""
    pts = _grid(8)
    shifted = pts + np.array([0.05, 0.0, 0.0])
    result_unmasked = evaluate_geometry(
        pts, shifted, samples=2000, seed=0, thresholds=[0.05],
    )
    result_none = evaluate_geometry(
        pts, shifted, samples=2000, seed=0, thresholds=[0.05],
        pred_mask=None,
    )
    assert result_unmasked.chamfer == pytest.approx(result_none.chamfer)
    assert result_unmasked.accuracy == pytest.approx(result_none.accuracy)
    assert result_unmasked.completeness == pytest.approx(result_none.completeness)
    assert result_unmasked.masked is False


def test_occlusion_mask_improves_accuracy() -> None:
    """Masking outlier pred points improves accuracy vs unmasked."""
    gt_pts = _grid(6)  # dense grid around origin [-1,1]
    # Add far-away outliers to prediction — these are "hallucinated" in occluded regions.
    rng = np.random.default_rng(0)
    outliers = rng.uniform(low=10, high=15, size=(500, 3))
    pred_with_outliers = np.concatenate([gt_pts, outliers], axis=0)

    # Build a mask where the central region is visible (value 0)
    # and far-away regions are occluded (value 1).
    dim = 20
    grid = np.ones((dim, dim, dim), dtype=np.float64)
    cx = cy = cz = np.linspace(-5, 5, dim)
    gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
    grid[gx**2 + gy**2 + gz**2 < 4.0] = 0.0
    w2g = np.eye(4)
    w2g[:3, 3] = (dim - 1) / 2  # world 0 → grid centre
    mask = OcclusionMask(grid=grid, T_mask_scene=w2g)

    result_masked = evaluate_geometry(
        pred_with_outliers, gt_pts, samples=3000, seed=42, thresholds=[0.05],
        pred_mask=mask,
    )
    result_unmasked = evaluate_geometry(
        pred_with_outliers, gt_pts, samples=3000, seed=42, thresholds=[0.05],
    )

    # Mask filters far-away outliers → accuracy improves (lower is better).
    assert result_masked.accuracy < result_unmasked.accuracy
    # Completeness (gt→pred) uses unfiltered pred in both paths.
    assert result_masked.completeness == pytest.approx(result_unmasked.completeness, rel=1e-3)
    assert result_masked.masked is True
    assert result_masked.visible_points < result_masked.total_pred_points


def test_gt_mask_kwarg_removed() -> None:
    pred = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    grid = np.array([[[0.0]], [[1.0]]], dtype=np.float64)
    gt_mask = OcclusionMask(grid=grid, T_mask_scene=np.eye(4))

    with pytest.raises(TypeError):
        evaluate_geometry(
            pred,
            gt,
            samples=2,
            seed=0,
            sample_method="uniform",
            align_mode="none",
            thresholds=[0.1],
            gt_mask=gt_mask,
        )


def test_crop_volume_kwarg_removed() -> None:
    vol = CropVolume(
        polygon_2d=np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        ),
        axis_min=0.0,
        axis_max=1.0,
        orthogonal_axis=1,
    )
    with pytest.raises(TypeError):
        evaluate_geometry(
            np.zeros((1, 3), dtype=np.float64),
            np.zeros((1, 3), dtype=np.float64),
            samples=1,
            crop_volume=vol,
        )


def test_occlusion_mask_chamfer_l1_mean() -> None:
    """Masked chamfer (l1_mean_bidirectional) is computed from split directions."""
    pts = _grid(6)
    outliers = np.array([[100.0, 0.0, 0.0]])
    pred = np.concatenate([pts, outliers], axis=0)
    grid = np.ones((10, 10, 10), dtype=np.float64)
    grid[4:6, 4:6, 4:6] = 0.0  # small visible cube near origin
    w2g = np.eye(4)
    w2g[:3, 3] = 4.5  # center
    mask = OcclusionMask(grid=grid, T_mask_scene=w2g)

    result = evaluate_geometry(
        pts, pred, samples=1000, seed=0, thresholds=[0.05],
        chamfer_variant="l1_mean_bidirectional", pred_mask=mask,
    )
    assert result.chamfer_variant == "l1_mean_bidirectional"
    assert result.masked is True


def test_occlusion_mask_chamfer_l2_squared() -> None:
    """Masked chamfer with l2_squared variant works correctly."""
    pts = _grid(6)
    outliers = np.array([[100.0, 0.0, 0.0]])
    pred = np.concatenate([pts, outliers], axis=0)
    grid = np.ones((10, 10, 10), dtype=np.float64)
    grid[4:6, 4:6, 4:6] = 0.0
    w2g = np.eye(4)
    w2g[:3, 3] = 4.5
    mask = OcclusionMask(grid=grid, T_mask_scene=w2g)

    result = evaluate_geometry(
        pts, pred, samples=1000, seed=0, thresholds=[0.05],
        chamfer_variant="l2_squared", pred_mask=mask,
    )
    assert result.chamfer_variant == "l2_squared"
    assert result.masked is True
    assert result.chamfer >= 0.0
