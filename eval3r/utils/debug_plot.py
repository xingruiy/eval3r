"""Debug plot of aligned point clouds using matplotlib."""

from __future__ import annotations

import numpy as np

from eval3r.utils.optional import optional_import
from eval3r.utils.typing import Points


def save_debug_plot(
    gt_pts: Points,
    pred_aligned: Points,
    out_path: str,
    align_mode: str,
    scale: float,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    pred_poses: np.ndarray | None = None,
    gt_poses: np.ndarray | None = None,
    pred_convention: str = "unspecified",
    gt_convention: str = "unspecified",
    matched_pred_idx: np.ndarray | None = None,
    matched_gt_idx: np.ndarray | None = None,
    *,
    max_points: int = 2000,
) -> None:
    """Downsample and export a 3D scatter plot of aligned prediction vs. GT.

    Args:
        gt_pts: Ground-truth points (N, 3).
        pred_aligned: Aligned prediction points (M, 3). May differ in length
            from the original sampled prediction when post-alignment filters
            (crop volume, occlusion mask) drop rows.
        out_path: Output image path (PNG).
        align_mode: Alignment mode label for the title.
        scale: Estimated scale factor for the title.
        max_points: Maximum points per cloud after downsampling.
    """
    plt = optional_import("matplotlib.pyplot", extra="render")
    optional_import("mpl_toolkits.mplot3d", extra="render")  # registers '3d' projection

    gt_pts = np.asarray(gt_pts, dtype=np.float64)
    pred_aligned = np.asarray(pred_aligned, dtype=np.float64)

    if len(pred_aligned) > max_points:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(pred_aligned), size=max_points, replace=False)
        pred_aligned = pred_aligned[idx]
    if len(gt_pts) > max_points:
        rng = np.random.default_rng(seed=123)
        idx = rng.choice(len(gt_pts), size=max_points, replace=False)
        gt_pts = gt_pts[idx]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")

    ax.scatter(
        gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2],
        c="#2ecc71", s=4, alpha=0.7, label="GT",
    )
    ax.scatter(
        pred_aligned[:, 0], pred_aligned[:, 1], pred_aligned[:, 2],
        c="#3498db", s=4, alpha=0.7, label="Pred (aligned)",
    )

    # Equal aspect ratio
    all_pts = np.concatenate([gt_pts, pred_aligned], axis=0)
    center = all_pts.mean(axis=0)
    half_span = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 + 0.01
    ax.set_xlim3d(center[0] - half_span, center[0] + half_span)
    ax.set_ylim3d(center[1] - half_span, center[1] + half_span)
    ax.set_zlim3d(center[2] - half_span, center[2] + half_span)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Alignment: {align_mode}  |  scale = {scale:.4f}")
    ax.legend(loc="upper right")

    if pred_poses is not None and gt_poses is not None:
        from eval3r.align.trajectory import cam_positions

        if pred_convention not in ("T_wc", "T_cw") or gt_convention not in ("T_wc", "T_cw"):
            # Pose metadata can be present for non-trajectory workflows where
            # conventions are intentionally unspecified; keep the debug plot
            # for point clouds and skip camera overlays in that case.
            pred_poses = None
            gt_poses = None

    if pred_poses is not None and gt_poses is not None:

        def _cam_dirs(poses: np.ndarray, convention: str) -> np.ndarray:
            poses = np.asarray(poses, dtype=np.float64)
            R = poses[:, :3, :3]
            if convention == "T_wc":
                return R[:, :, 2]
            if convention == "T_cw":
                return R.transpose(0, 2, 1)[:, :, 2]
            raise ValueError(
                f"Unknown pose convention: {convention!r}; expected 'T_wc' or 'T_cw'"
            )

        pred_centers = cam_positions(pred_poses, pred_convention)
        gt_centers = cam_positions(gt_poses, gt_convention)
        pred_dirs = _cam_dirs(pred_poses, pred_convention)
        gt_dirs = _cam_dirs(gt_poses, gt_convention)

        R = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
        t = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
        pred_centers = (scale * pred_centers @ R.T) + t
        pred_dirs = pred_dirs @ R.T

        if matched_pred_idx is None or matched_gt_idx is None:
            n = min(len(pred_centers), len(gt_centers))
            matched_pred_idx = np.arange(n)
            matched_gt_idx = np.arange(n)

        p = pred_centers[matched_pred_idx]
        g = gt_centers[matched_gt_idx]
        pd = pred_dirs[matched_pred_idx]
        gd = gt_dirs[matched_gt_idx]
        frustum_len = max(half_span * 0.05, 0.02)

        for c, d in zip(p, pd):
            tip = c + frustum_len * d
            ax.plot([c[0], tip[0]], [c[1], tip[1]], [c[2], tip[2]], c="#1f77b4", alpha=0.8)
            ax.scatter(c[0], c[1], c[2], c="#1f77b4", s=14, alpha=0.8)
        for c, d in zip(g, gd):
            tip = c + frustum_len * d
            ax.plot([c[0], tip[0]], [c[1], tip[1]], [c[2], tip[2]], c="#2ca02c", alpha=0.8)
            ax.scatter(c[0], c[1], c[2], c="#2ca02c", s=14, alpha=0.8)
        for pc, gc in zip(p, g):
            ax.plot([pc[0], gc[0]], [pc[1], gc[1]], [pc[2], gc[2]], c="#e67e22", alpha=0.4, lw=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
