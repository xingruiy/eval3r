"""Debug plot of aligned point clouds using matplotlib."""

from __future__ import annotations

import numpy as np

from eval3r.utils.optional import optional_import
from eval3r.utils.typing import Points


def save_debug_plot(
    pred_pts: Points,
    gt_pts: Points,
    pred_aligned: Points,
    out_path: str,
    align_mode: str,
    scale: float,
    *,
    max_points: int = 2000,
) -> None:
    """Downsample and export a 3D scatter plot of aligned prediction vs. GT.

    Args:
        pred_pts: Original prediction points (N, 3).
        gt_pts: Ground-truth points (N, 3).
        pred_aligned: Aligned prediction points (N, 3).
        out_path: Output image path (PNG).
        align_mode: Alignment mode label for the title.
        scale: Estimated scale factor for the title.
        max_points: Maximum points per cloud after downsampling.
    """
    plt = optional_import("matplotlib.pyplot", extra="render")
    optional_import("mpl_toolkits.mplot3d", extra="render")  # registers '3d' projection

    pred_pts = np.asarray(pred_pts, dtype=np.float64)
    gt_pts = np.asarray(gt_pts, dtype=np.float64)
    pred_aligned = np.asarray(pred_aligned, dtype=np.float64)

    # Downsample
    if len(pred_pts) > max_points:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(pred_pts), size=max_points, replace=False)
        pred_pts = pred_pts[idx]
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

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
