"""Visualization helpers for dense occlusion-mask grids."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.ndimage import binary_erosion

MaskValue = Literal["occluded", "visible"]


def _selected_voxels(grid: np.ndarray, value: MaskValue) -> np.ndarray:
    if value == "occluded":
        return np.asarray(grid) >= 0.5
    if value == "visible":
        return np.asarray(grid) < 0.5
    raise ValueError(f"value must be 'occluded' or 'visible', got {value!r}")


def _block_factors(shape: tuple[int, int, int], max_dim: int) -> tuple[int, int, int]:
    if max_dim < 1:
        raise ValueError("max_dim must be >= 1")
    return tuple(max(1, int(np.ceil(s / max_dim))) for s in shape)


def _block_mean(mask: np.ndarray, factors: tuple[int, int, int]) -> np.ndarray:
    trim_shape = tuple((s // f) * f for s, f in zip(mask.shape, factors, strict=True))
    slices = tuple(slice(0, s) for s in trim_shape)
    trimmed = mask[slices]
    reshaped = trimmed.reshape(
        trim_shape[0] // factors[0],
        factors[0],
        trim_shape[1] // factors[1],
        factors[1],
        trim_shape[2] // factors[2],
        factors[2],
    )
    return reshaped.mean(axis=(1, 3, 5), dtype=np.float32)


def downsample_mask(
    grid: np.ndarray,
    *,
    value: MaskValue = "occluded",
    max_dim: int = 160,
    threshold: float = 0.25,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return a compact boolean occupancy grid and original-grid scale factors."""
    selected = _selected_voxels(grid, value)
    factors = _block_factors(selected.shape, max_dim)
    if factors == (1, 1, 1):
        return selected.astype(bool, copy=False), factors
    density = _block_mean(selected, factors)
    return density >= threshold, factors


def save_mask_overview(
    grid: np.ndarray,
    out_path: str | Path,
    *,
    value: MaskValue = "occluded",
) -> Path:
    """Save central slices and density projections as a compact PNG sheet."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/eval3r-matplotlib")
    from eval3r.utils.optional import optional_import

    plt = optional_import("matplotlib.pyplot", extra="render")

    selected = _selected_voxels(grid, value)
    centers = tuple(s // 2 for s in selected.shape)
    projections = [
        (selected.mean(axis=0).T, "density over x"),
        (selected.mean(axis=1).T, "density over y"),
        (selected.mean(axis=2).T, "density over z"),
    ]
    slices = [
        (selected[centers[0], :, :].T, f"x slice {centers[0]}"),
        (selected[:, centers[1], :].T, f"y slice {centers[1]}"),
        (selected[:, :, centers[2]].T, f"z slice {centers[2]}"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, (image, title) in zip(axes.flat, projections + slices, strict=True):
        im = ax.imshow(image, origin="lower", interpolation="nearest", cmap="magma")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if title.startswith("density"):
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{value} voxels: shape={tuple(int(s) for s in grid.shape)}, "
        f"fraction={selected.mean():.4f}"
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_mask_point_cloud(
    grid: np.ndarray,
    out_path: str | Path,
    *,
    value: MaskValue = "occluded",
    max_dim: int = 160,
    threshold: float = 0.25,
    max_points: int = 500_000,
    t_mask_scene: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[Path, int, tuple[int, int, int]]:
    """Save a downsampled boundary point cloud as PLY.

    Points are emitted at selected downsampled surface cells. When
    ``t_mask_scene`` is provided, coordinates are written in scene/world space;
    otherwise they are written in original voxel coordinates.
    """
    import trimesh

    occupancy, factors = downsample_mask(
        grid, value=value, max_dim=max_dim, threshold=threshold
    )
    if not occupancy.any():
        raise ValueError("No voxels selected for visualization")

    eroded = binary_erosion(occupancy, structure=np.ones((3, 3, 3), dtype=bool))
    surface = occupancy & ~eroded
    coords = np.argwhere(surface).astype(np.float64)

    if len(coords) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(coords), size=max_points, replace=False)
        coords = coords[keep]

    scale = np.asarray(factors, dtype=np.float64)
    points = (coords + 0.5) * scale
    if t_mask_scene is not None:
        T_inv = np.linalg.inv(np.asarray(t_mask_scene, dtype=np.float64))
        homog = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        points = (T_inv @ homog.T).T[:, :3]

    color = np.array([230, 92, 46, 255], dtype=np.uint8)
    if value == "visible":
        color = np.array([48, 156, 214, 255], dtype=np.uint8)
    colors = np.broadcast_to(color, (len(points), 4))
    cloud = trimesh.PointCloud(points, colors=colors)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cloud.export(out)
    return out, int(len(points)), factors
