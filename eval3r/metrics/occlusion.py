"""Voxel occlusion mask for filtering predicted points in unseen regions."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from eval3r.utils.typing import Points


@dataclass
class OcclusionMask:
    """3D voxel grid marking occluded regions (1=occluded, 0=visible).

    The ``world2grid`` matrix maps world-space homogeneous coordinates
    ``[x, y, z, 1]`` to voxel-index space ``[i, j, k]``, where each
    integer coordinate selects a voxel centre.
    """

    grid: np.ndarray
    """3D float array of shape (Dx, Dy, Dz); 1.0 = occluded, 0.0 = visible."""

    world2grid: np.ndarray
    """4x4 affine matrix mapping world → voxel-index coordinates."""

    source: str = ""
    """Path to the mask file, for provenance."""


def load_occlusion_mask(
    mask_path: str | Path,
    world2grid_path: str | Path,
) -> OcclusionMask:
    """Load occlusion mask and world2grid transform from disk.

    Args:
        mask_path: Path to a ``.npy`` file containing the 3D occlusion grid.
        world2grid_path: Path to a whitespace-delimited 4×4 text file.

    Returns:
        OcclusionMask with the loaded data.
    """
    mask_path = Path(mask_path)
    grid = np.load(mask_path)
    if grid.ndim != 3:
        raise ValueError(
            f"Occlusion mask must be 3D, got shape {grid.shape}"
        )

    world2grid = np.loadtxt(world2grid_path)
    if world2grid.shape != (4, 4):
        raise ValueError(
            f"world2grid must be 4×4, got shape {world2grid.shape}"
        )

    return OcclusionMask(
        grid=grid.astype(np.float64, copy=False),
        world2grid=world2grid.astype(np.float64),
        source=str(mask_path),
    )


def filter_visible_points(
    points: Points,
    mask: OcclusionMask,
) -> tuple[Points, int, int]:
    """Filter predicted points to only those in visible (non-occluded) voxels.

    Trilinearly interpolates the occlusion grid at each point's grid-space
    location.  Points outside the grid bounds are treated as occluded.

    Returns:
        ``(visible_points, n_visible, n_total)``.  If every point is
        occluded the original points are returned unchanged (matching
        TransformerFusion's fallback), with a warning emitted.
    """
    n_total = len(points)

    # Transform world-space points to grid-index coordinates.
    homogeneous = np.column_stack([points, np.ones(n_total)])
    grid_coords = (mask.world2grid @ homogeneous.T).T[:, :3]  # (N, 3)

    # Trilinear interpolation (order=1) over the occlusion grid.
    # map_coordinates expects coords as a tuple of 1-D arrays, one per axis.
    # Mode 'constant' with cval=1.0 treats out-of-bounds as occluded.
    sampled = map_coordinates(
        mask.grid,
        (grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]),
        order=1,
        mode="constant",
        cval=1.0,
    )

    is_visible = sampled < 0.5
    n_visible = int(is_visible.sum())

    if n_visible == 0:
        warnings.warn(
            "All predicted points are occluded; keeping all points to avoid "
            "penalising the sample unfairly."
        )
        return points, n_total, n_total

    return points[is_visible], n_visible, n_total
