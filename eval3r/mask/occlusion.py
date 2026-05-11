"""Voxel occlusion mask for filtering predicted points in unseen regions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from eval3r.utils.typing import Points


@dataclass
class OcclusionMask:
    """3D voxel grid marking occluded regions (1=occluded, 0=visible).

    The ``T_mask_scene`` matrix maps homogeneous scene coordinates
    ``[x, y, z, 1]`` to continuous voxel coordinates ``[i, j, k, 1]``, where each
    integer coordinate selects a voxel centre.
    """

    grid: np.ndarray
    """3D float array of shape (Dx, Dy, Dz); 1.0 = occluded, 0.0 = visible."""

    T_mask_scene: np.ndarray
    """4x4 affine matrix mapping scene/world → occlusion-mask voxel coordinates."""

    source: str = ""
    """Path to the mask file, for provenance."""

    def filter_points(self, points: Points) -> tuple[Points, int, int]:
        return filter_visible_points(points, self)


def load_occlusion_mask(
    mask_path: str | Path,
    T_mask_scene_path: str | Path,
) -> OcclusionMask:
    """Load occlusion mask and ``T_mask_scene`` transform from disk.

    Args:
        mask_path: Path to a ``.npy`` file containing the 3D occlusion grid.
        T_mask_scene_path: Path to a whitespace-delimited 4×4 text file.

    Returns:
        OcclusionMask with the loaded data.
    """
    mask_path = Path(mask_path)
    grid = np.load(mask_path)
    if grid.ndim != 3:
        raise ValueError(
            f"Occlusion mask must be 3D, got shape {grid.shape}"
        )

    T_mask_scene = np.loadtxt(T_mask_scene_path)
    if T_mask_scene.shape != (4, 4):
        raise ValueError(
            f"T_mask_scene must be 4×4, got shape {T_mask_scene.shape}"
        )

    return OcclusionMask(
        grid=grid.astype(np.float64, copy=False),
        T_mask_scene=T_mask_scene.astype(np.float64),
        source=str(mask_path),
    )


def save_occlusion_mask(
    mask: OcclusionMask,
    out_dir: str | Path,
    *,
    mask_name: str = "occlusion_mask.npy",
    t_mask_scene_name: str = "T_mask_scene.txt",
) -> tuple[Path, Path]:
    """Save an occlusion mask to ``out_dir`` as a ``.npy`` + ``.txt`` pair.

    Mirrors the layout :func:`load_occlusion_mask` reads. Creates ``out_dir``
    if it does not already exist.

    Returns:
        ``(mask_path, T_mask_scene_path)``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid = np.asarray(mask.grid, dtype=np.float64)
    if grid.ndim != 3:
        raise ValueError(f"Occlusion mask must be 3D, got shape {grid.shape}")
    T = np.asarray(mask.T_mask_scene, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_mask_scene must be 4×4, got shape {T.shape}")
    mask_path = out / mask_name
    t_path = out / t_mask_scene_name
    np.save(mask_path, grid)
    np.savetxt(t_path, T)
    return mask_path, t_path


def filter_visible_points(
    points: Points,
    mask: OcclusionMask,
) -> tuple[Points, int, int]:
    """Filter a point set to only those in visible (non-occluded) voxels.

    Trilinearly interpolates the occlusion grid at each point's grid-space
    location.  Points outside the grid bounds are treated as occluded.

    Returns:
        ``(visible_points, n_visible, n_total)``.

    Raises:
        ValueError: If every point is occluded (including out-of-bounds),
            which usually indicates a bad occlusion transform, coordinate
            frame mismatch, or unit-scale mismatch.
    """
    n_total = len(points)

    # Transform world-space points to grid-index coordinates.
    homogeneous = np.column_stack([points, np.ones(n_total)])
    grid_coords = (mask.T_mask_scene @ homogeneous.T).T[:, :3]  # (N, 3)

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
        raise ValueError(
            "All points in the evaluated set were marked occluded by the occlusion mask "
            "(including out-of-bounds treated as occluded). This usually "
            "indicates an invalid T_mask_scene, a coordinate frame mismatch, "
            "or a unit-scale mismatch."
        )

    return points[is_visible], n_visible, n_total
