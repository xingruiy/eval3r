"""Fixture builders for synthetic ScanNet-style trees."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eval3r.io.geometry import save_mesh_ply

CUBE_VERTS = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=np.float64,
)
CUBE_FACES = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [1, 2, 6],
        [1, 6, 5],
        [3, 0, 4],
        [3, 4, 7],
    ],
    dtype=np.int64,
)


def write_intrinsics(path: Path, fx: float = 500.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    K = np.array(
        [[fx, 0.0, 320.0, 0.0], [0.0, fx, 240.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0, 0, 0, 1]]
    )
    np.savetxt(path, K, fmt="%.6f")


def write_pose(path: Path, idx: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    T = np.eye(4)
    T[:3, 3] = [idx * 0.01, 0, 0]
    np.savetxt(path, T, fmt="%.6f")


def write_depth(path: Path) -> None:
    """Write a 4x4 uint16 depth map of 1000 mm everywhere."""
    import imageio.v3 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.ones((4, 4)) * 1000).astype(np.uint16)
    imageio.imwrite(path, arr)


def write_color(path: Path, shape: tuple[int, int] = (4, 4)) -> None:
    import imageio.v3 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = shape
    arr = (np.ones((h, w, 3)) * 128).astype(np.uint8)
    imageio.imwrite(path, arr)


def make_scene(
    root: Path,
    scene_id: str,
    *,
    n_frames: int = 2,
    color_subdir: str = "color",
    depth_subdir: str = "depth",
    pose_subdir: str = "pose",
    intrinsics_subdir: str = "intrinsic",
    color_shape: tuple[int, int] = (4, 4),
    mesh_filename: str | None = None,
) -> Path:
    sd = root / "scans" / scene_id
    sd.mkdir(parents=True, exist_ok=True)
    save_mesh_ply(sd / (mesh_filename or f"{scene_id}_vh_clean_2.ply"), CUBE_VERTS, CUBE_FACES)
    for i in range(n_frames):
        write_pose(sd / pose_subdir / f"{i}.txt", i)
        write_depth(sd / depth_subdir / f"{i}.png")
        write_color(sd / color_subdir / f"{i}.jpg", color_shape)
    write_intrinsics(sd / intrinsics_subdir / "intrinsic_depth.txt", fx=500.0)
    write_intrinsics(sd / intrinsics_subdir / "intrinsic_color.txt", fx=700.0)
    return sd


def make_scannet_root(tmp_path: Path, scene_ids: list[str], **kw) -> Path:
    """Build a fake ScanNet root and a matching split file."""
    for sid in scene_ids:
        make_scene(tmp_path, sid, **kw)
    split_path = tmp_path / "split.txt"
    split_path.write_text("\n".join(scene_ids))
    return split_path
