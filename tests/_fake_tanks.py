"""Fixture builder for synthetic Tanks & Temples trees."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eval3r.io.geometry import save_point_cloud_ply


def make_tanks_scene(root: Path, scene_id: str) -> Path:
    sd = root / scene_id
    sd.mkdir(parents=True, exist_ok=True)
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    save_point_cloud_ply(sd / "point_cloud.ply", points)
    # Images
    img_dir = sd / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as imageio
    arr = (np.ones((4, 4, 3)) * 128).astype(np.uint8)
    imageio.imwrite(img_dir / "0000.jpg", arr)
    # Poses
    pose_dir = sd / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        T = np.eye(4)
        T[:3, 3] = [i * 0.01, 0, 0]
        np.savetxt(pose_dir / f"{i:04d}.txt", T, fmt="%.6f")
    # Intrinsics
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    np.savetxt(sd / "intrinsics.txt", K, fmt="%.6f")
    return sd


def make_tanks_root(tmp_path: Path, scene_ids: list[str], **kw) -> Path:
    for sid in scene_ids:
        make_tanks_scene(tmp_path, sid, **kw)
    split_path = tmp_path / "split.txt"
    split_path.write_text("\n".join(scene_ids))
    return split_path
