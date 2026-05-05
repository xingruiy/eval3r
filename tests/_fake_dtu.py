"""Fixture builder for synthetic DTU trees."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eval3r.io.geometry import save_point_cloud_ply


def write_empty_image(path: Path) -> None:
    import imageio.v3 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.ones((4, 4, 3)) * 128).astype(np.uint8)
    imageio.imwrite(path, arr)


def make_dtu_scan(root: Path, scan_id: int) -> Path:
    sd = root / "scans" / f"scan{scan_id}"
    sd.mkdir(parents=True, exist_ok=True)
    # Point cloud GT.
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    save_point_cloud_ply(sd / "points.ply", points)
    # Images — create so probing works.
    img_dir = sd / "image"
    write_empty_image(img_dir / "000000.png")
    # Camera poses: one 3x4 projection matrix per line.
    cam_dir = root / "Cameras" / f"scan{scan_id}"
    cam_dir.mkdir(parents=True, exist_ok=True)
    P = np.array([
        [2892.33, 0.0, 800.0, 0.0],
        [0.0, 2883.18, 600.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    np.savetxt(cam_dir / "camera_poses.txt", P.reshape(1, 12), fmt="%.6f")
    return sd


def make_dtu_root(tmp_path: Path, scan_ids: list[int], **kw) -> Path:
    for sid in scan_ids:
        make_dtu_scan(tmp_path, sid, **kw)
    split_path = tmp_path / "split.txt"
    split_path.write_text("\n".join(str(s) for s in scan_ids))
    return split_path
