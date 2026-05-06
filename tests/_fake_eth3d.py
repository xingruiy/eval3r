"""Fixture builder for synthetic ETH3D trees."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eval3r.io.geometry import save_mesh_ply
from tests._fake_scannet import CUBE_FACES, CUBE_VERTS


def make_eth3d_scene(root: Path, scene_id: str, *, track: str = "dslr") -> Path:
    sd = root / track / scene_id
    sd.mkdir(parents=True, exist_ok=True)
    save_mesh_ply(sd / "scan.ply", CUBE_VERTS, CUBE_FACES)
    # Images
    img_dir = sd / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as imageio
    arr = (np.ones((4, 4, 3)) * 128).astype(np.uint8)
    imageio.imwrite(img_dir / "0000.jpg", arr)
    # COLMAP calibration
    calib_dir = sd / "dslr_calibration_jpg"
    calib_dir.mkdir(parents=True, exist_ok=True)
    # images.txt: COLMAP format
    images_lines = [
        "# Image list with two lines per image: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# empty line follows each image",
        "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 0000.jpg",
        "",
        "2 1.0 0.0 0.0 0.0 0.01 0.0 0.0 1 0001.jpg",
        "",
    ]
    (calib_dir / "images.txt").write_text("\n".join(images_lines))
    # cameras.txt
    (calib_dir / "cameras.txt").write_text(
        "1 PINHOLE 640 480 500.0 500.0 320.0 240.0\n"
    )
    (calib_dir / "calibration.txt").write_text(
        "500.0 0.0 320.0\n0.0 500.0 240.0\n0.0 0.0 1.0\n"
    )
    return sd


def make_eth3d_root(tmp_path: Path, scene_ids: list[str], **kw) -> Path:
    for sid in scene_ids:
        make_eth3d_scene(tmp_path, sid, **kw)
    split_path = tmp_path / "split.txt"
    split_path.write_text("\n".join(scene_ids))
    return split_path
