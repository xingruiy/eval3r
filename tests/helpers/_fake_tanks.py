"""Fixture builder for synthetic Tanks & Temples trees."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

from eval3r.io.geometry import save_point_cloud_ply

# Default fixture alignment: +10 m translation in X. Non-identity so tests
# can verify load_poses() actually applied it; simple enough that the
# expected aligned camera center is `raw + [10, 0, 0]` per frame.
_DEFAULT_ALIGNMENT: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0, 10.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


# Default fixture crop volume: a unit-cube prism (XZ unit square × Y∈[0,1])
# that covers the synthetic GT triangle [(0,0,0),(1,0,0),(0,1,0)].
_DEFAULT_CROP_JSON: dict = {
    "class_name": "SelectionPolygonVolume",
    "orthogonal_axis": "Y",
    "axis_min": 0.0,
    "axis_max": 1.0,
    "bounding_polygon": [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ],
    "version_major": 1,
    "version_minor": 0,
}


def make_tanks_scene(
    root: Path,
    scene_id: str,
    *,
    alignment: np.ndarray | Literal["skip"] | None = None,
    crop: dict | Literal["skip"] | None = None,
) -> Path:
    sd = root / scene_id
    sd.mkdir(parents=True, exist_ok=True)
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    save_point_cloud_ply(sd / f"{scene_id}.ply", points)
    # Images
    img_dir = sd / "image"
    img_dir.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as imageio
    arr = (np.ones((4, 4, 3)) * 128).astype(np.uint8)
    imageio.imwrite(img_dir / "000001.jpg", arr)
    # Poses (COLMAP_SfM.log style)
    Ts = []
    for i in range(2):
        T = np.eye(4)
        T[:3, 3] = [i * 0.01, 0, 0]
        Ts.append(T)
    pose_log = sd / f"{scene_id}_COLMAP_SfM.log"
    lines = []
    for i, T in enumerate(Ts):
        lines.append(f"{i} {i} 0")
        lines.extend(" ".join(f"{x:.12f}" for x in row) for row in T)
    pose_log.write_text("\n".join(lines) + "\n")
    # Alignment matrix ({scene}_trans.txt is the SfM→laser registration
    # shipped with T&T training scenes). Pass alignment="skip" to omit.
    if alignment != "skip":
        mat = _DEFAULT_ALIGNMENT if alignment is None else np.asarray(alignment, dtype=np.float64)
        np.savetxt(sd / f"{scene_id}_trans.txt", mat)
    # Crop volume ({scene}.json is the Open3D SelectionPolygonVolume the
    # T&T eval toolkit uses to clip predictions). Pass crop="skip" to omit.
    if crop != "skip":
        payload = _DEFAULT_CROP_JSON if crop is None else crop
        (sd / f"{scene_id}.json").write_text(json.dumps(payload))
    return sd


def make_tanks_root(tmp_path: Path, scene_ids: list[str], **kw) -> Path:
    for sid in scene_ids:
        make_tanks_scene(tmp_path, sid, **kw)
    split_path = tmp_path / "split.txt"
    split_path.write_text("\n".join(scene_ids))
    return split_path
