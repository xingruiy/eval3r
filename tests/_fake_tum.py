"""Fixture builder for synthetic TUM RGB-D trees."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eval3r.io.trajectory import save_trajectory_tum

TUM_SAMPLE_TRAJECTORY = """\
# ground truth trajectory
# timestamp tx ty tz qx qy qz qw
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000
1.000000 0.010000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000
"""


def write_tum_depth(path: Path, value_mm: int = 5000) -> None:
    """Write a 4x4 uint16 depth map of *value_mm* everywhere."""
    import imageio.v3 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.ones((4, 4)) * value_mm).astype(np.uint16)
    imageio.imwrite(path, arr)


def write_tum_color(path: Path) -> None:
    import imageio.v3 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.ones((4, 4, 3)) * 128).astype(np.uint8)
    imageio.imwrite(path, arr)


def make_tum_sequence(root: Path, seq_name: str, *, n_frames: int = 2) -> Path:
    sd = root / seq_name
    rgb_dir = sd / "rgb"
    depth_dir = sd / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        ts = f"{i:.6f}"
        write_tum_color(rgb_dir / f"{ts}.png")
        write_tum_depth(depth_dir / f"{ts}.png")
    (sd / "groundtruth.txt").write_text(TUM_SAMPLE_TRAJECTORY)
    return sd


def make_tum_root(tmp_path: Path, seq_names: list[str], **kw) -> Path:
    for name in seq_names:
        make_tum_sequence(tmp_path, name, **kw)
    split_path = tmp_path / "split.txt"
    split_path.write_text("\n".join(seq_names))
    return split_path
