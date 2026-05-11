"""Fixture builder for synthetic Replica-style trees."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eval3r.io.geometry import save_mesh_ply
from tests.helpers._fake_scannet import CUBE_FACES, CUBE_VERTS, write_color, write_depth

REPLICA_TRAJECTORY = """\
0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 1.000000000
1.000000000 0.010000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 1.000000000
"""


def make_replica_scene(root: Path, scene_id: str, *, with_results: bool = True) -> Path:
    sd = root / scene_id
    sd.mkdir(parents=True, exist_ok=True)
    save_mesh_ply(sd / "mesh.ply", CUBE_VERTS, CUBE_FACES)
    if with_results:
        results_dir = sd / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            write_depth(results_dir / f"depth{i}.png")
            write_color(results_dir / f"rgb{i}.png")
    (sd / "trajectory.txt").write_text(REPLICA_TRAJECTORY)
    return sd


def make_replica_root(tmp_path: Path, scene_ids: list[str], **kw) -> Path:
    for sid in scene_ids:
        make_replica_scene(tmp_path, sid, **kw)
    split_path = tmp_path / "split.txt"
    split_path.write_text("\n".join(scene_ids))
    return split_path
