from __future__ import annotations

from pathlib import Path

import numpy as np

from eval3r import PredictionWriter
from eval3r.io.geometry import save_mesh_ply, save_point_cloud_ply
from eval3r.prediction import PredictionLocator
from tests._fake_scannet import CUBE_FACES, CUBE_VERTS


def _scene_dir(root: Path, sid: str) -> Path:
    p = root / sid
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_locates_manifest_directory(tmp_path: Path) -> None:
    root = tmp_path / "preds"
    with PredictionWriter(
        root / "scene_a", scene_id="scene_a", dataset="d", method="m",
        unit="m", coordinate_system="opengl", pose_convention="T_wc",
    ) as w:
        w.save_mesh(CUBE_VERTS, CUBE_FACES)
    loc = PredictionLocator(preds_root=root)
    rp = loc.resolve("scene_a")
    assert rp is not None
    assert rp["kind"] == "manifest"
    assert rp["reader"] is not None


def test_falls_back_to_named_mesh(tmp_path: Path) -> None:
    root = tmp_path / "preds"
    save_mesh_ply(_scene_dir(root, "scene_b") / "scene_b_mesh.ply", CUBE_VERTS, CUBE_FACES)
    rp = PredictionLocator(preds_root=root).resolve("scene_b")
    assert rp is not None
    assert rp["kind"] == "mesh_file"
    assert rp["path"].name == "scene_b_mesh.ply"


def test_priority_manifest_over_raw(tmp_path: Path) -> None:
    root = tmp_path / "preds"
    sd = _scene_dir(root, "scene_c")
    save_mesh_ply(sd / "mesh.ply", CUBE_VERTS, CUBE_FACES)
    with PredictionWriter(
        sd, scene_id="scene_c", dataset="d", method="m",
        unit="m", coordinate_system="opengl", pose_convention="T_wc",
        overwrite=True,
    ) as w:
        w.save_mesh(CUBE_VERTS, CUBE_FACES)
    rp = PredictionLocator(preds_root=root).resolve("scene_c")
    assert rp is not None
    assert rp["kind"] == "manifest"


def test_extra_pattern_wins(tmp_path: Path) -> None:
    root = tmp_path / "preds"
    sd = _scene_dir(root, "scene_d")
    save_mesh_ply(sd / "custom" / "out.ply", CUBE_VERTS, CUBE_FACES)
    save_mesh_ply(sd / "mesh.ply", CUBE_VERTS, CUBE_FACES)
    rp = PredictionLocator(preds_root=root, extra_patterns=("custom/out.ply",)).resolve(
        "scene_d"
    )
    assert rp is not None
    assert rp["path"].as_posix().endswith("custom/out.ply")


def test_classifies_point_cloud(tmp_path: Path) -> None:
    root = tmp_path / "preds"
    save_point_cloud_ply(_scene_dir(root, "scene_e") / "points.ply", CUBE_VERTS)
    rp = PredictionLocator(preds_root=root).resolve("scene_e")
    assert rp is not None
    assert rp["kind"] == "point_cloud_file"


def test_returns_none_when_missing(tmp_path: Path) -> None:
    rp = PredictionLocator(preds_root=tmp_path / "preds").resolve("scene_x")
    assert rp is None
