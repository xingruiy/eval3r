"""CLI tests for `e3r benchmark <dataset>`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from eval3r import PredictionWriter
from eval3r.cli.main import app
from eval3r.io.geometry import save_mesh_ply
from tests.helpers._fake_scannet import CUBE_FACES, CUBE_VERTS, make_scannet_root

runner = CliRunner(env={"NO_COLOR": "1"})


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    split = make_scannet_root(tmp_path / "ds", ["s1", "s2"])
    preds = tmp_path / "preds"
    with PredictionWriter(
        preds / "s1", scene_id="s1", dataset="d", method="m",
        unit="m", coordinate_system="opengl", pose_convention="T_wc",
    ) as w:
        w.save_mesh(CUBE_VERTS, CUBE_FACES)
    save_mesh_ply(preds / "s2" / "mesh.ply", CUBE_VERTS, CUBE_FACES)
    return tmp_path / "ds", split, preds


def test_benchmark_scannet_json_out(tmp_path: Path) -> None:
    ds_root, split, preds = _setup(tmp_path)
    out = tmp_path / "result.json"
    result = runner.invoke(
        app,
        [
            "benchmark", "scannet",
            "--pred-root", str(preds),
            "--gt-root", str(ds_root),
            "--split", str(split),
            "--samples", "2048",
            "--seed", "0",
            "--workers", "1",
            "--metric", "chamfer",
            "--metric", "fscore@0.05",
            "--out", str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(out.read_text())
    assert payload["dataset"] == "scannet"
    assert payload["coverage"]["n_evaluated"] == 2
    assert payload["summary"]["chamfer"]["n"] == 2
    assert payload["summary_all"]["chamfer"]["n"] == 2


def test_benchmark_scannet_default_metrics(tmp_path: Path) -> None:
    ds_root, split, preds = _setup(tmp_path)
    result = runner.invoke(
        app,
        [
            "benchmark", "scannet",
            "--pred-root", str(preds),
            "--gt-root", str(ds_root),
            "--split", str(split),
            "--samples", "2048",
            "--seed", "0",
            "--workers", "1",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["dataset"] == "scannet"
    assert "chamfer" in payload["summary"]


def test_benchmark_help_lists_dataset_subcommands() -> None:
    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0, result.stdout
    for name in [
        "scannet",
        "tanks-temples",
        "tum-rgbd",
        "replica",
        "dtu",
        "eth3d",
        "generic",
    ]:
        assert name in result.stdout


def test_benchmark_mask_dir_stored_in_config(tmp_path: Path) -> None:
    ds_root, split, preds = _setup(tmp_path)
    out = tmp_path / "result.json"
    mask_root = tmp_path / "masks"
    (mask_root / "s1").mkdir(parents=True)
    np.save(mask_root / "s1" / "occlusion_mask.npy", np.zeros((1, 1, 1), dtype=np.uint8))
    np.savetxt(mask_root / "s1" / "T_mask_scene.txt", np.eye(4))

    result = runner.invoke(
        app,
        [
            "benchmark", "scannet",
            "--pred-root", str(preds),
            "--gt-root", str(ds_root),
            "--split", str(split),
            "--samples", "2048",
            "--seed", "0",
            "--workers", "1",
            "--mask-dir", str(mask_root),
            "--out", str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(out.read_text())
    assert payload["config"]["mask_dir"] == str(mask_root)
    assert payload["config"]["mask_pattern"] == "{scene_id}/occlusion_mask.npy"
    assert payload["config"]["t_mask_scene_pattern"] == "{scene_id}/T_mask_scene.txt"


def test_benchmark_mask_patterns_must_be_relative(tmp_path: Path) -> None:
    ds_root, split, preds = _setup(tmp_path)
    result = runner.invoke(
        app,
        [
            "benchmark", "scannet",
            "--pred-root", str(preds),
            "--gt-root", str(ds_root),
            "--split", str(split),
            "--samples", "2048",
            "--seed", "0",
            "--workers", "1",
            "--mask-dir", str(tmp_path / "masks"),
            "--mask-pattern", str(tmp_path / "mask.npy"),
        ],
    )
    assert result.exit_code != 0
    assert "--mask-pattern must be relative to --mask-dir" in result.output


def test_benchmark_generic_mode(tmp_path: Path) -> None:
    gt_root = tmp_path / "ds"
    save_mesh_ply(gt_root / "s1" / "gt.ply", CUBE_VERTS, CUBE_FACES)
    save_mesh_ply(gt_root / "s2" / "gt.ply", CUBE_VERTS, CUBE_FACES)

    preds = tmp_path / "preds"
    save_mesh_ply(preds / "s1" / "mesh.ply", CUBE_VERTS, CUBE_FACES)
    save_mesh_ply(preds / "s2" / "mesh.ply", CUBE_VERTS, CUBE_FACES)

    out = tmp_path / "manual.json"
    result = runner.invoke(
        app,
        [
            "benchmark", "generic",
            "--pred-root", str(preds),
            "--gt-root", str(gt_root),
            "--gt-path", "{scene_id}/gt.ply",
            "--scenes", "s1,s2",
            "--samples", "2048",
            "--seed", "0",
            "--workers", "1",
            "--metric", "chamfer",
            "--metric", "fscore@0.05",
            "--out", str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(out.read_text())
    assert payload["dataset"] == "generic"
    assert payload["coverage"]["n_evaluated"] == 2


def test_benchmark_generic_requires_scene_source(tmp_path: Path) -> None:
    gt_root = tmp_path / "ds"
    save_mesh_ply(gt_root / "s1" / "gt.ply", CUBE_VERTS, CUBE_FACES)
    preds = tmp_path / "preds"
    save_mesh_ply(preds / "s1" / "mesh.ply", CUBE_VERTS, CUBE_FACES)

    result = runner.invoke(
        app,
        [
            "benchmark", "generic",
            "--pred-root", str(preds),
            "--gt-root", str(gt_root),
            "--gt-path", "{scene_id}/gt.ply",
        ],
    )
    assert result.exit_code != 0
