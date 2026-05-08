from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from eval3r import PredictionWriter
from eval3r.cli.main import app
from eval3r.io.geometry import save_mesh_ply
from tests._fake_scannet import CUBE_FACES, CUBE_VERTS, make_scannet_root

runner = CliRunner()


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
            "benchmark", str(preds),
            "--dataset", "scannet",
            "--root", str(ds_root),
            "--split", str(split),
            "--samples", "2048",
            "--seed", "0",
            "--workers", "1",
            "--thresholds", "0.05",
            "--out", str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(out.read_text())
    assert payload["dataset"] == "scannet"
    assert payload["coverage"]["n_evaluated"] == 2
    assert payload["summary"]["chamfer"]["n"] == 2
    assert payload["summary_all"]["chamfer"]["n"] == 2  # all scenes present


def test_datasets_list_and_validate(tmp_path: Path) -> None:
    ds_root, split, _ = _setup(tmp_path)
    list_result = runner.invoke(app, ["datasets", "list"])
    assert list_result.exit_code == 0
    assert "scannet" in list_result.stdout

    val_result = runner.invoke(
        app,
        [
            "datasets", "validate", "scannet",
            "--root", str(ds_root),
            "--split", str(split),
            "--scenes", "1",
        ],
    )
    assert val_result.exit_code == 0, val_result.stdout


def test_datasets_validate_fails_on_renamed_color(tmp_path: Path) -> None:
    split = make_scannet_root(tmp_path / "ds", ["s1"], color_subdir="images")
    val_result = runner.invoke(
        app,
        [
            "datasets", "validate", "scannet",
            "--root", str(tmp_path / "ds"),
            "--split", str(split),
            "--scenes", "1",
        ],
    )
    # Default color_subdir=color is wrong → validation should report error.
    assert val_result.exit_code == 1, val_result.stdout


def test_benchmark_json_marks_missing_scene_mask(tmp_path: Path) -> None:
    ds_root, split, preds = _setup(tmp_path)
    out = tmp_path / "result_mask.json"
    mask_root = tmp_path / "masks"
    (mask_root / "s1").mkdir(parents=True)
    np.save(mask_root / "s1" / "occlusion_mask.npy", np.zeros((1, 1, 1), dtype=np.uint8))
    np.savetxt(mask_root / "s1" / "T_mask_scene.txt", np.eye(4))

    result = runner.invoke(
        app,
        [
            "benchmark", str(preds),
            "--dataset", "scannet",
            "--root", str(ds_root),
            "--split", str(split),
            "--samples", "2048",
            "--seed", "0",
            "--workers", "1",
            "--thresholds", "0.05",
            "--mask-dir", str(mask_root),
            "--out", str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(out.read_text())
    by_scene = {s["scene_id"]: s for s in payload["scenes"]}
    assert by_scene["s1"]["mask_missing"] is False
    assert by_scene["s2"]["mask_missing"] is True


def test_benchmark_manual_mode(tmp_path: Path) -> None:
    """Omitting --dataset enters manual mode via GenericAdapter + --gt-path."""
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
            "benchmark", str(preds),
            "--root", str(gt_root),
            "--gt-path", "{scene_id}/gt.ply",
            "--scenes", "s1,s2",
            "--samples", "2048",
            "--seed", "0",
            "--workers", "1",
            "--thresholds", "0.05",
            "--out", str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(out.read_text())
    assert payload["dataset"] == "generic"
    assert payload["coverage"]["n_evaluated"] == 2


def test_benchmark_manual_mode_requires_thresholds(tmp_path: Path) -> None:
    """Manual mode without --thresholds errors out (no preset to fall back on)."""
    gt_root = tmp_path / "ds"
    save_mesh_ply(gt_root / "s1" / "gt.ply", CUBE_VERTS, CUBE_FACES)
    preds = tmp_path / "preds"
    save_mesh_ply(preds / "s1" / "mesh.ply", CUBE_VERTS, CUBE_FACES)

    result = runner.invoke(
        app,
        [
            "benchmark", str(preds),
            "--root", str(gt_root),
            "--gt-path", "{scene_id}/gt.ply",
            "--scenes", "s1",
        ],
    )
    assert result.exit_code != 0
    assert "thresholds" in result.stdout.lower() or "thresholds" in (result.stderr or "").lower()


def test_benchmark_unknown_dataset_errors(tmp_path: Path) -> None:
    """Unknown --dataset name surfaces as a hard error, no silent fallback."""
    result = runner.invoke(
        app,
        [
            "benchmark", str(tmp_path),
            "--dataset", "definitely_not_a_dataset",
            "--root", str(tmp_path),
        ],
    )
    assert result.exit_code != 0
