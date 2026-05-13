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


def test_benchmark_reports_run_context_by_default(tmp_path: Path) -> None:
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
            "--thresholds", "0.05",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "benchmark: dataset=scannet" in result.output
    assert "scenes=2" in result.output
    assert f"benchmark: pred_root={preds}" in result.output
    assert f"benchmark: gt_root={ds_root}" in result.output
    assert "workers=1 samples=2048 seed=0" in result.output
    assert "thresholds=[0.05]" in result.output
    assert json.loads(result.stdout)["dataset"] == "scannet"


def test_datasets_list_and_validate(tmp_path: Path) -> None:
    ds_root, split, _ = _setup(tmp_path)
    list_result = runner.invoke(app, ["dataset", "list"])
    assert list_result.exit_code == 0
    assert "scannet" in list_result.stdout

    val_result = runner.invoke(
        app,
        [
            "dataset", "validate", "scannet",
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
            "dataset", "validate", "scannet",
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
            "benchmark", "scannet",
            "--pred-root", str(preds),
            "--gt-root", str(ds_root),
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
    assert payload["config"]["mask_pattern"] == "{scene_id}/occlusion_mask.npy"
    assert payload["config"]["t_mask_scene_pattern"] == "{scene_id}/T_mask_scene.txt"
    by_scene = {s["scene_id"]: s for s in payload["scenes"]}
    assert by_scene["s1"]["mask_missing"] is False
    assert by_scene["s2"]["mask_missing"] is True


def test_benchmark_mask_patterns_support_flat_layout(tmp_path: Path) -> None:
    ds_root, split, preds = _setup(tmp_path)
    out = tmp_path / "result_flat_mask.json"
    mask_root = tmp_path / "masks"
    mask_root.mkdir()
    np.save(mask_root / "s1_mask.npy", np.zeros((1, 1, 1), dtype=np.uint8))
    np.savetxt(mask_root / "s1_T_mask_scene.txt", np.eye(4))

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
            "--thresholds", "0.05",
            "--mask-dir", str(mask_root),
            "--mask-pattern", "{scene_id}_mask.npy",
            "--t-mask-scene-pattern", "{scene_id}_T_mask_scene.txt",
            "--out", str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(out.read_text())
    assert payload["config"]["mask_pattern"] == "{scene_id}_mask.npy"
    assert payload["config"]["t_mask_scene_pattern"] == "{scene_id}_T_mask_scene.txt"
    by_scene = {s["scene_id"]: s for s in payload["scenes"]}
    assert by_scene["s1"]["mask_missing"] is False
    assert by_scene["s2"]["mask_missing"] is True


def test_benchmark_mask_patterns_support_nested_layout(tmp_path: Path) -> None:
    ds_root, split, preds = _setup(tmp_path)
    out = tmp_path / "result_nested_mask.json"
    mask_root = tmp_path / "masks"
    (mask_root / "volumes" / "s1").mkdir(parents=True)
    (mask_root / "transforms").mkdir(parents=True)
    np.save(
        mask_root / "volumes" / "s1" / "visible.npy",
        np.zeros((1, 1, 1), dtype=np.uint8),
    )
    np.savetxt(mask_root / "transforms" / "s1.txt", np.eye(4))

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
            "--thresholds", "0.05",
            "--mask-dir", str(mask_root),
            "--mask-pattern", "volumes/{scene_id}/visible.npy",
            "--t-mask-scene-pattern", "transforms/{scene_id}.txt",
            "--out", str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(out.read_text())
    assert payload["config"]["mask_pattern"] == "volumes/{scene_id}/visible.npy"
    assert payload["config"]["t_mask_scene_pattern"] == "transforms/{scene_id}.txt"
    by_scene = {s["scene_id"]: s for s in payload["scenes"]}
    assert by_scene["s1"]["mask_missing"] is False
    assert by_scene["s2"]["mask_missing"] is True


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
            "--thresholds", "0.05",
            "--mask-dir", str(tmp_path / "masks"),
            "--mask-pattern", str(tmp_path / "mask.npy"),
        ],
    )

    assert result.exit_code != 0
    assert "--mask-pattern must be relative to --mask-dir" in result.output


def test_benchmark_generic_mode(tmp_path: Path) -> None:
    """The generic subcommand benchmarks a manual `{scene_id}` GT layout."""
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
            "benchmark", "generic",
            "--pred-root", str(preds),
            "--gt-root", str(gt_root),
            "--gt-path", "{scene_id}/gt.ply",
            "--scenes", "s1",
        ],
    )
    assert result.exit_code != 0
    assert "thresholds" in result.stdout.lower() or "thresholds" in (result.stderr or "").lower()


def test_benchmark_flat_command_removed(tmp_path: Path) -> None:
    """The old flat benchmark command is intentionally no longer accepted."""
    result = runner.invoke(
        app,
        [
            "benchmark", str(tmp_path),
            "--dataset", "definitely_not_a_dataset",
            "--root", str(tmp_path),
        ],
    )
    assert result.exit_code != 0


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


def test_benchmark_scannet_help_uses_new_roots_and_no_verbose() -> None:
    result = runner.invoke(
        app,
        ["benchmark", "scannet", "--help"],
        terminal_width=120,
    )
    assert result.exit_code == 0, result.stdout
    assert "--pred-root" in result.stdout
    assert "--gt-root" in result.stdout
    assert "--verbose" not in result.stdout
