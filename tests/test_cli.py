"""CLI smoke tests via typer.testing.CliRunner."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from eval3r import PredictionWriter
from eval3r.cli.main import app
from eval3r.io.geometry import save_point_cloud_ply

runner = CliRunner()


def _write_pred(tmp_path, gaussian_cloud) -> str:
    out = tmp_path / "pred"
    with PredictionWriter(
        out, scene_id="s", dataset="d", method="m", unit="m",
        coordinate_system="opengl", pose_convention="T_wc",
    ) as w:
        w.save_point_cloud(gaussian_cloud)
    return str(out)


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    from eval3r._version import __version__

    assert __version__ in result.stdout


def test_validate_clean(tmp_path, gaussian_cloud) -> None:
    pred = _write_pred(tmp_path, gaussian_cloud)
    result = runner.invoke(app, ["validate", pred, "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True


def test_inspect_summary(tmp_path, gaussian_cloud) -> None:
    pred = _write_pred(tmp_path, gaussian_cloud)
    result = runner.invoke(app, ["inspect", pred, "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["scene_id"] == "s"
    assert payload["points.count"] == len(gaussian_cloud)


def test_metric_all_json(tmp_path, gaussian_cloud) -> None:
    pred = _write_pred(tmp_path, gaussian_cloud)
    gt_path = tmp_path / "gt.ply"
    save_point_cloud_ply(gt_path, gaussian_cloud)
    result = runner.invoke(
        app,
        [
            "metric", "all", pred,
            "--gt", str(gt_path),
            "--samples", "2048",
            "--seed", "0",
            "--align", "none",
            "--thresholds", "0.01",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["chamfer"] >= 0.0
    assert "0.01" in payload["fscore"]


def test_preset_show() -> None:
    result = runner.invoke(app, ["preset", "show", "scannet"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["dataset"] == "scannet"


def test_preset_list() -> None:
    result = runner.invoke(app, ["preset", "list"])
    assert result.exit_code == 0
    assert "scannet" in result.stdout


def test_metric_depth_png_scales(tmp_path) -> None:
    imageio = pytest.importorskip("imageio.v3")
    pred_path = tmp_path / "pred.png"
    gt_path = tmp_path / "gt.png"
    imageio.imwrite(pred_path, np.full((4, 4), 2000, dtype=np.uint16))
    imageio.imwrite(gt_path, np.full((4, 4), 2500, dtype=np.uint16))

    unscaled = runner.invoke(
        app, ["metric", "depth", str(pred_path), "--gt", str(gt_path), "--json"]
    )
    assert unscaled.exit_code == 0, unscaled.stdout
    unscaled_payload = json.loads(unscaled.stdout)

    scaled = runner.invoke(
        app,
        [
            "metric",
            "depth",
            str(pred_path),
            "--gt",
            str(gt_path),
            "--pred-scale",
            "0.001",
            "--gt-scale",
            "0.001",
            "--json",
        ],
    )
    assert scaled.exit_code == 0, scaled.stdout
    scaled_payload = json.loads(scaled.stdout)

    assert scaled_payload["abs_rel"] == pytest.approx(unscaled_payload["abs_rel"])
    assert scaled_payload["sq_rel"] == pytest.approx(unscaled_payload["sq_rel"] * 0.001)
    assert scaled_payload["rmse"] == pytest.approx(unscaled_payload["rmse"] * 0.001)


def test_metric_all_explicit_mask_requires_both_paths(tmp_path, gaussian_cloud) -> None:
    pred = _write_pred(tmp_path, gaussian_cloud)
    gt_path = tmp_path / "gt.ply"
    save_point_cloud_ply(gt_path, gaussian_cloud)
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, np.zeros((3, 3, 3), dtype=np.float64))

    result = runner.invoke(
        app,
        ["metric", "all", pred, "--gt", str(gt_path), "--mask", str(mask_path)],
        env={"NO_COLOR": "1", "TERM": "dumb"},
    )
    assert result.exit_code != 0
    assert "both --mask and --t-mask-scene" in result.output


def test_metric_all_explicit_mask_paths_are_used(tmp_path, gaussian_cloud) -> None:
    pred_path = tmp_path / "pred_mesh.ply"
    gt_path = tmp_path / "gt_mesh.ply"
    save_point_cloud_ply(pred_path, gaussian_cloud)
    save_point_cloud_ply(gt_path, gaussian_cloud)
    mask_path = tmp_path / "explicit_mask.npy"
    t_mask_scene_path = tmp_path / "T_mask_scene.txt"
    np.save(mask_path, np.zeros((5, 5, 5), dtype=np.float64))
    np.savetxt(t_mask_scene_path, np.eye(4))

    result = runner.invoke(
        app,
        [
            "metric", "all", str(pred_path),
            "--gt", str(gt_path),
            "--mask", str(mask_path),
            "--t-mask-scene", str(t_mask_scene_path),
            "--samples", "1024",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["masked"] is True


def test_metric_chamfer_validates_align_option(tmp_path, gaussian_cloud) -> None:
    pred = _write_pred(tmp_path, gaussian_cloud)
    gt_path = tmp_path / "gt.ply"
    save_point_cloud_ply(gt_path, gaussian_cloud)

    result = runner.invoke(
        app,
        ["metric", "chamfer", pred, "--gt", str(gt_path), "--align", "bad_align"],
    )
    assert result.exit_code == 2


def test_metric_chamfer_validates_chamfer_variant_option(tmp_path, gaussian_cloud) -> None:
    pred = _write_pred(tmp_path, gaussian_cloud)
    gt_path = tmp_path / "gt.ply"
    save_point_cloud_ply(gt_path, gaussian_cloud)

    result = runner.invoke(
        app,
        ["metric", "chamfer", pred, "--gt", str(gt_path), "--chamfer-variant", "bad_variant"],
    )
    assert result.exit_code == 2


def test_metric_fscore_validates_align_option(tmp_path, gaussian_cloud) -> None:
    pred = _write_pred(tmp_path, gaussian_cloud)
    gt_path = tmp_path / "gt.ply"
    save_point_cloud_ply(gt_path, gaussian_cloud)

    result = runner.invoke(
        app,
        ["metric", "fscore", pred, "--gt", str(gt_path), "--align", "bad_align"],
    )
    assert result.exit_code == 2
