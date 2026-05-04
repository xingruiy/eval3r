"""CLI smoke tests via typer.testing.CliRunner."""

from __future__ import annotations

import json

import numpy as np
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
