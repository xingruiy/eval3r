"""End-to-end smoke test: write → validate → inspect → metric."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from eval3r import PredictionWriter
from eval3r.cli.main import app
from eval3r.io.geometry import save_mesh_ply

runner = CliRunner(env={"NO_COLOR": "1"})


def test_e2e_metric_json(tmp_path, cube_mesh) -> None:
    v, f = cube_mesh
    rng = np.random.default_rng(0)
    perturbed = v + rng.normal(scale=0.005, size=v.shape)

    pred_dir = tmp_path / "pred"
    with PredictionWriter(
        pred_dir,
        scene_id="cube",
        dataset="synthetic",
        method="e2e",
        unit="m",
        coordinate_system="opengl",
        pose_convention="T_wc",
    ) as w:
        w.save_mesh(perturbed, f)

    gt_path = tmp_path / "gt_mesh.ply"
    save_mesh_ply(gt_path, v, f)

    val = runner.invoke(app, ["validate", str(pred_dir), "--json"])
    assert val.exit_code == 0
    assert json.loads(val.stdout)["ok"] is True

    insp = runner.invoke(app, ["inspect", str(pred_dir), "--json"])
    assert insp.exit_code == 0
    insp_payload = json.loads(insp.stdout)
    assert insp_payload["mesh.vertices"] == 8
    assert insp_payload["mesh.faces"] == 12

    metric = runner.invoke(
        app,
        [
            "metric", "geometry", str(pred_dir),
            "--gt", str(gt_path),
            "--samples", "8000",
            "--seed", "0",
            "--align", "none",
            "--thresholds", "0.05",
            "--json",
        ],
    )
    assert metric.exit_code == 0, metric.stdout
    payload = json.loads(metric.stdout)
    assert payload["chamfer"] < 0.05
    assert payload["fscore"]["0.05"]["f"] > 0.95


@pytest.mark.skipif(
    pytest.importorskip("pyrender", reason="pyrender not installed") is None, reason=""
)
def test_e2e_render_smoke(tmp_path, cube_mesh) -> None:
    v, f = cube_mesh
    mesh_path = tmp_path / "mesh.ply"
    save_mesh_ply(mesh_path, v, f)
    out = tmp_path / "render.png"
    result = runner.invoke(
        app,
        ["render", "mesh", str(mesh_path), "--out", str(out), "--headless"],
    )
    assert result.exit_code == 0, result.stdout
    assert out.exists() and out.stat().st_size > 0
