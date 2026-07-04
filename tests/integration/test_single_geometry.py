"""Task 007 integration tests: the single-file geometry path end-to-end.

Covers the Python API and the ``e3r metric geometry`` CLI on tiny committed PLY
fixtures, and checks that a complete, schema-valid run directory is written.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from eval3r import evaluate_geometry
from eval3r.cli.main import app
from eval3r.reports.json import read_run_result_json

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "geom"
runner = CliRunner()

# Fields .agent/reproducibility.md requires in results.json.
REQUIRED = [
    "schema_version", "protocol", "protocol_hash", "dataset", "split",
    "ground_truth", "local_evaluation", "n_scenes_expected", "n_scenes_evaluated",
    "alignment", "masking", "sampling", "metric_definitions", "confidence_policy",
    "failure_policy", "backend_versions", "command", "environment",
]


def test_identical_pointclouds_score_perfectly(tmp_path: Path) -> None:
    run = evaluate_geometry(
        FIX / "pred.ply", FIX / "gt.ply",
        input_type="pointcloud", gt_type="pointcloud",
        threshold=0.05, out_dir=tmp_path / "run", return_run=True,
    )
    m = run.result.metrics
    assert m["accuracy"] == 0.0
    assert m["completeness"] == 0.0
    assert m["fscore"] == 1.0
    assert run.result.n_scenes_evaluated == 1
    assert not run.result.failed_scenes


def test_run_directory_is_complete_and_schema_valid(tmp_path: Path) -> None:
    out = tmp_path / "run"
    evaluate_geometry(FIX / "pred.ply", FIX / "gt.ply", threshold=0.05, out_dir=out)

    for name in (
        "results.json", "results.csv", "per_scene.csv", "failures.json",
        "environment.json", "backend_versions.json", "logs.txt",
        "protocol.yaml", "config.yaml", "alignment_transforms.json",
    ):
        assert (out / name).is_file(), f"missing {name}"

    result = read_run_result_json(out / "results.json")
    data = json.loads((out / "results.json").read_text())
    missing = [f for f in REQUIRED if f not in data]
    assert not missing, f"results.json missing required fields: {missing}"
    assert result.protocol == "single_geometry"
    assert result.backend_versions["nearest_neighbor"]["name"] == "scipy"

    # environment.json is minimal and non-identifying (no cwd/git/timestamp/timezone).
    env = json.loads((out / "environment.json").read_text())
    for banned in ("cwd", "working_dir", "git", "git_commit", "timestamp", "timezone"):
        assert banned not in env


def test_mesh_sampling_is_deterministic(tmp_path: Path) -> None:
    kwargs = dict(input_type="mesh", gt_type="mesh", threshold=0.01, sample=4000)
    r1 = evaluate_geometry(FIX / "pred_mesh.ply", FIX / "gt_mesh.ply", **kwargs)
    r2 = evaluate_geometry(FIX / "pred_mesh.ply", FIX / "gt_mesh.ply", **kwargs)
    assert r1.metrics == r2.metrics
    assert r1.protocol_hash == r2.protocol_hash


def test_cli_geometry_writes_run_directory(tmp_path: Path) -> None:
    out = tmp_path / "cli_run"
    result = runner.invoke(
        app,
        [
            "metric", "geometry", str(FIX / "pred.ply"),
            "--gt", str(FIX / "gt.ply"), "--threshold", "0.05",
            "--out", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "single_geometry" in result.output
    assert (out / "results.json").is_file()
    parsed = read_run_result_json(out / "results.json")
    assert parsed.metrics["fscore"] == 1.0


def test_cli_missing_file_fails_loudly(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["metric", "geometry", str(tmp_path / "nope.ply"), "--gt", str(FIX / "gt.ply")],
    )
    assert result.exit_code == 1
    # the reason is surfaced, not hidden behind a bare exit code
    assert "nope.ply" in result.output or "failed" in result.output.lower()


def test_cli_rejects_invalid_input_kind() -> None:
    result = runner.invoke(
        app,
        ["metric", "geometry", str(FIX / "pred.ply"), "--gt", str(FIX / "gt.ply"),
         "--input", "voxelgrid"],
    )
    assert result.exit_code == 2
    assert "invalid --input" in result.output
