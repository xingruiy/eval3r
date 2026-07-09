"""Task 010 integration: DTU native benchmark through e3r benchmark run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from eval3r import run_benchmark
from eval3r.cli.main import app
from eval3r.core.errors import SceneEvaluationError
from eval3r.core.schema import FailurePolicySpec
from eval3r.datasets import DTUAdapter
from eval3r.pipeline.benchmark import run_benchmark_geometry
from eval3r.protocols import load_protocol
from eval3r.reports.json import read_run_result_json

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "dtu_tiny" / "dataset_root"
PREDS = Path(__file__).resolve().parents[1] / "fixtures" / "dtu_tiny" / "preds"
runner = CliRunner()


def test_native_run_scores_in_millimetres(tmp_path: Path) -> None:
    run = run_benchmark(
        PREDS, dataset="dtu", split="one", protocol="dtu_native_pointcloud",
        root=ROOT, out_dir=tmp_path / "run", return_run=True,
    )
    # scan 1 prediction offset +5 mm; the official path stays in mm (no metre conversion).
    assert run.result.metrics["accuracy"] == pytest.approx(5.0)
    assert run.result.metrics["completeness"] == pytest.approx(5.0)
    assert run.result.metrics["overall"] == pytest.approx(5.0)
    assert run.result.fidelity == "native"

    acc = next(m for m in run.result.per_scene_metrics if m.name == "accuracy")
    assert acc.unit == "mm"
    assert acc.metadata["evaluator_method"] == "validated_official_port"
    assert acc.metadata["plane_available"] is True

    # evaluator backend is recorded in results.json
    parsed = read_run_result_json(tmp_path / "run" / "results.json")
    assert parsed.backend_versions["official_eval"]["name"] == "dtu"
    data = json.loads((tmp_path / "run" / "results.json").read_text())
    assert data["metrics"]["overall"] == pytest.approx(5.0)


def test_missing_plane_scene_fails_not_silently_scored() -> None:
    # 'test' split includes scan 4, which has no Plane file; default policy is abort.
    with pytest.raises(SceneEvaluationError) as exc:
        run_benchmark(
            PREDS, dataset="dtu", split="test",
            protocol="dtu_native_pointcloud", root=ROOT,
        )
    assert exc.value.scene_id == "4"


def test_missing_plane_partial_coverage_under_skip(tmp_path: Path) -> None:
    proto = load_protocol("dtu_native_pointcloud").model_copy(deep=True)
    proto.failure_policy = FailurePolicySpec(policy="skip_and_flag")
    run = run_benchmark_geometry(PREDS, DTUAdapter(ROOT), proto, "test")
    assert run.result.n_scenes_evaluated == 1
    assert run.result.n_scenes_expected == 2
    assert [f.scene_id for f in run.result.failed_scenes] == ["4"]
    assert "plane" in run.result.failed_scenes[0].reason


def test_cli_native_run(tmp_path: Path) -> None:
    out = tmp_path / "run"
    result = runner.invoke(
        app,
        [
            "benchmark", "run", str(PREDS),
            "--dataset", "dtu", "--split", "one",
            "--protocol", "dtu_native_pointcloud", "--root", str(ROOT), "--out", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "results.json").is_file()
