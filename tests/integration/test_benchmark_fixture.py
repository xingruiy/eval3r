"""Task 008 integration tests: end-to-end benchmark on a fixture adapter + CLI.

Registers a tiny 'fixture' adapter (the generic custom adapter pointed at committed
point-cloud fixtures) so the documented verification command shape works in-process.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from eval3r import run_benchmark
from eval3r.cli.main import app
from eval3r.datasets import CustomAdapter, default_registry
from eval3r.reports.json import read_run_result_json

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "benchmark"
DATA = ROOT / "dataset_root"
PREDS = ROOT / "preds"
runner = CliRunner()


@pytest.fixture(autouse=True)
def _register_fixture_adapter() -> None:
    # A test-only adapter name that bakes in the committed dataset root, so
    # `--dataset fixture --split ...` needs no --root.
    default_registry().register("fixture", lambda root: CustomAdapter(DATA))


def test_run_benchmark_api_writes_full_run_directory(tmp_path: Path) -> None:
    out = tmp_path / "run"
    result = run_benchmark(
        PREDS, dataset="fixture", split="pair", protocol="single_geometry", out_dir=out
    )
    assert result.n_scenes_expected == 2
    assert result.n_scenes_evaluated == 2
    assert result.metrics["fscore"] == 1.0

    for name in ("results.json", "per_scene.csv", "manifest.yaml", "backend_versions.json"):
        assert (out / name).is_file(), f"missing {name}"

    # manifest was inferred and recorded as such.
    parsed = read_run_result_json(out / "results.json")
    assert parsed.metadata["manifest_inferred"] is True
    data = json.loads((out / "results.json").read_text())
    assert data["n_scenes_evaluated"] == 2


def test_cli_benchmark_run_pair(tmp_path: Path) -> None:
    out = tmp_path / "cli_run"
    result = runner.invoke(
        app,
        [
            "benchmark", "run", str(PREDS),
            "--dataset", "fixture", "--split", "pair",
            "--protocol", "single_geometry", "--out", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "single_geometry" in result.output
    assert "scene_a" in result.output  # per-scene progress is visible
    assert (out / "results.json").is_file()


def test_cli_benchmark_run_abort_on_missing_prediction(tmp_path: Path) -> None:
    # 'tiny' includes scene_missing (GT but no prediction); default policy is abort.
    result = runner.invoke(
        app,
        [
            "benchmark", "run", str(PREDS),
            "--dataset", "fixture", "--split", "tiny",
            "--protocol", "single_geometry", "--out", str(tmp_path / "r"),
        ],
    )
    assert result.exit_code == 1
    assert "scene_missing" in result.output


def test_cli_benchmark_validate_reports_missing(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "benchmark", "validate", str(PREDS),
            "--dataset", "fixture", "--split", "tiny", "--protocol", "single_geometry",
        ],
    )
    assert result.exit_code == 1
    assert "scene_missing" in result.output
    assert "2/3 scenes resolve" in result.output


def test_cli_dataset_inspect_shows_capabilities() -> None:
    result = runner.invoke(app, ["dataset", "inspect", "fixture", "--split", "pair"])
    assert result.exit_code == 0, result.output
    assert "dense_geometry" in result.output
    assert "scene_a" in result.output


def test_cli_benchmark_run_unknown_dataset_fails() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark", "run", str(PREDS),
            "--dataset", "no_such_dataset", "--split", "pair", "--protocol", "single_geometry",
        ],
    )
    assert result.exit_code == 1
    assert "no_such_dataset" in result.output
