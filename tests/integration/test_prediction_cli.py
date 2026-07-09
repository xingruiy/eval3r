"""Task 019 integration: `e3r prediction` CLI + writer output through `run_benchmark`.

Drives the real CLI in-process on directories produced by ``PredictionWriter``,
and proves the official layout is consumed unchanged by the benchmark loop (the
written ``manifest.yaml`` is honored as a *declared* manifest, not re-inferred).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from eval3r import run_benchmark
from eval3r.cli.main import app
from eval3r.datasets import CustomAdapter, default_registry
from eval3r.predictions import PredictionWriter

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "benchmark"
runner = CliRunner()


@pytest.fixture(autouse=True)
def _register_fixture_adapter() -> None:
    default_registry().register("fixture", lambda root: CustomAdapter(FIXTURES / "dataset_root"))


@pytest.fixture
def written_dir(tmp_path: Path) -> Path:
    root = tmp_path / "preds"
    with PredictionWriter(
        root,
        method="writer_e2e",
        dataset="fixture",
        split="pair",
        prediction_modality="pointcloud",
        scale="metric",
        coordinate_frame="world",
    ) as writer:
        for scene_id in ("scene_a", "scene_b"):
            writer.add_scene(scene_id, pointcloud=FIXTURES / "preds" / f"{scene_id}.ply")
    return root


def test_cli_validate_ok(written_dir: Path) -> None:
    result = runner.invoke(app, ["prediction", "validate", str(written_dir)])
    assert result.exit_code == 0, result.output
    assert "writer_e2e" in result.output
    assert "scene_a" in result.output and "scene_b" in result.output
    assert "2/2 scenes valid" in result.output
    assert "fingerprints verified" in result.output


def test_cli_validate_reports_tampered_and_missing_files(written_dir: Path) -> None:
    (written_dir / "scene_a" / "pointcloud.ply").write_bytes(b"tampered")
    (written_dir / "scene_b" / "pointcloud.ply").unlink()

    result = runner.invoke(app, ["prediction", "validate", str(written_dir)])
    assert result.exit_code == 1
    assert "0/2 scenes valid" in result.output
    assert "fingerprint mismatch" in result.output
    assert "missing file" in result.output

    # --no-verify skips hashing: the tampered file passes, the missing one still fails.
    result = runner.invoke(app, ["prediction", "validate", str(written_dir), "--no-verify"])
    assert result.exit_code == 1
    assert "1/2 scenes valid" in result.output
    assert "fingerprint" not in result.output.split("scenes valid")[1]


def test_cli_validate_fails_without_manifest(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(app, ["prediction", "validate", str(empty)], terminal_width=80)
    assert result.exit_code == 1
    assert "no manifest.yaml" in result.output


def test_cli_show_lists_files_and_fingerprints(written_dir: Path) -> None:
    result = runner.invoke(app, ["prediction", "show", str(written_dir)])
    assert result.exit_code == 0, result.output
    assert "writer_e2e" in result.output
    assert "eval3r-native-v1" in result.output
    assert "pointcloud.ply" in result.output
    assert "recorded" in result.output


def test_written_layout_runs_through_benchmark_unchanged(
    written_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "run"
    result = run_benchmark(
        written_dir, dataset="fixture", split="pair", protocol="single_geometry", out_dir=out
    )
    assert result.n_scenes_evaluated == 2
    assert result.metrics["fscore"] == 1.0

    # The written manifest was honored as declared, not re-inferred.
    parsed = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert parsed["metadata"]["manifest_inferred"] is False
    run_manifest = yaml.safe_load((out / "manifest.yaml").read_text(encoding="utf-8"))
    assert run_manifest["method"] == "writer_e2e"
    assert run_manifest["metadata"]["layout"] == "eval3r-native-v1"
    assert run_manifest["scenes"]["scene_a"]["pointcloud"] == "scene_a/pointcloud.ply"
