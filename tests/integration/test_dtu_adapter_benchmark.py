"""Task 009 DTU integration: eval3r-native benchmark run with mm normalization.

The run must NOT claim official-like fidelity (that is task 010): it uses the
eval3r-native ``single_geometry`` protocol and the adapter declares
``official_local_eval = False``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from eval3r import run_benchmark
from eval3r.cli.main import app

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "dtu_tiny" / "dataset_root"
PREDS = Path(__file__).resolve().parents[1] / "fixtures" / "dtu_tiny" / "preds"
runner = CliRunner()


def test_benchmark_normalizes_mm_to_meters(tmp_path: Path) -> None:
    run = run_benchmark(
        PREDS, dataset="dtu", split="one", protocol="single_geometry",
        root=ROOT, out_dir=tmp_path / "run", return_run=True,
    )
    # scan 1 prediction is offset +50 mm in x; normalized to metres accuracy = 0.05 m.
    acc = run.result.metrics["accuracy"]
    assert acc == pytest.approx(0.05, abs=1e-6)
    # eval3r-native fidelity: the result does not claim official DTU numbers.
    assert run.result.fidelity == "eval3r_native"


def test_benchmark_full_split_coverage(tmp_path: Path) -> None:
    result = run_benchmark(
        PREDS, dataset="dtu", split="test", protocol="single_geometry",
        root=ROOT, out_dir=tmp_path / "run",
    )
    assert result.n_scenes_expected == 2
    assert result.n_scenes_evaluated == 2
    assert not result.failed_scenes


def test_cli_dtu_benchmark_run(tmp_path: Path) -> None:
    out = tmp_path / "run"
    result = runner.invoke(
        app,
        [
            "benchmark", "run", str(PREDS),
            "--dataset", "dtu", "--split", "one",
            "--protocol", "single_geometry", "--root", str(ROOT), "--out", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "results.json").is_file()


def test_cli_dtu_inspect_reports_independent_gt() -> None:
    result = runner.invoke(
        app, ["dataset", "inspect", "dtu", "--root", str(ROOT), "--split", "test"]
    )
    assert result.exit_code == 0, result.output
    assert "independent_gt" in result.output
    assert "official_local_eval" in result.output
