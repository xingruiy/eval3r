"""Task 013 integration: ETH3D official-wrapper benchmark + server-only refusal.

The training run drives the **real** ``ETH3D/multi-view-evaluation`` binary (no fake
stand-in, per the "Official code / toolbox rule") on the analytic ``eth3d_tiny``
fixture: artifact resolution, subprocess invocation, output parsing, and per-scene
tolerance/metadata recording are exercised end-to-end, and the scores are checked
against the fixture's hand-derived expectations. Skips cleanly when the binary is
not configured (set EVAL3R_ETH3D_TOOL). The server-only refusal test needs no tool
and always runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from eval3r import run_benchmark
from eval3r.core.errors import BenchmarkError
from eval3r.reports.json import read_run_result_json

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "eth3d_tiny"
ROOT = FIX / "dataset_root"
PREDS = FIX / "preds"

# Analytic fixture expectation (reproduced exactly by the official binary): see the
# fixture construction notes in tests/unit/test_eth3d_official.py.
EXPECTED = {"1cm": 0.5, "2cm": 0.5, "5cm": 1.0, "10cm": 1.0, "20cm": 1.0, "50cm": 1.0}


def _real_tool() -> str | None:
    tool = os.environ.get("EVAL3R_ETH3D_TOOL") or os.environ.get("ETH3D_MULTI_VIEW_EVALUATION")
    return tool if tool and Path(tool).is_file() else None


def test_training_run_wraps_real_official_tool(tmp_path: Path) -> None:
    tool = _real_tool()
    if tool is None:
        pytest.skip(
            "real ETH3D multi-view-evaluation binary not configured; set EVAL3R_ETH3D_TOOL "
            "to a built ETH3DMultiViewEvaluation to run it."
        )
    run = run_benchmark(
        PREDS, dataset="eth3d", split="training",
        protocol="eth3d_training_official",
        root=ROOT, out_dir=tmp_path / "run", return_run=True,
    )
    r = run.result
    assert r.n_scenes_evaluated == 1 and not r.failed_scenes
    assert r.fidelity == "official"

    # All 18 official metrics (accuracy/completeness/F1 at 6 tolerances) match the
    # analytic fixture values.
    for suffix, expected in EXPECTED.items():
        for kind in ("accuracy", "completeness", "fscore"):
            assert r.metrics[f"{kind}_{suffix}"] == pytest.approx(expected), f"{kind}_{suffix}"

    # Per-scene provenance: official tool command, tolerances, and free-space/voxel
    # parameters are recorded (CLAUDE.md official-code + reproducibility rules).
    f2 = next(m for m in r.per_scene_metrics if m.name == "fscore_2cm")
    assert f2.metadata["evaluator_method"] == "official_script_wrapper"
    assert f2.metadata["tool_path"] == tool
    assert f2.metadata["threshold"] == pytest.approx(0.02)
    assert f2.metadata["voxel_size"] == pytest.approx(0.01)
    assert f2.metadata["tolerances"] == pytest.approx([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])

    written = read_run_result_json(tmp_path / "run" / "results.json")
    assert written.backend_versions["official_eval"]["name"] == "eth3d_official"
    assert written.ground_truth.provenance == "laser_scan"


def test_test_split_refused_server_only(tmp_path: Path) -> None:
    # No official protocol exists for the test split; the adapter refuses the split in
    # preflight regardless of the protocol used.
    with pytest.raises(BenchmarkError, match="server_only|server"):
        run_benchmark(
            PREDS, dataset="eth3d", split="test",
            protocol="eth3d_training_official",
            root=ROOT, out_dir=tmp_path / "run", return_run=True,
        )
