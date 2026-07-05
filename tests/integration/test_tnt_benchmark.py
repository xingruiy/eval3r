"""Task 012 integration: Tanks and Temples official-wrapper benchmark + server-only refusal.

The training run wraps a fake official toolbox fixture (via EVAL3R_TNT_TOOLBOX) so the
full benchmark path — artifact resolution, subprocess invocation, output parsing, and
per-scene threshold recording — is exercised without the heavy real toolbox.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval3r import run_benchmark
from eval3r.core.errors import BenchmarkError
from eval3r.reports.json import read_run_result_json

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "tanks_temples_tiny"
ROOT = FIX / "dataset_root"
PREDS = FIX / "preds"
FAKE_TOOLBOX = FIX / "fake_toolbox"


def test_training_run_wraps_official_toolbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVAL3R_TNT_TOOLBOX", str(FAKE_TOOLBOX))
    run = run_benchmark(
        PREDS, dataset="tanks_temples", split="training",
        protocol="tanks_temples_training_official",
        root=ROOT, out_dir=tmp_path / "run", return_run=True,
    )
    r = run.result
    assert r.n_scenes_evaluated == 1 and not r.failed_scenes
    assert r.fidelity == "official"
    assert r.metrics["precision"] == pytest.approx(0.85)
    assert r.metrics["recall"] == pytest.approx(0.75)
    assert r.metrics["fscore"] == pytest.approx(0.7969)

    # per-scene distance threshold comes from the official output, not a global constant.
    prec = next(m for m in r.per_scene_metrics if m.name == "precision")
    assert prec.metadata["distance_tau"] == pytest.approx(0.01)
    assert prec.metadata["evaluator_method"] == "official_script_wrapper"
    assert prec.metadata["toolbox_dir"] == str(FAKE_TOOLBOX)

    # backend version recorded in results.json
    written = read_run_result_json(tmp_path / "run" / "results.json")
    assert written.backend_versions["official_eval"]["name"] == "tnt_official"


def test_intermediate_split_refused_server_only(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="server_only|server"):
        run_benchmark(
            PREDS, dataset="tanks_temples", split="intermediate",
            protocol="tanks_temples_intermediate_server_only",
            root=ROOT, out_dir=tmp_path / "run", return_run=True,
        )
