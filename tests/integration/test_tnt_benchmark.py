"""Task 012 integration: Tanks and Temples official-wrapper benchmark + server-only refusal.

The training run drives the **real** ``isl-org/TanksAndTemples`` toolbox (no fake stand-in,
per the "Official code / toolbox rule"): the full benchmark path — artifact resolution,
subprocess invocation under the toolbox's pinned open3d==0.9 interpreter, output parsing,
and per-scene threshold recording — is exercised end-to-end and skips cleanly when the
toolbox is not configured. The server-only refusal test needs no toolbox and always runs.

Set EVAL3R_TNT_TOOLBOX / EVAL3R_TNT_PYTHON / EVAL3R_TNT_DATA to run the real path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from eval3r import run_benchmark
from eval3r.core.errors import BenchmarkError
from eval3r.reports.json import read_run_result_json

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "tanks_temples_tiny"
ROOT = FIX / "dataset_root"
PREDS = FIX / "preds"


def _real_toolbox_data() -> Path | None:
    """Barn scene dir if the real toolbox + data are configured, else None (skip)."""
    toolbox = os.environ.get("EVAL3R_TNT_TOOLBOX")
    python_exe = os.environ.get("EVAL3R_TNT_PYTHON")
    data = os.environ.get("EVAL3R_TNT_DATA")
    if not (toolbox and python_exe and data and (Path(toolbox) / "run.py").is_file()):
        return None
    scene = Path(data) / "Barn"
    needed = [scene / "Barn.ply", scene / "Barn.json", scene / "Barn_trans.txt",
              scene / "Barn_COLMAP_SfM.log", scene / "Barn_COLMAP.ply"]
    return scene if all(p.is_file() for p in needed) else None


def test_training_run_wraps_real_official_toolbox(tmp_path: Path) -> None:
    scene = _real_toolbox_data()
    if scene is None:
        pytest.skip(
            "real Tanks and Temples toolbox not configured; set EVAL3R_TNT_TOOLBOX, "
            "EVAL3R_TNT_PYTHON (pinned open3d==0.9) and EVAL3R_TNT_DATA to run it."
        )
    # Restrict the run to a single scene (Barn) via a root that only exposes Barn, so the
    # test does not evaluate all seven training scenes.
    root = tmp_path / "root"
    root.mkdir()
    (root / "Barn").symlink_to(scene, target_is_directory=True)
    preds = tmp_path / "preds"
    preds.mkdir()
    (preds / "Barn.ply").symlink_to(scene / "Barn_COLMAP.ply")

    run = run_benchmark(
        preds, dataset="tanks_temples", split="training",
        protocol="tanks_temples_training_official",
        root=root, out_dir=tmp_path / "run", return_run=True,
    )
    r = run.result
    assert r.n_scenes_evaluated == 1 and not r.failed_scenes
    assert r.fidelity == "official"
    for name in ("precision", "recall", "fscore"):
        assert 0.0 <= r.metrics[name] <= 1.0

    # per-scene distance threshold + provenance come from the official output/run.
    prec = next(m for m in r.per_scene_metrics if m.name == "precision")
    assert prec.metadata["distance_tau"] == pytest.approx(0.01)
    assert prec.metadata["evaluator_method"] == "official_script_wrapper"
    assert prec.metadata["python_executable"] == os.environ["EVAL3R_TNT_PYTHON"]

    written = read_run_result_json(tmp_path / "run" / "results.json")
    assert written.backend_versions["official_eval"]["name"] == "tnt_official"


def test_intermediate_split_refused_server_only(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="server_only|server"):
        run_benchmark(
            PREDS, dataset="tanks_temples", split="intermediate",
            protocol="tanks_temples_intermediate_server",
            root=ROOT, out_dir=tmp_path / "run", return_run=True,
        )
