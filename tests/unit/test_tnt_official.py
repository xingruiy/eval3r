"""Task 012 unit tests: Tanks and Temples official wrapper backend.

The wrapper invokes an external toolbox checkout via subprocess. Here it is driven
against a fake toolbox fixture that emits the official summary format, so the real
command construction + subprocess + output parsing are exercised without the heavy
open3d-based official code. The absent-toolbox path is tested to fail explicitly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval3r.backends.tnt_official import TntOfficialEval, parse_official_output
from eval3r.core.errors import MetricError
from eval3r.core.registry import BackendError, default_registry

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "tanks_temples_tiny"
FAKE_TOOLBOX = FIX / "fake_toolbox"
SCENE_DIR = FIX / "dataset_root" / "Barn"
PRED = FIX / "preds" / "Barn.ply"

OFFICIAL_SUMMARY = """
===========================
evaluation result : Ignatius
===========================
distance tau : 0.003
precision : 0.9033
recall : 0.9243
f-score : 0.9137
===========================
"""


def test_registered_in_backend_registry() -> None:
    assert "tnt_official" in default_registry().available("official_eval")


def test_parse_official_output() -> None:
    values = parse_official_output(OFFICIAL_SUMMARY)
    assert values["precision"] == pytest.approx(0.9033)
    assert values["recall"] == pytest.approx(0.9243)
    assert values["fscore"] == pytest.approx(0.9137)
    assert values["distance_tau"] == pytest.approx(0.003)


def test_parse_official_output_missing_fields_errors() -> None:
    with pytest.raises(MetricError, match="missing recall, fscore, distance_tau"):
        parse_official_output("precision : 0.5\n")


def test_absent_toolbox_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL3R_TNT_TOOLBOX", raising=False)
    monkeypatch.delenv("TANKSANDTEMPLES_TOOLBOX", raising=False)
    backend = TntOfficialEval()
    with pytest.raises(BackendError, match="official toolbox is not configured"):
        backend.evaluate_scene(
            "Barn",
            dataset_dir=SCENE_DIR,
            traj_path=SCENE_DIR / "Barn_COLMAP_SfM.log",
            ply_path=PRED,
            out_dir=SCENE_DIR / "_out",
        )


def test_toolbox_without_run_py_fails(tmp_path: Path) -> None:
    backend = TntOfficialEval(toolbox_dir=tmp_path)
    with pytest.raises(BackendError, match="has no run.py"):
        backend.evaluate_scene(
            "Barn", dataset_dir=SCENE_DIR, traj_path=SCENE_DIR / "Barn_COLMAP_SfM.log",
            ply_path=PRED, out_dir=tmp_path / "_out",
        )


def test_wrapper_runs_fake_toolbox_and_parses(tmp_path: Path) -> None:
    backend = TntOfficialEval(toolbox_dir=FAKE_TOOLBOX)
    result = backend.evaluate_scene(
        "Barn",
        dataset_dir=SCENE_DIR,
        traj_path=SCENE_DIR / "Barn_COLMAP_SfM.log",
        ply_path=PRED,
        out_dir=tmp_path / "out",
    )
    # Barn's official per-scene threshold is 0.01 (read from the toolbox output).
    assert result.distance_tau == pytest.approx(0.01)
    assert result.precision == pytest.approx(0.85)
    assert result.recall == pytest.approx(0.75)
    assert result.fscore == pytest.approx(0.7969)
    # provenance is recorded: the exact command and the toolbox dir.
    assert result.toolbox_dir == str(FAKE_TOOLBOX)
    assert "--ply-path" in result.command
    assert str(PRED) in result.command
    # the toolbox wrote its output into out_dir.
    assert (tmp_path / "out" / "Barn.evaluation.txt").is_file()


def test_wrapper_reports_toolbox_failure(tmp_path: Path) -> None:
    # Point at the real scene dir but a wrong scene name so the fake toolbox exits nonzero.
    backend = TntOfficialEval(toolbox_dir=FAKE_TOOLBOX)
    bad_dir = tmp_path / "Nope"
    bad_dir.mkdir()
    with pytest.raises(MetricError, match="failed for scene"):
        backend.evaluate_scene(
            "Nope", dataset_dir=bad_dir, traj_path=SCENE_DIR / "Barn_COLMAP_SfM.log",
            ply_path=PRED, out_dir=tmp_path / "out",
        )


def test_backend_info_unresolved_toolbox_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL3R_TNT_TOOLBOX", raising=False)
    monkeypatch.delenv("TANKSANDTEMPLES_TOOLBOX", raising=False)
    info = TntOfficialEval().backend_info()
    assert info.name == "tnt_official"
    assert info.kind == "official_eval"
    assert info.version == "external_checkout"
