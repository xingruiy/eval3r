"""Task 012 unit tests: Tanks and Temples official wrapper backend.

The wrapper invokes the **real** official toolbox checkout via subprocess. Per the
project's "Official code / toolbox rule", there is no fake stand-in toolbox: the
end-to-end run test drives the genuine ``isl-org/TanksAndTemples`` toolbox and skips
cleanly (like the MATLAB path) when it is not configured in the environment. Pure paths
(output parsing, toolbox/interpreter resolution, backend_info) are tested directly and
always run.

To exercise the real end-to-end path set:

    EVAL3R_TNT_TOOLBOX  -> the toolbox python_toolbox/evaluation dir (with run.py)
    EVAL3R_TNT_PYTHON   -> interpreter with the toolbox's pinned open3d==0.9
    EVAL3R_TNT_DATA     -> dataset root containing a Barn/ scene + Barn_COLMAP.ply
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from eval3r.backends.tnt_official import TntOfficialEval, parse_official_output
from eval3r.core.errors import MetricError
from eval3r.core.registry import BackendError, default_registry

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "tanks_temples_tiny"
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


def _real_toolbox_env() -> tuple[str, str, Path] | None:
    """Return (toolbox_dir, python_exe, barn_scene_dir) if the real toolbox is configured.

    All three of EVAL3R_TNT_TOOLBOX / EVAL3R_TNT_PYTHON / EVAL3R_TNT_DATA must be set and
    the real Barn artifacts + a Barn_COLMAP.ply prediction must exist; otherwise None so
    the caller skips (no fake toolbox is ever substituted).
    """
    toolbox = os.environ.get("EVAL3R_TNT_TOOLBOX")
    python_exe = os.environ.get("EVAL3R_TNT_PYTHON")
    data = os.environ.get("EVAL3R_TNT_DATA")
    if not (toolbox and python_exe and data):
        return None
    if not (Path(toolbox) / "run.py").is_file():
        return None
    scene = Path(data) / "Barn"
    needed = [scene / "Barn.ply", scene / "Barn.json", scene / "Barn_trans.txt",
              scene / "Barn_COLMAP_SfM.log", scene / "Barn_COLMAP.ply"]
    if not all(p.is_file() for p in needed):
        return None
    return toolbox, python_exe, scene


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


def test_resolve_python_prefers_explicit_then_env_then_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EVAL3R_TNT_PYTHON", raising=False)
    # default: current interpreter
    assert TntOfficialEval()._resolve_python() == sys.executable
    # env var overrides default
    monkeypatch.setenv("EVAL3R_TNT_PYTHON", "/opt/tnt/bin/python")
    assert TntOfficialEval()._resolve_python() == "/opt/tnt/bin/python"
    # explicit ctor arg wins over env var
    assert TntOfficialEval(python_executable="/explicit/python")._resolve_python() == (
        "/explicit/python"
    )


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


def test_backend_info_unresolved_toolbox_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL3R_TNT_TOOLBOX", raising=False)
    monkeypatch.delenv("TANKSANDTEMPLES_TOOLBOX", raising=False)
    info = TntOfficialEval().backend_info()
    assert info.name == "tnt_official"
    assert info.kind == "official_eval"
    assert info.version == "external_checkout"


# --- real official toolbox (skips cleanly when not configured; never faked) ----------


def test_real_toolbox_runs_and_parses_barn(tmp_path: Path) -> None:
    env = _real_toolbox_env()
    if env is None:
        pytest.skip(
            "real Tanks and Temples toolbox not configured; set EVAL3R_TNT_TOOLBOX, "
            "EVAL3R_TNT_PYTHON (pinned open3d==0.9) and EVAL3R_TNT_DATA to run it."
        )
    toolbox, python_exe, scene = env
    backend = TntOfficialEval(toolbox_dir=toolbox, python_executable=python_exe)
    result = backend.evaluate_scene(
        "Barn",
        dataset_dir=scene,
        traj_path=scene / "Barn_COLMAP_SfM.log",
        ply_path=scene / "Barn_COLMAP.ply",
        out_dir=tmp_path / "out",
        timeout=3000,
    )
    # Barn's official per-scene threshold is 0.01, read from the official output.
    assert result.distance_tau == pytest.approx(0.01)
    for score in (result.precision, result.recall, result.fscore):
        assert 0.0 <= score <= 1.0
    # provenance is recorded: exact command, toolbox dir/commit, and the interpreter used.
    assert result.toolbox_dir == str(toolbox)
    assert result.python_executable == python_exe
    assert "--ply-path" in result.command


def test_real_toolbox_reports_failure(tmp_path: Path) -> None:
    # Point at an unknown scene directory: the real run.py raises immediately
    # ("invalid dataset-dir, not in scenes_tau_dict") before any heavy loading, so this is
    # a fast, deterministic exercise of the nonzero-exit -> MetricError surfacing.
    env = _real_toolbox_env()
    if env is None:
        pytest.skip("real Tanks and Temples toolbox not configured; see EVAL3R_TNT_* env vars.")
    toolbox, python_exe, scene = env
    bad_dir = tmp_path / "NotAScene"
    bad_dir.mkdir()
    backend = TntOfficialEval(toolbox_dir=toolbox, python_executable=python_exe)
    with pytest.raises(MetricError, match="failed for scene"):
        backend.evaluate_scene(
            "NotAScene",
            dataset_dir=bad_dir,
            traj_path=scene / "Barn_COLMAP_SfM.log",
            ply_path=scene / "Barn_COLMAP.ply",
            out_dir=tmp_path / "out",
            timeout=300,
        )
