"""Task 013 unit tests: ETH3D official multi-view-evaluation wrapper backend.

The wrapper invokes the **real** official ``ETH3D/multi-view-evaluation`` binary via
subprocess. Per the project's "Official code / toolbox rule", there is no fake
stand-in evaluator: the end-to-end tests drive the genuine binary on the analytic
``eth3d_tiny`` fixture and skip cleanly (like the MATLAB and Tanks-and-Temples
paths) when it is not configured. Pure paths (output parsing, tool resolution,
backend_info) are tested directly and always run.

To exercise the real end-to-end path set:

    EVAL3R_ETH3D_TOOL -> a built ETH3DMultiViewEvaluation binary
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from eval3r.backends.eth3d_official import Eth3dOfficialEval, parse_official_output
from eval3r.core.errors import MetricError
from eval3r.core.registry import BackendError, default_registry

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "eth3d_tiny"
SCAN_MLP = FIX / "dataset_root" / "courtyard" / "dslr_scan_eval" / "scan_alignment.mlp"
PRED = FIX / "preds" / "courtyard.ply"

TOLERANCES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

# The analytic expectation for the eth3d_tiny fixture, reproduced exactly by the
# real official binary (see the fixture docstring in the adapter tests): two exact
# prediction points, two 3 cm in front of their scan points along the scan rays,
# one unobserved point 1 m behind a scan point.
EXPECTED = {0.01: 0.5, 0.02: 0.5, 0.05: 1.0, 0.1: 1.0, 0.2: 1.0, 0.5: 1.0}

OFFICIAL_SUMMARY = """\
Loading reconstruction: /tmp/pred.ply
Loading scan: /tmp/scan1.ply
Computing completeness
Computing accuracy
Tolerances: 0.01 0.02 0.05
Completenesses: 0.5 0.5 1
Accuracies: 0.25 0.5 1
F1-scores: 0.333333 0.5 1
"""


def _real_tool() -> str | None:
    """The real official binary path when configured, else None (skip; never faked)."""
    tool = os.environ.get("EVAL3R_ETH3D_TOOL") or os.environ.get("ETH3D_MULTI_VIEW_EVALUATION")
    if tool and Path(tool).is_file():
        return tool
    return None


def test_registered_in_backend_registry() -> None:
    assert "eth3d_official" in default_registry().available("official_eval")


def test_parse_official_output() -> None:
    parsed = parse_official_output(OFFICIAL_SUMMARY, [0.01, 0.02, 0.05])
    assert parsed["tolerances"] == [0.01, 0.02, 0.05]
    assert parsed["completenesses"] == [0.5, 0.5, 1.0]
    assert parsed["accuracies"] == [0.25, 0.5, 1.0]
    assert parsed["f1_scores"] == pytest.approx([0.333333, 0.5, 1.0])


def test_parse_official_output_missing_lines_errors() -> None:
    with pytest.raises(MetricError, match="missing line\\(s\\): accuracies, f1_scores"):
        parse_official_output("Tolerances: 0.01\nCompletenesses: 0.5\n", [0.01])


def test_parse_official_output_misaligned_errors() -> None:
    bad = OFFICIAL_SUMMARY.replace("Accuracies: 0.25 0.5 1", "Accuracies: 0.25 0.5")
    with pytest.raises(MetricError, match="misaligned: 3 tolerances but 2 accuracies"):
        parse_official_output(bad, [0.01])


def test_parse_official_output_missing_requested_tolerance_errors() -> None:
    with pytest.raises(MetricError, match="requested tolerance 0.2"):
        parse_official_output(OFFICIAL_SUMMARY, [0.01, 0.2])


def test_absent_tool_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL3R_ETH3D_TOOL", raising=False)
    monkeypatch.delenv("ETH3D_MULTI_VIEW_EVALUATION", raising=False)
    backend = Eth3dOfficialEval()
    with pytest.raises(BackendError, match="official evaluation tool is not configured"):
        backend.evaluate_scene(
            "courtyard", scan_mlp_path=SCAN_MLP, ply_path=PRED, tolerances=TOLERANCES
        )


def test_nonexistent_tool_path_fails(tmp_path: Path) -> None:
    backend = Eth3dOfficialEval(tool_path=tmp_path / "missing_binary")
    with pytest.raises(BackendError, match="does not exist"):
        backend.evaluate_scene(
            "courtyard", scan_mlp_path=SCAN_MLP, ply_path=PRED, tolerances=TOLERANCES
        )


def test_empty_tolerances_refused() -> None:
    with pytest.raises(MetricError, match="no evaluation tolerances"):
        Eth3dOfficialEval().evaluate_scene(
            "courtyard", scan_mlp_path=SCAN_MLP, ply_path=PRED, tolerances=[]
        )


def test_backend_info_unresolved_tool_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL3R_ETH3D_TOOL", raising=False)
    monkeypatch.delenv("ETH3D_MULTI_VIEW_EVALUATION", raising=False)
    info = Eth3dOfficialEval().backend_info()
    assert info.name == "eth3d_official"
    assert info.kind == "official_eval"
    assert info.version == "external_build"


# --- real official tool (skips cleanly when not configured; never faked) -------------


def test_real_tool_reproduces_analytic_fixture_scores() -> None:
    tool = _real_tool()
    if tool is None:
        pytest.skip(
            "real ETH3D multi-view-evaluation binary not configured; set EVAL3R_ETH3D_TOOL "
            "to a built ETH3DMultiViewEvaluation to run it."
        )
    backend = Eth3dOfficialEval(tool_path=tool)
    result = backend.evaluate_scene(
        "courtyard", scan_mlp_path=SCAN_MLP, ply_path=PRED, tolerances=TOLERANCES
    )
    for tol, expected in EXPECTED.items():
        assert backend.lookup(result.accuracies, tol) == pytest.approx(expected)
        assert backend.lookup(result.completenesses, tol) == pytest.approx(expected)
        assert backend.lookup(result.f1_scores, tol) == pytest.approx(expected)
    # Provenance is recorded: exact command and tool path (+ commit when in a checkout).
    assert result.tool_path == str(tool)
    assert "--ground_truth_mlp_path" in result.command
    assert result.voxel_size == pytest.approx(0.01)


def test_real_tool_failure_surfaces_stderr(tmp_path: Path) -> None:
    tool = _real_tool()
    if tool is None:
        pytest.skip(
            "real ETH3D multi-view-evaluation binary not configured; see EVAL3R_ETH3D_TOOL."
        )
    missing_pred = tmp_path / "nope.ply"
    backend = Eth3dOfficialEval(tool_path=tool)
    with pytest.raises(MetricError, match="failed for scene 'courtyard'"):
        backend.evaluate_scene(
            "courtyard", scan_mlp_path=SCAN_MLP, ply_path=missing_pred, tolerances=TOLERANCES
        )
