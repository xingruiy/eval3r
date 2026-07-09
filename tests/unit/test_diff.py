"""Task 016 tests: run diffing — strict hash refusal, loose labeling, warnings.

Comparability warning triggers follow ``.agent/reproducibility.md`` "Report
comparability": GT provenance, local evaluation status, scene coverage, failure
policy, alignment mode, confidence policy, sampling counts, backend officialness.
"""

from __future__ import annotations

import pytest

from eval3r.core.errors import RunComparisonError
from eval3r.core.result import SceneFailure
from eval3r.core.schema import (
    AlignmentSpec,
    ConfidenceSpec,
    FailurePolicySpec,
    GroundTruthSpec,
    LocalEvaluationSpec,
    SamplingSideSpec,
    SamplingSpec,
)
from eval3r.reports import write_run_directory
from eval3r.reports.diff import diff_runs, load_run_result


def _write_run(result, path):
    return write_run_directory(result, path)


@pytest.fixture
def two_runs(make_run_result, tmp_path):
    """Two comparable full-coverage runs with slightly different scores."""

    def build(overrides_a: dict | None = None, overrides_b: dict | None = None):
        full = dict(n_scenes_expected=2, n_scenes_evaluated=2, failed_scenes=[])
        b_defaults = dict(method="method_b", metrics={"accuracy": 0.05, "fscore": 0.55})
        a = make_run_result(**{**full, **(overrides_a or {})})
        b = make_run_result(**{**full, **b_defaults, **(overrides_b or {})})
        dir_a = _write_run(a, tmp_path / "run_a")
        dir_b = _write_run(b, tmp_path / "run_b")
        return dir_a, dir_b

    return build


# --- strict / loose behavior -------------------------------------------------------


def test_strict_diff_refuses_mismatched_protocol_hashes(two_runs) -> None:
    dir_a, dir_b = two_runs(overrides_b={"protocol_hash": "sha256:bbb"})
    with pytest.raises(RunComparisonError) as exc:
        diff_runs(dir_a, dir_b)
    msg = str(exc.value)
    assert "sha256:aaa" in msg and "sha256:bbb" in msg
    assert "--loose" in msg  # tells the user how to request a non-strict comparison


def test_loose_diff_is_labeled_non_strict_and_warns_on_hash(two_runs) -> None:
    dir_a, dir_b = two_runs(overrides_b={"protocol_hash": "sha256:bbb"})
    diff = diff_runs(dir_a, dir_b, loose=True)
    assert diff.strict is False
    warned_fields = {w.field for w in diff.warnings}
    assert "protocol_hash" in warned_fields


def test_matching_runs_diff_cleanly(two_runs) -> None:
    dir_a, dir_b = two_runs()
    diff = diff_runs(dir_a, dir_b)
    assert diff.strict is True
    assert diff.warnings == []
    deltas = {m.name: m for m in diff.metrics}
    assert deltas["accuracy"].value_a == pytest.approx(0.03)
    assert deltas["accuracy"].value_b == pytest.approx(0.05)
    assert deltas["accuracy"].delta == pytest.approx(0.02)
    assert deltas["fscore"].delta == pytest.approx(-0.06)


def test_per_scene_deltas_for_common_scenes(two_runs) -> None:
    dir_a, dir_b = two_runs()
    diff = diff_runs(dir_a, dir_b)
    assert "scan001" in diff.per_scene
    scene = {m.name: m for m in diff.per_scene["scan001"]}
    # both fixture runs carry the same per-scene values for scan001
    assert scene["accuracy"].delta == pytest.approx(0.0)
    assert diff.scenes_only_in_a == []
    assert diff.scenes_only_in_b == []


def test_none_metric_values_give_none_delta(two_runs) -> None:
    dir_a, dir_b = two_runs(overrides_b={"metrics": {"accuracy": None, "fscore": 0.55}})
    diff = diff_runs(dir_a, dir_b)
    deltas = {m.name: m for m in diff.metrics}
    assert deltas["accuracy"].value_b is None
    assert deltas["accuracy"].delta is None


# --- comparability warning triggers -------------------------------------------------


@pytest.mark.parametrize(
    ("field", "overrides_b"),
    [
        (
            "ground_truth_provenance",
            {
                "ground_truth": GroundTruthSpec(
                    modality="pointcloud",
                    provenance="reconstructed",
                    independence="reconstruction_derived",
                    density="dense_surface",
                )
            },
        ),
        (
            "local_evaluation_status",
            {"local_evaluation": LocalEvaluationSpec(status="server_only")},
        ),
        (
            "scene_coverage",
            {
                "n_scenes_evaluated": 1,
                "failed_scenes": [
                    SceneFailure(scene_id="scan002", stage="load", reason="missing")
                ],
            },
        ),
        ("failure_policy", {"failure_policy": FailurePolicySpec(policy="score_worst")}),
        (
            "alignment_mode",
            {
                "alignment": AlignmentSpec(
                    mode="sim3", estimate_on="pointcloud", solver="umeyama"
                )
            },
        ),
        (
            "confidence_policy",
            {"confidence_policy": ConfidenceSpec(policy="threshold", threshold=0.5)},
        ),
        (
            "sampling",
            {
                "sampling": SamplingSpec(
                    pred=SamplingSideSpec(method="random_points", n_points=2000),
                    gt=SamplingSideSpec(),
                )
            },
        ),
        ("backend_officialness", {"fidelity": "official"}),
        (
            "backend_officialness",
            {
                "backend_versions": {
                    "nearest_neighbor": {
                        "name": "scipy", "library": "scipy", "version": "1.15.3"
                    },
                    "official_eval": {
                        "name": "dtu_eval", "library": "dtu_eval", "version": "1.0"
                    },
                }
            },
        ),
    ],
)
def test_comparability_warning_triggers(two_runs, field, overrides_b) -> None:
    dir_a, dir_b = two_runs(overrides_b=overrides_b)
    diff = diff_runs(dir_a, dir_b)
    warned = {w.field for w in diff.warnings}
    assert field in warned, f"expected warning '{field}', got {warned}"
    warning = next(w for w in diff.warnings if w.field == field)
    assert warning.value_a != warning.value_b
    assert warning.reason  # every warning explains why it matters


# --- loading errors ------------------------------------------------------------------


def test_missing_run_directory_fails_with_reason(tmp_path) -> None:
    with pytest.raises(RunComparisonError, match="results.json"):
        load_run_result(tmp_path / "nonexistent_run")


def test_diff_accepts_direct_results_json_path(two_runs) -> None:
    dir_a, dir_b = two_runs()
    diff = diff_runs(dir_a / "results.json", dir_b / "results.json")
    assert diff.strict is True
