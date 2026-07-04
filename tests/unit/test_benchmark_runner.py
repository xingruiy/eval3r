"""Task 008 benchmark-orchestration unit tests: preflight, manifest, coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval3r.core.errors import BenchmarkError, SceneEvaluationError
from eval3r.core.schema import DatasetCapabilities, FailurePolicySpec, LocalEvaluationSpec
from eval3r.datasets import CustomAdapter
from eval3r.pipeline.benchmark import (
    load_or_infer_manifest,
    preflight,
    run_benchmark_geometry,
)
from eval3r.protocols import load_protocol

DATA = Path(__file__).resolve().parents[1] / "fixtures" / "benchmark" / "dataset_root"
PREDS = Path(__file__).resolve().parents[1] / "fixtures" / "benchmark" / "preds"


def _proto(policy: str = "abort", worst: dict | None = None):
    proto = load_protocol("single_geometry").model_copy(deep=True)
    proto.failure_policy = FailurePolicySpec(policy=policy, worst_values=worst or {})
    return proto


# --- preflight -----------------------------------------------------------------


class _ServerOnlyAdapter:
    name = "server_only"
    capabilities = DatasetCapabilities(server_only_eval=True)

    def local_evaluation(self, split, protocol):  # noqa: ANN001, ANN201
        return LocalEvaluationSpec(status="server_only")


class _UnsupportedAdapter:
    name = "unsup"
    capabilities = DatasetCapabilities()

    def local_evaluation(self, split, protocol):  # noqa: ANN001, ANN201
        return LocalEvaluationSpec(status="requires_external_renderer", reason="needs a renderer")


def test_preflight_refuses_server_only() -> None:
    with pytest.raises(BenchmarkError) as exc:
        preflight(_ServerOnlyAdapter(), "test", _proto())
    assert "server-only" in str(exc.value)


def test_preflight_refuses_unsupported_local_status() -> None:
    with pytest.raises(BenchmarkError) as exc:
        preflight(_UnsupportedAdapter(), "training", _proto())
    assert "requires_external_renderer" in str(exc.value)
    assert "needs a renderer" in str(exc.value)


def test_preflight_allows_supported_custom() -> None:
    preflight(CustomAdapter(DATA), "pair", _proto())  # no raise


# --- manifest ------------------------------------------------------------------


def test_manifest_inference_marks_inferred() -> None:
    proto = _proto()
    manifest, data, inferred = load_or_infer_manifest(PREDS, ["scene_a", "scene_b"], proto)
    assert inferred is True
    assert manifest is not None
    assert set(manifest.scenes) == {"scene_a", "scene_b"}
    assert data["metadata"]["inferred"] is True


# --- coverage / failure policy -------------------------------------------------


def test_all_scenes_pass_aggregate_mean() -> None:
    run = run_benchmark_geometry(PREDS, CustomAdapter(DATA), _proto(), "pair")
    assert run.result.n_scenes_expected == 2
    assert run.result.n_scenes_evaluated == 2
    assert not run.result.failed_scenes
    assert run.result.metrics["fscore"] == 1.0
    # two per-scene results per metric (scene_a + scene_b)
    fscores = [m for m in run.result.per_scene_metrics if m.name == "fscore"]
    assert {m.scene_id for m in fscores} == {"scene_a", "scene_b"}


def test_abort_policy_stops_on_unresolvable_scene() -> None:
    with pytest.raises(SceneEvaluationError) as exc:
        run_benchmark_geometry(PREDS, CustomAdapter(DATA), _proto("abort"), "tiny")
    assert exc.value.scene_id == "scene_missing"
    assert exc.value.stage == "resolve"


def test_skip_and_flag_reports_partial_coverage() -> None:
    run = run_benchmark_geometry(PREDS, CustomAdapter(DATA), _proto("skip_and_flag"), "tiny")
    assert run.result.n_scenes_expected == 3
    assert run.result.n_scenes_evaluated == 2
    assert [f.scene_id for f in run.result.failed_scenes] == ["scene_missing"]
    assert run.result.failed_scenes[0].stage == "resolve"
    # aggregate is over the 2 resolvable scenes; coverage is visible via the counts.
    assert run.result.metrics["fscore"] == 1.0


def test_score_worst_injects_worst_values() -> None:
    proto = _proto("score_worst", {"accuracy": 5.0, "fscore": 0.0})
    run = run_benchmark_geometry(PREDS, CustomAdapter(DATA), proto, "tiny")
    assert run.result.n_scenes_evaluated == 2
    assert [f.scene_id for f in run.result.failed_scenes] == ["scene_missing"]
    # the worst-scored scene contributes to the aggregate: mean accuracy over
    # scene_a(0), scene_b(0), scene_missing(5) = 5/3.
    assert run.result.metrics["accuracy"] == pytest.approx(5.0 / 3.0)
    worst = [m for m in run.result.per_scene_metrics if m.scene_id == "scene_missing"]
    assert all(m.metadata.get("scored_worst") for m in worst)
