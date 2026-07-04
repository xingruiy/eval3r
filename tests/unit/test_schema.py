"""Task 002 schema tests: round-trip serialization, fixtures, and validation errors.

Every model must round-trip (model -> JSON -> model), the sample fixtures under
``tests/fixtures/schema/`` must validate, and ``RunResult`` must carry every field
``.agent/reproducibility.md`` requires in ``results.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

import eval3r.core as core
from eval3r.core import (
    AggregationSpec,
    AlignmentSpec,
    ConfidenceManifestSpec,
    ConfidenceSpec,
    CullingSpec,
    DatasetCapabilities,
    DatasetVariant,
    EvalProtocol,
    FailurePolicySpec,
    GroundTruthSpec,
    LocalEvaluationSpec,
    MaskingSpec,
    MetricResult,
    MetricSpec,
    PredictionManifest,
    Reconstruction,
    ReportingSpec,
    RunResult,
    SamplingSideSpec,
    SamplingSpec,
    SceneData,
    SceneFailure,
    ScenePredictionEntry,
    UsesGTSpec,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "schema"


# --- example instances used for round-trip coverage ----------------------------


def _ground_truth() -> GroundTruthSpec:
    return GroundTruthSpec(
        modality="pointcloud",
        provenance="laser_scan",
        independence="independent",
        density="dense_surface",
    )


def _local_eval() -> LocalEvaluationSpec:
    return LocalEvaluationSpec(status="supported", reason="GT provided directly.")


def _dataset() -> DatasetVariant:
    return DatasetVariant(dataset="custom", variant="example", split="val")


def _protocol() -> EvalProtocol:
    return EvalProtocol(
        schema_version=1,
        protocol_version="1.0.0",
        name="example_single_geometry",
        fidelity="eval3r_native",
        dataset=_dataset(),
        prediction_modality="pointcloud",
        ground_truth=_ground_truth(),
        local_evaluation=_local_eval(),
        alignment=AlignmentSpec(mode="none"),
        confidence=ConfidenceSpec(policy="none"),
        masking=MaskingSpec(),
        sampling=SamplingSpec(
            pred=SamplingSideSpec(method="uniform_points", n_points=200_000),
            gt=SamplingSideSpec(method="all_points"),
        ),
        metrics=[
            MetricSpec(name="chamfer_distance", statistic="mean", reduction="mean"),
            MetricSpec(name="f_score", threshold=0.05),
        ],
        aggregation=AggregationSpec(),
        failure_policy=FailurePolicySpec(policy="skip_and_flag"),
        reporting=ReportingSpec(),
        backend_preferences={"nearest_neighbor": "scipy"},
    )


def _manifest() -> PredictionManifest:
    return PredictionManifest.model_validate(
        {
            "method": "example_method",
            "dataset": {"dataset": "custom"},
            "prediction_modality": "pointcloud",
            "coordinate_frame": "world",
            "scale": "metric",
            "scenes": {"scene0": {"pointcloud": "preds/scene0.ply"}},
        }
    )


def _run_result() -> RunResult:
    proto = _protocol()
    return RunResult(
        schema_version=1,
        eval3r_version="0.3.0",
        method="example_method",
        method_version="0.1.0",
        dataset=proto.dataset,
        split="val",
        protocol=proto.name,
        protocol_version=proto.protocol_version,
        protocol_hash="sha256:0000000000000000000000000000000000000000000000000000000000000000",
        fidelity=proto.fidelity,
        ground_truth=proto.ground_truth,
        local_evaluation=proto.local_evaluation,
        n_scenes_expected=2,
        n_scenes_evaluated=1,
        failed_scenes=[
            SceneFailure(scene_id="scene1", stage="load", reason="prediction file missing")
        ],
        failure_policy=proto.failure_policy,
        metrics={"chamfer_distance": 0.0123, "f_score": 0.87},
        metric_definitions=proto.metrics,
        per_scene_metrics=[
            MetricResult(
                name="chamfer_distance",
                value=0.0123,
                unit="m",
                statistic="mean",
                scene_id="scene0",
                protocol=proto.name,
                protocol_hash="sha256:00",
                backend="scipy",
                n_points_pred=200_000,
                n_points_gt=180_000,
            )
        ],
        confidence_policy=proto.confidence,
        alignment=proto.alignment,
        masking=proto.masking,
        sampling=proto.sampling,
        aggregation=proto.aggregation,
        uses_gt=UsesGTSpec(),
        backend_versions={"scipy": "1.11.0"},
        environment={"python": "3.10.12"},
        command="e3r metric geometry ...",
        timestamp="2026-07-04T00:00:00Z",
    )


ALL_INSTANCES: list[BaseModel] = [
    DatasetCapabilities(),
    _dataset(),
    _ground_truth(),
    _local_eval(),
    Reconstruction(modality="pointcloud", coordinate_frame="world", scale="metric"),
    SceneData(scene_id="scene0", dataset="custom", ground_truth=_ground_truth()),
    AlignmentSpec(),
    SamplingSideSpec(),
    SamplingSpec(pred=SamplingSideSpec(), gt=SamplingSideSpec()),
    CullingSpec(),
    MaskingSpec(),
    ConfidenceSpec(),
    MetricSpec(name="chamfer_distance"),
    AggregationSpec(),
    FailurePolicySpec(),
    ReportingSpec(),
    UsesGTSpec(),
    ConfidenceManifestSpec(),
    ScenePredictionEntry(pointcloud=Path("preds/scene0.ply")),
    _manifest(),
    _protocol(),
    MetricResult(name="f_score", value=0.9, protocol="p", protocol_hash="sha256:00"),
    SceneFailure(scene_id="scene0", stage="metric", reason="nan values"),
    _run_result(),
]


@pytest.mark.parametrize("instance", ALL_INSTANCES, ids=lambda m: type(m).__name__)
def test_model_round_trips_through_json(instance: BaseModel) -> None:
    rebuilt = type(instance).model_validate_json(instance.model_dump_json())
    assert rebuilt == instance


def test_all_exported_models_are_covered() -> None:
    covered = {type(m).__name__ for m in ALL_INSTANCES}
    exported = set(core.__all__) - {"E3RModel"}
    missing = exported - covered
    assert not missing, f"exported models without a round-trip instance: {sorted(missing)}"


# --- fixtures ------------------------------------------------------------------


def test_protocol_fixture_validates() -> None:
    data = yaml.safe_load((FIXTURES / "protocol.yaml").read_text())
    proto = EvalProtocol.model_validate(data)
    assert proto.name == "example_single_geometry"
    assert [m.name for m in proto.metrics] == ["chamfer_distance", "f_score"]
    # unset fields fell back to documented defaults
    assert proto.masking.ignore_invalid_depth is True
    assert proto.reporting.formats == ["json", "csv"]


def test_manifest_fixture_validates() -> None:
    data = yaml.safe_load((FIXTURES / "manifest.yaml").read_text())
    manifest = PredictionManifest.model_validate(data)
    assert set(manifest.scenes) == {"scene0", "scene1"}
    assert manifest.normalized_convention == "cam_to_world_opencv_meters"


def test_results_fixture_validates_and_round_trips() -> None:
    data = json.loads((FIXTURES / "results.json").read_text())
    result = RunResult.model_validate(data)
    assert result.n_scenes_expected == 2
    assert result.n_scenes_evaluated == 1
    assert result.failed_scenes[0].scene_id == "scene1"
    assert RunResult.model_validate_json(result.model_dump_json()) == result


# --- required RunResult fields (.agent/reproducibility.md) ----------------------

REQUIRED_RUN_RESULT_FIELDS = [
    "schema_version",
    "protocol",
    "protocol_hash",
    "dataset",
    "split",
    "ground_truth",
    "local_evaluation",
    "n_scenes_expected",
    "n_scenes_evaluated",
    "alignment",
    "masking",
    "sampling",
    "metric_definitions",
    "confidence_policy",
    "failure_policy",
    "backend_versions",
    "command",
    "environment",
]


def test_run_result_has_all_required_fields() -> None:
    fields = set(RunResult.model_fields)
    missing = [f for f in REQUIRED_RUN_RESULT_FIELDS if f not in fields]
    assert not missing, f"RunResult missing required reproducibility fields: {missing}"


# --- validation errors ---------------------------------------------------------


def test_bad_enum_value_rejected() -> None:
    with pytest.raises(ValidationError):
        GroundTruthSpec(
            modality="pointcloud",
            provenance="not_a_real_provenance",  # type: ignore[arg-type]
            independence="independent",
            density="dense_surface",
        )


def test_missing_required_field_rejected() -> None:
    with pytest.raises(ValidationError):
        # 'density' is required and omitted.
        GroundTruthSpec.model_validate(
            {"modality": "pointcloud", "provenance": "laser_scan", "independence": "independent"}
        )


def test_wrong_shape_rejected() -> None:
    with pytest.raises(ValidationError):
        # 'scenes' must be a mapping of ScenePredictionEntry, not a list.
        PredictionManifest.model_validate(
            {
                "method": "m",
                "dataset": {"dataset": "custom"},
                "prediction_modality": "pointcloud",
                "coordinate_frame": "world",
                "scale": "metric",
                "scenes": ["scene0"],
            }
        )


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        DatasetVariant.model_validate({"dataset": "custom", "typo_field": 1})
