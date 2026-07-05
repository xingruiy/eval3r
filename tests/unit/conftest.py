"""Shared fixtures for unit tests (task 016: report/diff RunResult factory)."""

from __future__ import annotations

from typing import Any

import pytest

from eval3r.core.result import MetricResult, RunResult, SceneFailure
from eval3r.core.schema import (
    AggregationSpec,
    AlignmentSpec,
    ConfidenceSpec,
    DatasetVariant,
    FailurePolicySpec,
    GroundTruthSpec,
    LocalEvaluationSpec,
    MaskingSpec,
    MetricSpec,
    SamplingSideSpec,
    SamplingSpec,
)


def _metric(name: str, value: float | None, scene_id: str, **kwargs: Any) -> MetricResult:
    return MetricResult(
        name=name,
        value=value,
        scene_id=scene_id,
        protocol="single_geometry",
        protocol_hash="sha256:aaa",
        **kwargs,
    )


def _base_run_result(**overrides: Any) -> RunResult:
    """A complete two-scene RunResult (one ok, one failed) for report/diff tests.

    Keyword overrides replace top-level ``RunResult`` fields, so tests can flip
    exactly the field a comparability trigger or banner check needs.
    """
    fields: dict[str, Any] = dict(
        schema_version=1,
        eval3r_version="0.3.0",
        method="method_a",
        method_version=None,
        dataset=DatasetVariant(dataset="dtu", variant="mvs_2014", split="test"),
        split="test",
        protocol="single_geometry",
        protocol_version="0.1.0",
        protocol_hash="sha256:aaa",
        fidelity="eval3r_native",
        ground_truth=GroundTruthSpec(
            modality="pointcloud",
            provenance="laser_scan",
            independence="independent",
            density="dense_surface",
        ),
        local_evaluation=LocalEvaluationSpec(status="supported"),
        n_scenes_expected=2,
        n_scenes_evaluated=1,
        failed_scenes=[
            SceneFailure(
                scene_id="scan002",
                stage="load",
                reason="prediction file missing: preds/scan002.ply",
            )
        ],
        failure_policy=FailurePolicySpec(policy="skip_and_flag"),
        metrics={"accuracy": 0.03, "fscore": 0.61},
        metric_definitions=[
            MetricSpec(name="accuracy", statistic="mean"),
            MetricSpec(name="fscore", threshold=0.05),
        ],
        per_scene_metrics=[
            _metric("accuracy", 0.03, "scan001", unit="m", statistic="mean"),
            _metric("fscore", 0.61, "scan001", threshold=0.05),
        ],
        confidence_policy=ConfidenceSpec(),
        alignment=AlignmentSpec(mode="none"),
        masking=MaskingSpec(),
        sampling=SamplingSpec(
            pred=SamplingSideSpec(method="random_points", n_points=1000),
            gt=SamplingSideSpec(),
        ),
        aggregation=AggregationSpec(),
        backend_versions={
            "nearest_neighbor": {"name": "scipy", "library": "scipy", "version": "1.15.3"},
        },
        environment={"python_version": "3.10", "os": "Linux"},
        command="e3r metric geometry pred.ply --gt gt.ply",
        timestamp="2026-07-06T00:00:00Z",
    )
    fields.update(overrides)
    return RunResult(**fields)


@pytest.fixture
def make_run_result():
    """Factory: ``make_run_result(**field_overrides) -> RunResult``.

    Defaults to partial coverage (1 of 2 scenes evaluated, one recorded failure).
    Pass ``n_scenes_evaluated=2, n_scenes_expected=2, failed_scenes=[]`` for a
    full-coverage run.
    """
    return _base_run_result
