"""The ``EvalProtocol`` model, transcribed from ``.agent/schema.md``.

A protocol is the source of evaluation meaning: it pins the dataset variant,
prediction modality, ground-truth spec, and the alignment / masking / sampling /
confidence / metric / aggregation / failure / reporting policies. YAML loading and
canonical hashing are task 003; this module owns only the model.
"""

from __future__ import annotations

from eval3r.core.schema import (
    AggregationSpec,
    AlignmentSpec,
    ConfidenceSpec,
    DatasetVariant,
    E3RModel,
    FailurePolicySpec,
    GroundTruthSpec,
    LocalEvaluationSpec,
    MaskingSpec,
    MetricSpec,
    ReportingSpec,
    SamplingSpec,
)
from eval3r.core.types import Fidelity, PredictionModality


class EvalProtocol(E3RModel):
    schema_version: int
    protocol_version: str
    name: str
    fidelity: Fidelity
    dataset: DatasetVariant
    prediction_modality: PredictionModality
    ground_truth: GroundTruthSpec
    local_evaluation: LocalEvaluationSpec
    alignment: AlignmentSpec
    confidence: ConfidenceSpec
    masking: MaskingSpec
    sampling: SamplingSpec
    metrics: list[MetricSpec]
    aggregation: AggregationSpec
    failure_policy: FailurePolicySpec
    reporting: ReportingSpec
    # Keys must be backend registry kinds from .agent/backends.md
    # (mesh, pointcloud, nearest_neighbor, registration, trajectory, camera,
    # depth_io, official_eval); values are registered backend names.
    backend_preferences: dict[str, str] = {}
    notes: list[str] = []
