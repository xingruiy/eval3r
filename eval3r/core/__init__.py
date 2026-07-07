"""Core schema, protocol, manifest, result, and registry primitives.

Re-exports the Pydantic schema models so callers can use a single import surface
(``from eval3r.core import EvalProtocol, PredictionManifest, RunResult``) instead
of reaching into individual modules.
"""

from __future__ import annotations

from eval3r.core.adaptation import AdaptationOverride, AdaptationRecord
from eval3r.core.manifest import (
    ConfidenceManifestSpec,
    PredictionManifest,
    ScenePredictionEntry,
    UsesGTSpec,
)
from eval3r.core.protocol import EvalProtocol
from eval3r.core.result import MetricResult, RunResult, SceneFailure
from eval3r.core.schema import (
    AggregationSpec,
    AlignmentSpec,
    ConfidenceSpec,
    CullingSpec,
    DatasetCapabilities,
    DatasetVariant,
    E3RModel,
    FailurePolicySpec,
    GroundTruthSpec,
    LocalEvaluationSpec,
    MaskingSpec,
    MetricSpec,
    Reconstruction,
    ReportingSpec,
    SamplingSideSpec,
    SamplingSpec,
    SceneData,
)

__all__ = [
    # base
    "E3RModel",
    # dataset
    "DatasetCapabilities",
    "DatasetVariant",
    "SceneData",
    "Reconstruction",
    # ground truth / local eval
    "GroundTruthSpec",
    "LocalEvaluationSpec",
    # policy specs
    "AlignmentSpec",
    "SamplingSideSpec",
    "SamplingSpec",
    "CullingSpec",
    "MaskingSpec",
    "ConfidenceSpec",
    "MetricSpec",
    "AggregationSpec",
    "FailurePolicySpec",
    "ReportingSpec",
    "AdaptationOverride",
    "AdaptationRecord",
    # manifest
    "UsesGTSpec",
    "ConfidenceManifestSpec",
    "ScenePredictionEntry",
    "PredictionManifest",
    # protocol
    "EvalProtocol",
    # results
    "MetricResult",
    "SceneFailure",
    "RunResult",
]
