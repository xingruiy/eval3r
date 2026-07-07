"""Result models, transcribed from ``.agent/schema.md``.

``RunResult`` carries every field ``.agent/reproducibility.md`` requires in
``results.json`` — protocol identity and hash, dataset/variant/split, ground-truth
provenance, local-evaluation status, the alignment / masking / sampling /
confidence / metric / aggregation / failure policies actually used, scene
coverage, backend versions, and command/environment — so a result file is
self-describing without the original protocol file. Writing results to disk is
task 006; this module owns only the in-memory models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from eval3r.core.adaptation import AdaptationRecord
from eval3r.core.manifest import UsesGTSpec
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
    SamplingSpec,
)
from eval3r.core.types import Fidelity


class MetricResult(E3RModel):
    name: str
    value: float | None
    unit: str | None = None
    threshold: float | None = None
    statistic: str | None = None
    reduction: str | None = None
    scene_id: str | None = None
    frame_id: str | None = None
    protocol: str
    protocol_hash: str
    backend: str | None = None
    n_points_pred: int | None = None
    n_points_gt: int | None = None
    n_pixels_valid: int | None = None
    valid_fraction: float | None = None
    culled_fraction: float | None = None
    metadata: dict = {}


class SceneFailure(E3RModel):
    scene_id: str
    stage: Literal[
        "resolve",
        "convert",
        "load",
        "normalize",
        "align",
        "mask",
        "sample",
        "metric",
        "aggregate",
        "report",
    ]
    reason: str
    traceback: str | None = None
    recoverable: bool = True


class RunResult(E3RModel):
    schema_version: int
    eval3r_version: str
    method: str | None = None
    method_version: str | None = None
    dataset: DatasetVariant
    split: str | None = None
    protocol: str
    protocol_version: str
    protocol_hash: str
    fidelity: Fidelity
    ground_truth: GroundTruthSpec
    local_evaluation: LocalEvaluationSpec
    n_scenes_expected: int
    n_scenes_evaluated: int
    failed_scenes: list[SceneFailure] = []
    failure_policy: FailurePolicySpec
    metrics: dict[str, float | None]
    metric_definitions: list[MetricSpec] = []
    per_scene_metrics: list[MetricResult] = []
    confidence_policy: ConfidenceSpec
    alignment: AlignmentSpec
    adaptation: AdaptationRecord | None = None
    masking: MaskingSpec
    sampling: SamplingSpec
    aggregation: AggregationSpec
    uses_gt: UsesGTSpec | None = None
    # kind -> backend metadata; a metadata dict ({name, library, version, approximate})
    # per .agent/backends.md, or a plain version string for simple backends.
    backend_versions: dict[str, Any] = {}
    environment: dict = {}
    command: str | None = None
    manifest_path: Path | None = None
    protocol_path: Path | None = None
    git_commit: str | None = None
    timestamp: str
    metadata: dict = {}
