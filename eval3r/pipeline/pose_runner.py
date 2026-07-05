"""Staged single-trajectory pose runner.

Mirrors the task-007/014 runners (stage tracking, failure-policy handling,
RunResult assembly) with trajectory-shaped stages: resolve (both TUM files
exist) → load + associate + align + metric (delegated to the ``trajectory``
backend, evo) → aggregate. One prediction trajectory is evaluated against one
ground-truth trajectory; alignment is estimated once per pair (``per_scene``).

The estimated alignment (mode, rotation, translation, scale, |ln s|) is recorded
in the run's alignment records and every metric's metadata — ATE under Sim3 is
not comparable to ATE under SE3 or to unaligned ATE.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from eval3r.backends.trajectory_evo import RPE_POSE_RELATIONS
from eval3r.core.errors import (
    AlignmentError,
    Eval3rError,
    InvalidTrajectoryError,
    SceneEvaluationError,
)
from eval3r.core.hashing import protocol_hash as compute_protocol_hash
from eval3r.core.protocol import EvalProtocol
from eval3r.core.registry import BackendRegistry, default_registry
from eval3r.core.result import MetricResult, RunResult, SceneFailure
from eval3r.metrics.pose import (
    alignment_scale_error,
    check_pose_alignment_spec,
    pose_metric_result,
    require_rpe_parameters,
)

PoseStage = Literal["resolve", "load", "align", "metric", "aggregate"]

# CLI/API shorthands accepted for --align, normalized to first-class modes.
POSE_ALIGN_ALIASES = {
    "none": "none",
    "se3": "trajectory_se3",
    "sim3": "trajectory_sim3",
    "trajectory_se3": "trajectory_se3",
    "trajectory_sim3": "trajectory_sim3",
}


@dataclass
class PoseSceneOutcome:
    """Metrics + the estimated alignment on success, a structured failure otherwise."""

    scene_id: str
    scene_metrics: list[MetricResult] = field(default_factory=list)
    alignment_record: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    failure: SceneFailure | None = None


@dataclass
class PoseRunOutput:
    """Everything the writer/CLI needs after a single-trajectory pose run."""

    result: RunResult
    protocol: EvalProtocol
    protocol_hash: str
    alignment_records: list[dict[str, Any]]
    overrides: dict[str, Any]
    config: dict[str, Any]


# --- overrides ---------------------------------------------------------------------


def apply_pose_overrides(
    protocol: EvalProtocol,
    *,
    align: str | None,
    associate_max_diff: float | None,
    backend: str | None,
) -> tuple[EvalProtocol, dict[str, Any]]:
    """Apply CLI/API pose flags to a protocol copy and record what changed.

    Alignment mode and association tolerance change evaluation behavior, so the
    caller recomputes the canonical hash. Overrides are refused when the
    protocol's alignment spec disallows them (e.g. a metric-scale dataset
    protocol pinning SE3: Sim3 must never be enabled silently there).
    """
    proto = protocol.model_copy(deep=True)
    overrides: dict[str, Any] = {}

    if align is not None:
        if align not in POSE_ALIGN_ALIASES:
            raise AlignmentError(
                f"--align '{align}' is not a trajectory alignment mode; choose one "
                f"of: {', '.join(POSE_ALIGN_ALIASES)} (se3/sim3 are shorthands for "
                f"trajectory_se3/trajectory_sim3)."
            )
        mode = POSE_ALIGN_ALIASES[align]
        if mode != proto.alignment.mode and not proto.alignment.allow_override:
            raise AlignmentError(
                f"protocol '{proto.name}' pins alignment mode "
                f"'{proto.alignment.mode}' and disallows overrides "
                f"(allow_override: false); refusing --align {align}."
            )
        proto.alignment.mode = mode  # type: ignore[assignment]
        proto.alignment.estimate_on = "trajectory" if mode != "none" else "none"
        proto.alignment.solver = "evo" if mode != "none" else "none"
        overrides["align"] = mode
    if associate_max_diff is not None:
        if associate_max_diff <= 0:
            raise InvalidTrajectoryError(
                f"--associate-max-diff must be a positive number of seconds; "
                f"got {associate_max_diff}."
            )
        proto.alignment.parameters["associate_max_diff"] = float(associate_max_diff)
        overrides["associate_max_diff"] = float(associate_max_diff)
    if backend is not None:
        proto.backend_preferences["trajectory"] = backend
        overrides["backend"] = backend
    return proto, overrides


# --- per-scene evaluation ------------------------------------------------------------


def evaluate_pose_scene(
    scene_id: str,
    pred_path: Path,
    gt_path: Path,
    *,
    protocol: EvalProtocol,
    protocol_hash: str,
    registry: BackendRegistry,
) -> PoseSceneOutcome:
    """Run load/associate → align → metric for one trajectory pair."""
    backend = registry.require(
        "trajectory", protocol.backend_preferences.get("trajectory", "evo")
    )
    alignment = protocol.alignment
    association = dict(alignment.parameters)

    stage: PoseStage = "load"
    try:
        check_pose_alignment_spec(alignment)

        # ATE first: its backend result also carries the association counts and
        # the estimated alignment reused for the alignment_scale_error diagnostic.
        stage = "align"
        ate_result = backend.evaluate_ate(
            pred_path, gt_path, alignment.mode, association
        )

        stage = "metric"
        results: list[MetricResult] = []
        for spec in protocol.metrics:
            if spec.name in RPE_POSE_RELATIONS:
                delta, delta_unit, all_pairs = require_rpe_parameters(spec)
                backend_result = backend.evaluate_rpe(
                    pred_path, gt_path, alignment.mode, association,
                    pose_relation=RPE_POSE_RELATIONS[spec.name],
                    delta=delta, delta_unit=delta_unit, all_pairs=all_pairs,
                )
            else:
                backend_result = ate_result
            results.append(
                pose_metric_result(
                    spec, backend_result,
                    protocol=protocol.name, protocol_hash=protocol_hash,
                    backend_name=backend.name, scene_id=scene_id,
                )
            )

        stage = "aggregate"
        align_rec = dict(ate_result["alignment"])
        align_rec["scale_error"] = alignment_scale_error(float(align_rec["scale"]))
        align_rec["scene_id"] = scene_id
        return PoseSceneOutcome(
            scene_id=scene_id,
            scene_metrics=results,
            alignment_record=align_rec,
            metadata={
                "n_pred_poses": ate_result["n_pred_poses"],
                "n_gt_poses": ate_result["n_gt_poses"],
                "n_associated": ate_result["n_associated"],
                "n_dropped_pred": ate_result["n_dropped_pred"],
                "n_dropped_gt": ate_result["n_dropped_gt"],
                "association": ate_result["association"],
            },
        )
    except Eval3rError as exc:
        failure = SceneFailure(
            scene_id=scene_id,
            stage=stage,
            reason=str(exc),
            traceback=traceback.format_exc(),
            recoverable=stage != "load",
        )
        return PoseSceneOutcome(scene_id=scene_id, failure=failure)


# --- run assembly ---------------------------------------------------------------------


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _aggregate_run_metrics(
    outcome: PoseSceneOutcome, protocol: EvalProtocol
) -> dict[str, float | None]:
    if outcome.failure is None:
        return {m.name: m.value for m in outcome.scene_metrics}
    if protocol.failure_policy.policy == "score_worst":
        worst = protocol.failure_policy.worst_values
        return {m.name: worst.get(m.name) for m in protocol.metrics}
    return {m.name: None for m in protocol.metrics}


def run_single_file_pose(
    pred_path: str | Path,
    gt_path: str | Path,
    protocol: EvalProtocol,
    *,
    align: str | None = None,
    associate_max_diff: float | None = None,
    backend: str | None = None,
    method: str | None = None,
    scene_id: str | None = None,
    registry: BackendRegistry | None = None,
    command: str | None = None,
    environment: dict[str, Any] | None = None,
) -> PoseRunOutput:
    """Evaluate one predicted trajectory against ground truth under ``protocol``."""
    registry = registry or default_registry()
    pred_path = Path(pred_path)
    gt_path = Path(gt_path)
    scene_id = scene_id or pred_path.stem or "scene"

    proto, overrides = apply_pose_overrides(
        protocol, align=align, associate_max_diff=associate_max_diff, backend=backend
    )
    phash = compute_protocol_hash(proto)

    stage_resolve_failure: SceneFailure | None = None
    missing = [
        (role, path)
        for role, path in (("prediction", pred_path), ("ground-truth", gt_path))
        if not path.is_file()
    ]
    if missing:
        reason = "; ".join(
            f"{role} trajectory file does not exist or is not a file: {path}"
            for role, path in missing
        ) + ". Expected TUM-format text files (timestamp x y z qx qy qz qw)."
        if proto.failure_policy.policy == "abort":
            raise SceneEvaluationError(scene_id, "resolve", reason)
        stage_resolve_failure = SceneFailure(
            scene_id=scene_id, stage="resolve", reason=reason, recoverable=False
        )

    if stage_resolve_failure is not None:
        outcome = PoseSceneOutcome(scene_id=scene_id, failure=stage_resolve_failure)
    else:
        outcome = evaluate_pose_scene(
            scene_id, pred_path, gt_path,
            protocol=proto, protocol_hash=phash, registry=registry,
        )

    if outcome.failure is not None and proto.failure_policy.policy == "abort":
        raise SceneEvaluationError(scene_id, outcome.failure.stage, outcome.failure.reason)

    backend_versions = registry.backend_versions(
        {"trajectory": proto.backend_preferences.get("trajectory", "evo")}
    )
    metrics = _aggregate_run_metrics(outcome, proto)
    alignment_records = (
        [outcome.alignment_record] if outcome.alignment_record is not None else []
    )

    result = RunResult(
        schema_version=proto.schema_version,
        eval3r_version=_eval3r_version(),
        method=method,
        method_version=None,
        dataset=proto.dataset,
        split=proto.dataset.split,
        protocol=proto.name,
        protocol_version=proto.protocol_version,
        protocol_hash=phash,
        fidelity=proto.fidelity,
        ground_truth=proto.ground_truth,
        local_evaluation=proto.local_evaluation,
        n_scenes_expected=1,
        n_scenes_evaluated=1 if outcome.failure is None else 0,
        failed_scenes=[outcome.failure] if outcome.failure else [],
        failure_policy=proto.failure_policy,
        metrics=metrics,
        metric_definitions=proto.metrics,
        per_scene_metrics=outcome.scene_metrics,
        confidence_policy=proto.confidence,
        alignment=proto.alignment,
        masking=proto.masking,
        sampling=proto.sampling,
        aggregation=proto.aggregation,
        backend_versions=backend_versions,
        environment=environment if environment is not None else {},
        command=command,
        timestamp=_now_utc_iso(),
        metadata=outcome.metadata,
    )

    config = {
        "protocol": proto.name,
        "protocol_hash": phash,
        "inputs": {
            "pred": str(pred_path),
            "gt": str(gt_path),
            "format": "tum",
        },
        "overrides": overrides,
    }

    return PoseRunOutput(
        result=result,
        protocol=proto,
        protocol_hash=phash,
        alignment_records=alignment_records,
        overrides=overrides,
        config=config,
    )


def _eval3r_version() -> str:
    from eval3r import __version__

    return __version__
