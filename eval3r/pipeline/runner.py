"""Staged single-file geometry runner.

Sequences the pipeline stages (load → align → mask/cull → sample → metric) for one
prediction/ground-truth file pair and assembles a complete :class:`RunResult`. The
stage sequence and every choice inside it come from the protocol; the runner adds
only orchestration, failure-policy handling, and metadata capture, so the same
skeleton can later serve depth/pose (tasks 014/015) without drifting.

Failure policies (``.agent/plan.md`` failure accounting):

- ``abort``         — re-raise a :class:`SceneEvaluationError` naming the scene/stage.
- ``skip_and_flag`` — record the failure, emit no metric values.
- ``score_worst``   — record the failure and fill metrics from the protocol's
  ``failure_policy.worst_values`` (missing entries stay ``None``, visibly).
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from eval3r.core.errors import Eval3rError, SceneEvaluationError
from eval3r.core.hashing import protocol_hash as compute_protocol_hash
from eval3r.core.protocol import EvalProtocol
from eval3r.core.registry import BackendRegistry, default_registry
from eval3r.core.result import MetricResult, RunResult, SceneFailure
from eval3r.metrics.diagnostics import build_diagnostic_metrics, partition_specs
from eval3r.metrics.geometry import DirectionalDistances, compute_directional_distances
from eval3r.pipeline.stages.align import (
    AlignmentResult,
    AlignmentVisData,
    align_geometry,
    capture_alignment_vis,
)
from eval3r.pipeline.stages.load import GeometryKind, load_geometry
from eval3r.pipeline.stages.mask import apply_culling
from eval3r.pipeline.stages.metric import compute_scene_metrics
from eval3r.pipeline.stages.normalize import normalize_to_meters
from eval3r.pipeline.stages.sample import DEFAULT_BASE_SEED, sample_geometry

# Stages the single-file geometry path runs, in order. A subset of SceneFailure.stage.
GeometryStage = Literal["load", "normalize", "align", "mask", "sample", "metric"]
_THRESHOLD_METRICS = frozenset({"precision", "recall", "fscore", "coverage"})


@dataclass
class SceneOutcome:
    """Result of evaluating one scene: metrics on success, a failure otherwise.

    ``debug`` carries the per-point directional distances (with the cleaned point
    arrays) when the protocol's reporting spec requests debug outputs; ``None``
    otherwise, and always ``None`` for official-toolbox evaluation paths, which own
    their distance computation internally. ``alignment_vis`` carries the subsampled
    before/after/gt capture whenever a non-``none`` alignment actually ran, so the
    mandatory visualization artifacts can be written with the run directory.
    """

    scene_id: str
    metrics: list[MetricResult] = field(default_factory=list)
    failure: SceneFailure | None = None
    alignment: AlignmentResult | None = None
    debug: DirectionalDistances | None = None
    alignment_vis: AlignmentVisData | None = None


@dataclass
class GeometryRunOutput:
    """Everything the writer/CLI needs after a single-file geometry run."""

    result: RunResult
    protocol: EvalProtocol
    protocol_hash: str
    alignment_transforms: list[dict[str, Any]]
    overrides: dict[str, Any]
    config: dict[str, Any]
    debug_scenes: list[tuple[str, DirectionalDistances]] = field(default_factory=list)
    alignment_vis: list[AlignmentVisData] = field(default_factory=list)


# --- overrides -----------------------------------------------------------------


def apply_geometry_overrides(
    protocol: EvalProtocol,
    *,
    input_type: GeometryKind,
    gt_type: GeometryKind,
    threshold: float | None,
    sample: int | None,
) -> tuple[EvalProtocol, dict[str, Any]]:
    """Map CLI/API flags onto a copy of the protocol and record what changed.

    Overriding a threshold or sample count changes evaluation behavior, so the
    returned protocol's canonical hash changes accordingly (recomputed by the caller).
    Input types are recorded and drive loading/sampling; the mesh path forces surface
    sampling with a default count of 200_000 when none is given.
    """
    proto = protocol.model_copy(deep=True)
    overrides: dict[str, Any] = {"input_type": input_type, "gt_type": gt_type}

    proto.prediction_modality = input_type

    if threshold is not None:
        for metric in proto.metrics:
            if metric.name in _THRESHOLD_METRICS:
                metric.threshold = threshold
        overrides["threshold"] = threshold

    _override_sampling(proto.sampling.pred, kind=input_type, sample=sample)
    _override_sampling(proto.sampling.gt, kind=gt_type, sample=sample)
    if sample is not None:
        overrides["sample"] = sample

    return proto, overrides


def _override_sampling(side: Any, *, kind: GeometryKind, sample: int | None) -> None:
    if kind == "mesh":
        side.method = "surface_area"
        side.n_points = sample if sample is not None else (side.n_points or 200_000)
    elif sample is not None:
        side.method = "random_points"
        side.n_points = sample


# --- per-scene evaluation ------------------------------------------------------


def evaluate_geometry_scene(
    scene_id: str,
    pred_path: Path,
    gt_path: Path,
    *,
    protocol: EvalProtocol,
    protocol_hash: str,
    input_type: GeometryKind,
    gt_type: GeometryKind,
    registry: BackendRegistry,
    base_seed: int = DEFAULT_BASE_SEED,
    pred_unit: str | None = None,
    gt_unit: str | None = None,
    pred_trajectory: Path | None = None,
    gt_trajectory: Path | None = None,
) -> SceneOutcome:
    """Run the stages for one scene, returning metrics or a structured failure.

    ``pred_unit`` / ``gt_unit`` are the source length units of the loaded files; each
    is normalized to metres in the ``normalize`` stage before alignment. ``None`` (the
    single-file default) means the geometry is already metric.
    ``pred_trajectory`` / ``gt_trajectory`` feed trajectory-first alignment
    (``estimate_on: trajectory``); a missing one is an explicit failure at stage
    ``align`` when that alignment is requested.
    """
    prefs = protocol.backend_preferences
    mesh_backend = registry.require("mesh", prefs.get("mesh", "trimesh"))
    pc_backend = registry.require("pointcloud", prefs.get("pointcloud", "plyfile"))
    nn_backend = registry.require("nearest_neighbor", prefs.get("nearest_neighbor", "scipy"))
    align_spec = protocol.alignment
    registration_backend = (
        registry.require("registration", prefs.get("registration", "open3d"))
        if align_spec.mode != "none" and align_spec.solver == "icp"
        else None
    )
    trajectory_backend = (
        registry.require("trajectory", prefs.get("trajectory", "evo"))
        if align_spec.mode != "none" and align_spec.estimate_on == "trajectory"
        else None
    )
    metric_scale = (protocol.ground_truth.unit or "m") == "m"

    stage: GeometryStage = "load"
    try:
        pred = load_geometry(
            pred_path, input_type, mesh_backend=mesh_backend, pointcloud_backend=pc_backend
        )
        gt = load_geometry(
            gt_path, gt_type, mesh_backend=mesh_backend, pointcloud_backend=pc_backend
        )

        stage = "normalize"
        pred = normalize_to_meters(pred, pred_unit)
        gt = normalize_to_meters(gt, gt_unit)

        stage = "align"
        pred_before_align = pred
        pred, alignment = align_geometry(
            pred, gt, protocol.alignment, scene_id=scene_id, metric_scale=metric_scale,
            registration_backend=registration_backend,
            trajectory_backend=trajectory_backend,
            pred_trajectory=pred_trajectory, gt_trajectory=gt_trajectory,
        )
        # Mandatory visualization: capture whenever an alignment actually ran, so
        # the run directory always gets the before/after overlays.
        alignment_vis = (
            capture_alignment_vis(pred_before_align, gt, alignment)
            if protocol.alignment.mode != "none"
            else None
        )

        stage = "mask"
        pred = apply_culling(pred, protocol.masking.pred_culling, role="pred")
        gt = apply_culling(gt, protocol.masking.gt_culling, role="gt")

        stage = "sample"
        pred_points, _ = sample_geometry(
            pred, protocol.sampling.pred, scene_id=scene_id, role="pred",
            mesh_backend=mesh_backend, base_seed=base_seed,
        )
        gt_points, _ = sample_geometry(
            gt, protocol.sampling.gt, scene_id=scene_id, role="gt",
            mesh_backend=mesh_backend, base_seed=base_seed,
        )

        stage = "metric"
        geometry_specs, diagnostic_specs = partition_specs(protocol.metrics)
        # Debug outputs (error-colored cloud / histogram) need per-point distances;
        # compute them once here and hand the identical values to the metric layer.
        reporting = protocol.reporting
        capture_debug = reporting.save_colored_errors or reporting.save_distance_histogram
        distances = (
            compute_directional_distances(pred_points, gt_points, nn_backend)
            if capture_debug
            else None
        )
        metrics = compute_scene_metrics(
            pred_points, gt_points, geometry_specs,
            protocol=protocol.name, protocol_hash=protocol_hash,
            nn_backend=nn_backend, scene_id=scene_id, distances=distances,
        )
        if diagnostic_specs:
            # The single-file / no-cull path removes nothing (mask method 'none'); the
            # only surviving loss is NaN/Inf removal, captured by valid_fraction.
            valid_fraction = metrics[0].valid_fraction if metrics else 1.0
            metrics = metrics + build_diagnostic_metrics(
                diagnostic_specs,
                {"culled_fraction": 0.0, "valid_fraction": valid_fraction},
                scene_id=scene_id, protocol=protocol.name, protocol_hash=protocol_hash,
            )
        return SceneOutcome(
            scene_id=scene_id, metrics=metrics, alignment=alignment, debug=distances,
            alignment_vis=alignment_vis,
        )
    except Eval3rError as exc:
        failure = SceneFailure(
            scene_id=scene_id,
            stage=stage,
            reason=str(exc),
            traceback=traceback.format_exc(),
            recoverable=stage != "load",
        )
        return SceneOutcome(scene_id=scene_id, failure=failure)


# --- run assembly --------------------------------------------------------------


def _now_utc_iso() -> str:
    # UTC with 'Z'; deliberately no local timezone offset (privacy: no timezone leak).
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _used_backend_versions(
    registry: BackendRegistry, protocol: EvalProtocol, *, kinds: set[str]
) -> dict[str, Any]:
    prefs = protocol.backend_preferences
    defaults = {
        "mesh": "trimesh",
        "pointcloud": "plyfile",
        "nearest_neighbor": "scipy",
        "registration": "open3d",
        "trajectory": "evo",
    }
    resolved = {kind: prefs.get(kind, defaults.get(kind, "")) for kind in kinds if kind}
    return registry.backend_versions(resolved)


def _aggregate_metrics(
    outcome: SceneOutcome, protocol: EvalProtocol
) -> dict[str, float | None]:
    """Single-file aggregate: each metric's value is its (only) scene value.

    On failure, values come from ``failure_policy.worst_values`` when the policy is
    ``score_worst``; otherwise they are ``None`` so partial coverage stays visible.
    """
    if outcome.failure is None:
        return {m.name: m.value for m in outcome.metrics}
    if protocol.failure_policy.policy == "score_worst":
        worst = protocol.failure_policy.worst_values
        return {m.name: worst.get(m.name) for m in protocol.metrics}
    return {m.name: None for m in protocol.metrics}


def run_single_file_geometry(
    pred_path: str | Path,
    gt_path: str | Path,
    protocol: EvalProtocol,
    *,
    input_type: GeometryKind = "pointcloud",
    gt_type: GeometryKind = "pointcloud",
    threshold: float | None = None,
    sample: int | None = None,
    method: str | None = None,
    scene_id: str | None = None,
    registry: BackendRegistry | None = None,
    command: str | None = None,
    environment: dict[str, Any] | None = None,
    pred_trajectory: str | Path | None = None,
    gt_trajectory: str | Path | None = None,
) -> GeometryRunOutput:
    """Evaluate one prediction file against one GT file under ``protocol``."""
    registry = registry or default_registry()
    pred_path = Path(pred_path)
    gt_path = Path(gt_path)
    scene_id = scene_id or pred_path.stem or "scene"

    proto, overrides = apply_geometry_overrides(
        protocol, input_type=input_type, gt_type=gt_type, threshold=threshold, sample=sample
    )
    phash = compute_protocol_hash(proto)

    outcome = evaluate_geometry_scene(
        scene_id, pred_path, gt_path,
        protocol=proto, protocol_hash=phash,
        input_type=input_type, gt_type=gt_type, registry=registry,
        pred_trajectory=Path(pred_trajectory) if pred_trajectory else None,
        gt_trajectory=Path(gt_trajectory) if gt_trajectory else None,
    )

    # abort policy: fail loudly with scene/stage context.
    if outcome.failure is not None and proto.failure_policy.policy == "abort":
        raise SceneEvaluationError(scene_id, outcome.failure.stage, outcome.failure.reason)

    kinds = {"nearest_neighbor"}
    if input_type == "mesh" or gt_type == "mesh":
        kinds.add("mesh")
    if input_type == "pointcloud" or gt_type == "pointcloud":
        kinds.add("pointcloud")
    if proto.alignment.mode != "none" and proto.alignment.solver == "icp":
        kinds.add("registration")
    if proto.alignment.mode != "none" and proto.alignment.estimate_on == "trajectory":
        kinds.add("trajectory")
    backend_versions = _used_backend_versions(registry, proto, kinds=kinds)

    metrics = _aggregate_metrics(outcome, proto)
    evaluated = 1 if outcome.failure is None else 0
    alignment_transforms = [outcome.alignment.as_dict()] if outcome.alignment else []

    env = environment if environment is not None else {}

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
        n_scenes_evaluated=evaluated,
        failed_scenes=[outcome.failure] if outcome.failure else [],
        failure_policy=proto.failure_policy,
        metrics=metrics,
        metric_definitions=proto.metrics,
        per_scene_metrics=outcome.metrics,
        confidence_policy=proto.confidence,
        alignment=proto.alignment,
        masking=proto.masking,
        sampling=proto.sampling,
        aggregation=proto.aggregation,
        backend_versions=backend_versions,
        environment=env,
        command=command,
        timestamp=_now_utc_iso(),
    )

    config = {
        "protocol": proto.name,
        "protocol_hash": phash,
        "inputs": {
            "pred": str(pred_path),
            "gt": str(gt_path),
            "input_type": input_type,
            "gt_type": gt_type,
        },
        "overrides": overrides,
    }

    return GeometryRunOutput(
        result=result,
        protocol=proto,
        protocol_hash=phash,
        alignment_transforms=alignment_transforms,
        overrides=overrides,
        config=config,
        debug_scenes=(
            [(outcome.scene_id, outcome.debug)] if outcome.debug is not None else []
        ),
        alignment_vis=(
            [outcome.alignment_vis] if outcome.alignment_vis is not None else []
        ),
    )


def _eval3r_version() -> str:
    from eval3r import __version__

    return __version__
