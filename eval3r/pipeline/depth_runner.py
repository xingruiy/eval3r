"""Staged single-file / sequence depth runner.

Mirrors the task-007 geometry runner (same stage order, failure-policy handling,
and RunResult assembly) with depth-shaped stages: load (``depth_io`` backend) →
mask (protocol ``MaskingSpec``) → align (protocol ``AlignmentSpec`` scale modes) →
metric → per-frame aggregation. A prediction is either one depth file
(``single_depth``) or a directory of frames (``depth_sequence``) matched to a GT
directory by filename stem.

Depth sequences are only ever aggregated over frames; they are never integrated
into meshes, fused point clouds, or TSDF volumes (hard boundary).

Scale alignment granularity:

- ``per_frame``    — scale (and shift for affine) estimated independently per frame.
- ``per_sequence`` — one scale estimated over all frames' pooled valid pixels.
- ``per_scene``    — identical pooling on this single-scene path, recorded as
  ``per_scene`` so results stay comparable with dataset runs.

Every estimated scale/shift is recorded (alignment records + per-metric metadata),
because AbsRel under per-frame median scaling is not comparable to AbsRel under
per-sequence least-squares scaling.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from eval3r.core.adaptation import (
    AdaptationRecord,
    adaptation_override_from_legacy,
    resolve_adaptation,
)
from eval3r.core.errors import Eval3rError, InvalidDepthError, SceneEvaluationError
from eval3r.core.hashing import protocol_hash as compute_protocol_hash
from eval3r.core.protocol import EvalProtocol
from eval3r.core.registry import BackendRegistry, default_registry
from eval3r.core.result import MetricResult, RunResult, SceneFailure
from eval3r.metrics.aggregation import mean as scene_mean
from eval3r.metrics.depth import (
    DEPTH_ALIGNMENT_GRANULARITIES,
    DepthScaleAlignment,
    DepthValidity,
    check_depth_alignment_spec,
    compute_depth_validity,
    estimate_depth_scale,
    evaluate_depth_metrics,
    validate_depth_pair,
)

DepthStage = Literal["resolve", "load", "mask", "align", "metric", "aggregate"]


@dataclass(frozen=True)
class DepthFrame:
    """One matched (pred, gt) depth-file pair."""

    frame_id: str
    pred_path: Path
    gt_path: Path


@dataclass
class DepthSceneOutcome:
    """Per-frame + aggregated metrics on success, a structured failure otherwise."""

    scene_id: str
    frame_metrics: list[MetricResult] = field(default_factory=list)
    scene_metrics: list[MetricResult] = field(default_factory=list)
    alignments: list[DepthScaleAlignment] = field(default_factory=list)
    failure: SceneFailure | None = None


@dataclass
class DepthRunOutput:
    """Everything the writer/CLI needs after a single-file depth run."""

    result: RunResult
    protocol: EvalProtocol
    protocol_hash: str
    alignment_records: list[dict[str, Any]]
    overrides: dict[str, Any]
    config: dict[str, Any]
    n_frames: int
    adaptation: AdaptationRecord | None = None


# --- frame resolution ------------------------------------------------------------


def resolve_depth_frames(pred_path: Path, gt_path: Path, *, scene_id: str) -> list[DepthFrame]:
    """Resolve one file pair or match two frame directories by filename stem."""
    if pred_path.is_file() and gt_path.is_file():
        return [DepthFrame(frame_id=pred_path.stem, pred_path=pred_path, gt_path=gt_path)]
    if pred_path.is_dir() and gt_path.is_dir():
        return _match_sequence_dirs(pred_path, gt_path, scene_id=scene_id)
    for role, path in (("prediction", pred_path), ("ground-truth", gt_path)):
        if not path.exists():
            raise InvalidDepthError(f"{role} depth path does not exist: {path}.")
    raise InvalidDepthError(
        f"prediction and ground truth must both be files (single frame) or both be "
        f"directories (depth sequence); got pred={pred_path} "
        f"({'dir' if pred_path.is_dir() else 'file'}) vs gt={gt_path} "
        f"({'dir' if gt_path.is_dir() else 'file'})."
    )


def _index_depth_dir(directory: Path, *, role: str) -> dict[str, Path]:
    files = sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith("."))
    if not files:
        raise InvalidDepthError(f"{role} depth directory contains no files: {directory}.")
    index: dict[str, Path] = {}
    for path in files:
        if path.stem in index:
            raise InvalidDepthError(
                f"{role} depth directory {directory} has two files with the same stem "
                f"'{path.stem}' ({index[path.stem].name} and {path.name}); frame matching "
                f"by stem is ambiguous."
            )
        index[path.stem] = path
    return index


def _match_sequence_dirs(pred_dir: Path, gt_dir: Path, *, scene_id: str) -> list[DepthFrame]:
    pred_index = _index_depth_dir(pred_dir, role="prediction")
    gt_index = _index_depth_dir(gt_dir, role="ground-truth")
    missing_gt = sorted(set(pred_index) - set(gt_index))
    missing_pred = sorted(set(gt_index) - set(pred_index))
    if missing_gt or missing_pred:
        raise InvalidDepthError(
            f"depth sequence frames do not match between {pred_dir} and {gt_dir} "
            f"(scene '{scene_id}'): {len(missing_gt)} pred frame(s) without GT "
            f"{missing_gt[:5]}{'…' if len(missing_gt) > 5 else ''}, "
            f"{len(missing_pred)} GT frame(s) without pred "
            f"{missing_pred[:5]}{'…' if len(missing_pred) > 5 else ''}. "
            f"Frames are matched by filename stem."
        )
    return [
        DepthFrame(frame_id=stem, pred_path=pred_index[stem], gt_path=gt_index[stem])
        for stem in sorted(pred_index)
    ]


# --- overrides -------------------------------------------------------------------


def apply_depth_overrides(
    protocol: EvalProtocol,
    *,
    modality: Literal["single_depth", "depth_sequence"],
    align_granularity: str | None,
    pred_depth_unit: float | None,
    gt_depth_unit: float | None,
) -> tuple[EvalProtocol, dict[str, Any]]:
    """Apply CLI/API depth flags to a protocol copy and record what changed.

    Alignment granularity changes evaluation behavior, so the caller recomputes
    the canonical hash. Alignment mode itself is resolved as non-hashed
    prediction adaptation. Depth units are inputs, not protocol fields; they are
    recorded as overrides and in result metadata.
    """
    proto = protocol.model_copy(deep=True)
    overrides: dict[str, Any] = {"modality": modality}
    proto.prediction_modality = modality

    if align_granularity is not None:
        if align_granularity not in DEPTH_ALIGNMENT_GRANULARITIES:
            raise InvalidDepthError(
                f"--align-granularity '{align_granularity}' is invalid; choose one of: "
                f"{', '.join(DEPTH_ALIGNMENT_GRANULARITIES)}."
            )
        proto.alignment.granularity = align_granularity  # type: ignore[assignment]
        overrides["align_granularity"] = align_granularity
    if pred_depth_unit is not None:
        overrides["pred_depth_unit"] = pred_depth_unit
    if gt_depth_unit is not None:
        overrides["gt_depth_unit"] = gt_depth_unit
    return proto, overrides


# --- per-scene evaluation ----------------------------------------------------------


def evaluate_depth_scene(
    scene_id: str,
    frames: list[DepthFrame],
    *,
    protocol: EvalProtocol,
    protocol_hash: str,
    registry: BackendRegistry,
    pred_depth_unit: float | None,
    gt_depth_unit: float | None,
) -> DepthSceneOutcome:
    """Run load → mask → align → metric → aggregate for one scene's frames."""
    depth_backend = registry.require(
        "depth_io", protocol.backend_preferences.get("depth_io", "imageio")
    )
    alignment = protocol.alignment
    unit_metadata = {
        "depth_unit_pred": pred_depth_unit if pred_depth_unit is not None else 1.0,
        "depth_unit_gt": gt_depth_unit if gt_depth_unit is not None else 1.0,
    }

    stage: DepthStage = "load"
    try:
        check_depth_alignment_spec(alignment)

        loaded: list[tuple[DepthFrame, np.ndarray, np.ndarray, DepthValidity]] = []
        for frame in frames:
            stage = "load"
            pred = depth_backend.load_depth(frame.pred_path, pred_depth_unit)
            gt = depth_backend.load_depth(frame.gt_path, gt_depth_unit)
            pred, gt = validate_depth_pair(pred, gt, frame=frame.frame_id)
            stage = "mask"
            validity = compute_depth_validity(
                pred, gt, protocol.masking, frame=frame.frame_id
            )
            loaded.append((frame, pred, gt, validity))

        stage = "align"
        alignments: list[DepthScaleAlignment] = []
        pooled_scale, pooled_shift = 1.0, 0.0
        if alignment.mode != "none" and alignment.granularity in ("per_sequence", "per_scene"):
            pooled_pred = np.concatenate([p[v.mask] for _, p, _, v in loaded])
            pooled_gt = np.concatenate([g[v.mask] for _, _, g, v in loaded])
            pooled_scale, pooled_shift = estimate_depth_scale(
                pooled_pred, pooled_gt, alignment.mode,
                context=f"scene '{scene_id}', {len(loaded)} frame(s) pooled",
            )
            alignments.append(
                DepthScaleAlignment(
                    mode=alignment.mode, granularity=alignment.granularity,
                    scale=pooled_scale, shift=pooled_shift,
                    n_pixels_used=int(pooled_pred.size), frame_id=None,
                )
            )

        stage = "metric"
        frame_results: list[MetricResult] = []
        for frame, pred, gt, validity in loaded:
            if alignment.mode == "none":
                frame_align = DepthScaleAlignment(
                    mode="none", granularity=alignment.granularity,
                    scale=1.0, shift=0.0,
                    n_pixels_used=validity.n_pixels_valid, frame_id=frame.frame_id,
                )
            elif alignment.granularity == "per_frame":
                scale, shift = estimate_depth_scale(
                    pred[validity.mask], gt[validity.mask], alignment.mode,
                    context=f"scene '{scene_id}', frame '{frame.frame_id}'",
                )
                frame_align = DepthScaleAlignment(
                    mode=alignment.mode, granularity="per_frame",
                    scale=scale, shift=shift,
                    n_pixels_used=validity.n_pixels_valid, frame_id=frame.frame_id,
                )
                alignments.append(frame_align)
            else:
                frame_align = DepthScaleAlignment(
                    mode=alignment.mode, granularity=alignment.granularity,
                    scale=pooled_scale, shift=pooled_shift,
                    n_pixels_used=validity.n_pixels_valid, frame_id=frame.frame_id,
                )
            aligned = pred * frame_align.scale + frame_align.shift
            frame_results.extend(
                evaluate_depth_metrics(
                    aligned, gt, validity, protocol.metrics,
                    alignment=frame_align,
                    protocol=protocol.name, protocol_hash=protocol_hash,
                    backend_name=depth_backend.name,
                    scene_id=scene_id, frame_id=frame.frame_id,
                    extra_metadata=unit_metadata,
                )
            )

        stage = "aggregate"
        scene_results = _aggregate_frames_to_scene(
            frame_results, protocol=protocol, protocol_hash=protocol_hash,
            scene_id=scene_id, n_frames=len(loaded),
            backend_name=depth_backend.name, unit_metadata=unit_metadata,
        )
        return DepthSceneOutcome(
            scene_id=scene_id,
            frame_metrics=frame_results,
            scene_metrics=scene_results,
            alignments=alignments,
        )
    except Eval3rError as exc:
        failure = SceneFailure(
            scene_id=scene_id,
            stage=stage,
            reason=str(exc),
            traceback=traceback.format_exc(),
            recoverable=stage != "load",
        )
        return DepthSceneOutcome(scene_id=scene_id, failure=failure)


def _aggregate_frames_to_scene(
    frame_results: list[MetricResult],
    *,
    protocol: EvalProtocol,
    protocol_hash: str,
    scene_id: str,
    n_frames: int,
    backend_name: str,
    unit_metadata: dict[str, Any],
) -> list[MetricResult]:
    """Mean over frames per metric name (the protocol's per_frame aggregation)."""
    scene_results: list[MetricResult] = []
    for spec in protocol.metrics:
        values: list[float] = [
            m.value for m in frame_results if m.name == spec.name and m.value is not None
        ]
        per_frame = [m for m in frame_results if m.name == spec.name]
        fractions = [m.valid_fraction for m in per_frame if m.valid_fraction is not None]
        scene_results.append(
            MetricResult(
                name=spec.name,
                value=scene_mean(values),
                unit=per_frame[0].unit if per_frame else None,
                threshold=spec.threshold,
                statistic=spec.statistic,
                scene_id=scene_id,
                frame_id=None,
                protocol=protocol.name,
                protocol_hash=protocol_hash,
                backend=backend_name,
                n_pixels_valid=sum(m.n_pixels_valid or 0 for m in per_frame),
                valid_fraction=float(np.mean(fractions)) if fractions else None,
                metadata={
                    "aggregation": "per_frame_mean",
                    "n_frames": n_frames,
                    "alignment_mode": protocol.alignment.mode,
                    "alignment_granularity": protocol.alignment.granularity,
                    **unit_metadata,
                },
            )
        )
    return scene_results


# --- run assembly ------------------------------------------------------------------


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _aggregate_run_metrics(
    outcome: DepthSceneOutcome, protocol: EvalProtocol
) -> dict[str, float | None]:
    if outcome.failure is None:
        return {m.name: m.value for m in outcome.scene_metrics}
    if protocol.failure_policy.policy == "score_worst":
        worst = protocol.failure_policy.worst_values
        return {m.name: worst.get(m.name) for m in protocol.metrics}
    return {m.name: None for m in protocol.metrics}


def run_single_file_depth(
    pred_path: str | Path,
    gt_path: str | Path,
    protocol: EvalProtocol,
    *,
    pred_depth_unit: float | None = None,
    gt_depth_unit: float | None = None,
    align: str | None = None,
    adapt: str | None = None,
    align_granularity: str | None = None,
    method: str | None = None,
    scene_id: str | None = None,
    registry: BackendRegistry | None = None,
    command: str | None = None,
    environment: dict[str, Any] | None = None,
) -> DepthRunOutput:
    """Evaluate one depth prediction (file or frame directory) under ``protocol``."""
    registry = registry or default_registry()
    pred_path = Path(pred_path)
    gt_path = Path(gt_path)
    scene_id = scene_id or pred_path.stem or "scene"

    stage_resolve_failure: SceneFailure | None = None
    try:
        frames = resolve_depth_frames(pred_path, gt_path, scene_id=scene_id)
    except Eval3rError as exc:
        if protocol.failure_policy.policy == "abort":
            raise SceneEvaluationError(scene_id, "resolve", str(exc)) from exc
        frames = []
        stage_resolve_failure = SceneFailure(
            scene_id=scene_id, stage="resolve", reason=str(exc),
            traceback=traceback.format_exc(), recoverable=False,
        )

    modality: Literal["single_depth", "depth_sequence"] = (
        "depth_sequence" if len(frames) > 1 or pred_path.is_dir() else "single_depth"
    )
    proto, overrides = apply_depth_overrides(
        protocol, modality=modality,
        align_granularity=align_granularity,
        pred_depth_unit=pred_depth_unit, gt_depth_unit=gt_depth_unit,
    )
    phash = compute_protocol_hash(proto)
    adaptation_override = adaptation_override_from_legacy(
        adapt=adapt,
        align=align,
        unit="m",
        scale="metric",
    )
    run_proto, adaptation = resolve_adaptation(proto, None, adaptation_override)
    if align is not None:
        overrides["align"] = align
    if adapt is not None:
        overrides["adapt"] = adapt

    if stage_resolve_failure is not None:
        outcome = DepthSceneOutcome(scene_id=scene_id, failure=stage_resolve_failure)
    else:
        outcome = evaluate_depth_scene(
            scene_id, frames,
            protocol=run_proto, protocol_hash=phash, registry=registry,
            pred_depth_unit=pred_depth_unit, gt_depth_unit=gt_depth_unit,
        )

    if outcome.failure is not None and run_proto.failure_policy.policy == "abort":
        raise SceneEvaluationError(scene_id, outcome.failure.stage, outcome.failure.reason)

    backend_versions = registry.backend_versions(
        {"depth_io": run_proto.backend_preferences.get("depth_io", "imageio")}
    )
    metrics = _aggregate_run_metrics(outcome, run_proto)
    alignment_records = [a.as_dict() for a in outcome.alignments]

    result = RunResult(
        schema_version=run_proto.schema_version,
        eval3r_version=_eval3r_version(),
        method=method,
        method_version=None,
        dataset=run_proto.dataset,
        split=run_proto.dataset.split,
        protocol=run_proto.name,
        protocol_version=run_proto.protocol_version,
        protocol_hash=phash,
        fidelity=run_proto.fidelity,
        ground_truth=run_proto.ground_truth,
        local_evaluation=run_proto.local_evaluation,
        n_scenes_expected=1,
        n_scenes_evaluated=1 if outcome.failure is None else 0,
        failed_scenes=[outcome.failure] if outcome.failure else [],
        failure_policy=run_proto.failure_policy,
        metrics=metrics,
        metric_definitions=run_proto.metrics,
        per_scene_metrics=outcome.scene_metrics + outcome.frame_metrics,
        confidence_policy=run_proto.confidence,
        alignment=run_proto.alignment,
        adaptation=adaptation,
        masking=run_proto.masking,
        sampling=run_proto.sampling,
        aggregation=run_proto.aggregation,
        backend_versions=backend_versions,
        environment=environment if environment is not None else {},
        command=command,
        timestamp=_now_utc_iso(),
        metadata={
            "n_frames": len(frames),
            "depth_unit_pred": pred_depth_unit if pred_depth_unit is not None else 1.0,
            "depth_unit_gt": gt_depth_unit if gt_depth_unit is not None else 1.0,
        },
    )

    config = {
        "protocol": run_proto.name,
        "protocol_hash": phash,
        "inputs": {
            "pred": str(pred_path),
            "gt": str(gt_path),
            "modality": modality,
            "pred_depth_unit": pred_depth_unit,
            "gt_depth_unit": gt_depth_unit,
        },
        "overrides": overrides,
        "adaptation": adaptation.model_dump(mode="json"),
    }

    return DepthRunOutput(
        result=result,
        protocol=run_proto,
        protocol_hash=phash,
        alignment_records=alignment_records,
        overrides=overrides,
        config=config,
        n_frames=len(frames),
        adaptation=adaptation,
    )


def _eval3r_version() -> str:
    from eval3r import __version__

    return __version__
