"""Generic benchmark orchestration over a :class:`DatasetAdapter`.

Runs the task-007 geometry scene evaluation across every scene in a split, applies
the protocol's failure policy per scene, aggregates per-scene metrics, and assembles
a benchmark :class:`RunResult` with full coverage accounting. No dataset-specific or
official behavior lives here — that is owned by real adapters (tasks 009+).

Preflight gating refuses a run *before* loading predictions when the split is not
locally evaluable (server-only, missing public GT, external assets/renderer, or
unsupported), so the CLI never emits official-looking numbers for a split that
cannot be evaluated locally (``.agent/datasets.md`` local-evaluation rules).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from eval3r.core.errors import (
    BenchmarkError,
    DatasetError,
    InvalidGeometryError,
    SceneEvaluationError,
)
from eval3r.core.hashing import protocol_hash as compute_protocol_hash
from eval3r.core.manifest import PredictionManifest
from eval3r.core.protocol import EvalProtocol
from eval3r.core.registry import BackendRegistry, default_registry
from eval3r.core.result import MetricResult, RunResult, SceneFailure
from eval3r.datasets.base import DatasetAdapter, gt_geometry, prediction_kind
from eval3r.pipeline.runner import SceneOutcome, evaluate_geometry_scene

_SUPPORTED_STATUS = "supported"


@dataclass
class BenchmarkRunOutput:
    """Everything the writer/CLI needs after a benchmark run."""

    result: RunResult
    protocol: EvalProtocol
    protocol_hash: str
    alignment_transforms: list[dict[str, Any]]
    manifest: dict[str, Any]
    manifest_inferred: bool
    config: dict[str, Any]


# --- manifest ------------------------------------------------------------------


def load_or_infer_manifest(
    pred_root: Path,
    scenes: list[str],
    protocol: EvalProtocol,
    *,
    manifest_path: Path | None = None,
) -> tuple[PredictionManifest | None, dict[str, Any], bool]:
    """Return ``(manifest, manifest_dict, inferred)``.

    Uses an explicit ``manifest_path``, else ``<pred_root>/manifest.yaml`` if present,
    else infers a simple ``<scene>.ply`` layout and marks it inferred so the run
    directory records that predictions were resolved by inference, not declaration.
    """
    src = manifest_path or (pred_root / "manifest.yaml")
    if src.is_file():
        data = yaml.safe_load(src.read_text(encoding="utf-8"))
        manifest = PredictionManifest.model_validate(data)
        return manifest, manifest.model_dump(mode="json"), False

    inferred = {
        "method": "inferred",
        "dataset": {"dataset": protocol.dataset.dataset, "variant": protocol.dataset.variant},
        "prediction_modality": protocol.prediction_modality,
        "coordinate_frame": "world",
        "scale": "metric",
        "scenes": {s: {"pointcloud": f"{s}.ply"} for s in scenes},
        "metadata": {"inferred": True},
    }
    manifest = PredictionManifest.model_validate(inferred)
    return manifest, manifest.model_dump(mode="json"), True


# --- preflight -----------------------------------------------------------------


def preflight(adapter: DatasetAdapter, split: str, protocol: EvalProtocol) -> None:
    """Refuse the run with an explicit reason if the split is not locally evaluable."""
    if adapter.capabilities.server_only_eval:
        raise BenchmarkError(
            f"dataset '{adapter.name}' declares this evaluation server-only; local numbers "
            f"would be unofficial and misleading. Submit predictions to the official server."
        )
    local = adapter.local_evaluation(split, protocol)
    if local.status != _SUPPORTED_STATUS:
        reason = local.reason or "no local evaluation path is available."
        raise BenchmarkError(
            f"dataset '{adapter.name}' split '{split}' is not locally evaluable "
            f"(status: {local.status}): {reason}"
        )


# --- per-scene resolution + evaluation -----------------------------------------


def _worst_results(
    scene_id: str, protocol: EvalProtocol, protocol_hash: str
) -> list[MetricResult]:
    worst = protocol.failure_policy.worst_values
    results: list[MetricResult] = []
    for spec in protocol.metrics:
        results.append(
            MetricResult(
                name=spec.name,
                value=worst.get(spec.name),
                scene_id=scene_id,
                protocol=protocol.name,
                protocol_hash=protocol_hash,
                metadata={"scored_worst": True},
            )
        )
    return results


def _evaluate_scene(
    adapter: DatasetAdapter,
    pred_root: Path,
    manifest: PredictionManifest | None,
    scene_id: str,
    protocol: EvalProtocol,
    protocol_hash: str,
    registry: BackendRegistry,
) -> SceneOutcome:
    """Resolve prediction + GT, then run the task-007 geometry stages for one scene."""
    try:
        recon = adapter.resolve_prediction(pred_root, scene_id, manifest)
        if recon.path is None:
            raise DatasetError(
                f"adapter '{adapter.name}' resolved scene '{scene_id}' without a prediction path."
            )
        pred_path = recon.path
        pred_kind = prediction_kind(recon)
        scene = adapter.load_scene(scene_id)
        gt_path, gt_kind = gt_geometry(scene)
    except (DatasetError, InvalidGeometryError) as exc:
        return SceneOutcome(
            scene_id=scene_id,
            failure=SceneFailure(scene_id=scene_id, stage="resolve", reason=str(exc)),
        )

    return evaluate_geometry_scene(
        scene_id, pred_path, gt_path,
        protocol=protocol, protocol_hash=protocol_hash,
        input_type=pred_kind, gt_type=gt_kind, registry=registry,  # type: ignore[arg-type]
    )


# --- run assembly --------------------------------------------------------------


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_benchmark_geometry(
    pred_root: str | Path,
    adapter: DatasetAdapter,
    protocol: EvalProtocol,
    split: str,
    *,
    manifest_path: str | Path | None = None,
    registry: BackendRegistry | None = None,
    command: str | None = None,
    environment: dict[str, Any] | None = None,
    method: str | None = None,
    progress: Any | None = None,
) -> BenchmarkRunOutput:
    """Evaluate every scene in ``split`` and assemble a benchmark ``RunResult``."""
    registry = registry or default_registry()
    pred_root = Path(pred_root)

    preflight(adapter, split, protocol)

    scenes = list(adapter.iter_scenes(split))
    phash = compute_protocol_hash(protocol)
    manifest, manifest_dict, inferred = load_or_infer_manifest(
        pred_root, scenes, protocol,
        manifest_path=Path(manifest_path) if manifest_path else None,
    )

    per_scene: list[MetricResult] = []
    failures: list[SceneFailure] = []
    alignment_transforms: list[dict[str, Any]] = []
    evaluated = 0

    for scene_id in scenes:
        outcome = _evaluate_scene(
            adapter, pred_root, manifest, scene_id, protocol, phash, registry
        )
        if progress is not None:
            progress(scene_id, outcome)

        if outcome.failure is None:
            evaluated += 1
            per_scene.extend(outcome.metrics)
            if outcome.alignment is not None:
                alignment_transforms.append(outcome.alignment.as_dict())
            continue

        # failed scene: apply the protocol's failure policy.
        policy = protocol.failure_policy.policy
        if policy == "abort":
            raise SceneEvaluationError(scene_id, outcome.failure.stage, outcome.failure.reason)
        failures.append(outcome.failure)
        if policy == "score_worst":
            per_scene.extend(_worst_results(scene_id, protocol, phash))

    from eval3r.pipeline.stages.aggregate import aggregate_scene_metrics

    metrics = aggregate_scene_metrics(per_scene, protocol.metrics)
    backend_versions = registry.backend_versions(
        {"nearest_neighbor": protocol.backend_preferences.get("nearest_neighbor", "scipy"),
         "pointcloud": protocol.backend_preferences.get("pointcloud", "plyfile"),
         "mesh": protocol.backend_preferences.get("mesh", "trimesh")}
    )

    result = RunResult(
        schema_version=protocol.schema_version,
        eval3r_version=_eval3r_version(),
        method=method or (manifest.method if manifest else None),
        method_version=manifest.version if manifest else None,
        dataset=protocol.dataset,
        split=split,
        protocol=protocol.name,
        protocol_version=protocol.protocol_version,
        protocol_hash=phash,
        fidelity=protocol.fidelity,
        ground_truth=protocol.ground_truth,
        local_evaluation=protocol.local_evaluation,
        n_scenes_expected=len(scenes),
        n_scenes_evaluated=evaluated,
        failed_scenes=failures,
        failure_policy=protocol.failure_policy,
        metrics=metrics,
        metric_definitions=protocol.metrics,
        per_scene_metrics=per_scene,
        confidence_policy=protocol.confidence,
        alignment=protocol.alignment,
        masking=protocol.masking,
        sampling=protocol.sampling,
        aggregation=protocol.aggregation,
        uses_gt=manifest.uses_gt if manifest else None,
        backend_versions=backend_versions,
        environment=environment or {},
        command=command,
        manifest_path=Path(manifest_path) if manifest_path else None,
        timestamp=_now_utc_iso(),
        metadata={"manifest_inferred": inferred},
    )

    config = {
        "protocol": protocol.name,
        "protocol_hash": phash,
        "dataset": adapter.name,
        "split": split,
        "pred_root": str(pred_root),
        "manifest_inferred": inferred,
        "n_scenes": len(scenes),
    }

    return BenchmarkRunOutput(
        result=result,
        protocol=protocol,
        protocol_hash=phash,
        alignment_transforms=alignment_transforms,
        manifest=manifest_dict,
        manifest_inferred=inferred,
        config=config,
    )


def _eval3r_version() -> str:
    from eval3r import __version__

    return __version__
