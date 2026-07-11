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

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

from eval3r.core.adaptation import (
    AdaptationRecord,
    adaptation_override_from_legacy,
    resolve_adaptation,
)
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
from eval3r.core.types import WorldAxes
from eval3r.datasets.base import DatasetAdapter, gt_geometry, prediction_kind
from eval3r.pipeline.runner import SceneOutcome, evaluate_geometry_scene
from eval3r.pipeline.stages.sample import DEFAULT_BASE_SEED

_SUPPORTED_STATUS = "supported"


@dataclass
class BenchmarkRunOutput:
    """Everything the writer/CLI needs after a benchmark run.

    ``debug_scenes`` holds per-scene directional distances captured when the
    protocol's reporting spec requests debug outputs; only the eval3r-native
    geometry path produces them (official-toolbox paths own their distances).
    ``alignment_vis`` holds the per-scene before/after captures for the mandatory
    alignment visualization whenever a non-``none`` alignment ran.
    """

    result: RunResult
    protocol: EvalProtocol
    protocol_hash: str
    alignment_transforms: list[dict[str, Any]]
    manifest: dict[str, Any]
    manifest_inferred: bool
    config: dict[str, Any]
    adaptation: AdaptationRecord | None = None
    debug_scenes: list[tuple[str, Any]] = field(default_factory=list)
    alignment_vis: list[Any] = field(default_factory=list)
    trajectory_alignment_vis: list[Any] = field(default_factory=list)


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


def _evaluate_scene_official(
    adapter: DatasetAdapter,
    pred_root: Path,
    manifest: PredictionManifest | None,
    scene_id: str,
    protocol: EvalProtocol,
    protocol_hash: str,
    registry: BackendRegistry,
    official_name: str,
    base_seed: int,
) -> SceneOutcome:
    """Evaluate one scene with a dataset's official-like evaluator (e.g. DTU dtu_eval).

    Points are used in their native units (DTU is millimetres): the official ObsMask,
    Plane, and distance cap are all in that frame, so this path does **not** normalize
    to metres. Visibility (ObsMask/Plane) comes from the adapter.
    """
    from eval3r.core.errors import MetricError
    from eval3r.pipeline.stages.sample import derive_seed

    if not hasattr(adapter, "load_visibility_data"):
        raise BenchmarkError(
            f"protocol '{protocol.name}' needs the official evaluator '{official_name}', but "
            f"dataset '{adapter.name}' provides no visibility (ObsMask/Plane) data."
        )
    pc_backend = registry.require(
        "pointcloud", protocol.backend_preferences.get("pointcloud", "plyfile")
    )
    evaluator = registry.require("official_eval", official_name)

    try:
        recon = adapter.resolve_prediction(pred_root, scene_id, manifest)
        if recon.path is None:
            raise DatasetError(f"scene '{scene_id}' resolved without a prediction path.")
        if recon.modality != "pointcloud":
            raise DatasetError(
                f"official DTU-like evaluation takes point-cloud predictions; scene '{scene_id}' "
                f"is '{recon.modality}'. Sample the mesh to a point cloud first."
            )
        pred_points = pc_backend.load_pointcloud(recon.path)
        scene = adapter.load_scene(scene_id)
        gt_path, gt_kind = gt_geometry(scene)
        if gt_kind != "pointcloud":
            raise DatasetError(f"DTU GT must be a point cloud; scene '{scene_id}' GT is {gt_kind}.")
        gt_points = pc_backend.load_pointcloud(gt_path)
        visibility = adapter.load_visibility_data(scene_id, protocol)
    except (DatasetError, InvalidGeometryError) as exc:
        return SceneOutcome(
            scene_id=scene_id,
            failure=SceneFailure(scene_id=scene_id, stage="resolve", reason=str(exc)),
        )

    try:
        seed = derive_seed("derive", scene_id, base_seed=base_seed)
        result = evaluator.evaluate(pred_points, gt_points, visibility, seed=seed)
    except MetricError as exc:
        return SceneOutcome(
            scene_id=scene_id,
            failure=SceneFailure(scene_id=scene_id, stage="metric", reason=str(exc)),
        )

    values = {
        "accuracy": result.accuracy,
        "completeness": result.completeness,
        "overall": result.overall,
    }
    meta = {
        "evaluator_method": getattr(evaluator, "method", official_name),
        "plane_available": visibility.get("plane_available"),
        "n_data_in_obs": result.n_data_in_obs,
        "n_stl_above": result.n_stl_above,
    }
    metrics = [
        MetricResult(
            name=spec.name,
            value=values.get(spec.name),
            unit="mm",
            scene_id=scene_id,
            protocol=protocol.name,
            protocol_hash=protocol_hash,
            backend=official_name,
            n_points_pred=result.n_data_in_obs,
            n_points_gt=result.n_stl_above,
            metadata=meta,
        )
        for spec in protocol.metrics
        if spec.name in values
    ]
    return SceneOutcome(scene_id=scene_id, metrics=metrics)


def _evaluate_scene_tnt_official(
    adapter: DatasetAdapter,
    pred_root: Path,
    manifest: PredictionManifest | None,
    scene_id: str,
    protocol: EvalProtocol,
    protocol_hash: str,
    registry: BackendRegistry,
    official_name: str,
    out_root: Path,
) -> SceneOutcome:
    """Evaluate one scene by wrapping a file-based official toolbox (Tanks and Temples).

    Unlike the DTU port (which takes point arrays + ObsMask), this official backend
    reads the prediction/GT/crop/trajectory files itself and does its own alignment,
    ICP, cropping, and per-scene thresholding. eval3r resolves the five per-scene
    artifacts and the prediction path, invokes the toolbox, and records the official
    precision/recall/F-score plus the per-scene distance threshold, command, and
    toolbox version. The per-scene threshold comes from the official output, never a
    hardcoded global constant (``.agent/datasets.md`` Tanks and Temples rules).
    """
    from eval3r.core.errors import MetricError

    if not hasattr(adapter, "official_artifacts"):
        raise BenchmarkError(
            f"protocol '{protocol.name}' needs the official evaluator '{official_name}', but "
            f"dataset '{adapter.name}' provides no official_artifacts() to resolve the GT point "
            f"cloud, crop volume, alignment transform, and .log trajectory."
        )
    evaluator = registry.require("official_eval", official_name)

    try:
        recon = adapter.resolve_prediction(pred_root, scene_id, manifest)
        if recon.path is None:
            raise DatasetError(f"scene '{scene_id}' resolved without a prediction path.")
        art = adapter.official_artifacts(scene_id)
    except (DatasetError, InvalidGeometryError) as exc:
        return SceneOutcome(
            scene_id=scene_id,
            failure=SceneFailure(scene_id=scene_id, stage="resolve", reason=str(exc)),
        )

    try:
        result = evaluator.evaluate_scene(
            scene_id,
            dataset_dir=art["dataset_dir"],
            traj_path=art["trajectory_log"],
            ply_path=recon.path,
            out_dir=out_root / scene_id,
        )
    except MetricError as exc:
        return SceneOutcome(
            scene_id=scene_id,
            failure=SceneFailure(scene_id=scene_id, stage="metric", reason=str(exc)),
        )

    values = {
        "precision": result.precision,
        "recall": result.recall,
        "fscore": result.fscore,
    }
    meta = {
        "evaluator_method": getattr(evaluator, "method", official_name),
        # Per-scene threshold resolved by the official backend, not a global constant.
        "distance_tau": result.distance_tau,
        "command": result.command,
        "toolbox_dir": result.toolbox_dir,
        "toolbox_commit": result.toolbox_commit,
        "python_executable": result.python_executable,
    }
    metrics = [
        MetricResult(
            name=spec.name,
            value=values.get(spec.name),
            scene_id=scene_id,
            protocol=protocol.name,
            protocol_hash=protocol_hash,
            backend=official_name,
            metadata={**meta, "threshold": result.distance_tau},
        )
        for spec in protocol.metrics
        if spec.name in values
    ]
    return SceneOutcome(scene_id=scene_id, metrics=metrics)


def _evaluate_scene_eth3d_official(
    adapter: DatasetAdapter,
    pred_root: Path,
    manifest: PredictionManifest | None,
    scene_id: str,
    protocol: EvalProtocol,
    protocol_hash: str,
    registry: BackendRegistry,
    official_name: str,
) -> SceneOutcome:
    """Evaluate one scene by wrapping the official ETH3D multi-view-evaluation tool.

    The official binary consumes the prediction PLY and the scene's
    ``scan_alignment.mlp`` directly and reports accuracy / completeness / F1 at a
    tolerance list. The protocol's metric specs carry the official tolerance set as
    per-metric thresholds (e.g. ``fscore_2cm`` with ``threshold: 0.02``); the tool is
    invoked once per scene with all tolerances and each spec is filled from the
    matching column. Voxel-normalization and beam-based free-space handling happen
    inside the official tool, never here.
    """
    from eval3r.core.errors import MetricError

    if not hasattr(adapter, "official_artifacts"):
        raise BenchmarkError(
            f"protocol '{protocol.name}' needs the official evaluator '{official_name}', but "
            f"dataset '{adapter.name}' provides no official_artifacts() to resolve the "
            f"scan_alignment.mlp ground truth."
        )
    evaluator = registry.require("official_eval", official_name)

    # Metric spec -> (official output column, tolerance). Fails explicitly on a
    # metric the official tool does not report or on a missing threshold.
    kinds = {"accuracy": "accuracies", "completeness": "completenesses", "fscore": "f1_scores"}
    spec_columns: list[tuple[str, str, float]] = []
    for spec in protocol.metrics:
        kind = spec.name.rsplit("_", 1)[0] if "_" in spec.name else spec.name
        if kind not in kinds:
            raise BenchmarkError(
                f"protocol '{protocol.name}' metric '{spec.name}' does not map to an ETH3D "
                f"official output (accuracy/completeness/fscore at a tolerance)."
            )
        if spec.threshold is None:
            raise BenchmarkError(
                f"protocol '{protocol.name}' metric '{spec.name}' has no threshold; the ETH3D "
                f"official protocol must pin every tolerance explicitly."
            )
        spec_columns.append((spec.name, kinds[kind], spec.threshold))
    tolerances = sorted({tol for _, _, tol in spec_columns})

    try:
        recon = adapter.resolve_prediction(pred_root, scene_id, manifest)
        if recon.path is None:
            raise DatasetError(f"scene '{scene_id}' resolved without a prediction path.")
        art = adapter.official_artifacts(scene_id)
    except (DatasetError, InvalidGeometryError) as exc:
        return SceneOutcome(
            scene_id=scene_id,
            failure=SceneFailure(scene_id=scene_id, stage="resolve", reason=str(exc)),
        )

    try:
        result = evaluator.evaluate_scene(
            scene_id,
            scan_mlp_path=art["scan_mlp"],
            ply_path=recon.path,
            tolerances=tolerances,
        )
        columns = {
            "accuracies": result.accuracies,
            "completenesses": result.completenesses,
            "f1_scores": result.f1_scores,
        }
        values = {
            name: evaluator.lookup(columns[column], tol)
            for name, column, tol in spec_columns
        }
    except MetricError as exc:
        return SceneOutcome(
            scene_id=scene_id,
            failure=SceneFailure(scene_id=scene_id, stage="metric", reason=str(exc)),
        )

    meta = {
        "evaluator_method": getattr(evaluator, "method", official_name),
        "tolerances": result.tolerances,
        "command": result.command,
        "tool_path": result.tool_path,
        "tool_commit": result.tool_commit,
        # Official free-space / voxel-normalization parameters in effect (tool defaults).
        "voxel_size": result.voxel_size,
        "beam_start_radius_meters": result.beam_start_radius_meters,
        "beam_divergence_halfangle_deg": result.beam_divergence_halfangle_deg,
    }
    metrics = [
        MetricResult(
            name=spec.name,
            value=values.get(spec.name),
            scene_id=scene_id,
            protocol=protocol.name,
            protocol_hash=protocol_hash,
            backend=official_name,
            metadata={**meta, "threshold": spec.threshold},
        )
        for spec in protocol.metrics
        if spec.name in values
    ]
    return SceneOutcome(scene_id=scene_id, metrics=metrics)


def _evaluate_scene_visibility_culled(
    adapter: DatasetAdapter,
    pred_root: Path,
    manifest: PredictionManifest | None,
    scene_id: str,
    protocol: EvalProtocol,
    protocol_hash: str,
    registry: BackendRegistry,
    base_seed: int,
    pred_unit_override: str | None = None,
    pred_world_frame_override: WorldAxes | None = None,
) -> SceneOutcome:
    """Evaluate one scene with render+TSDF visibility culling of the prediction.

    Implements the ScanNet single-/double-layer convention: the prediction mesh is
    trimmed to the region observed by the GT camera trajectory (``visibility`` backend)
    before surface sampling and scoring. ``culled_fraction`` and the renderer/TSDF
    metadata are recorded per scene (CLAUDE.md visibility-culling exception).
    """
    from eval3r.core.errors import CullingError, Eval3rError, MetricError
    from eval3r.metrics.diagnostics import build_diagnostic_metrics, partition_specs
    from eval3r.metrics.geometry import compute_directional_distances
    from eval3r.pipeline.stages.load import LoadedGeometry, load_geometry
    from eval3r.pipeline.stages.metric import compute_scene_metrics
    from eval3r.pipeline.stages.normalize import normalize_to_meters, normalize_world_frame
    from eval3r.pipeline.stages.sample import sample_geometry

    if not hasattr(adapter, "load_trajectory"):
        raise BenchmarkError(
            f"protocol '{protocol.name}' requests gt_visibility culling, but dataset "
            f"'{adapter.name}' exposes no load_trajectory() to supply the GT trajectory."
        )
    mesh_backend = registry.require("mesh", protocol.backend_preferences.get("mesh", "trimesh"))
    pc_backend = registry.require(
        "pointcloud", protocol.backend_preferences.get("pointcloud", "plyfile")
    )
    nn_backend = registry.require(
        "nearest_neighbor", protocol.backend_preferences.get("nearest_neighbor", "scipy")
    )
    vis_backend = registry.require(
        "visibility", protocol.backend_preferences.get("visibility", "render_tsdf")
    )
    cull_spec = protocol.masking.pred_culling
    tolerance = cull_spec.tolerance
    if tolerance is None:
        raise BenchmarkError(
            f"protocol '{protocol.name}' visibility culling needs an explicit "
            f"masking.pred_culling.tolerance (keep radius, metres)."
        )
    params = dict(cull_spec.parameters)

    stage: Literal["resolve", "load", "normalize", "mask", "sample", "metric"] = "resolve"
    try:
        recon = adapter.resolve_prediction(pred_root, scene_id, manifest)
        if recon.path is None:
            raise DatasetError(f"scene '{scene_id}' resolved without a prediction path.")
        if recon.modality != "mesh":
            raise DatasetError(
                f"visibility-culled evaluation takes mesh predictions; scene '{scene_id}' is "
                f"'{recon.modality}'."
            )
        scene = adapter.load_scene(scene_id)
        gt_path, gt_kind = gt_geometry(scene)
        if gt_kind != "mesh":
            raise DatasetError(f"ScanNet GT must be a mesh; scene '{scene_id}' GT is {gt_kind}.")
        trajectory = adapter.load_trajectory(scene_id)

        stage = "load"
        pred = load_geometry(
            recon.path, "mesh", mesh_backend=mesh_backend, pointcloud_backend=pc_backend
        )
        gt = load_geometry(
            gt_path, "mesh", mesh_backend=mesh_backend, pointcloud_backend=pc_backend
        )

        stage = "normalize"
        pred = normalize_to_meters(pred, pred_unit_override or recon.unit)
        gt = normalize_to_meters(gt, scene.ground_truth.unit)
        # Rotate an OpenGL-world prediction into the internal frame before the GT
        # trajectory renders/culls it (the culling trajectory is already internal).
        pred = normalize_world_frame(pred, pred_world_frame_override or recon.world_frame)

        stage = "mask"
        cull = vis_backend.cull(pred.mesh, trajectory, tolerance=tolerance, **params)
        trimmed = LoadedGeometry(kind="mesh", path=pred.path, mesh=cull.trimmed_mesh)

        stage = "sample"
        pred_points, _ = sample_geometry(
            trimmed, protocol.sampling.pred, scene_id=scene_id, role="pred",
            mesh_backend=mesh_backend, base_seed=base_seed,
        )
        gt_points, _ = sample_geometry(
            gt, protocol.sampling.gt, scene_id=scene_id, role="gt",
            mesh_backend=mesh_backend, base_seed=base_seed,
        )

        stage = "metric"
        geometry_specs, diagnostic_specs = partition_specs(protocol.metrics)
        # Debug outputs (error-colored cloud / histogram) need per-point distances,
        # exactly as on the native runner path; the culled path must honor the
        # protocol's reporting spec too (found in task 028: it silently didn't).
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
        valid_fraction = metrics[0].valid_fraction if metrics else 1.0
        metrics = metrics + build_diagnostic_metrics(
            diagnostic_specs,
            {"culled_fraction": cull.culled_fraction, "valid_fraction": valid_fraction},
            scene_id=scene_id, protocol=protocol.name, protocol_hash=protocol_hash,
            backend=vis_backend.name, metadata=cull.metadata,
        )
    except (DatasetError, InvalidGeometryError, CullingError, MetricError, Eval3rError) as exc:
        return SceneOutcome(
            scene_id=scene_id,
            failure=SceneFailure(scene_id=scene_id, stage=stage, reason=str(exc)),
        )
    return SceneOutcome(scene_id=scene_id, metrics=metrics, debug=distances)


def _evaluate_scene(
    adapter: DatasetAdapter,
    pred_root: Path,
    manifest: PredictionManifest | None,
    scene_id: str,
    protocol: EvalProtocol,
    protocol_hash: str,
    registry: BackendRegistry,
    pred_unit_override: str | None = None,
    pred_world_frame_override: WorldAxes | None = None,
    base_seed: int = DEFAULT_BASE_SEED,
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
        pred_unit = pred_unit_override or recon.unit
        pred_world_frame = pred_world_frame_override or recon.world_frame
        scene = adapter.load_scene(scene_id)
        gt_path, gt_kind = gt_geometry(scene)
        gt_unit = scene.ground_truth.unit
        # Trajectory-first alignment inputs (task 018): the prediction trajectory
        # comes from the manifest entry, the GT trajectory from the scene data. A
        # missing one when the protocol requests trajectory alignment fails
        # explicitly at stage 'align' inside the geometry stages.
        pred_trajectory: Path | None = None
        if manifest is not None and scene_id in manifest.scenes:
            traj = manifest.scenes[scene_id].trajectory
            if traj is not None:
                pred_trajectory = traj if traj.is_absolute() else pred_root / traj
        gt_trajectory = scene.gt_trajectory
    except (DatasetError, InvalidGeometryError) as exc:
        return SceneOutcome(
            scene_id=scene_id,
            failure=SceneFailure(scene_id=scene_id, stage="resolve", reason=str(exc)),
        )

    return evaluate_geometry_scene(
        scene_id, pred_path, gt_path,
        protocol=protocol, protocol_hash=protocol_hash,
        input_type=pred_kind, gt_type=gt_kind, registry=registry,  # type: ignore[arg-type]
        pred_unit=pred_unit, gt_unit=gt_unit, pred_world_frame=pred_world_frame,
        pred_trajectory=pred_trajectory, gt_trajectory=gt_trajectory,
        base_seed=base_seed,
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
    adapt: str | None = None,
    progress: Any | None = None,
    base_seed: int = DEFAULT_BASE_SEED,
) -> BenchmarkRunOutput:
    """Evaluate every scene in ``split`` and assemble a benchmark ``RunResult``.

    ``base_seed`` is run configuration (not protocol state): protocols with
    ``seed: derive`` derive each scene's sampling seed from this base, so overriding
    it quantifies sampling sensitivity across repeated runs. The value used is
    recorded in the run config and result metadata. Protocols pinning an explicit
    integer seed ignore it.
    """
    registry = registry or default_registry()
    pred_root = Path(pred_root)

    preflight(adapter, split, protocol)

    # Adapters whose GT geometry depends on the protocol (e.g. Neural-RGBD's culled vs source
    # mesh variant) can bind the protocol up front, because the interface's load_scene() is not
    # handed a protocol. This is an opt-in hook; the dataset-specific decision stays in the
    # adapter and adapters without protocol-dependent GT simply do not implement it.
    bind_protocol = getattr(adapter, "bind_protocol", None)
    if callable(bind_protocol):
        bind_protocol(protocol)

    scenes = list(adapter.iter_scenes(split))
    phash = compute_protocol_hash(protocol)
    manifest, manifest_dict, inferred = load_or_infer_manifest(
        pred_root, scenes, protocol,
        manifest_path=Path(manifest_path) if manifest_path else None,
    )
    # An inferred manifest is only a naive <scene>.ply fallback; it must not override
    # an adapter's own filename resolution (e.g. DTU's <method>NNN_l3.ply). Adapters
    # get None when inferred and fall back to their own inference; the inferred
    # manifest is still written to the run directory for the record.
    resolve_manifest = None if inferred else manifest
    adaptation_override = adaptation_override_from_legacy(adapt=adapt)
    run_protocol, adaptation = resolve_adaptation(
        protocol, resolve_manifest, adaptation_override
    )
    has_manifest_adaptation = resolve_manifest is not None
    pred_unit_override = (
        adaptation.unit
        if has_manifest_adaptation
        or (adaptation_override is not None and adaptation_override.unit is not None)
        else None
    )
    pred_world_frame_override = (
        adaptation.world_frame
        if has_manifest_adaptation
        or (
            adaptation_override is not None
            and (
                adaptation_override.world_frame is not None
                or adaptation_override.axes is not None
            )
        )
        else None
    )
    if adapt is not None:
        manifest_dict["adapt_override"] = adapt

    official_name = run_protocol.backend_preferences.get("official_eval")
    # Official evaluators come in three shapes: point-array (DTU port, ObsMask),
    # file-based/artifacts (Tanks and Temples toolbox, which reads files + runs ICP),
    # and scan-MLP (ETH3D multi-view-evaluation binary on prediction PLY + .mlp).
    official_input_mode = "point_arrays"
    if official_name:
        official_input_mode = getattr(
            registry.require("official_eval", official_name), "input_mode", "point_arrays"
        )
    # gt_visibility culling is realized by the render+TSDF-trim 'visibility' backend.
    visibility_culling = run_protocol.masking.pred_culling.method == "gt_visibility"

    import tempfile

    per_scene: list[MetricResult] = []
    failures: list[SceneFailure] = []
    alignment_transforms: list[dict[str, Any]] = []
    debug_scenes: list[tuple[str, Any]] = []
    alignment_vis: list[Any] = []
    trajectory_alignment_vis: list[Any] = []
    evaluated = 0
    tnt_out_root = (
        Path(tempfile.mkdtemp(prefix="eval3r_tnt_"))
        if official_name and official_input_mode == "artifacts"
        else None
    )

    for scene_id in scenes:
        if official_name and official_input_mode == "artifacts":
            assert tnt_out_root is not None
            outcome = _evaluate_scene_tnt_official(
                adapter, pred_root, resolve_manifest, scene_id, run_protocol, phash,
                registry, official_name, tnt_out_root,
            )
        elif official_name and official_input_mode == "scan_mlp":
            outcome = _evaluate_scene_eth3d_official(
                adapter, pred_root, resolve_manifest, scene_id, run_protocol, phash,
                registry, official_name,
            )
        elif official_name:
            outcome = _evaluate_scene_official(
                adapter, pred_root, resolve_manifest, scene_id, run_protocol, phash,
                registry, official_name, base_seed,
            )
        elif visibility_culling:
            outcome = _evaluate_scene_visibility_culled(
                adapter, pred_root, resolve_manifest, scene_id, run_protocol, phash,
                registry, base_seed,
                pred_unit_override=pred_unit_override,
                pred_world_frame_override=pred_world_frame_override,
            )
        else:
            outcome = _evaluate_scene(
                adapter, pred_root, resolve_manifest, scene_id, run_protocol, phash, registry,
                pred_unit_override=pred_unit_override,
                pred_world_frame_override=pred_world_frame_override,
                base_seed=base_seed,
            )
        if progress is not None:
            progress(scene_id, outcome)

        if outcome.failure is None:
            evaluated += 1
            per_scene.extend(outcome.metrics)
            if outcome.alignment is not None:
                alignment_transforms.append(outcome.alignment.as_dict())
            if outcome.debug is not None:
                debug_scenes.append((scene_id, outcome.debug))
            if outcome.alignment_vis is not None:
                alignment_vis.append(outcome.alignment_vis)
            if outcome.trajectory_alignment_vis is not None:
                trajectory_alignment_vis.append(outcome.trajectory_alignment_vis)
            continue

        # failed scene: apply the protocol's failure policy.
        policy = run_protocol.failure_policy.policy
        if policy == "abort":
            raise SceneEvaluationError(scene_id, outcome.failure.stage, outcome.failure.reason)
        failures.append(outcome.failure)
        if policy == "score_worst":
            per_scene.extend(_worst_results(scene_id, run_protocol, phash))

    from eval3r.pipeline.stages.aggregate import aggregate_scene_metrics

    metrics = aggregate_scene_metrics(per_scene, run_protocol.metrics)
    used_backends = {
        "nearest_neighbor": run_protocol.backend_preferences.get("nearest_neighbor", "scipy"),
        "pointcloud": run_protocol.backend_preferences.get("pointcloud", "plyfile"),
        "mesh": run_protocol.backend_preferences.get("mesh", "trimesh"),
    }
    if official_name:
        used_backends["official_eval"] = official_name
    if visibility_culling:
        used_backends["visibility"] = run_protocol.backend_preferences.get(
            "visibility", "render_tsdf"
        )
    if run_protocol.alignment.mode != "none" and run_protocol.alignment.solver == "icp":
        used_backends["registration"] = run_protocol.backend_preferences.get(
            "registration", "open3d"
        )
    if run_protocol.alignment.mode != "none" and run_protocol.alignment.estimate_on == "trajectory":
        used_backends["trajectory"] = run_protocol.backend_preferences.get("trajectory", "evo")
    backend_versions = registry.backend_versions(used_backends)

    result = RunResult(
        schema_version=run_protocol.schema_version,
        eval3r_version=_eval3r_version(),
        method=method or (manifest.method if manifest else None),
        method_version=manifest.version if manifest else None,
        dataset=run_protocol.dataset,
        split=split,
        protocol=run_protocol.name,
        protocol_version=run_protocol.protocol_version,
        protocol_hash=phash,
        fidelity=run_protocol.fidelity,
        ground_truth=run_protocol.ground_truth,
        local_evaluation=run_protocol.local_evaluation,
        n_scenes_expected=len(scenes),
        n_scenes_evaluated=evaluated,
        failed_scenes=failures,
        failure_policy=run_protocol.failure_policy,
        metrics=metrics,
        metric_definitions=run_protocol.metrics,
        per_scene_metrics=per_scene,
        confidence_policy=run_protocol.confidence,
        alignment=run_protocol.alignment,
        adaptation=adaptation,
        masking=run_protocol.masking,
        sampling=run_protocol.sampling,
        aggregation=run_protocol.aggregation,
        uses_gt=manifest.uses_gt if manifest else None,
        backend_versions=backend_versions,
        environment=environment or {},
        command=command,
        manifest_path=Path(manifest_path) if manifest_path else None,
        timestamp=_now_utc_iso(),
        metadata={"manifest_inferred": inferred, "base_seed": base_seed},
    )

    config = {
        "protocol": run_protocol.name,
        "protocol_hash": phash,
        "dataset": adapter.name,
        "split": split,
        "pred_root": str(pred_root),
        "manifest_inferred": inferred,
        "n_scenes": len(scenes),
        "adaptation": adaptation.model_dump(mode="json"),
        "base_seed": base_seed,
    }

    return BenchmarkRunOutput(
        result=result,
        protocol=run_protocol,
        protocol_hash=phash,
        alignment_transforms=alignment_transforms,
        manifest=manifest_dict,
        manifest_inferred=inferred,
        config=config,
        adaptation=adaptation,
        debug_scenes=debug_scenes,
        alignment_vis=alignment_vis,
        trajectory_alignment_vis=trajectory_alignment_vis,
    )


def _eval3r_version() -> str:
    from eval3r import __version__

    return __version__
