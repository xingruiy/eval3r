"""Benchmark orchestrator: dataset adapter + locator + per-scene evaluate_geometry."""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

import numpy as np

from eval3r.align import AlignMode
from eval3r.benchmark.aggregate import aggregate, aggregate_all
from eval3r.datasets.base import Asset, DatasetAdapter
from eval3r.mask.base import CropToGT
from eval3r.mask.crop import CropVolume
from eval3r.io.geometry import (
    MeshData,
    PointCloudData,
    load_mesh,
    load_point_cloud,
)
from eval3r.metric.geometry import (
    ChamferVariant,
    GeometryEvalResult,
    evaluate_geometry,
)
from eval3r.metric.sampling import SampleMethod
from eval3r.prediction.discovery import PredictionLocator, ResolvedPrediction
from eval3r.utils.errors import MissingArtifactError, NotSupportedError
from eval3r.utils.logging import get_logger
from eval3r.utils.typing import PathLike

_log = get_logger(__name__)


@dataclass
class BenchmarkConfig:
    samples: int = 200_000
    seed: int = 42
    sample_method: SampleMethod = "area"
    align: AlignMode = "none"
    thresholds: tuple[float, ...] = (0.05,)
    chamfer_variant: ChamferVariant = "l1_mean_bidirectional"
    crop_to_gt_bbox: bool = False
    bbox_margin: float = 0.10
    # If the dataset adapter exposes a per-scene crop volume (e.g. T&T's
    # ``{scene}.json``), clip the prediction to it before metrics. No-op
    # for adapters whose ``load_crop_volume`` raises NotSupportedError.
    crop_to_eval_region: bool = True
    # If True and the dataset adapter exposes ``load_thresholds(scene_id)``
    # (e.g. T&T's published per-scene τ), use those instead of
    # ``cfg.thresholds`` for that scene. When scenes use heterogeneous τ,
    # the aggregator pools them under canonical ``f`` / ``precision`` /
    # ``recall`` columns so every scene contributes its own scene-specific
    # f-score to the same list (matches the official T&T protocol).
    use_dataset_thresholds: bool = True
    # Scalar applied to every threshold (per-scene τ from the adapter or
    # the global ``thresholds`` fallback) before metrics. Useful for
    # sensitivity studies — e.g. ``threshold_multiplier=2.0`` scores T&T
    # at 2× the published τ. Set to 1.0 to disable.
    threshold_multiplier: float = 1.0
    debug_plot: bool = False
    fail_on_missing: bool = False
    workers: int = field(default_factory=lambda: min(8, os.cpu_count() or 1))
    # Defaults applied to ``summary_all`` for scenes whose status != "ok".
    # Distance metrics get penalised; f-score / precision / recall go to 0.
    missing_distance_default: float = 1.0
    missing_fscore_default: float = 0.0
    # External pose directory for non-manifest predictions.
    pred_pose_dir: str | None = None
    pred_pose_file: str = "{scene_id}.txt"
    pred_pose_convention: str = "unspecified"
    verbose: bool = False
    # Occlusion mask for filtering predicted points in unobserved regions.
    mask_dir: str | None = None
    mask_pattern: str = "{scene_id}/occlusion_mask.npy"
    t_mask_scene_pattern: str = "{scene_id}/T_mask_scene.txt"


SceneStatus = Literal["ok", "missing_pred", "missing_gt", "failed"]


@dataclass
class SceneOutcome:
    scene_id: str
    status: SceneStatus
    result: GeometryEvalResult | None = None
    error: str | None = None
    pred_path: Path | None = None
    gt_path: Path | None = None
    mask_missing: bool = False


@dataclass
class BenchmarkResult:
    dataset: str
    split: str
    scenes: list[SceneOutcome]
    summary: dict[str, dict[str, float]]
    """Mean / median / std / n over **successful** scenes only."""
    summary_all: dict[str, dict[str, float]]
    """Mean / median / std / n over **all** scenes; missing scenes get the
    configured defaults (distance → ``missing_distance_default``,
    f-score / precision / recall → ``missing_fscore_default``)."""
    coverage: dict[str, int]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "scenes": [
                {
                    "scene_id": o.scene_id,
                    "status": o.status,
                    "pred_path": str(o.pred_path) if o.pred_path else None,
                    "gt_path": str(o.gt_path) if o.gt_path else None,
                    "error": o.error,
                    "mask_missing": o.mask_missing,
                    "result": o.result.to_dict() if o.result else None,
                }
                for o in self.scenes
            ],
            "summary": self.summary,
            "summary_all": self.summary_all,
            "coverage": self.coverage,
            "config": self.config,
        }


# ---------------------------------------------------------------------------
# Worker — must be top-level so multiprocessing can pickle it.
# ---------------------------------------------------------------------------


def _resolve_mask_pattern(mask_dir: str, pattern: str, scene_id: str) -> Path:
    rel = Path(pattern.format(scene_id=scene_id))
    if rel.is_absolute():
        raise ValueError("Mask path patterns must be relative to mask_dir.")
    return Path(mask_dir) / rel


def _load_pred_geometry(
    rp: ResolvedPrediction,
) -> MeshData | PointCloudData:
    if rp["kind"] == "manifest":
        reader = rp["reader"]
        assert reader is not None
        try:
            return reader.mesh
        except MissingArtifactError:
            return reader.points
    if rp["kind"] == "mesh_file":
        return load_mesh(rp["path"])
    return load_point_cloud(rp["path"])


def _evaluate_one(
    scene_id: str,
    pred_descriptor: dict[str, Any] | None,
    gt_path: Path | None,
    gt_asset: Asset,
    gt_pose_loader_payload: _PoseLoaderPayload | None = None,
    crop_volume: CropVolume | None = None,
    scene_thresholds: tuple[float, ...] | None = None,
    config: BenchmarkConfig | None = None,
) -> SceneOutcome:
    if pred_descriptor is None:
        return SceneOutcome(scene_id=scene_id, status="missing_pred")
    if gt_path is None or not gt_path.exists():
        return SceneOutcome(
            scene_id=scene_id, status="missing_gt", gt_path=gt_path
        )
    try:
        # Re-hydrate ResolvedPrediction across the process boundary.
        rp: ResolvedPrediction = ResolvedPrediction(
            scene_id=scene_id,
            kind=pred_descriptor["kind"],
            path=Path(pred_descriptor["path"]),
            reader=None,
        )
        if rp["kind"] == "manifest":
            from eval3r.prediction.reader import PredictionReader

            rp["reader"] = PredictionReader(rp["path"], verify_hashes=False)

        pred_geom = _load_pred_geometry(rp)
        if gt_asset is Asset.POINT_CLOUD:
            gt_geom = load_point_cloud(gt_path)
        else:
            gt_geom = load_mesh(gt_path)

        # Trajectory-based alignment: load poses in the worker so the
        # parent process does not eagerly parse pose files for every scene.
        pred_poses: np.ndarray | None = None
        pred_pose_convention = "unspecified"
        pred_timestamps: np.ndarray | None = None
        gt_poses: np.ndarray | None = None
        gt_pose_convention = "unspecified"
        gt_timestamps: np.ndarray | None = None
        if isinstance(config.align, str) and config.align.startswith("traj_"):
            if gt_pose_loader_payload is not None:
                cls_mod = importlib.import_module(gt_pose_loader_payload["module"])
                cls_obj: Any = cls_mod
                for part in gt_pose_loader_payload["qualname"].split("."):
                    cls_obj = getattr(cls_obj, part)
                adapter = cls_obj.__new__(cls_obj)
                adapter.__dict__.update(gt_pose_loader_payload["state"])
                gt_traj = adapter.load_poses(scene_id)
                gt_poses = gt_traj.poses
                gt_pose_convention = gt_traj.convention
                gt_timestamps = gt_traj.timestamps

            if rp["kind"] == "manifest" and rp["reader"] is not None:
                try:
                    traj = rp["reader"].poses
                    pred_poses = traj.poses
                    pred_pose_convention = traj.convention
                    pred_timestamps = traj.timestamps
                except MissingArtifactError:
                    pass
            elif config.pred_pose_dir is not None:
                from eval3r.io.trajectory import load_trajectory_auto

                pose_path = Path(config.pred_pose_dir) / config.pred_pose_file.format(
                    scene_id=scene_id
                )
                if not pose_path.exists():
                    raise MissingArtifactError(
                        f"Pose file not found at {pose_path} for scene {scene_id!r}. "
                        f"Check --pred-pose-dir and --pred-pose-file."
                    )
                traj = load_trajectory_auto(
                    pose_path, convention=config.pred_pose_convention
                )
                pred_poses = traj.poses
                pred_pose_convention = traj.convention
                pred_timestamps = traj.timestamps
            else:
                raise MissingArtifactError(
                    f"Trajectory alignment ({config.align}) requires poses, "
                    f"but prediction for scene {scene_id!r} is a raw file "
                    f"(no manifest). Provide --pred-pose-dir so poses can "
                    f"be located."
                )

        debug_plot_path: str | None = None
        if config.debug_plot:
            os.makedirs("debug_plots", exist_ok=True)
            debug_plot_path = f"debug_plots/{scene_id}.png"

        pred_mask = None
        mask_missing = False
        if config.mask_dir is not None:
            from eval3r.mask.occlusion import load_occlusion_mask

            mask_path = _resolve_mask_pattern(
                config.mask_dir, config.mask_pattern, scene_id
            )
            w2g_path = _resolve_mask_pattern(
                config.mask_dir, config.t_mask_scene_pattern, scene_id
            )
            if mask_path.exists() and w2g_path.exists():
                pred_mask = load_occlusion_mask(mask_path, w2g_path)
            else:
                mask_missing = True
        if pred_mask is None and crop_volume is not None:
            pred_mask = crop_volume
        if pred_mask is None and config.crop_to_gt_bbox:
            gv = gt_geom.vertices if isinstance(gt_geom, MeshData) else gt_geom.points
            pred_mask = CropToGT(
                bbox_min=gv.min(axis=0),
                bbox_max=gv.max(axis=0),
                margin=config.bbox_margin,
            )

        result = evaluate_geometry(
            pred_geom,
            gt_geom,
            samples=config.samples,
            seed=config.seed,
            sample_method=config.sample_method,
            align_mode=config.align,
            thresholds=(
                scene_thresholds if scene_thresholds is not None else config.thresholds
            ),
            chamfer_variant=config.chamfer_variant,
            debug_plot_path=debug_plot_path,
            pred_poses=pred_poses,
            gt_poses=gt_poses,
            pred_convention=pred_pose_convention,
            gt_convention=gt_pose_convention,
            pred_timestamps=pred_timestamps,
            gt_timestamps=gt_timestamps,
            pred_mask=pred_mask,
        )
        return SceneOutcome(
            scene_id=scene_id,
            status="ok",
            result=result,
            pred_path=rp["path"],
            gt_path=gt_path,
            mask_missing=mask_missing,
        )
    except Exception:
        return SceneOutcome(
            scene_id=scene_id,
            status="failed",
            error=traceback.format_exc(limit=8),
            pred_path=Path(pred_descriptor["path"]) if pred_descriptor else None,
            gt_path=gt_path,
        )


def _pred_descriptor(rp: ResolvedPrediction | None) -> dict[str, Any] | None:
    if rp is None:
        return None
    return {"kind": rp["kind"], "path": str(rp["path"])}


class _PoseLoaderPayload(TypedDict):
    module: str
    qualname: str
    state: dict[str, Any]


BenchmarkJob = tuple[
    str,
    dict[str, Any] | None,
    Path | None,
    Asset,
    _PoseLoaderPayload | None,
    CropVolume | None,
    tuple[float, ...] | None,
]


def _evaluate_one_process(
    result_queue: mp.Queue, job: BenchmarkJob, config: BenchmarkConfig
) -> None:
    result_queue.put(_evaluate_one(*job, config))


def _abrupt_worker_failure(job: BenchmarkJob, exitcode: int | None) -> SceneOutcome:
    return SceneOutcome(
        scene_id=job[0],
        status="failed",
        error=f"Worker process exited abruptly with exit code {exitcode}.",
        pred_path=Path(job[1]["path"]) if job[1] else None,
        gt_path=job[2],
    )


def _run_jobs_parallel(
    jobs: list[BenchmarkJob],
    cfg: BenchmarkConfig,
    *,
    on_outcome: Callable[[SceneOutcome], None],
) -> list[SceneOutcome]:
    ctx = mp.get_context()
    active: list[tuple[mp.Process, mp.Queue, BenchmarkJob]] = []
    outcomes: list[SceneOutcome] = []
    next_job = 0
    max_workers = max(1, cfg.workers)

    def start_more() -> None:
        nonlocal next_job
        while next_job < len(jobs) and len(active) < max_workers:
            job = jobs[next_job]
            next_job += 1
            result_queue = ctx.Queue(maxsize=1)
            proc = ctx.Process(
                target=_evaluate_one_process,
                args=(result_queue, job, cfg),
            )
            proc.start()
            active.append((proc, result_queue, job))

    def finish(
        proc: mp.Process,
        result_queue: mp.Queue,
        job: BenchmarkJob,
        outcome: SceneOutcome,
    ) -> None:
        proc.join()
        result_queue.close()
        result_queue.join_thread()
        active.remove((proc, result_queue, job))
        outcomes.append(outcome)
        on_outcome(outcome)
        start_more()

    try:
        start_more()
        while active:
            made_progress = False
            for proc, result_queue, job in list(active):
                try:
                    outcome = result_queue.get_nowait()
                except queue.Empty:
                    outcome = None

                if outcome is not None:
                    finish(proc, result_queue, job, outcome)
                    made_progress = True
                    continue

                if not proc.is_alive():
                    proc.join()
                    try:
                        outcome = result_queue.get(timeout=0.2)
                    except queue.Empty:
                        outcome = _abrupt_worker_failure(job, proc.exitcode)
                    result_queue.close()
                    result_queue.join_thread()
                    active.remove((proc, result_queue, job))
                    outcomes.append(outcome)
                    on_outcome(outcome)
                    start_more()
                    made_progress = True

            if not made_progress and active:
                time.sleep(0.05)
    except BaseException:
        for proc, result_queue, _job in active:
            if proc.is_alive():
                proc.terminate()
            proc.join()
            result_queue.close()
            result_queue.join_thread()
        raise

    return outcomes


def _gt_pose_loader_payload(
    dataset: DatasetAdapter, scene_id: str, align: AlignMode
) -> _PoseLoaderPayload | None:
    if not (isinstance(align, str) and align.startswith("traj_")):
        return None
    # Avoid resolving Asset.POSES in the parent process. Some adapters derive
    # trajectories without exposing a standalone pose asset path, and workers
    # should still use adapter-specific load_poses(scene_id) parsing.
    return {
        "module": dataset.__class__.__module__,
        "qualname": dataset.__class__.__qualname__,
        "state": dict(dataset.__dict__),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_benchmark(
    dataset: DatasetAdapter,
    preds_root: PathLike,
    *,
    locator: PredictionLocator | None = None,
    config: BenchmarkConfig | None = None,
    split: str | PathLike | None = None,
    progress: bool = True,
) -> BenchmarkResult:
    cfg = config or BenchmarkConfig()
    if cfg.threshold_multiplier <= 0.0:
        raise ValueError(
            f"threshold_multiplier must be positive, got {cfg.threshold_multiplier}"
        )
    # Effective fallback thresholds = cfg.thresholds × multiplier. Used
    # when the adapter doesn't expose per-scene τ, and as the seed list
    # for the aggregator's pre-seeded f@<thr> columns.
    effective_default_thresholds = tuple(
        float(t) * cfg.threshold_multiplier for t in cfg.thresholds
    )
    loc = locator or PredictionLocator(preds_root=Path(preds_root))
    scenes = dataset.list_scenes(split)
    if split is not None:
        split_label = str(split)
    else:
        adapter_split = getattr(dataset, "_split", None)
        split_label = str(adapter_split) if adapter_split is not None else "auto"

    # Resolve preds + GT paths upfront. Heavier per-scene data such as
    # trajectory arrays is loaded inside the worker.
    gt_asset = _gt_asset(dataset)
    jobs: list[BenchmarkJob] = []
    for sid in scenes:
        rp = loc.resolve(sid)
        if rp is None and cfg.fail_on_missing:
            raise MissingArtifactError(
                f"No prediction found for scene {sid!r} under {loc.preds_root}"
            )
        try:
            gt_path = dataset.asset_path(sid, gt_asset)
        except Exception:
            gt_path = None
        crop_vol: CropVolume | None = None
        if cfg.crop_to_eval_region and gt_path is not None:
            try:
                crop_vol = dataset.load_crop_volume(sid)
            except NotSupportedError:
                pass
            except Exception:
                # Treat any other resolve failure (missing file, malformed
                # JSON) the same way per-scene: log via verbose path and
                # carry on without cropping rather than aborting the run.
                if cfg.verbose:
                    _log.info(
                        "benchmark: %s -> crop volume unavailable", sid
                    )
        scene_thr: tuple[float, ...] | None = None
        if cfg.use_dataset_thresholds and gt_path is not None:
            try:
                scene_thr = tuple(
                    float(t) * cfg.threshold_multiplier
                    for t in dataset.load_thresholds(sid)
                )
            except NotSupportedError:
                pass
        # If the adapter didn't supply τ, fall back to the global
        # thresholds with the multiplier applied. Keeping this in the
        # job tuple means workers don't need to know about the
        # multiplier separately.
        if scene_thr is None and cfg.threshold_multiplier != 1.0:
            scene_thr = effective_default_thresholds
        jobs.append(
            (
                sid,
                _pred_descriptor(rp),
                gt_path,
                gt_asset,
                _gt_pose_loader_payload(dataset, sid, cfg.align),
                crop_vol,
                scene_thr,
            )
        )

    def _log_outcome(o: SceneOutcome) -> None:
        _log.info("benchmark: %s -> %s", o.scene_id, o.status)
        # `failed` is exceptional — always surface the traceback so the
        # user can act on it. Other statuses are routine; gate their
        # details behind --verbose to avoid noise on large datasets.
        if o.status == "failed" and o.error:
            for line in o.error.rstrip().split("\n"):
                _log.warning("benchmark:   %s", line)
        elif cfg.verbose:
            if o.status == "missing_pred":
                _log.info("benchmark:   prediction not found")
            elif o.status == "missing_gt":
                _log.info("benchmark:   %s", o.gt_path)

    outcomes: list[SceneOutcome] = []
    if cfg.workers <= 1 or len(jobs) <= 1:
        for j in jobs:
            outcomes.append(_evaluate_one(*j, cfg))  # type: ignore[arg-type]
            if progress:
                _log_outcome(outcomes[-1])
    else:
        outcomes = _run_jobs_parallel(
            jobs,
            cfg,
            on_outcome=_log_outcome if progress else lambda _: None,
        )
        outcomes.sort(key=lambda o: scenes.index(o.scene_id))

    coverage = {
        "n_total": len(scenes),
        "n_evaluated": sum(1 for o in outcomes if o.status == "ok"),
        "n_missing_pred": sum(1 for o in outcomes if o.status == "missing_pred"),
        "n_missing_gt": sum(1 for o in outcomes if o.status == "missing_gt"),
        "n_failed": sum(1 for o in outcomes if o.status == "failed"),
    }

    # Pool f-score columns across τ values when scenes used heterogeneous
    # per-scene thresholds (e.g. T&T scene-tau). Pooling collapses all
    # per-τ buckets into canonical ``f`` / ``precision`` / ``recall`` so
    # each scene contributes its own scene-τ result to the same list.
    used_thresholds: set[float] = set(effective_default_thresholds)
    for j in jobs:
        st = j[6]  # scene_thresholds slot
        if st is not None:
            used_thresholds.update(float(t) for t in st)
    pooled = (
        any(j[6] is not None for j in jobs)
        and any(
            j[6] is not None and tuple(j[6]) != effective_default_thresholds
            for j in jobs
        )
        and len(used_thresholds) > 1
    )

    return BenchmarkResult(
        dataset=dataset.name,
        split=split_label,
        scenes=outcomes,
        summary=aggregate(
            outcomes, thresholds=effective_default_thresholds, pooled=pooled
        ),
        summary_all=aggregate_all(
            outcomes,
            n_total=len(scenes),
            thresholds=effective_default_thresholds,
            pooled=pooled,
            distance_default=cfg.missing_distance_default,
            fscore_default=cfg.missing_fscore_default,
        ),
        coverage=coverage,
        config={
            "samples": cfg.samples,
            "seed": cfg.seed,
            "sample_method": cfg.sample_method,
            "align": cfg.align,
            "thresholds": list(cfg.thresholds),
            "chamfer_variant": cfg.chamfer_variant,
            "crop_to_gt_bbox": cfg.crop_to_gt_bbox,
            "bbox_margin": cfg.bbox_margin,
            "crop_to_eval_region": cfg.crop_to_eval_region,
            "use_dataset_thresholds": cfg.use_dataset_thresholds,
            "threshold_multiplier": cfg.threshold_multiplier,
            "thresholds_pooled": pooled,
            "workers": cfg.workers,
            "missing_distance_default": cfg.missing_distance_default,
            "missing_fscore_default": cfg.missing_fscore_default,
            "pred_pose_dir": cfg.pred_pose_dir,
            "pred_pose_file": cfg.pred_pose_file,
            "pred_pose_convention": cfg.pred_pose_convention,
            "verbose": cfg.verbose,
            "mask_dir": cfg.mask_dir,
            "mask_pattern": cfg.mask_pattern,
            "t_mask_scene_pattern": cfg.t_mask_scene_pattern,
        },
    )


def _gt_asset(dataset: DatasetAdapter):  # type: ignore[no-untyped-def]
    from eval3r.datasets.base import Asset

    if dataset.supports(Asset.MESH):
        return Asset.MESH
    if dataset.supports(Asset.POINT_CLOUD):
        return Asset.POINT_CLOUD
    raise MissingArtifactError(
        f"Dataset adapter {dataset.name!r} supports neither mesh nor point_cloud GT."
    )
