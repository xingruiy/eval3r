"""Benchmark orchestrator: dataset adapter + locator + per-scene evaluate_geometry."""

from __future__ import annotations

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from eval3r.align import AlignMode
from eval3r.benchmark.aggregate import aggregate, aggregate_all
from eval3r.datasets.base import Asset, DatasetAdapter
from eval3r.io.geometry import (
    MeshData,
    PointCloudData,
    load_mesh,
    load_point_cloud,
)
from eval3r.metrics.geometry import (
    ChamferVariant,
    GeometryEvalResult,
    MaskMode,
    evaluate_geometry,
)
from eval3r.metrics.sampling import SampleMethod
from eval3r.prediction.discovery import PredictionLocator, ResolvedPrediction
from eval3r.utils.errors import MissingArtifactError
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
    mask_name: str = "occlusion_mask.npy"
    t_mask_scene_name: str = "T_mask_scene.txt"
    mask_mode: MaskMode = "pred"


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
# Worker — must be top-level so ProcessPoolExecutor can pickle it.
# ---------------------------------------------------------------------------


def _crop_to_bbox(
    points: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, margin: float
) -> np.ndarray:
    lo = bbox_min - margin
    hi = bbox_max + margin
    mask = np.all((points >= lo) & (points <= hi), axis=1)
    return points[mask]


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
    gt_poses: np.ndarray | None = None,
    gt_pose_convention: str = "unspecified",
    gt_timestamps: np.ndarray | None = None,
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

        if config.crop_to_gt_bbox:
            gv = gt_geom.vertices if isinstance(gt_geom, MeshData) else gt_geom.points
            bbox_min, bbox_max = gv.min(axis=0), gv.max(axis=0)
            if isinstance(pred_geom, MeshData):
                kept = _crop_to_bbox(pred_geom.vertices, bbox_min, bbox_max, config.bbox_margin)
                pred_geom = PointCloudData(points=kept)
            else:
                pred_geom = PointCloudData(
                    points=_crop_to_bbox(
                        pred_geom.points, bbox_min, bbox_max, config.bbox_margin
                    )
                )

        # Trajectory-based alignment: load pred poses from manifest or
        # from an explicit external pose directory.
        pred_poses: np.ndarray | None = None
        pred_pose_convention = "unspecified"
        pred_timestamps: np.ndarray | None = None
        if isinstance(config.align, str) and config.align.startswith("traj_"):
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
        gt_mask = None
        mask_missing = False
        if config.mask_dir is not None:
            from eval3r.metrics.occlusion import load_occlusion_mask

            mask_path = Path(config.mask_dir) / scene_id / config.mask_name
            w2g_path = Path(config.mask_dir) / scene_id / config.t_mask_scene_name
            if mask_path.exists() and w2g_path.exists():
                mask_obj = load_occlusion_mask(mask_path, w2g_path)
                if config.mask_mode == "pred":
                    pred_mask = mask_obj
                elif config.mask_mode == "gt":
                    gt_mask = mask_obj
                elif config.mask_mode == "both":
                    pred_mask = mask_obj
                    gt_mask = mask_obj
            else:
                mask_missing = True

        result = evaluate_geometry(
            pred_geom,
            gt_geom,
            samples=config.samples,
            seed=config.seed,
            sample_method=config.sample_method,
            align_mode=config.align,
            thresholds=config.thresholds,
            chamfer_variant=config.chamfer_variant,
            debug_plot_path=debug_plot_path,
            pred_poses=pred_poses,
            gt_poses=gt_poses,
            pred_convention=pred_pose_convention,
            gt_convention=gt_pose_convention,
            pred_timestamps=pred_timestamps,
            gt_timestamps=gt_timestamps,
            pred_mask=pred_mask,
            gt_mask=gt_mask,
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
    loc = locator or PredictionLocator(preds_root=Path(preds_root))
    scenes = dataset.list_scenes(split)
    if split is not None:
        split_label = str(split)
    else:
        adapter_split = getattr(dataset, "_split", None)
        split_label = str(adapter_split) if adapter_split is not None else "auto"

    # Resolve preds + GT paths upfront; that way workers don't share adapter
    # state across processes.
    want_traj = isinstance(cfg.align, str) and cfg.align.startswith("traj_")
    gt_asset = _gt_asset(dataset)
    jobs: list[tuple[str, dict[str, Any] | None, Path | None, Asset, np.ndarray | None, str, np.ndarray | None]] = []
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
        gt_poses_arr: np.ndarray | None = None
        gt_pose_conv = "unspecified"
        gt_ts_arr: np.ndarray | None = None
        if want_traj and gt_path is not None:
            try:
                traj = dataset.load_poses(sid)
                gt_poses_arr = traj.poses
                gt_pose_conv = traj.convention
                gt_ts_arr = traj.timestamps
            except Exception:
                pass
        jobs.append((sid, _pred_descriptor(rp), gt_path, gt_asset, gt_poses_arr, gt_pose_conv, gt_ts_arr))

    outcomes: list[SceneOutcome] = []
    if cfg.workers <= 1 or len(jobs) <= 1:
        for j in jobs:
            outcomes.append(_evaluate_one(*j, cfg))  # type: ignore[arg-type]
            if progress:
                o = outcomes[-1]
                _log.info("benchmark: %s -> %s", o.scene_id, o.status)
                if cfg.verbose:
                    if o.status == "failed" and o.error:
                        for line in o.error.rstrip().split("\n"):
                            _log.info("benchmark:   %s", line)
                    elif o.status == "missing_pred":
                        _log.info("benchmark:   prediction not found")
                    elif o.status == "missing_gt":
                        _log.info("benchmark:   %s", o.gt_path)
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futures = {ex.submit(_evaluate_one, *j, cfg): j[0] for j in jobs}
            for fut in as_completed(futures):
                outcomes.append(fut.result())
                if progress:
                    o = outcomes[-1]
                    _log.info("benchmark: %s -> %s", o.scene_id, o.status)
                    if cfg.verbose:
                        if o.status == "failed" and o.error:
                            for line in o.error.rstrip().split("\n"):
                                _log.info("benchmark:   %s", line)
                        elif o.status == "missing_pred":
                            _log.info("benchmark:   prediction not found")
                        elif o.status == "missing_gt":
                            _log.info("benchmark:   %s", o.gt_path)
        outcomes.sort(key=lambda o: scenes.index(o.scene_id))

    coverage = {
        "n_total": len(scenes),
        "n_evaluated": sum(1 for o in outcomes if o.status == "ok"),
        "n_missing_pred": sum(1 for o in outcomes if o.status == "missing_pred"),
        "n_missing_gt": sum(1 for o in outcomes if o.status == "missing_gt"),
        "n_failed": sum(1 for o in outcomes if o.status == "failed"),
    }
    return BenchmarkResult(
        dataset=dataset.name,
        split=split_label,
        scenes=outcomes,
        summary=aggregate(outcomes, thresholds=cfg.thresholds),
        summary_all=aggregate_all(
            outcomes,
            n_total=len(scenes),
            thresholds=cfg.thresholds,
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
            "workers": cfg.workers,
            "missing_distance_default": cfg.missing_distance_default,
            "missing_fscore_default": cfg.missing_fscore_default,
            "pred_pose_dir": cfg.pred_pose_dir,
            "pred_pose_file": cfg.pred_pose_file,
            "pred_pose_convention": cfg.pred_pose_convention,
            "verbose": cfg.verbose,
            "mask_dir": cfg.mask_dir,
            "mask_name": cfg.mask_name,
            "t_mask_scene_name": cfg.t_mask_scene_name,
            "mask_mode": cfg.mask_mode,
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
