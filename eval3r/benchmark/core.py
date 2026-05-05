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
from eval3r.benchmark.aggregate import aggregate
from eval3r.datasets.base import DatasetAdapter
from eval3r.io.geometry import (
    MeshData,
    PointCloudData,
    load_mesh,
    load_point_cloud,
)
from eval3r.metrics.geometry import (
    ChamferVariant,
    GeometryEvalResult,
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
    fail_on_missing: bool = False
    workers: int = field(default_factory=lambda: min(8, os.cpu_count() or 1))


SceneStatus = Literal["ok", "missing_pred", "missing_gt", "failed"]


@dataclass
class SceneOutcome:
    scene_id: str
    status: SceneStatus
    result: GeometryEvalResult | None = None
    error: str | None = None
    pred_path: Path | None = None
    gt_path: Path | None = None


@dataclass
class BenchmarkResult:
    dataset: str
    split: str
    scenes: list[SceneOutcome]
    summary: dict[str, dict[str, float]]
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
                    "result": o.result.to_dict() if o.result else None,
                }
                for o in self.scenes
            ],
            "summary": self.summary,
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
    config: BenchmarkConfig,
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
        gt_geom = load_mesh(gt_path)

        if config.crop_to_gt_bbox:
            gv = gt_geom.vertices
            bbox_min, bbox_max = gv.min(axis=0), gv.max(axis=0)
            if isinstance(pred_geom, MeshData):
                kept = _crop_to_bbox(pred_geom.vertices, bbox_min, bbox_max, config.bbox_margin)
                # Cropping a mesh in vertex-space breaks face indices, so fall
                # back to a point cloud after cropping.
                pred_geom = PointCloudData(points=kept)
            else:
                pred_geom = PointCloudData(
                    points=_crop_to_bbox(
                        pred_geom.points, bbox_min, bbox_max, config.bbox_margin
                    )
                )

        result = evaluate_geometry(
            pred_geom,
            gt_geom,
            samples=config.samples,
            seed=config.seed,
            sample_method=config.sample_method,
            align_mode=config.align,
            thresholds=config.thresholds,
            chamfer_variant=config.chamfer_variant,
        )
        return SceneOutcome(
            scene_id=scene_id,
            status="ok",
            result=result,
            pred_path=rp["path"],
            gt_path=gt_path,
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
    jobs: list[tuple[str, dict[str, Any] | None, Path | None]] = []
    for sid in scenes:
        rp = loc.resolve(sid)
        if rp is None and cfg.fail_on_missing:
            raise MissingArtifactError(
                f"No prediction found for scene {sid!r} under {loc.preds_root}"
            )
        try:
            gt_path = dataset.asset_path(sid, _gt_asset(dataset))
        except Exception:
            gt_path = None
        jobs.append((sid, _pred_descriptor(rp), gt_path))

    outcomes: list[SceneOutcome] = []
    if cfg.workers <= 1 or len(jobs) <= 1:
        for j in jobs:
            outcomes.append(_evaluate_one(*j, cfg))
            if progress:
                _log.info("benchmark: %s -> %s", outcomes[-1].scene_id, outcomes[-1].status)
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futures = {ex.submit(_evaluate_one, *j, cfg): j[0] for j in jobs}
            for fut in as_completed(futures):
                outcomes.append(fut.result())
                if progress:
                    _log.info(
                        "benchmark: %s -> %s",
                        outcomes[-1].scene_id,
                        outcomes[-1].status,
                    )
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
        summary=aggregate(outcomes),
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
