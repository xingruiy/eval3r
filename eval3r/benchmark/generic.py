"""Generic benchmark — config, worker, and class.

Handles arbitrary ``{scene_id}``-template ground-truth layouts.
Only geometry (mesh or point cloud) evaluation is supported.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path

from eval3r.alignment.base import IdentityAligner
from eval3r.alignment.icp import ICPAligner
from eval3r.benchmark.base import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkJob,
    SceneOutcome,
    _load_pred_geometry,
    _resolve_mask_pattern,
)
from eval3r.filtering.base import BaseFilter
from eval3r.io.geometry import load_mesh, load_point_cloud
from eval3r.io.trajectory import load_trajectory_auto
from eval3r.metrics.base import GeometryMetric
from eval3r.metrics.metric3d import Accuracy, ChamferDistance, Completeness, FScore
from eval3r.pipeline import EvalConfig, Pipeline
from eval3r.sampling.base import PointSampler
from eval3r.sampling.importance import ImportanceSampler
from eval3r.sampling.uniform import UniformSampler
from eval3r.utils.errors import MissingArtifactError
from eval3r.utils.typing import PathLike


@dataclass
class GenericBenchmarkConfig(BenchmarkConfig):
    gt_path: str = "{scene_id}/gt.ply"
    mask_dir: str | None = None
    mask_pattern: str = "{scene_id}/occlusion_mask.npy"
    t_mask_scene_pattern: str = "{scene_id}/T_mask_scene.txt"


def _load_gt_geom(gt_root: Path, gt_path_tmpl: str, scene_id: str):
    rel = gt_path_tmpl.format(scene_id=scene_id)
    path = gt_root / rel
    if not path.exists():
        return None, path
    try:
        return load_mesh(path), path
    except Exception:
        return load_point_cloud(path), path


def _try_load_occlusion(
    scene_id: str, mask_dir: str, mask_pattern: str, t_mask_pattern: str
) -> BaseFilter | None:
    from eval3r.filtering.occlusion.mask import load_occlusion_mask

    mask_path = _resolve_mask_pattern(mask_dir, mask_pattern, scene_id)
    w2g_path = _resolve_mask_pattern(mask_dir, t_mask_pattern, scene_id)
    if mask_path.exists() and w2g_path.exists():
        return load_occlusion_mask(mask_path, w2g_path)
    return None


def _evaluate_scene(
    job: BenchmarkJob,
    *,
    gt_root: Path,
    cfg: GenericBenchmarkConfig,
    work_dir: Path,
) -> SceneOutcome:
    scene_id, pred_desc = job
    if pred_desc is None:
        return SceneOutcome(scene_id=scene_id, status="missing_pred")
    try:
        gt_geom, gt_path = _load_gt_geom(gt_root, cfg.gt_path, scene_id)
        if gt_geom is None:
            return SceneOutcome(scene_id=scene_id, status="missing_gt", gt_path=gt_path)

        # Filters
        filters: list[BaseFilter] = []
        if cfg.mask_dir:
            occ = _try_load_occlusion(
                scene_id, cfg.mask_dir, cfg.mask_pattern, cfg.t_mask_scene_pattern
            )
            if occ:
                filters.append(occ)

        # Aligner
        aligner: IdentityAligner | ICPAligner = IdentityAligner()
        if cfg.aligner == "none":
            aligner = IdentityAligner()
        elif cfg.aligner == "icp_se3":
            aligner = ICPAligner(estimate_scale=False)
        elif cfg.aligner == "icp_sim3":
            aligner = ICPAligner(estimate_scale=True)
        elif cfg.aligner in ("traj_se3", "traj_sim3"):
            if pred_desc["kind"] == "manifest":
                from eval3r.manifest.reader import PredictionReader

                reader = PredictionReader(Path(pred_desc["path"]), verify_hashes=False)
                try:
                    pred_traj = reader.poses
                except MissingArtifactError:
                    pred_traj = None
            else:
                pred_traj = None

            if pred_traj is None and cfg.pred_pose_dir is not None:
                pose_path = Path(cfg.pred_pose_dir) / cfg.pred_pose_file.format(
                    scene_id=scene_id
                )
                pred_traj = load_trajectory_auto(
                    pose_path, convention=cfg.pred_pose_convention
                )
            elif pred_traj is None:
                raise MissingArtifactError(
                    f"Trajectory alignment requires poses for scene {scene_id!r}."
                )

            raise MissingArtifactError(
                "Generic benchmark: trajectory alignment requires GT poses. "
                "Load them manually and construct TrajectoryAligner directly."
            )
        else:
            raise ValueError(f"unknown aligner: {cfg.aligner!r}")

        # Sampler
        sampler: PointSampler
        if cfg.sampler in ("area", "uniform"):
            sampler = ImportanceSampler()
        elif cfg.sampler == "vertex":
            sampler = UniformSampler()
        else:
            raise ValueError(f"unknown sampler: {cfg.sampler!r}")

        # Metrics
        metrics: list[GeometryMetric] = []
        for mname in cfg.metrics:
            if mname == "chamfer":
                metrics.append(ChamferDistance())
            elif mname == "accuracy":
                metrics.append(Accuracy())
            elif mname == "completeness":
                metrics.append(Completeness())
            elif mname.startswith("fscore@"):
                metrics.append(FScore(float(mname[7:])))

        pipe = Pipeline(
            sampler=sampler,
            aligner=aligner,
            filters=filters,
            metrics=metrics,
            config=EvalConfig(samples=cfg.samples, seed=cfg.seed),
        )

        pred_geom = _load_pred_geometry(pred_desc, scene_id)
        debug_plot_path = work_dir / "debug_plots" / f"{scene_id}.png" if cfg.debug_plot else None
        _pred, _gt = pred_geom, gt_geom
        del pred_geom, gt_geom
        pr = pipe.evaluate(_pred, _gt, debug_plot_path=debug_plot_path)

        (work_dir / "scene_results" / f"{scene_id}.json").write_text(json.dumps(pr.to_dict(), indent=2))
        return SceneOutcome(
            scene_id=scene_id,
            status="ok",
            result=pr,
            pred_path=Path(pred_desc["path"]),
            gt_path=gt_path,
        )
    except Exception:
        tb = traceback.format_exc()
        exc_line = tb.strip().splitlines()[-1]
        (work_dir / "scene_results" / f"{scene_id}.json").write_text(
            json.dumps({"scene_id": scene_id, "status": "failed", "exception": exc_line, "traceback": tb}, indent=2)
        )
        return SceneOutcome(
            scene_id=scene_id,
            status="failed",
            error=tb,
            pred_path=Path(pred_desc["path"]) if pred_desc else None,
        )


class GenericBenchmark(BaseBenchmark):
    dataset_name = "generic"

    def __init__(
        self,
        gt_root: PathLike,
        pred_root: PathLike,
        *,
        cfg: GenericBenchmarkConfig | None = None,
        scenes: list[str] | None = None,
    ) -> None:
        super().__init__(gt_root, pred_root, cfg=cfg or GenericBenchmarkConfig())
        self._scenes_override = scenes

    def _default_config(self) -> GenericBenchmarkConfig:
        return GenericBenchmarkConfig()

    def _list_scenes(self, split: str | None) -> list[str]:
        if self._scenes_override is not None:
            return list(self._scenes_override)
        if split is not None:
            p = Path(split)
            if not p.exists():
                raise MissingArtifactError(f"Generic split file not found: {p}")
            return [line.strip() for line in p.read_text().splitlines() if line.strip()]
        raise MissingArtifactError(
            "GenericBenchmark requires a scene source: pass split=<path> or "
            "scenes=[...] to the constructor."
        )

    @property
    def _worker(self):
        return _evaluate_scene


__all__ = ["GenericBenchmarkConfig", "GenericBenchmark"]
