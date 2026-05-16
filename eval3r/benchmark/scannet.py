"""ScanNet benchmark — config, worker, and class."""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval3r.alignment.base import IdentityAligner
from eval3r.alignment.icp import ICPAligner
from eval3r.alignment.trajectory import TrajectoryAligner
from eval3r.benchmark.base import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkJob,
    SceneOutcome,
    _load_pred_geometry,
    _resolve_mask_pattern,
)
from eval3r.filtering.base import BaseFilter
from eval3r.filtering.bbox import BBoxFilter
from eval3r.io.geometry import load_mesh
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
class ScanNetBenchmarkConfig(BenchmarkConfig):
    scene_subdir: str = "scans/{scene_id}"
    mesh_filename: str = "{scene_id}_vh_clean_2.ply"
    pose_subdir: str = "pose"
    pose_format: str = "{frame}.txt"
    mask_dir: str | None = None
    mask_pattern: str = "{scene_id}/occlusion_mask.npy"
    t_mask_scene_pattern: str = "{scene_id}/T_mask_scene.txt"
    crop_to_gt_bbox: bool = False
    bbox_margin: float = 0.10


def _load_scannet_poses(gt_root: Path, scene_id: str, cfg: ScanNetBenchmarkConfig):
    from eval3r.io.trajectory import Trajectory
    scene_dir = gt_root / cfg.scene_subdir.format(scene_id=scene_id)
    pose_dir = scene_dir / cfg.pose_subdir
    if not pose_dir.is_dir():
        raise MissingArtifactError(f"ScanNet pose dir not found: {pose_dir}")
    files = sorted(
        (p for p in pose_dir.iterdir() if p.suffix == ".txt"),
        key=lambda p: int(p.stem),
    )
    if not files:
        raise MissingArtifactError(f"ScanNet: no pose files in {pose_dir}")
    poses = np.stack([np.loadtxt(p) for p in files], axis=0)
    timestamps = np.array([float(p.stem) for p in files], dtype=np.float64)
    return Trajectory(poses=poses, timestamps=timestamps, convention="T_wc")


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
    cfg: ScanNetBenchmarkConfig,
    work_dir: Path,
) -> SceneOutcome:
    scene_id, pred_desc = job
    if pred_desc is None:
        return SceneOutcome(scene_id=scene_id, status="missing_pred")
    try:
        gt_path = (
            gt_root
            / cfg.scene_subdir.format(scene_id=scene_id)
            / cfg.mesh_filename.format(scene_id=scene_id)
        )
        if not gt_path.exists():
            return SceneOutcome(scene_id=scene_id, status="missing_gt", gt_path=gt_path)
        gt_geom = load_mesh(gt_path)

        # Filters
        filters: list[BaseFilter] = []
        if cfg.mask_dir:
            occ = _try_load_occlusion(
                scene_id, cfg.mask_dir, cfg.mask_pattern, cfg.t_mask_scene_pattern
            )
            if occ:
                filters.append(occ)
        if not filters and cfg.crop_to_gt_bbox:
            gv = gt_geom.vertices
            filters.append(BBoxFilter(gv.min(0), gv.max(0), margin=cfg.bbox_margin))

        # Aligner
        aligner: IdentityAligner | ICPAligner | TrajectoryAligner = IdentityAligner()
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
                    f"Trajectory alignment ({cfg.aligner}) requires poses for "
                    f"scene {scene_id!r}. Provide --pred-pose-dir."
                )

            gt_traj = _load_scannet_poses(gt_root, scene_id, cfg)
            aligner = TrajectoryAligner(
                pred_traj.poses,
                gt_traj.poses,
                pred_timestamps=pred_traj.timestamps,
                gt_timestamps=gt_traj.timestamps,
                estimate_scale=(cfg.aligner == "traj_sim3"),
                pred_convention=pred_traj.convention,
                gt_convention=gt_traj.convention,
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


class ScanNetBenchmark(BaseBenchmark):
    dataset_name = "scannet"

    def __init__(
        self,
        gt_root: PathLike,
        pred_root: PathLike,
        *,
        cfg: ScanNetBenchmarkConfig | None = None,
    ) -> None:
        super().__init__(gt_root, pred_root, cfg=cfg or ScanNetBenchmarkConfig())

    def _default_config(self) -> ScanNetBenchmarkConfig:
        return ScanNetBenchmarkConfig()

    def _list_scenes(self, split: str | None) -> list[str]:
        if split is None:
            scans_dir = self.gt_root / self.cfg.scene_subdir.split("/", 1)[0]  # type: ignore[attr-defined]
            if not scans_dir.is_dir():
                raise MissingArtifactError(
                    f"ScanNet auto-discovery: {scans_dir} does not exist. "
                    f"Pass split=<path-to-list>."
                )
            return sorted(p.name for p in scans_dir.iterdir() if p.is_dir())
        p = Path(split)
        if not p.exists():
            raise MissingArtifactError(f"ScanNet split file not found: {p}")
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]

    @property
    def _worker(self):
        return _evaluate_scene


__all__ = ["ScanNetBenchmarkConfig", "ScanNetBenchmark"]
