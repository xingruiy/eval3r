"""Replica benchmark — config, worker, and class."""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path

from eval3r.alignment.base import IdentityAligner
from eval3r.alignment.icp import ICPAligner
from eval3r.alignment.trajectory import TrajectoryAligner
from eval3r.benchmark.base import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkJob,
    SceneOutcome,
    _load_pred_geometry,
)
from eval3r.io.geometry import load_mesh
from eval3r.io.trajectory import load_trajectory_auto, load_trajectory_tum
from eval3r.metrics.base import GeometryMetric
from eval3r.metrics.metric3d import Accuracy, ChamferDistance, Completeness, FScore
from eval3r.pipeline import EvalConfig, Pipeline
from eval3r.sampling.importance import ImportanceSampler
from eval3r.sampling.uniform import UniformSampler
from eval3r.utils.errors import MissingArtifactError
from eval3r.utils.typing import PathLike


@dataclass
class ReplicaBenchmarkConfig(BenchmarkConfig):
    mesh_filename: str = "mesh.ply"
    results_subdir: str = "results"
    trajectory_filename: str = "trajectory.txt"


def _load_replica_poses(gt_root: Path, scene_id: str, cfg: ReplicaBenchmarkConfig):
    path = gt_root / scene_id / cfg.trajectory_filename
    if not path.exists():
        raise MissingArtifactError(f"Replica trajectory not found: {path}")
    return load_trajectory_tum(path, convention="T_wc")


def _evaluate_scene(
    job: BenchmarkJob,
    *,
    gt_root: Path,
    cfg: ReplicaBenchmarkConfig,
    work_dir: Path,
) -> SceneOutcome:
    scene_id, pred_desc = job
    if pred_desc is None:
        return SceneOutcome(scene_id=scene_id, status="missing_pred")
    try:
        gt_path = gt_root / scene_id / cfg.mesh_filename
        if not gt_path.exists():
            return SceneOutcome(scene_id=scene_id, status="missing_gt", gt_path=gt_path)
        gt_geom = load_mesh(gt_path)

        # Aligner
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

            gt_traj = _load_replica_poses(gt_root, scene_id, cfg)
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
            filters=[],
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


class ReplicaBenchmark(BaseBenchmark):
    dataset_name = "replica"

    def __init__(
        self,
        gt_root: PathLike,
        pred_root: PathLike,
        *,
        cfg: ReplicaBenchmarkConfig | None = None,
    ) -> None:
        super().__init__(gt_root, pred_root, cfg=cfg or ReplicaBenchmarkConfig())

    def _default_config(self) -> ReplicaBenchmarkConfig:
        return ReplicaBenchmarkConfig()

    def _list_scenes(self, split: str | None) -> list[str]:
        cfg: ReplicaBenchmarkConfig = self.cfg  # type: ignore[assignment]
        if split is not None:
            p = Path(split)
            if not p.exists():
                raise MissingArtifactError(f"Replica split file not found: {p}")
            return [line.strip() for line in p.read_text().splitlines() if line.strip()]
        return sorted(
            p.name
            for p in self.gt_root.iterdir()
            if p.is_dir() and (p / cfg.mesh_filename).exists()
        )

    @property
    def _worker(self):
        return _evaluate_scene


__all__ = ["ReplicaBenchmarkConfig", "ReplicaBenchmark"]
