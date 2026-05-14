"""DTU benchmark — config, worker, and class."""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
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
)
from eval3r.filtering.base import BaseFilter
from eval3r.io.geometry import load_point_cloud
from eval3r.io.trajectory import load_trajectory_auto
from eval3r.metrics.base import GeometryMetric
from eval3r.metrics.metric3d import Accuracy, ChamferDistance, Completeness, FScore
from eval3r.pipeline import EvalConfig, Pipeline
from eval3r.sampling.importance import ImportanceSampler
from eval3r.sampling.uniform import UniformSampler
from eval3r.utils.errors import MissingArtifactError
from eval3r.utils.typing import PathLike

_DTU_EVAL_SCANS = [
    1, 4, 9, 15, 24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122,
]


@dataclass
class DTUBenchmarkConfig(BenchmarkConfig):
    scan_subdir: str = "scans"
    scan_format: str = "scan{scan_id}"
    point_cloud_filename: str = "points.ply"
    eval_scans: list[int] | None = None
    metrics: list[str] = field(
        default_factory=lambda: ["chamfer", "fscore@1.0", "fscore@2.0", "fscore@5.0"]
    )


def _evaluate_scene(
    job: BenchmarkJob,
    *,
    gt_root: Path,
    cfg: DTUBenchmarkConfig,
    work_dir: Path,
) -> SceneOutcome:
    scene_id, pred_desc = job
    if pred_desc is None:
        return SceneOutcome(scene_id=scene_id, status="missing_pred")
    try:
        scan_name = cfg.scan_format.format(scan_id=int(scene_id))
        gt_path = gt_root / cfg.scan_subdir / scan_name / cfg.point_cloud_filename
        if not gt_path.exists():
            return SceneOutcome(scene_id=scene_id, status="missing_gt", gt_path=gt_path)
        gt_geom = load_point_cloud(gt_path)

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

            # DTU GT poses would need to be loaded separately — not supported by default
            raise MissingArtifactError(
                "Trajectory alignment for DTU requires GT poses. "
                "Load them manually and construct TrajectoryAligner directly."
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


class DTUBenchmark(BaseBenchmark):
    dataset_name = "dtu"

    def __init__(
        self,
        gt_root: PathLike,
        pred_root: PathLike,
        *,
        cfg: DTUBenchmarkConfig | None = None,
    ) -> None:
        super().__init__(gt_root, pred_root, cfg=cfg or DTUBenchmarkConfig())

    def _default_config(self) -> DTUBenchmarkConfig:
        return DTUBenchmarkConfig()

    def _list_scenes(self, split: str | None) -> list[str]:
        cfg: DTUBenchmarkConfig = self.cfg  # type: ignore[assignment]
        if split is None:
            eval_scans = cfg.eval_scans or _DTU_EVAL_SCANS
            return [str(s) for s in eval_scans]
        p = Path(split)
        if not p.exists():
            raise MissingArtifactError(f"DTU split file not found: {p}")
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]

    @property
    def _worker(self):
        return _evaluate_scene


__all__ = ["DTUBenchmarkConfig", "DTUBenchmark"]
