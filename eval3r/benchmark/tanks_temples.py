"""Tanks & Temples benchmark — config, worker, and class."""

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
)
from eval3r.filtering.base import BaseFilter
from eval3r.filtering.polygon import PolygonFilter
from eval3r.io.geometry import load_point_cloud
from eval3r.io.trajectory import Trajectory, load_trajectory_auto
from eval3r.metrics.base import GeometryMetric
from eval3r.metrics.metric3d import Accuracy, ChamferDistance, Completeness, FScore
from eval3r.pipeline import EvalConfig, Pipeline
from eval3r.sampling.base import PointSampler
from eval3r.sampling.importance import ImportanceSampler
from eval3r.sampling.uniform import UniformSampler
from eval3r.utils.errors import MissingArtifactError
from eval3r.utils.typing import PathLike

_TRAINING_SCENES = [
    "Barn", "Caterpillar", "Church", "Courthouse",
    "Ignatius", "Meetingroom", "Truck",
]
_INTERMEDIATE_SCENES = [
    "Family", "Francis", "Horse", "Lighthouse",
    "M60", "Panther", "Playground", "Train",
]
_ADVANCED_SCENES = [
    "Auditorium", "Ballroom", "Courtroom", "Museum", "Palace", "Temple",
]

_SUBSETS: dict[str, list[str]] = {
    "training": _TRAINING_SCENES,
    "intermediate": _INTERMEDIATE_SCENES,
    "advanced": _ADVANCED_SCENES,
    "all": _TRAINING_SCENES + _INTERMEDIATE_SCENES + _ADVANCED_SCENES,
}

_SCENES_TAU: dict[str, float] = {
    "Barn": 0.01,
    "Caterpillar": 0.005,
    "Church": 0.025,
    "Courthouse": 0.025,
    "Ignatius": 0.003,
    "Meetingroom": 0.01,
    "Truck": 0.005,
}


@dataclass
class TanksTemplesBenchmarkConfig(BenchmarkConfig):
    point_cloud_filename: str = "{scene_id}.ply"
    crop_filename: str = "{scene_id}.json"
    crop_to_eval_region: bool = True
    use_dataset_thresholds: bool = True
    threshold_multiplier: float = 1.0
    subset: str | None = None


def _load_alignment(scene_dir: Path, scene_id: str) -> np.ndarray | None:
    path = scene_dir / f"{scene_id}_trans.txt"
    if not path.exists():
        return None
    mat = np.loadtxt(path, dtype=np.float64)
    return mat if mat.shape == (4, 4) else None


def _load_tt_poses(scene_dir: Path, scene_id: str) -> Trajectory:

    pose_file = scene_dir / f"{scene_id}_COLMAP_SfM.log"
    if not pose_file.exists():
        raise MissingArtifactError(f"T&T pose log not found: {pose_file}")
    lines = [ln.strip() for ln in pose_file.read_text().splitlines() if ln.strip()]
    if len(lines) % 5 != 0:
        raise MissingArtifactError(f"T&T pose log malformed: {pose_file}")
    poses_list = []
    timestamps = []
    for i in range(0, len(lines), 5):
        hdr = lines[i].split()
        timestamps.append(float(hdr[0]))
        mat = np.array(
            [[float(x) for x in lines[i + j].split()] for j in range(1, 5)],
            dtype=np.float64,
        )
        poses_list.append(mat)
    poses = np.stack(poses_list, axis=0)
    T_align = _load_alignment(scene_dir, scene_id)
    if T_align is not None:
        poses = T_align @ poses
    return Trajectory(
        poses=poses,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        convention="T_wc",
    )


def _evaluate_scene(
    job: BenchmarkJob,
    *,
    gt_root: Path,
    cfg: TanksTemplesBenchmarkConfig,
    work_dir: Path,
) -> SceneOutcome:
    scene_id, pred_desc = job
    if pred_desc is None:
        return SceneOutcome(scene_id=scene_id, status="missing_pred")
    try:
        scene_dir = gt_root / scene_id
        gt_filename = cfg.point_cloud_filename.format(scene_id=scene_id)
        gt_path = scene_dir / gt_filename
        if not gt_path.exists():
            return SceneOutcome(scene_id=scene_id, status="missing_gt", gt_path=gt_path)
        gt_geom = load_point_cloud(gt_path)

        # Filters — load crop volume lazily inside worker
        filters: list[BaseFilter] = []
        if cfg.crop_to_eval_region and cfg.crop_filename:
            crop_path = scene_dir / cfg.crop_filename.format(scene_id=scene_id)
            if crop_path.exists():
                filters.append(PolygonFilter.load_from_json(crop_path))

        # Determine thresholds for this scene
        if cfg.use_dataset_thresholds and scene_id in _SCENES_TAU:
            scene_thresholds = [_SCENES_TAU[scene_id] * cfg.threshold_multiplier]
        else:
            # extract from metrics list
            scene_thresholds = [
                float(m[7:]) for m in cfg.metrics if m.startswith("fscore@")
            ]

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
                    f"Trajectory alignment requires poses for scene {scene_id!r}."
                )

            gt_traj = _load_tt_poses(scene_dir, scene_id)
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

        # Metrics — use scene thresholds for fscore
        metrics: list[GeometryMetric] = []
        for mname in cfg.metrics:
            if mname == "chamfer":
                metrics.append(ChamferDistance())
            elif mname == "accuracy":
                metrics.append(Accuracy())
            elif mname == "completeness":
                metrics.append(Completeness())
        for thr in scene_thresholds:
            metrics.append(FScore(thr))

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


class TanksTemplesBenchmark(BaseBenchmark):
    dataset_name = "tanks_temples"

    def __init__(
        self,
        gt_root: PathLike,
        pred_root: PathLike,
        *,
        cfg: TanksTemplesBenchmarkConfig | None = None,
    ) -> None:
        super().__init__(gt_root, pred_root, cfg=cfg or TanksTemplesBenchmarkConfig())

    def _default_config(self) -> TanksTemplesBenchmarkConfig:
        return TanksTemplesBenchmarkConfig()

    def _list_scenes(self, split: str | None) -> list[str]:
        cfg: TanksTemplesBenchmarkConfig = self.cfg  # type: ignore[assignment]
        if split is not None:
            p = Path(split)
            if not p.exists():
                raise MissingArtifactError(f"T&T split file not found: {p}")
            return [line.strip() for line in p.read_text().splitlines() if line.strip()]
        if cfg.subset is not None:
            subset_key = cfg.subset.lower()
            if subset_key not in _SUBSETS:
                raise MissingArtifactError(
                    f"T&T: unknown subset {cfg.subset!r}. Available: {sorted(_SUBSETS)}"
                )
            return list(_SUBSETS[subset_key])
        # enumerate directories that contain the point cloud file
        return sorted(
            p.name
            for p in self.gt_root.iterdir()
            if p.is_dir()
            and (p / cfg.point_cloud_filename.format(scene_id=p.name)).exists()
        )

    @property
    def _worker(self):
        return _evaluate_scene


__all__ = ["TanksTemplesBenchmarkConfig", "TanksTemplesBenchmark"]
