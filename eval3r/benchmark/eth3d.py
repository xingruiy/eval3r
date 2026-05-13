"""ETH3D benchmark — config, worker, and class."""

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
from eval3r.io.geometry import load_mesh, load_point_cloud
from eval3r.io.trajectory import Trajectory, load_trajectory_auto, _quat_to_rot
from eval3r.metrics.base import GeometryMetric
from eval3r.metrics.metric3d import Accuracy, ChamferDistance, Completeness, FScore
from eval3r.pipeline import EvalConfig, Pipeline
from eval3r.sampling.importance import ImportanceSampler
from eval3r.sampling.uniform import UniformSampler
from eval3r.utils.errors import MissingArtifactError
from eval3r.utils.typing import PathLike

_DSLR_TRAINING_SCENES = [
    "courtyard", "delivery_area", "electro", "facade", "kicker",
    "meadow", "office", "pipes", "playground", "relief",
    "relief_2", "terrace", "terrains",
]
_RIG_TRAINING_SCENES = [
    "courtyard", "delivery_area", "electro", "facade", "kicker",
    "meadow", "office", "playground", "relief", "relief_2", "terrace", "terrains",
]


@dataclass
class ETH3DBenchmarkConfig(BenchmarkConfig):
    track: str = "dslr"
    mesh_filename: str = "scan.ply"
    point_cloud_filename: str = "scan_points.ply"
    calibration_subdir: str = "dslr_calibration_jpg"


def _parse_colmap_images_txt(path: Path) -> Trajectory:
    lines = path.read_text().splitlines()
    data_lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    n = len(data_lines)
    if n == 0:
        raise MissingArtifactError(f"COLMAP images.txt is empty: {path}")
    timestamps = np.arange(n, dtype=np.float64)
    poses = np.tile(np.eye(4), (n, 1, 1))
    for i, line in enumerate(data_lines):
        parts = line.split()
        if len(parts) < 9:
            continue
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        poses[i, :3, :3] = _quat_to_rot(qx, qy, qz, qw)
        poses[i, :3, 3] = [tx, ty, tz]
    return Trajectory(poses=poses, timestamps=timestamps, convention="T_cw")


def _load_eth3d_gt(
    gt_root: Path, scene_id: str, cfg: ETH3DBenchmarkConfig
):
    scene_dir = gt_root / cfg.track / scene_id
    pc_path = scene_dir / cfg.point_cloud_filename
    if pc_path.exists():
        return load_point_cloud(pc_path), pc_path
    mesh_path = scene_dir / cfg.mesh_filename
    if mesh_path.exists():
        return load_mesh(mesh_path), mesh_path
    return None, scene_dir / cfg.point_cloud_filename


def _load_eth3d_poses(gt_root: Path, scene_id: str, cfg: ETH3DBenchmarkConfig) -> Trajectory:
    scene_dir = gt_root / cfg.track / scene_id
    images_txt = scene_dir / cfg.calibration_subdir / "images.txt"
    if not images_txt.exists():
        raise MissingArtifactError(f"ETH3D COLMAP images.txt not found: {images_txt}")
    return _parse_colmap_images_txt(images_txt)


def _evaluate_scene(
    job: BenchmarkJob,
    *,
    gt_root: Path,
    cfg: ETH3DBenchmarkConfig,
    work_dir: Path,
) -> SceneOutcome:
    scene_id, pred_desc = job
    if pred_desc is None:
        return SceneOutcome(scene_id=scene_id, status="missing_pred")
    try:
        gt_geom, gt_path = _load_eth3d_gt(gt_root, scene_id, cfg)
        if gt_geom is None:
            return SceneOutcome(scene_id=scene_id, status="missing_gt", gt_path=gt_path)

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

            gt_traj = _load_eth3d_poses(gt_root, scene_id, cfg)
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
        pr = pipe.evaluate(pred_geom, gt_geom, debug_plot_path=debug_plot_path)

        (work_dir / "scene_results" / f"{scene_id}.json").write_text(json.dumps(pr.to_dict(), indent=2))
        return SceneOutcome(
            scene_id=scene_id,
            status="ok",
            result=pr,
            pred_path=Path(pred_desc["path"]),
            gt_path=gt_path,
        )
    except Exception:
        return SceneOutcome(
            scene_id=scene_id,
            status="failed",
            error=traceback.format_exc(limit=8),
            pred_path=Path(pred_desc["path"]) if pred_desc else None,
        )


class ETH3DBenchmark(BaseBenchmark):
    dataset_name = "eth3d"

    def __init__(
        self,
        gt_root: PathLike,
        pred_root: PathLike,
        *,
        cfg: ETH3DBenchmarkConfig | None = None,
    ) -> None:
        super().__init__(gt_root, pred_root, cfg=cfg or ETH3DBenchmarkConfig())

    def _default_config(self) -> ETH3DBenchmarkConfig:
        return ETH3DBenchmarkConfig()

    def _list_scenes(self, split: str | None) -> list[str]:
        cfg: ETH3DBenchmarkConfig = self.cfg  # type: ignore[assignment]
        if split is not None:
            p = Path(split)
            if not p.exists():
                raise MissingArtifactError(f"ETH3D split file not found: {p}")
            return [line.strip() for line in p.read_text().splitlines() if line.strip()]
        track_dir = self.gt_root / cfg.track
        if not track_dir.is_dir():
            raise MissingArtifactError(
                f"ETH3D track directory not found: {track_dir}. "
                f"Pass track='dslr' or track='rig'."
            )
        return sorted(
            p.name
            for p in track_dir.iterdir()
            if p.is_dir()
            and (
                (p / cfg.point_cloud_filename).exists()
                or (p / cfg.mesh_filename).exists()
            )
        )

    @property
    def _worker(self):
        return _evaluate_scene


__all__ = ["ETH3DBenchmarkConfig", "ETH3DBenchmark"]
