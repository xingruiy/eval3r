"""Trajectory metrics — ATE (Absolute Trajectory Error)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from eval3r.align.trajectory import align_trajectory, cam_positions
from eval3r.utils.typing import Poses

TrajAlignMode = Literal["traj_se3", "traj_sim3"]


@dataclass
class TrajectoryEvalResult:
    ate_rmse: float
    ate_mean: float
    ate_median: float
    ate_std: float
    align_rotation: np.ndarray  # (3, 3)
    align_translation: np.ndarray  # (3,)
    align_scale: float
    align_mode: str


def evaluate_trajectory(
    pred_poses: Poses,
    gt_poses: Poses,
    *,
    mode: TrajAlignMode = "traj_sim3",
    pred_convention: str = "unspecified",
    gt_convention: str = "unspecified",
    pred_timestamps: np.ndarray | None = None,
    gt_timestamps: np.ndarray | None = None,
) -> TrajectoryEvalResult:
    """Compute ATE between *pred_poses* and *gt_poses*.

    Aligns the prediction trajectory to GT (Sim(3) or SE(3) depending on
    *mode*), then computes the RMSE / mean / median / std of the per-frame
    position errors.

    When *pred_timestamps* and *gt_timestamps* are provided and the
    trajectories have different lengths, frames are matched by nearest
    timestamp; ATE is computed only on matched pairs.
    """
    from eval3r.align.trajectory import _match_by_timestamp

    al = align_trajectory(
        pred_poses, gt_poses, mode,
        pred_convention=pred_convention,
        gt_convention=gt_convention,
        pred_timestamps=pred_timestamps,
        gt_timestamps=gt_timestamps,
    )
    pred_pos = cam_positions(pred_poses, pred_convention)
    gt_pos = cam_positions(gt_poses, gt_convention)

    if pred_pos.shape[0] != gt_pos.shape[0]:
        if pred_timestamps is None or gt_timestamps is None:
            raise ValueError(
                f"Trajectory lengths differ ({pred_pos.shape[0]} vs {gt_pos.shape[0]}) "
                f"and no timestamps provided."
            )
        pred_idx, gt_idx = _match_by_timestamp(
            np.asarray(pred_timestamps, dtype=np.float64),
            np.asarray(gt_timestamps, dtype=np.float64),
        )
        pred_pos = pred_pos[pred_idx]
        gt_pos = gt_pos[gt_idx]

    pred_aligned = (al.scale * pred_pos @ al.rotation.T) + al.translation
    errors = np.linalg.norm(pred_aligned - gt_pos, axis=1)
    return TrajectoryEvalResult(
        ate_rmse=float(np.sqrt(np.mean(errors**2))),
        ate_mean=float(np.mean(errors)),
        ate_median=float(np.median(errors)),
        ate_std=float(np.std(errors)),
        align_rotation=al.rotation,
        align_translation=al.translation,
        align_scale=al.scale,
        align_mode=al.mode,
    )
