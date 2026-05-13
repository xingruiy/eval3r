"""Trajectory-based alignment using Umeyama on camera positions."""

from __future__ import annotations

from typing import Literal

import numpy as np

from eval3r.alignment.base import AlignResult, umeyama
from eval3r.utils.typing import Poses


def cam_positions(poses: Poses, convention: str) -> np.ndarray:
    """Extract camera centers in world frame from (T, 4, 4) poses.

    *convention* must be ``"T_wc"`` (camera-to-world) or ``"T_cw"``
    (world-to-camera).
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (T, 4, 4), got {poses.shape}")
    t = poses[:, :3, 3]
    if convention == "T_wc":
        return t.copy()
    if convention == "T_cw":
        R = poses[:, :3, :3]
        return -(R.transpose(0, 2, 1) @ t[..., None]).squeeze(-1)
    raise ValueError(f"Unknown pose convention: {convention!r}; expected 'T_wc' or 'T_cw'")


def _match_by_timestamp(
    pred_ts: np.ndarray, gt_ts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """For each pred timestamp, find the nearest GT timestamp.

    Returns ``(pred_idx, gt_idx)`` arrays of matched pairs.
    """
    gt_idx = np.array([np.argmin(np.abs(gt_ts - t)) for t in pred_ts])
    return np.arange(len(pred_ts)), gt_idx


class TrajectoryAligner:
    """Align a predicted camera trajectory to ground truth using Umeyama.

    Args:
        estimate_scale: When True, solves Sim(3) (``traj_sim3``).
            When False, solves SE(3) (``traj_se3``).
        pred_convention: Pose convention for prediction (``"T_wc"`` or ``"T_cw"``).
        gt_convention: Pose convention for ground truth (``"T_wc"`` or ``"T_cw"``).
    """

    def __init__(
        self,
        *,
        estimate_scale: bool = False,
        pred_convention: str = "unspecified",
        gt_convention: str = "unspecified",
    ) -> None:
        self.estimate_scale = estimate_scale
        self.pred_convention = pred_convention
        self.gt_convention = gt_convention

    def align(
        self,
        pred_poses: Poses,
        gt_poses: Poses,
        *,
        pred_timestamps: np.ndarray | None = None,
        gt_timestamps: np.ndarray | None = None,
    ) -> AlignResult:
        """Estimate the Sim(3)/SE(3) transform that aligns *pred_poses* to *gt_poses*.

        Extracts camera positions from both trajectories according to their
        conventions, then calls the closed-form Umeyama solver on the
        resulting point sets.

        When *pred_timestamps* and *gt_timestamps* are provided and the
        trajectories have different lengths, frames are matched by nearest
        timestamp before alignment.
        """
        pred_pos = cam_positions(pred_poses, self.pred_convention)
        gt_pos = cam_positions(gt_poses, self.gt_convention)

        pred_idx = np.arange(pred_pos.shape[0])
        gt_idx = np.arange(gt_pos.shape[0])

        if pred_pos.shape[0] != gt_pos.shape[0]:
            if pred_timestamps is None or gt_timestamps is None:
                raise ValueError(
                    f"Trajectory lengths differ ({pred_pos.shape[0]} vs {gt_pos.shape[0]}) "
                    f"and no timestamps provided. Provide per-frame timestamps so "
                    f"corresponding frames can be matched."
                )
            pred_idx, gt_idx = _match_by_timestamp(
                np.asarray(pred_timestamps, dtype=np.float64),
                np.asarray(gt_timestamps, dtype=np.float64),
            )
            pred_pos = pred_pos[pred_idx]
            gt_pos = gt_pos[gt_idx]

        mode: Literal["traj_sim3", "traj_se3"] = (
            "traj_sim3" if self.estimate_scale else "traj_se3"
        )
        umeyama_mode: Literal["se3", "sim3"] = "sim3" if self.estimate_scale else "se3"
        result = umeyama(pred_pos, gt_pos, mode=umeyama_mode)
        result.mode = mode
        result.matched_pred_idx = pred_idx
        result.matched_gt_idx = gt_idx
        return result
