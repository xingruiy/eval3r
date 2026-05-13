"""Alignment dispatch — explicit modes only, never silent."""

from __future__ import annotations

from typing import Literal

import numpy as np

from eval3r.alignment.icp import ICPAligner
from eval3r.alignment.base import AlignResult, umeyama
from eval3r.alignment.trajectory import TrajectoryAligner
from eval3r.utils.errors import AlignmentError
from eval3r.utils.typing import Points, Poses

AlignMode = Literal["none", "scale", "se3", "sim3", "icp", "traj_se3", "traj_sim3"]


def align(
    source: Points,
    target: Points,
    *,
    mode: AlignMode = "none",
    correspondences: bool = False,
    pred_poses: Poses | None = None,
    gt_poses: Poses | None = None,
    pred_convention: str = "unspecified",
    gt_convention: str = "unspecified",
    pred_timestamps: np.ndarray | None = None,
    gt_timestamps: np.ndarray | None = None,
) -> AlignResult:
    """Estimate an alignment that maps ``source`` onto ``target``.

    With ``correspondences=True`` the two arrays are assumed to be
    point-to-point matched (Umeyama). Otherwise ICP is used after the
    closed-form initialisation in ``mode``.

    Use ``mode=\"traj_sim3\"`` or ``mode=\"traj_se3\"`` to align using
    camera trajectories. In this case *pred_poses* and *gt_poses* are
    required, each of shape ``(T, 4, 4)``, with their respective
    *pred_convention* / *gt_convention* (``\"T_wc\"`` or ``\"T_cw\"``).
    """
    if mode == "none":
        return AlignResult(scale=1.0, rotation=np.eye(3), translation=np.zeros(3), mode="none")
    if mode in ("traj_se3", "traj_sim3"):
        if pred_poses is None or gt_poses is None:
            raise AlignmentError(
                f"align mode '{mode}' requires pred_poses and gt_poses "
                f"(both (T, 4, 4) arrays)."
            )
        return TrajectoryAligner(
            estimate_scale=(mode == "traj_sim3"),
            pred_convention=pred_convention,
            gt_convention=gt_convention,
        ).align(pred_poses, gt_poses, pred_timestamps=pred_timestamps, gt_timestamps=gt_timestamps)
    if correspondences:
        if mode in ("scale", "se3", "sim3"):
            return umeyama(source, target, mode=mode)
        raise AlignmentError(
            f"align mode '{mode}' is not valid with correspondences=True; "
            f"use 'scale', 'se3', or 'sim3'."
        )
    if mode == "icp":
        return ICPAligner().align(source, target)
    if mode == "se3":
        return ICPAligner(estimate_scale=False).align(source, target)
    if mode == "sim3":
        # Rigid ICP first to lock correspondences, then refine with scale.
        # Single-pass ICP-with-scale collapses to wrong local minima when
        # source and target differ in scale by more than a small factor.
        rigid = ICPAligner(estimate_scale=False).align(source, target)
        return ICPAligner(estimate_scale=True).align(source, target, init=rigid)
    if mode == "scale":
        # crude isotropic-scale-only fit: ratio of bbox extents.
        src = np.asarray(source, dtype=np.float64)
        tgt = np.asarray(target, dtype=np.float64)
        s_ext = float(np.linalg.norm(src.max(0) - src.min(0)))
        t_ext = float(np.linalg.norm(tgt.max(0) - tgt.min(0)))
        if s_ext == 0:
            raise AlignmentError("scale alignment: source has zero extent")
        s = t_ext / s_ext
        return AlignResult(scale=s, rotation=np.eye(3), translation=np.zeros(3), mode="scale")
    raise AlignmentError(f"Unknown alignment mode: {mode!r}")


__all__ = ["align", "AlignMode", "AlignResult", "umeyama", "ICPAligner", "TrajectoryAligner"]
