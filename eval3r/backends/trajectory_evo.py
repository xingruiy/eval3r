"""evo trajectory backend (``trajectory`` registry kind).

Delegates trajectory loading, timestamp association, SE3/Sim3 alignment, and
ATE/RPE computation to `evo <https://github.com/MichaelGrupp/evo>`_
(``.agent/backends.md`` "Trajectory backend"): eval3r never reimplements
trajectory association or pose-error math.

Formats: TUM (``timestamp x y z qx qy qz qw``) only for now; further source
formats arrive with their dataset adapters (task 015 scope).

Every result dict is self-describing: it carries the statistics, the association
policy and pose counts (associated + dropped on both sides), the alignment mode
with the estimated rotation/translation/scale, and — for RPE — the delta /
delta_unit / all_pairs actually used. Nothing is chosen silently: the alignment
mode and the RPE delta parameters must be passed in explicitly by the caller
(the protocol pins them).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from eval3r.core.errors import AlignmentError, InvalidTrajectoryError, MetricError
from eval3r.core.registry import BackendInfo

# Alignment modes this backend implements (first-class AlignmentSpec.mode values).
TRAJECTORY_ALIGNMENT_MODES = ("none", "trajectory_se3", "trajectory_sim3")

# Metric-name → evo pose relation for the RPE metrics defined in .agent/metrics.md.
RPE_POSE_RELATIONS = {
    "rpe_translation": "translation_part",
    "rpe_rotation": "rotation_angle_deg",
}

_RPE_DELTA_UNITS = ("frames", "seconds", "meters")


class EvoTrajectoryBackend:
    """ATE / RPE via the evo library (in-process API, not the evo CLI)."""

    name = "evo"
    kind = "trajectory"

    def backend_info(self) -> BackendInfo:
        import evo

        return BackendInfo(
            kind=self.kind,
            name=self.name,
            library="evo",
            version=str(evo.__version__).lstrip("v"),
        )

    # --- loading / association / alignment -------------------------------------

    def load_trajectory(self, path: Path) -> Any:
        """Load one TUM-format trajectory (``timestamp x y z qx qy qz qw``)."""
        from evo.core.trajectory import TrajectoryException
        from evo.tools import file_interface

        try:
            return file_interface.read_tum_trajectory_file(path)
        except (file_interface.FileInterfaceException, TrajectoryException) as exc:
            raise InvalidTrajectoryError(
                f"could not load TUM trajectory file {path}: {exc}. Expected one "
                f"'timestamp x y z qx qy qz qw' line per pose (TUM format)."
            ) from exc

    def _associate_and_align(
        self,
        pred_path: Path,
        gt_path: Path,
        align: str,
        association: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Load, associate by timestamp, align; return (gt, pred, metadata)."""
        from evo.core import sync

        if align not in TRAJECTORY_ALIGNMENT_MODES:
            raise AlignmentError(
                f"trajectory alignment mode '{align}' is not supported by the evo "
                f"backend; choose one of: {', '.join(TRAJECTORY_ALIGNMENT_MODES)}."
            )
        max_diff = association.get("associate_max_diff")
        if max_diff is None:
            raise InvalidTrajectoryError(
                "the association policy must set 'associate_max_diff' explicitly "
                "(seconds); it is never defaulted by the evo backend."
            )
        offset = float(association.get("offset", 0.0))

        pred = self.load_trajectory(pred_path)
        gt = self.load_trajectory(gt_path)
        n_pred, n_gt = pred.num_poses, gt.num_poses

        try:
            gt_assoc, pred_assoc = sync.associate_trajectories(
                gt,
                pred,
                max_diff=float(max_diff),
                offset_2=offset,
                first_name=f"ground truth ({gt_path})",
                snd_name=f"prediction ({pred_path})",
            )
        except sync.SyncException as exc:
            raise InvalidTrajectoryError(
                f"timestamp association failed between prediction {pred_path} "
                f"({n_pred} poses) and ground truth {gt_path} ({n_gt} poses) with "
                f"associate_max_diff={max_diff}s, offset={offset}s: {exc}"
            ) from exc

        n_associated = pred_assoc.num_poses
        pred_aligned = copy.deepcopy(pred_assoc)
        rotation = np.eye(3)
        translation = np.zeros(3)
        scale = 1.0
        if align != "none":
            try:
                rotation, translation, scale = pred_aligned.align(
                    gt_assoc, correct_scale=(align == "trajectory_sim3")
                )
            except Exception as exc:  # evo raises bare exceptions on degeneracy
                raise AlignmentError(
                    f"'{align}' alignment of prediction {pred_path} onto ground "
                    f"truth {gt_path} failed on {n_associated} associated pose(s): "
                    f"{exc}"
                ) from exc

        metadata: dict[str, Any] = {
            "n_pred_poses": n_pred,
            "n_gt_poses": n_gt,
            "n_associated": n_associated,
            "n_dropped_pred": n_pred - n_associated,
            "n_dropped_gt": n_gt - n_associated,
            "association": {
                "policy": "nearest_timestamp",
                "associate_max_diff": float(max_diff),
                "offset": offset,
            },
            "alignment": {
                "mode": align,
                "scale": float(scale),
                "rotation": np.asarray(rotation).tolist(),
                "translation": np.asarray(translation).tolist(),
                "n_poses_used": n_associated,
            },
        }
        return gt_assoc, pred_aligned, metadata

    def align_trajectories(
        self,
        pred_path: Path,
        gt_path: Path,
        align: str,
        association: dict[str, Any],
    ) -> dict[str, Any]:
        """Estimate the transform mapping the pred trajectory onto the gt trajectory.

        Same association + Umeyama alignment as the pose metrics (task 015), but
        returns the transform itself instead of an error statistic, so the geometry
        align stage (task 018) can propagate it to a prediction's geometry. The
        4x4 ``matrix`` maps prediction coordinates into gt coordinates
        (``x_gt ≈ s·R·x_pred + t``, evo's Umeyama convention). ``residual_rmse`` is
        the position RMSE over the associated poses after alignment — the quantity
        the transform actually minimized.
        """
        gt_assoc, pred_aligned, meta = self._associate_and_align(
            pred_path, gt_path, align, association
        )
        alignment = meta["alignment"]
        scale = float(alignment["scale"])
        matrix = np.eye(4)
        matrix[:3, :3] = scale * np.asarray(alignment["rotation"], dtype=np.float64)
        matrix[:3, 3] = np.asarray(alignment["translation"], dtype=np.float64)
        residuals = pred_aligned.positions_xyz - gt_assoc.positions_xyz
        return {
            "matrix": matrix.tolist(),
            "scale": scale,
            "rotation": alignment["rotation"],
            "translation": alignment["translation"],
            "residual_rmse": float(np.sqrt(np.mean(np.sum(residuals**2, axis=1)))),
            **meta,
        }

    # --- metrics ----------------------------------------------------------------

    def evaluate_ate(
        self,
        pred_path: Path,
        gt_path: Path,
        align: str,
        association: dict[str, Any],
    ) -> dict[str, Any]:
        """Absolute trajectory error (evo APE, translation part), in metres."""
        from evo.core import metrics

        gt_assoc, pred_aligned, meta = self._associate_and_align(
            pred_path, gt_path, align, association
        )
        ape = metrics.APE(metrics.PoseRelation.translation_part)
        ape.process_data((gt_assoc, pred_aligned))
        return {
            "metric": "ate",
            "pose_relation": "translation_part",
            "unit": "m",
            "stats": {k: float(v) for k, v in ape.get_all_statistics().items()},
            **meta,
        }

    def evaluate_rpe(
        self,
        pred_path: Path,
        gt_path: Path,
        align: str,
        association: dict[str, Any],
        *,
        pose_relation: str,
        delta: float,
        delta_unit: str,
        all_pairs: bool = False,
    ) -> dict[str, Any]:
        """Relative pose error (evo RPE) for an explicit pose relation and delta.

        ``delta`` / ``delta_unit`` come from the protocol's MetricSpec parameters —
        they are never defaulted here, because RPE at delta=1 frame is not
        comparable to RPE at delta=1 second.
        """
        from evo.core import metrics

        if pose_relation not in RPE_POSE_RELATIONS.values():
            raise MetricError(
                f"RPE pose relation '{pose_relation}' is not supported; choose one "
                f"of: {', '.join(sorted(set(RPE_POSE_RELATIONS.values())))}."
            )
        if delta_unit not in _RPE_DELTA_UNITS:
            raise MetricError(
                f"RPE delta_unit '{delta_unit}' is not supported; choose one of: "
                f"{', '.join(_RPE_DELTA_UNITS)}."
            )

        gt_assoc, pred_aligned, meta = self._associate_and_align(
            pred_path, gt_path, align, association
        )
        rpe = metrics.RPE(
            getattr(metrics.PoseRelation, pose_relation),
            delta=float(delta),
            delta_unit=getattr(metrics.Unit, delta_unit),
            all_pairs=all_pairs,
        )
        try:
            rpe.process_data((gt_assoc, pred_aligned))
        except metrics.MetricsException as exc:
            raise MetricError(
                f"RPE ({pose_relation}, delta={delta} {delta_unit}) failed on "
                f"{meta['n_associated']} associated pose(s) between {pred_path} "
                f"and {gt_path}: {exc}"
            ) from exc
        return {
            "metric": "rpe",
            "pose_relation": pose_relation,
            "unit": "deg" if pose_relation == "rotation_angle_deg" else "m",
            "delta": float(delta),
            "delta_unit": delta_unit,
            "all_pairs": all_pairs,
            "stats": {k: float(v) for k, v in rpe.get_all_statistics().items()},
            **meta,
        }
