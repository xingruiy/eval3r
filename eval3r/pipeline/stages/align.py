"""Align stage: protocol-defined point-set alignment.

Supported estimation paths (task 018; ``.agent/schema.md`` "Alignment schema"):

===============================================  =============================================
Protocol spec                                    Meaning
===============================================  =============================================
``mode: none``                                   identity, prediction untouched
``mode: se3|sim3, solver: none|umeyama``         corresponded Umeyama (equal counts, matched
(``estimate_on`` != ``trajectory``)              order — the task-007 single-file path)
``mode: se3|sim3, solver: icp``                  closest-point ICP (Open3D point-to-point,
                                                 ``with_scaling`` = mode is sim3) from a
                                                 recorded centroid/RMS-radius init
``mode: se3|sim3, solver: umeyama,``             pred trajectory associated + Umeyama-aligned
``estimate_on: trajectory``                      onto the gt trajectory (evo backend), the
                                                 resulting 4x4 propagated to the geometry
===============================================  =============================================

There is deliberately **no feature-based global registration** (FPFH etc.): geometry
alignment is closest-point ICP or trajectory-first propagation, and quirky alignments
are surfaced by the mandatory visualization artifacts rather than hidden behind a
fancier solver.

``mode: icp`` is refused: the mode must state the transform class (``se3`` or
``sim3`` + ``solver: icp``) so scale handling is always explicit. Umeyama estimation
assumes 1:1 correspondence between the pred and gt alignment points; unequal counts
fail with an explicit message pointing at the two correspondence-free paths.

Per ``.agent/plan.md`` alignment policy: ICP never runs by default (a protocol must
select ``solver: icp``), and Sim3 is disallowed for metric-scale protocols unless
explicitly allowed (via ``alignment.parameters.allow_sim3`` or
``alignment.allow_override``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from eval3r.core.errors import AlignmentError
from eval3r.core.schema import AlignmentSpec
from eval3r.pipeline.stages.load import LoadedGeometry

_SCALE_MODES = {"sim3", "trajectory_sim3"}
_RIGID_MODES = {"se3", "trajectory_se3"}

#: Resolved ICP defaults; every resolved value is recorded in the alignment result.
DEFAULT_ICP_MAX_ITERATIONS = 50
DEFAULT_ICP_MAX_POINTS = 200_000
ICP_SUBSAMPLE_SEED = 0


@dataclass
class AlignmentResult:
    """The transform applied to the prediction plus its provenance, for the writer."""

    scene_id: str
    mode: str
    solver: str
    estimate_on: str
    matrix: list[list[float]]
    scale: float
    residual_rmse: float | None = None
    n_correspondences: int | None = None
    fitness: float | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "mode": self.mode,
            "solver": self.solver,
            "estimate_on": self.estimate_on,
            "matrix": self.matrix,
            "scale": self.scale,
            "residual_rmse": self.residual_rmse,
            "n_correspondences": self.n_correspondences,
            "fitness": self.fitness,
            "parameters": self.parameters,
        }


@dataclass
class AlignmentVisData:
    """Subsampled point captures for the mandatory alignment visualization.

    Captured whenever a non-``none`` alignment runs, so the run directory (or
    ``e3r align`` output) can always show the before/after overlays. ``pred_after``
    is the same subsample as ``pred_before`` with the estimated transform applied,
    so the two overlays are point-for-point comparable.
    """

    scene_id: str
    pred_before: np.ndarray
    pred_after: np.ndarray
    gt: np.ndarray
    alignment: dict[str, Any]
    subsample_seed: int
    max_points: int


#: Per-side point cap for visualization captures (keeps benchmark memory bounded).
DEFAULT_VIS_MAX_POINTS = 100_000


def capture_alignment_vis(
    pred_before: LoadedGeometry,
    gt: LoadedGeometry,
    result: AlignmentResult,
    *,
    max_points: int = DEFAULT_VIS_MAX_POINTS,
    seed: int = ICP_SUBSAMPLE_SEED,
) -> AlignmentVisData:
    """Capture subsampled before/after/gt points for the visualization artifacts."""
    src = _subsample(pred_before.alignment_points(), max_points, seed)
    matrix = np.asarray(result.matrix, dtype=np.float64)
    return AlignmentVisData(
        scene_id=result.scene_id,
        pred_before=src,
        pred_after=src @ matrix[:3, :3].T + matrix[:3, 3],
        gt=_subsample(gt.alignment_points(), max_points, seed),
        alignment=result.as_dict(),
        subsample_seed=seed,
        max_points=max_points,
    )


def umeyama(src: np.ndarray, dst: np.ndarray, *, with_scale: bool) -> tuple[np.ndarray, float]:
    """Least-squares similarity transform mapping ``src`` onto ``dst`` (Umeyama 1991).

    Returns a 4x4 matrix ``[[s·R, t], [0, 1]]`` and the scale ``s`` (1.0 when
    ``with_scale`` is False). ``src`` and ``dst`` must be ``(N, 3)`` corresponded.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    u, sigma, vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0
    rot = u @ np.diag(d) @ vt
    if with_scale:
        var_src = (src_c**2).sum() / n
        scale = float((sigma * d).sum() / var_src) if var_src > 0 else 1.0
    else:
        scale = 1.0
    t = mu_dst - scale * rot @ mu_src
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rot
    matrix[:3, 3] = t
    return matrix, scale


def centroid_scale_init(
    src: np.ndarray, dst: np.ndarray, *, with_scale: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    """Coarse ICP init: translate centroids together, optionally match RMS radii.

    No rotation is estimated (that would need correspondences or features); ICP
    owns the rotation. Returns the 4x4 init and a record of how it was built so
    the choice is visible in result metadata.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    scale0 = 1.0
    if with_scale:
        rms_src = float(np.sqrt(np.mean(np.sum((src - c_src) ** 2, axis=1))))
        rms_dst = float(np.sqrt(np.mean(np.sum((dst - c_dst) ** 2, axis=1))))
        if rms_src <= 0.0 or rms_dst <= 0.0:
            raise AlignmentError(
                f"cannot derive an initial scale for ICP: RMS radius is zero "
                f"(pred {rms_src}, gt {rms_dst}). All points coincide on one side."
            )
        scale0 = rms_dst / rms_src
    init = np.eye(4)
    init[:3, :3] *= scale0
    init[:3, 3] = c_dst - scale0 * c_src
    record = {
        "method": "centroid" + ("+rms_radius_scale" if with_scale else ""),
        "initial_scale": scale0,
    }
    return init, record


def _residual_rmse(src: np.ndarray, dst: np.ndarray, matrix: np.ndarray) -> float:
    moved = src @ matrix[:3, :3].T + matrix[:3, 3]
    return float(np.sqrt(np.mean(np.sum((moved - dst) ** 2, axis=1))))


def _subsample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[np.sort(idx)]


# --- solver paths ----------------------------------------------------------------


def _align_corresponded_umeyama(
    pred: LoadedGeometry,
    gt: LoadedGeometry,
    alignment: AlignmentSpec,
    *,
    scene_id: str,
    with_scale: bool,
) -> tuple[LoadedGeometry, AlignmentResult]:
    src = pred.alignment_points()
    dst = gt.alignment_points()
    if src.shape[0] != dst.shape[0]:
        raise AlignmentError(
            f"alignment solver 'umeyama' (without estimate_on: trajectory) requires 1:1 "
            f"correspondence between prediction and ground-truth points (scene "
            f"'{scene_id}'): got {src.shape[0]} pred vs {dst.shape[0]} gt points. For "
            f"non-corresponded geometries use solver 'icp' (closest-point ICP), or "
            f"estimate_on: trajectory to align trajectories and propagate to the geometry."
        )
    matrix, scale = umeyama(src, dst, with_scale=with_scale)
    result = AlignmentResult(
        scene_id=scene_id,
        mode=alignment.mode,
        solver="umeyama",
        estimate_on=alignment.estimate_on if alignment.estimate_on != "none" else "pointcloud",
        matrix=matrix.tolist(),
        scale=scale,
        residual_rmse=_residual_rmse(src, dst, matrix),
        n_correspondences=int(src.shape[0]),
        parameters=dict(alignment.parameters),
    )
    return pred.transformed(matrix), result


def _align_icp(
    pred: LoadedGeometry,
    gt: LoadedGeometry,
    alignment: AlignmentSpec,
    *,
    scene_id: str,
    with_scale: bool,
    registration_backend: Any,
) -> tuple[LoadedGeometry, AlignmentResult]:
    if registration_backend is None:
        raise AlignmentError(
            f"alignment solver 'icp' needs a registration backend, but none was "
            f"supplied to the align stage (scene '{scene_id}'). This is a wiring bug "
            f"in the calling runner, not a data problem."
        )
    if alignment.estimate_on == "trajectory":
        raise AlignmentError(
            f"alignment solver 'icp' estimates on the geometries themselves; "
            f"estimate_on: trajectory requires solver 'umeyama' (scene '{scene_id}')."
        )
    params = alignment.parameters
    mcd = params.get("max_correspondence_distance")
    if not isinstance(mcd, (int, float)):
        raise AlignmentError(
            f"alignment solver 'icp' requires an explicit numeric "
            f"'max_correspondence_distance' (metres) in alignment.parameters (scene "
            f"'{scene_id}'); got {mcd!r}. It is never defaulted: pin it in the protocol, "
            f"or use `e3r align --max-corr-dist` (which resolves 'auto' to a recorded "
            f"value) to find one."
        )
    max_iterations = int(params.get("max_iterations", DEFAULT_ICP_MAX_ITERATIONS))
    max_points = int(params.get("max_points", DEFAULT_ICP_MAX_POINTS))
    if max_points < 1:
        raise AlignmentError(
            f"alignment parameter max_points must be >= 1 (scene '{scene_id}'); "
            f"got {max_points}."
        )

    src_full = pred.alignment_points()
    dst_full = gt.alignment_points()
    src = _subsample(src_full, max_points, ICP_SUBSAMPLE_SEED)
    dst = _subsample(dst_full, max_points, ICP_SUBSAMPLE_SEED)
    init, init_record = centroid_scale_init(src, dst, with_scale=with_scale)

    outcome = registration_backend.icp(
        src,
        dst,
        init_transform=init,
        max_correspondence_distance=float(mcd),
        max_iterations=max_iterations,
        with_scaling=with_scale,
    )

    resolved = dict(alignment.parameters)
    resolved.update(outcome.parameters)
    resolved["init"] = init_record
    resolved["max_points"] = max_points
    resolved["subsample_seed"] = ICP_SUBSAMPLE_SEED
    resolved["n_points_pred_total"] = int(src_full.shape[0])
    resolved["n_points_gt_total"] = int(dst_full.shape[0])
    result = AlignmentResult(
        scene_id=scene_id,
        mode=alignment.mode,
        solver="icp",
        estimate_on="pointcloud",
        matrix=np.asarray(outcome.matrix).tolist(),
        scale=outcome.scale,
        residual_rmse=outcome.inlier_rmse,
        n_correspondences=outcome.n_correspondences,
        fitness=outcome.fitness,
        parameters=resolved,
    )
    return pred.transformed(np.asarray(outcome.matrix)), result


def _align_trajectory(
    pred: LoadedGeometry,
    alignment: AlignmentSpec,
    *,
    scene_id: str,
    with_scale: bool,
    trajectory_backend: Any,
    pred_trajectory: Path | None,
    gt_trajectory: Path | None,
) -> tuple[LoadedGeometry, AlignmentResult]:
    if trajectory_backend is None:
        raise AlignmentError(
            f"trajectory-first alignment needs a trajectory backend, but none was "
            f"supplied to the align stage (scene '{scene_id}'). This is a wiring bug "
            f"in the calling runner, not a data problem."
        )
    missing = [
        name
        for name, path in (("prediction", pred_trajectory), ("ground-truth", gt_trajectory))
        if path is None
    ]
    if missing:
        raise AlignmentError(
            f"alignment estimate_on: trajectory requires both trajectories, but the "
            f"{' and '.join(missing)} trajectory is missing for scene '{scene_id}'. "
            f"Provide it in the prediction manifest / dataset (TUM format), or use "
            f"solver 'icp' to align on the geometries instead."
        )
    if "associate_max_diff" not in alignment.parameters:
        raise AlignmentError(
            f"alignment estimate_on: trajectory requires an explicit "
            f"'associate_max_diff' (seconds) in alignment.parameters (scene "
            f"'{scene_id}'); it is never defaulted."
        )

    assert pred_trajectory is not None and gt_trajectory is not None  # checked above
    traj_mode = "trajectory_sim3" if with_scale else "trajectory_se3"
    record = trajectory_backend.align_trajectories(
        Path(pred_trajectory),
        Path(gt_trajectory),
        traj_mode,
        dict(alignment.parameters),
    )

    resolved = dict(alignment.parameters)
    resolved.update(
        {
            "trajectory_alignment_mode": traj_mode,
            "pred_trajectory": str(pred_trajectory),
            "gt_trajectory": str(gt_trajectory),
            "association": record["association"],
            "n_pred_poses": record["n_pred_poses"],
            "n_gt_poses": record["n_gt_poses"],
            "n_associated": record["n_associated"],
            "n_dropped_pred": record["n_dropped_pred"],
            "n_dropped_gt": record["n_dropped_gt"],
        }
    )
    matrix = np.asarray(record["matrix"], dtype=np.float64)
    result = AlignmentResult(
        scene_id=scene_id,
        mode=alignment.mode,
        solver="umeyama",
        estimate_on="trajectory",
        matrix=matrix.tolist(),
        scale=float(record["scale"]),
        residual_rmse=float(record["residual_rmse"]),
        n_correspondences=int(record["n_associated"]),
        parameters=resolved,
    )
    return pred.transformed(matrix), result


# --- dispatch ----------------------------------------------------------------------


def align_geometry(
    pred: LoadedGeometry,
    gt: LoadedGeometry,
    alignment: AlignmentSpec,
    *,
    scene_id: str,
    metric_scale: bool,
    registration_backend: Any | None = None,
    trajectory_backend: Any | None = None,
    pred_trajectory: Path | None = None,
    gt_trajectory: Path | None = None,
) -> tuple[LoadedGeometry, AlignmentResult]:
    """Estimate and apply the protocol's alignment transform to ``pred``.

    Returns the (possibly transformed) prediction and an :class:`AlignmentResult`
    describing exactly what was applied. ``mode == "none"`` yields an identity
    transform and the unchanged prediction. The registration/trajectory backends and
    trajectory paths are only needed for the solver paths that use them; the runner
    supplies them from the registry / manifest.
    """
    mode = alignment.mode
    identity = AlignmentResult(
        scene_id=scene_id,
        mode=mode,
        solver=alignment.solver,
        estimate_on=alignment.estimate_on,
        matrix=np.eye(4).tolist(),
        scale=1.0,
        parameters=dict(alignment.parameters),
    )

    if mode == "none":
        return pred, identity

    if mode == "icp":
        raise AlignmentError(
            f"alignment mode 'icp' is refused because it does not state the transform "
            f"class (scene '{scene_id}'): use mode 'se3' or 'sim3' with solver 'icp' "
            f"so rigid-vs-similarity scale handling stays explicit."
        )
    if mode.startswith("trajectory_") or mode.startswith("scale_"):
        raise AlignmentError(
            f"alignment mode '{mode}' belongs to the pose/depth paths and is not "
            f"available on the geometry path (scene '{scene_id}'). Use mode 'se3' or "
            f"'sim3' (solver 'umeyama' with estimate_on: trajectory gives "
            f"trajectory-first geometry alignment)."
        )
    if mode not in _RIGID_MODES | _SCALE_MODES:
        raise AlignmentError(f"unsupported alignment mode '{mode}' (scene '{scene_id}').")

    with_scale = mode in _SCALE_MODES
    if with_scale and metric_scale:
        allowed_modes = list(alignment.allowed_modes) or [alignment.mode]
        legacy_allowed = bool(alignment.allow_override) or bool(
            alignment.parameters.get("allow_sim3")
        )
        allowed = (
            alignment.scale_resolution != "forbidden" and mode in allowed_modes
        ) or legacy_allowed
        if not allowed:
            raise AlignmentError(
                f"Sim3 alignment (mode '{mode}') rescales the prediction and is disallowed for "
                f"metric-scale protocols unless explicitly allowed by the protocol adaptation "
                f"envelope (scene '{scene_id}'). Set alignment.allowed_modes to include "
                f"'{mode}' and alignment.scale_resolution to 'allowed' when scale correction "
                f"is scientifically valid for this protocol."
            )

    solver = alignment.solver
    if solver == "icp":
        return _align_icp(
            pred, gt, alignment,
            scene_id=scene_id, with_scale=with_scale,
            registration_backend=registration_backend,
        )
    if solver in {"none", "umeyama"}:
        if alignment.estimate_on == "trajectory":
            return _align_trajectory(
                pred, alignment,
                scene_id=scene_id, with_scale=with_scale,
                trajectory_backend=trajectory_backend,
                pred_trajectory=pred_trajectory, gt_trajectory=gt_trajectory,
            )
        return _align_corresponded_umeyama(
            pred, gt, alignment, scene_id=scene_id, with_scale=with_scale
        )
    raise AlignmentError(
        f"alignment solver '{solver}' is not available on the geometry path (scene "
        f"'{scene_id}'). Supported: 'umeyama' (corresponded points, or estimate_on: "
        f"trajectory), 'icp' (closest-point ICP), 'none' (defaults to umeyama)."
    )
