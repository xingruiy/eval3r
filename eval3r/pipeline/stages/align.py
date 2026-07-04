"""Align stage: protocol-defined point-set alignment.

Supports the modes the single-file geometry path can honor without a trajectory or
registration backend: ``none``, ``se3`` (rigid Umeyama), and ``sim3`` (similarity
Umeyama, i.e. with scale). ICP (``icp``) and trajectory modes are deferred until
their backends exist (task 015+); requesting them here fails explicitly rather than
silently doing nothing.

Umeyama estimation assumes 1:1 correspondence between the pred and gt alignment
points (equal counts, matched order). That is the only correspondence a single-file
geometry comparison can assume without a registration/ICP backend, so unequal counts
fail with an explicit message instead of guessing a matching.

Per ``.agent/plan.md`` alignment policy: ICP never runs by default, and Sim3 is
disallowed for metric-scale protocols unless explicitly allowed (here via
``alignment.parameters.allow_sim3`` or ``alignment.allow_override``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eval3r.core.errors import AlignmentError
from eval3r.core.schema import AlignmentSpec
from eval3r.pipeline.stages.load import LoadedGeometry

_SCALE_MODES = {"sim3", "trajectory_sim3"}
_RIGID_MODES = {"se3", "trajectory_se3"}


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
            "parameters": self.parameters,
        }


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


def _residual_rmse(src: np.ndarray, dst: np.ndarray, matrix: np.ndarray) -> float:
    moved = src @ matrix[:3, :3].T + matrix[:3, 3]
    return float(np.sqrt(np.mean(np.sum((moved - dst) ** 2, axis=1))))


def align_geometry(
    pred: LoadedGeometry,
    gt: LoadedGeometry,
    alignment: AlignmentSpec,
    *,
    scene_id: str,
    metric_scale: bool,
) -> tuple[LoadedGeometry, AlignmentResult]:
    """Estimate and apply the protocol's alignment transform to ``pred``.

    Returns the (possibly transformed) prediction and an :class:`AlignmentResult`
    describing exactly what was applied. ``mode == "none"`` yields an identity
    transform and the unchanged prediction.
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

    if mode in {"icp"}:
        raise AlignmentError(
            f"alignment mode 'icp' is not available on the single-file geometry path "
            f"(scene '{scene_id}'): no registration/ICP backend is wired yet (task 015+). "
            f"Use 'none', 'se3', or 'sim3', or supply an aligned prediction."
        )
    if mode.startswith("trajectory_") or mode.startswith("scale_"):
        raise AlignmentError(
            f"alignment mode '{mode}' needs a trajectory/depth alignment source and is not "
            f"available on the single-file geometry path (scene '{scene_id}'). "
            f"Use 'none', 'se3', or 'sim3'."
        )
    if mode not in _RIGID_MODES | _SCALE_MODES:
        raise AlignmentError(f"unsupported alignment mode '{mode}' (scene '{scene_id}').")

    with_scale = mode in _SCALE_MODES
    if with_scale and metric_scale:
        allowed = bool(alignment.allow_override) or bool(alignment.parameters.get("allow_sim3"))
        if not allowed:
            raise AlignmentError(
                f"Sim3 alignment (mode '{mode}') rescales the prediction and is disallowed for "
                f"metric-scale protocols unless explicitly allowed (scene '{scene_id}'). "
                f"Set alignment.parameters.allow_sim3 = true (or alignment.allow_override = true) "
                f"in the protocol to permit scale correction, and know that the reported metrics "
                f"then no longer reflect metric-scale error."
            )

    src = pred.alignment_points()
    dst = gt.alignment_points()
    if src.shape[0] != dst.shape[0]:
        raise AlignmentError(
            f"alignment mode '{mode}' uses Umeyama, which requires 1:1 correspondence between "
            f"prediction and ground-truth points (scene '{scene_id}'): got "
            f"{src.shape[0]} pred vs {dst.shape[0]} gt points. Provide corresponded point sets, "
            f"or use mode 'none' (correspondence-free ICP is deferred to task 015+)."
        )

    matrix, scale = umeyama(src, dst, with_scale=with_scale)
    result = AlignmentResult(
        scene_id=scene_id,
        mode=mode,
        solver="umeyama",
        estimate_on=alignment.estimate_on if alignment.estimate_on != "none" else "pointcloud",
        matrix=matrix.tolist(),
        scale=scale,
        residual_rmse=_residual_rmse(src, dst, matrix),
        n_correspondences=int(src.shape[0]),
        parameters=dict(alignment.parameters),
    )
    return pred.transformed(matrix), result
