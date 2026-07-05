"""DTU official-like point-cloud evaluation backend (``official_eval`` kind).

A clean-room port of the official DTU point-cloud evaluation (accuracy/completeness
with ObsMask observability culling, ground-plane culling, 0.2 mm downsampling, and a
20 mm distance cap), matching the widely-used reference implementation
(``jzhangbs/DTUeval-python``, read for algorithm parity, not vendored).

Everything here works in the DTU **millimetre** model frame — ObsMask ``BB``/``Res``,
the ground plane, and the 20 mm cap are all millimetres — so the official protocol
must feed this backend millimetre points and must not normalize to metres first.

Determinism: the reference downsample randomly shuffles before a radius-dedup, which
makes it vary run-to-run by ~1e-4 mm. This port shuffles with an explicit seed so the
same inputs always give the same score; validated on real DTU data to land inside the
reference's own run-to-run band (see ``.agent/tasks/010-...`` Findings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy
from scipy.spatial import cKDTree

from eval3r.core.errors import MetricError
from eval3r.core.registry import BackendInfo

DEFAULT_DOWNSAMPLE_MM = 0.2
DEFAULT_MAX_DIST_MM = 20.0
DEFAULT_PATCH_MM = 60.0


@dataclass(frozen=True)
class DTUEvalResult:
    """Official-like DTU scores (millimetres) plus culling diagnostics."""

    accuracy: float
    completeness: float
    overall: float
    n_data_down: int
    n_data_in_obs: int
    n_stl_above: int


def _radius_dedup_downsample(points: np.ndarray, thresh: float, seed: int) -> np.ndarray:
    """Downsample so no two kept points are within ``thresh`` (mm), reference-style.

    Shuffle (seeded, for determinism), then greedily keep a point and drop every
    not-yet-kept neighbour within ``thresh``. Matches the reference radius-dedup.
    """
    if points.shape[0] == 0:
        return points
    rng = np.random.default_rng(seed)
    order = rng.permutation(points.shape[0])
    shuffled = points[order]
    tree = cKDTree(shuffled)
    neighbors = tree.query_ball_point(shuffled, r=thresh)
    keep = np.ones(shuffled.shape[0], dtype=bool)
    for curr, idxs in enumerate(neighbors):
        if keep[curr]:
            keep[idxs] = False
            keep[curr] = True
    return shuffled[keep]


def _obs_mask_cull(
    points: np.ndarray, obs_mask: np.ndarray, bb: np.ndarray, res: float, patch: float
) -> np.ndarray:
    """Keep points inside the observability volume (bounding box + ObsMask voxels)."""
    bb = bb.astype(np.float64)
    inbound = ((points >= bb[:1] - patch) & (points < bb[1:] + patch * 2)).sum(-1) == 3
    inside = points[inbound]
    grid = np.around((inside - bb[:1]) / res).astype(np.int64)
    grid_ok = ((grid >= 0) & (grid < np.array(obs_mask.shape)[None])).sum(-1) == 3
    grid_in = grid[grid_ok]
    observed = obs_mask[grid_in[:, 0], grid_in[:, 1], grid_in[:, 2]].astype(bool)
    return inside[grid_ok][observed]


def _plane_cull(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Keep points above the DTU ground plane: ``[x y z 1]·P > 0``."""
    hom = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
    above = (plane.reshape(1, 4) * hom).sum(-1) > 0
    return points[above]


def _capped_mean(distances: np.ndarray, max_dist: float) -> float:
    kept = distances[distances < max_dist]
    if kept.size == 0:
        raise MetricError(
            f"no points fall within the DTU {max_dist} mm distance cap; the prediction may be "
            f"in the wrong frame/units, or empty after ObsMask/Plane culling."
        )
    return float(kept.mean())


def evaluate_dtu(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    *,
    obs_mask: np.ndarray,
    bb: np.ndarray,
    res: float,
    plane: np.ndarray,
    downsample_mm: float = DEFAULT_DOWNSAMPLE_MM,
    max_dist_mm: float = DEFAULT_MAX_DIST_MM,
    patch_mm: float = DEFAULT_PATCH_MM,
    seed: int = 0,
) -> DTUEvalResult:
    """Official-like DTU accuracy/completeness (mm) for one (pred, gt) pair."""
    pred = np.ascontiguousarray(pred_points, dtype=np.float64)
    stl = np.ascontiguousarray(gt_points, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[1] != 3 or stl.ndim != 2 or stl.shape[1] != 3:
        raise MetricError("DTU evaluation needs (N, 3) pred and gt point arrays in millimetres.")

    data_down = _radius_dedup_downsample(pred, downsample_mm, seed)
    inbound_all = ((data_down >= bb.astype(np.float64)[:1] - patch_mm)
                   & (data_down < bb.astype(np.float64)[1:] + patch_mm * 2)).sum(-1) == 3
    data_in = data_down[inbound_all]  # used as the reference cloud for completeness
    data_in_obs = _obs_mask_cull(data_down, obs_mask, bb, res, patch_mm)
    if data_in_obs.shape[0] == 0 or data_in.shape[0] == 0:
        raise MetricError(
            "DTU prediction has no points inside the ObsMask observability volume; check the "
            "prediction frame/units and that the correct scan's ObsMask is used."
        )

    # accuracy: data -> stl
    d2s, _ = cKDTree(stl).query(data_in_obs, k=1)
    accuracy = _capped_mean(np.asarray(d2s), max_dist_mm)

    # completeness: stl (above plane) -> data_in
    stl_above = _plane_cull(stl, plane)
    if stl_above.shape[0] == 0:
        raise MetricError("DTU ground-plane culling removed all GT points; check the Plane file.")
    s2d, _ = cKDTree(data_in).query(stl_above, k=1)
    completeness = _capped_mean(np.asarray(s2d), max_dist_mm)

    return DTUEvalResult(
        accuracy=accuracy,
        completeness=completeness,
        overall=(accuracy + completeness) / 2.0,
        n_data_down=int(data_down.shape[0]),
        n_data_in_obs=int(data_in_obs.shape[0]),
        n_stl_above=int(stl_above.shape[0]),
    )


class DTUOfficialEval:
    """Validated Python port of the official DTU point-cloud evaluation."""

    name = "dtu"
    method = "validated_official_port"

    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            kind="official_eval",
            name=self.name,
            library=f"eval3r-port(scipy {scipy.__version__})",
            version=self.method,
            approximate=False,
        )

    def evaluate(
        self,
        pred_points: np.ndarray,
        gt_points: np.ndarray,
        visibility: dict[str, Any],
        *,
        downsample_mm: float = DEFAULT_DOWNSAMPLE_MM,
        max_dist_mm: float = DEFAULT_MAX_DIST_MM,
        patch_mm: float = DEFAULT_PATCH_MM,
        seed: int = 0,
    ) -> DTUEvalResult:
        """Evaluate one scene; ``visibility`` carries ObsMask/BB/Res and the Plane."""
        for key in ("obs_mask", "bb", "res", "plane"):
            if key not in visibility or visibility[key] is None:
                raise MetricError(
                    f"DTU official-like evaluation requires '{key}' (ObsMask/Plane data). "
                    f"It was missing for this scene, so official-like fidelity cannot be claimed. "
                    f"Provide the scan's ObsMask/Plane files or use an eval3r-native protocol."
                )
        return evaluate_dtu(
            pred_points, gt_points,
            obs_mask=visibility["obs_mask"], bb=visibility["bb"],
            res=float(visibility["res"]), plane=visibility["plane"],
            downsample_mm=downsample_mm, max_dist_mm=max_dist_mm, patch_mm=patch_mm, seed=seed,
        )
