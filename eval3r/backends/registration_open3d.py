"""Open3D registration backend (``registration`` registry kind).

Closest-point (point-to-point) ICP only. Per explicit project decision there is **no
FPFH / feature-descriptor global registration** in eval3r: geometry alignment is
either closest-point ICP from a recorded coarse init, or trajectory-first Umeyama
propagation (see ``pipeline/stages/align.py``). Quirky alignments are surfaced by the
mandatory visualization artifacts, not hidden behind a fancier solver.

Everything that shapes the estimate is explicit and returned for result metadata:
the init transform, the max correspondence distance (never defaulted here), the
iteration cap and convergence criteria, and the fitness / inlier RMSE /
correspondence count that Open3D reports. There is no RANSAC or seeded randomness
(nothing to record); results are reproducible up to Open3D's multithreaded
floating-point reduction order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eval3r.core.errors import AlignmentError
from eval3r.core.registry import BackendInfo


@dataclass
class RegistrationOutcome:
    """One ICP run's estimate plus every parameter that shaped it."""

    matrix: np.ndarray  # 4x4, maps source coordinates into target coordinates
    scale: float  # cbrt(det) of the linear part; 1.0 for rigid ICP
    fitness: float  # inlier fraction of source points (Open3D definition)
    inlier_rmse: float  # RMSE over inlier correspondences
    n_correspondences: int
    parameters: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "matrix": np.asarray(self.matrix).tolist(),
            "scale": self.scale,
            "fitness": self.fitness,
            "inlier_rmse": self.inlier_rmse,
            "n_correspondences": self.n_correspondences,
            "parameters": self.parameters,
        }


def _validate_points(points: np.ndarray, *, role: str) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise AlignmentError(
            f"ICP {role} points must be a non-empty (N, 3) array; got shape "
            f"{tuple(pts.shape)}. Provide at least one 3D point per side."
        )
    if not np.isfinite(pts).all():
        raise AlignmentError(
            f"ICP {role} points contain NaN/Inf values; clean the geometry before "
            f"alignment (the load/normalize stages keep such values visible on purpose)."
        )
    return pts


def matrix_scale(matrix: np.ndarray) -> float:
    """Uniform scale of a 4x4 similarity transform: cbrt(det) of the linear part."""
    det = float(np.linalg.det(np.asarray(matrix, dtype=np.float64)[:3, :3]))
    if det <= 0.0 or not np.isfinite(det):
        raise AlignmentError(
            f"estimated transform has a non-positive/non-finite linear determinant "
            f"({det!r}); the registration is degenerate (reflection or collapse)."
        )
    return float(np.cbrt(det))


class Open3dRegistrationBackend:
    """Point-to-point ICP via ``open3d.pipelines.registration.registration_icp``."""

    name = "open3d"
    kind = "registration"

    def backend_info(self) -> BackendInfo:
        import open3d

        return BackendInfo(
            kind=self.kind,
            name=self.name,
            library="open3d",
            version=str(open3d.__version__),
        )

    def icp(
        self,
        source: np.ndarray,
        target: np.ndarray,
        *,
        init_transform: np.ndarray,
        max_correspondence_distance: float,
        max_iterations: int,
        with_scaling: bool = False,
        relative_fitness: float = 1e-6,
        relative_rmse: float = 1e-6,
    ) -> RegistrationOutcome:
        """Closest-point ICP mapping ``source`` onto ``target``.

        ``max_correspondence_distance`` (metres, in target units) and
        ``max_iterations`` are required — the caller (protocol or CLI) owns those
        choices; this backend never fills them in. ``with_scaling`` enables scaled
        (Sim3) point-to-point estimation.
        """
        import open3d as o3d

        src = _validate_points(source, role="source")
        dst = _validate_points(target, role="target")
        init = np.asarray(init_transform, dtype=np.float64)
        if init.shape != (4, 4):
            raise AlignmentError(
                f"ICP init transform must be a 4x4 matrix; got shape {tuple(init.shape)}."
            )
        if not (np.isfinite(max_correspondence_distance) and max_correspondence_distance > 0):
            raise AlignmentError(
                f"ICP max_correspondence_distance must be a positive number of metres; "
                f"got {max_correspondence_distance!r}. Pin it in the protocol's alignment "
                f"parameters (or pass --max-corr-dist to `e3r align`)."
            )
        if max_iterations < 1:
            raise AlignmentError(
                f"ICP max_iterations must be >= 1; got {max_iterations!r}."
            )

        reg = o3d.pipelines.registration
        source_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
        target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst))
        result = reg.registration_icp(
            source_pcd,
            target_pcd,
            max_correspondence_distance,
            init,
            reg.TransformationEstimationPointToPoint(with_scaling=with_scaling),
            reg.ICPConvergenceCriteria(
                relative_fitness=relative_fitness,
                relative_rmse=relative_rmse,
                max_iteration=max_iterations,
            ),
        )

        matrix = np.asarray(result.transformation, dtype=np.float64)
        n_corr = int(np.asarray(result.correspondence_set).shape[0])
        if n_corr == 0:
            raise AlignmentError(
                f"ICP found no correspondences within max_correspondence_distance="
                f"{max_correspondence_distance} between {src.shape[0]} source and "
                f"{dst.shape[0]} target points. The init transform is too far off or "
                f"the distance is too small; inspect the alignment_before overlay and "
                f"increase --max-corr-dist / fix the coarse init."
            )
        return RegistrationOutcome(
            matrix=matrix,
            scale=matrix_scale(matrix) if with_scaling else 1.0,
            fitness=float(result.fitness),
            inlier_rmse=float(result.inlier_rmse),
            n_correspondences=n_corr,
            parameters={
                "estimation": "point_to_point",
                "with_scaling": with_scaling,
                "init_transform": init.tolist(),
                "max_correspondence_distance": float(max_correspondence_distance),
                "max_iterations": int(max_iterations),
                "relative_fitness": relative_fitness,
                "relative_rmse": relative_rmse,
                "n_source_points": int(src.shape[0]),
                "n_target_points": int(dst.shape[0]),
            },
        )
