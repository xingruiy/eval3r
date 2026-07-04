"""Sample stage: protocol-defined sampling into an ``(N, 3)`` point array.

Meshes are sampled on their surface via the mesh backend (raw vertices are never
used unless a protocol explicitly asks for them). Point clouds are returned whole
for ``all_points`` / ``none``, or randomly subsampled (without replacement) for a
declared point count. The sampling seed is protocol-pinned and recorded: an explicit
integer is used verbatim; ``derive`` / ``None`` derives a stable per-scene seed from
a base seed and the scene id (``.agent/metrics.md`` recommended default).
"""

from __future__ import annotations

import hashlib

import numpy as np

from eval3r.core.errors import MetricError
from eval3r.core.registry import MeshBackend
from eval3r.core.schema import SamplingSideSpec
from eval3r.pipeline.stages.load import LoadedGeometry

DEFAULT_BASE_SEED = 0


def derive_seed(
    spec_seed: int | str | None, scene_id: str, *, base_seed: int = DEFAULT_BASE_SEED
) -> int:
    """Resolve the seed for a sampling side.

    An explicit ``int`` is returned unchanged. ``"derive"`` or ``None`` derives a
    deterministic 32-bit seed from ``base_seed`` and ``scene_id`` so repeated runs
    match while different scenes differ.
    """
    if isinstance(spec_seed, int):
        return spec_seed
    digest = hashlib.sha256(f"{base_seed}:{scene_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def sample_geometry(
    geometry: LoadedGeometry,
    spec: SamplingSideSpec,
    *,
    scene_id: str,
    role: str,
    mesh_backend: MeshBackend,
    base_seed: int = DEFAULT_BASE_SEED,
) -> tuple[np.ndarray, int]:
    """Return ``(points, seed_used)`` for one geometry side under ``spec``."""
    seed = derive_seed(spec.seed, f"{scene_id}:{role}", base_seed=base_seed)

    if geometry.kind == "mesh":
        if spec.method in ("all_points", "none"):
            raise MetricError(
                f"{role} is a mesh but sampling method is '{spec.method}'. Meshes must be "
                f"surface-sampled before distance metrics; set sampling method to 'surface_area' "
                f"with a point count. Raw mesh vertices are not used unless a protocol explicitly "
                f"requests them."
            )
        if spec.method in ("surface_area", "uniform_points", "random_points"):
            if spec.n_points is None:
                raise MetricError(
                    f"{role} mesh surface sampling ('{spec.method}') requires an explicit point "
                    f"count (sampling.{role}.n_points). Pass --sample or set it in the protocol."
                )
            points = mesh_backend.sample_surface(geometry.mesh, spec.n_points, seed)
            return np.asarray(points, dtype=np.float64), seed
        raise MetricError(
            f"sampling method '{spec.method}' is not supported for a mesh input ({role})."
        )

    # point cloud
    assert geometry.points is not None
    points = np.asarray(geometry.points, dtype=np.float64)
    if spec.method in ("all_points", "none") or spec.n_points is None:
        return points, seed
    if spec.method in ("uniform_points", "random_points"):
        n = int(spec.n_points)
        if n >= points.shape[0]:
            return points, seed
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(points.shape[0], size=n, replace=False))
        return points[idx], seed
    if spec.method == "voxel_downsample":
        raise MetricError(
            f"voxel_downsample sampling ({role}) is not implemented on the single-file geometry "
            f"path (task 007). Use 'all_points' or a point-count subsample."
        )
    raise MetricError(f"unsupported sampling method '{spec.method}' for a point cloud ({role}).")
