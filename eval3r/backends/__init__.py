"""Backend adapters behind stable interfaces.

Use :func:`eval3r.core.registry.default_registry` to obtain the process-wide
registry with the built-in backends registered. Individual backend classes are
importable directly for typing or explicit construction.
"""

from __future__ import annotations

from eval3r.backends.dtu_eval import DTUOfficialEval
from eval3r.backends.mesh_trimesh import TrimeshMeshBackend
from eval3r.backends.nn_scipy import ScipyNNBackend
from eval3r.backends.pointcloud_plyfile import PlyfilePointCloudBackend

__all__ = [
    "TrimeshMeshBackend",
    "ScipyNNBackend",
    "PlyfilePointCloudBackend",
    "DTUOfficialEval",
]
