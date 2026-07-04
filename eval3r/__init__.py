"""eval3r: dataset-aware 3D reconstruction evaluation under explicit protocols.

Evaluate meshes, point clouds, depth predictions, and trajectories under explicit,
dataset-aware protocols.

The remaining public API (``evaluate_geometry``, ``run_benchmark``, ``diff_runs``)
is wired up in later task slices.
"""

from __future__ import annotations

from eval3r.protocols import load_protocol

__version__ = "0.3.0"

__all__ = ["load_protocol", "__version__"]
