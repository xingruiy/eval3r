"""ScanNet preset — deferred to v0.2."""

from __future__ import annotations

from eval3r.utils.errors import MissingOptionalDependencyError

# Spec snapshot (will become a real Pydantic config in v0.2):
SCANNET_PRESET = {
    "dataset": "scannet",
    "unit": "m",
    "align": "none",
    "sample_method": "area",
    "samples": 200_000,
    "seed": 42,
    "thresholds": [0.05],
    "chamfer_variant": "l1_mean_bidirectional",
}


def load_scannet_preset() -> dict:
    """Return the ScanNet preset as a plain dict.

    The full preset (with mask handling, GT path resolution, etc.) is deferred.
    """
    raise MissingOptionalDependencyError(
        "ScanNet preset is not implemented in eval3r v0.1. "
        "The static settings are available as eval3r.presets.scannet.SCANNET_PRESET."
    )
