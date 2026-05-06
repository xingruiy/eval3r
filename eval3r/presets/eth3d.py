"""ETH3D preset."""

from __future__ import annotations

ETH3D_PRESET = {
    "dataset": "eth3d",
    "unit": "m",
    "align": "none",
    "sample_method": "area",
    "samples": 200_000,
    "seed": 42,
    "thresholds": [0.05],
    "chamfer_variant": "l1_mean_bidirectional",
}
