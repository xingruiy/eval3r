"""DTU preset."""

from __future__ import annotations

DTU_PRESET = {
    "dataset": "dtu",
    "unit": "mm",
    "align": "none",
    "sample_method": "area",
    "samples": 200_000,
    "seed": 42,
    "thresholds": [1.0, 2.0, 5.0],
    "chamfer_variant": "l1_mean_bidirectional",
}
