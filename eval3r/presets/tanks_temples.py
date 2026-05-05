"""Tanks & Temples preset."""

from __future__ import annotations

TANKS_TEMPLES_PRESET = {
    "dataset": "tanks_temples",
    "unit": "m",
    "align": "none",
    "sample_method": "area",
    "samples": 200_000,
    "seed": 42,
    "thresholds": [0.05],
    "chamfer_variant": "l1_mean_bidirectional",
}
