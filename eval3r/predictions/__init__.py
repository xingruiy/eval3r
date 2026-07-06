"""Official eval3r-native prediction layout (task 019).

A self-contained, relocatable on-disk layout for a method's predictions:
``manifest.yaml`` at the root plus one directory per scene holding canonically
named files. :class:`~eval3r.predictions.writer.PredictionWriter` exports into
the layout; :func:`~eval3r.predictions.reader.read_prediction_dir` resolves and
verifies it. ``e3r benchmark run`` consumes the layout unchanged because
``load_or_infer_manifest`` already honors ``<pred_root>/manifest.yaml``.
"""

from __future__ import annotations

from eval3r.predictions.reader import (
    PredictionCheck,
    check_prediction_dir,
    read_prediction_dir,
)
from eval3r.predictions.writer import PredictionWriter

__all__ = [
    "PredictionWriter",
    "PredictionCheck",
    "check_prediction_dir",
    "read_prediction_dir",
]
