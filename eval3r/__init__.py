"""eval3r — toolkit for saving, evaluating, and visualizing 3D reconstruction predictions."""

from eval3r._version import __version__
from eval3r.metrics.geometry import Evaluator, evaluate_geometry
from eval3r.prediction.reader import PredictionReader
from eval3r.prediction.writer import PredictionWriter

__all__ = [
    "__version__",
    "PredictionWriter",
    "PredictionReader",
    "Evaluator",
    "evaluate_geometry",
]
