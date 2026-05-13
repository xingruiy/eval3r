"""High-level Pipeline that composes sampler / aligner / masker / metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from eval3r.alignment import AlignMode
from eval3r.filtering import BBoxFilter
from eval3r.filtering.base import BaseFilter
from eval3r.io.geometry import MeshData, PointCloudData
from eval3r.metrics.metric3d import ChamferVariant, EvalResult3D, evaluate_geometry
from eval3r.sampling import SampleMethod
from eval3r.utils.typing import Points

_VALID_SAMPLERS: frozenset[str] = frozenset({"area", "uniform", "vertex"})
_VALID_ALIGNERS: frozenset[str] = frozenset(
    {"none", "scale", "se3", "sim3", "icp", "traj_se3", "traj_sim3"}
)
_VALID_MASKERS: frozenset[str | None] = frozenset({None, "none", "bbox"})
_KNOWN_METRICS: frozenset[str] = frozenset(
    {"chamfer", "accuracy", "completeness", "normal_consistency"}
)


def _gt_bounds(
    gt: Points | PointCloudData | MeshData,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(gt, MeshData):
        pts = np.asarray(gt.vertices, dtype=np.float64)
    elif isinstance(gt, PointCloudData):
        pts = np.asarray(gt.points, dtype=np.float64)
    else:
        pts = np.asarray(gt, dtype=np.float64)
    return pts.min(axis=0), pts.max(axis=0)


def _parse_fscore_thresholds(metrics: Sequence[str]) -> list[float]:
    thresholds: list[float] = []
    for m in metrics:
        if m.startswith("fscore@"):
            tail = m[7:]
            try:
                thresholds.append(float(tail))
            except ValueError:
                raise ValueError(
                    f"Invalid fscore metric {m!r}; expected format 'fscore@<threshold>'"
                )
    return thresholds


@dataclass
class EvalConfig:
    """Fine-grained configuration knobs for Pipeline."""

    samples: int = 200_000
    seed: int = 42
    chamfer_variant: ChamferVariant = "l1_mean_bidirectional"
    bbox_margin: float = 0.0


@dataclass
class PipelineResult:
    """Result returned by :meth:`Pipeline.evaluate`."""

    raw: EvalResult3D
    metrics: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines: list[str] = []
        for m in self.metrics:
            if m == "chamfer":
                lines.append(f"chamfer:             {self.raw.chamfer:.6f}")
            elif m == "accuracy":
                lines.append(f"accuracy:            {self.raw.accuracy:.6f}")
            elif m == "completeness":
                lines.append(f"completeness:        {self.raw.completeness:.6f}")
            elif m.startswith("fscore@"):
                thr = float(m[7:])
                v = self.raw.fscore.get(thr, {})
                f = v.get("f", float("nan"))
                p = v.get("precision", float("nan"))
                r = v.get("recall", float("nan"))
                lines.append(f"{m:<20} f={f:.4f}  P={p:.4f}  R={r:.4f}")
            # "normal_consistency" and other unknown metrics: silently skipped
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for m in self.metrics:
            if m == "chamfer":
                d["chamfer"] = self.raw.chamfer
            elif m == "accuracy":
                d["accuracy"] = self.raw.accuracy
            elif m == "completeness":
                d["completeness"] = self.raw.completeness
            elif m.startswith("fscore@"):
                thr = float(m[7:])
                d[m] = self.raw.fscore.get(thr, {})
        return d


class Pipeline:
    """Composable 3D-reconstruction evaluation pipeline.

    Parameters
    ----------
    sampler:
        Point-sampling strategy. One of ``"area"``, ``"uniform"``, ``"vertex"``.
    aligner:
        Alignment mode passed to :func:`evaluate_geometry`. One of
        ``"none"``, ``"scale"``, ``"se3"``, ``"sim3"``, ``"icp"``,
        ``"traj_se3"``, ``"traj_sim3"``.
    masker:
        Optional filter applied to prediction points after alignment.
        ``"bbox"`` builds a :class:`BBoxFilter` from the GT geometry bounds.
        ``None`` / ``"none"`` disables filtering.
    metrics:
        Which metrics to report. Supported values: ``"chamfer"``,
        ``"accuracy"``, ``"completeness"``, ``"fscore@<threshold>"``
        (e.g. ``"fscore@0.05"``). ``"normal_consistency"`` is accepted
        but silently skipped (not yet implemented).
    config:
        Optional :class:`EvalConfig` for fine-grained control of sample
        count, seed, chamfer variant, and bbox margin.
    """

    def __init__(
        self,
        *,
        sampler: str = "area",
        aligner: str = "none",
        masker: str | None = None,
        metrics: Sequence[str] = ("chamfer",),
        config: EvalConfig | None = None,
    ) -> None:
        if sampler not in _VALID_SAMPLERS:
            raise ValueError(
                f"Unknown sampler {sampler!r}. Valid options: {sorted(_VALID_SAMPLERS)}"
            )
        if aligner not in _VALID_ALIGNERS:
            raise ValueError(
                f"Unknown aligner {aligner!r}. Valid options: {sorted(_VALID_ALIGNERS)}"
            )
        if masker not in _VALID_MASKERS:
            raise ValueError(
                f"Unknown masker {masker!r}. Valid options: {sorted(str(m) for m in _VALID_MASKERS)}"
            )
        # validate fscore threshold strings eagerly
        _parse_fscore_thresholds(metrics)

        self.sampler: SampleMethod = sampler  # type: ignore[assignment]
        self.aligner: AlignMode = aligner  # type: ignore[assignment]
        self.masker = masker
        self.metrics: list[str] = list(metrics)
        self.config: EvalConfig = config or EvalConfig()

    def evaluate(
        self,
        pred: Points | PointCloudData | MeshData,
        gt: Points | PointCloudData | MeshData,
    ) -> PipelineResult:
        """Run sample → align → mask → metric and return a :class:`PipelineResult`."""
        cfg = self.config

        pred_mask: BaseFilter | None = None
        if self.masker == "bbox":
            bbox_min, bbox_max = _gt_bounds(gt)
            pred_mask = BBoxFilter(
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                margin=cfg.bbox_margin,
            )

        thresholds = _parse_fscore_thresholds(self.metrics)

        raw = evaluate_geometry(
            pred,
            gt,
            samples=cfg.samples,
            seed=cfg.seed,
            sample_method=self.sampler,
            align_mode=self.aligner,
            thresholds=thresholds if thresholds else [],
            chamfer_variant=cfg.chamfer_variant,
            pred_mask=pred_mask,
        )

        return PipelineResult(raw=raw, metrics=self.metrics)

    def __repr__(self) -> str:
        return (
            f"Pipeline(sampler={self.sampler!r}, aligner={self.aligner!r}, "
            f"masker={self.masker!r}, metrics={self.metrics!r})"
        )


__all__ = ["EvalConfig", "Pipeline", "PipelineResult"]
