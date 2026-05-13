"""High-level Pipeline that composes sampler / aligner / filters / metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eval3r.alignment.base import AlignResult, IdentityAligner
from eval3r.filtering.base import BaseFilter
from eval3r.io.geometry import MeshData, PointCloudData
from eval3r.metrics.base import GeometryMetric
from eval3r.sampling.base import PointSampler
from eval3r.utils.typing import Points


@dataclass
class EvalConfig:
    """Fine-grained configuration knobs for Pipeline."""

    samples: int = 200_000
    seed: int = 42


@dataclass
class PipelineResult:
    """Result returned by :meth:`Pipeline.evaluate`."""

    values: dict[str, Any]
    """metric.name → raw output (float or 3-tuple for fscore)."""
    n_samples: int
    n_visible: int
    n_total: int
    align_mode: str
    align_scale: float

    def summary(self) -> str:
        lines: list[str] = []
        for k, v in self.values.items():
            if isinstance(v, tuple) and len(v) == 3:
                lines.append(f"{k:<20} f={v[0]:.4f}  P={v[1]:.4f}  R={v[2]:.4f}")
            else:
                lines.append(f"{k:<20} {v:.6f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k, v in self.values.items():
            if isinstance(v, tuple) and len(v) == 3:
                d[k] = {"f": v[0], "precision": v[1], "recall": v[2]}
            else:
                d[k] = v
        d.update({
            "n_samples": self.n_samples,
            "n_visible": self.n_visible,
            "n_total": self.n_total,
            "align_mode": self.align_mode,
            "align_scale": self.align_scale,
        })
        return d


@dataclass
class Pipeline:
    """Composable 3D-reconstruction evaluation pipeline.

    Parameters
    ----------
    sampler:
        A :class:`~eval3r.sampling.base.PointSampler` instance.
    aligner:
        Any object with ``.align(src, tgt) -> AlignResult``, or ``None``
        to skip alignment. Use :class:`~eval3r.alignment.base.IdentityAligner`
        for an explicit no-op.
    filters:
        Sequence of :class:`~eval3r.filtering.base.BaseFilter` applied in
        order to the aligned prediction points.
    metrics:
        Sequence of :class:`~eval3r.metrics.base.GeometryMetric` to compute.
        Each must have a ``.name`` attribute.
    config:
        Optional :class:`EvalConfig` for sample count and seed.
    """

    sampler: PointSampler
    aligner: object | None = None
    filters: list[BaseFilter] = field(default_factory=list)
    metrics: list[GeometryMetric] = field(default_factory=list)
    config: EvalConfig = field(default_factory=EvalConfig)

    def evaluate(
        self,
        pred: Points | PointCloudData | MeshData,
        gt: Points | PointCloudData | MeshData,
    ) -> PipelineResult:
        """Run sample → align → filter → metric and return a :class:`PipelineResult`."""
        cfg = self.config

        pred_pts = self.sampler.sample(pred, cfg.samples, seed=cfg.seed)
        gt_pts = self.sampler.sample(gt, cfg.samples, seed=cfg.seed + 1)

        if self.aligner is None:
            al = AlignResult(
                scale=1.0,
                rotation=np.eye(3),
                translation=np.zeros(3),
                mode="none",
            )
        else:
            al = self.aligner.align(pred_pts, gt_pts)  # type: ignore[union-attr]

        pred_aligned: np.ndarray = (
            al.transform(pred_pts) if al.mode != "none" else pred_pts
        )

        n_visible = n_total = len(pred_aligned)
        for f in self.filters:
            pred_aligned, n_visible, n_total = f.filter_points(pred_aligned)

        values: dict[str, Any] = {}
        for m in self.metrics:
            values[m.name] = m(pred_aligned, gt_pts)

        return PipelineResult(
            values=values,
            n_samples=cfg.samples,
            n_visible=n_visible,
            n_total=n_total,
            align_mode=al.mode,
            align_scale=al.scale,
        )

    def __repr__(self) -> str:
        return (
            f"Pipeline(sampler={self.sampler!r}, aligner={self.aligner!r}, "
            f"filters={self.filters!r}, metrics={[m.name for m in self.metrics]!r})"
        )


__all__ = ["EvalConfig", "Pipeline", "PipelineResult"]
