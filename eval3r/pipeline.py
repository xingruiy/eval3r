"""High-level Pipeline that composes sampler / aligner / filters / metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from eval3r.alignment.base import AlignResult
from eval3r.filtering.base import BaseFilter
from eval3r.io.geometry import MeshData, PointCloudData
from eval3r.metrics.base import GeometryMetric
from eval3r.metrics.metric3d import _nn_dists
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
        *,
        debug_plot_path: str | Path | None = None,
    ) -> PipelineResult:
        """Run sample → align → filter → metric and return a :class:`PipelineResult`."""
        cfg = self.config

        pred_pts = self.sampler.sample(pred, cfg.samples, seed=cfg.seed)
        gt_pts = self.sampler.sample(gt, cfg.samples, seed=cfg.seed + 1)
        del pred, gt  # release mesh/cloud memory before alignment + KDTree phase

        if self.aligner is None:
            al = AlignResult(
                scale=1.0,
                rotation=np.eye(3),
                translation=np.zeros(3),
                mode="none",
            )
        else:
            al = self.aligner.align(pred_pts, gt_pts)  # type: ignore[attr-defined]

        pred_aligned: np.ndarray = (
            al.transform(pred_pts) if al.mode != "none" else pred_pts
        )

        n_visible = n_total = len(pred_aligned)
        for f in self.filters:
            pred_aligned, n_visible, n_total = f.filter_points(pred_aligned)

        if debug_plot_path is not None:
            from eval3r.utils.debug_plot import save_debug_plot
            save_debug_plot(
                gt_pts, pred_aligned, str(debug_plot_path),
                al.mode, al.scale,
                rotation=al.rotation, translation=al.translation,
                pred_poses=al.pred_poses,
                gt_poses=al.gt_poses,
                pred_convention=al.pred_convention,
                gt_convention=al.gt_convention,
                matched_pred_idx=al.matched_pred_idx,
                matched_gt_idx=al.matched_gt_idx,
            )

        d_pg = _nn_dists(pred_aligned, gt_pts) if self.metrics else None
        d_gp = _nn_dists(gt_pts, pred_aligned) if self.metrics else None

        values: dict[str, Any] = {}
        for m in self.metrics:
            values[m.name] = m(pred_aligned, gt_pts, d_pg=d_pg, d_gp=d_gp)

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
