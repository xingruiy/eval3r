"""Depth metrics in image space (``.agent/metrics.md`` "Depth metrics").

Operates on prediction/ground-truth depth arrays that are already in metres (the
depth IO backend applies ``depth_unit``). Every choice — invalid-value handling,
scale alignment mode and granularity, δ thresholds — comes from the protocol's
``MaskingSpec`` / ``AlignmentSpec`` / ``MetricSpec``; nothing is chosen silently.

Metric names (the δ names disambiguate 1.25 / 1.25² / 1.25³, since aggregation
keys metrics by name):

```text
absrel    mean(|pred - gt| / gt)                       unitless
sqrel     mean((pred - gt)^2 / gt)                     metres
rmse      sqrt(mean((pred - gt)^2))                    metres
rmse_log  sqrt(mean((ln pred - ln gt)^2))              unitless
silog     sqrt(mean(d^2) - mean(d)^2), d = ln pred - ln gt   (Eigen et al.
          scale-invariant log error, reported unscaled — not multiplied by 100)
delta_1   fraction of pixels with max(pred/gt, gt/pred) < threshold (1.25)
delta_2   same at threshold 1.25² = 1.5625
delta_3   same at threshold 1.25³ = 1.953125
```

δ thresholds are still explicit in the protocol; the ``_1/_2/_3`` suffix is a
naming convention, not a hidden default — a ``delta_*`` spec without a threshold
fails loudly.

Depth sequences are aggregated over frames (see the depth runner). If an
evaluation workflow converts them into meshes, fused point clouds, or TSDF
volumes, that conversion belongs in an explicit pipeline/backend stage rather
than hidden metric behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from eval3r.core.errors import InvalidDepthError, MetricError
from eval3r.core.result import MetricResult
from eval3r.core.schema import AlignmentSpec, MaskingSpec, MetricSpec

# Metrics computable from a masked (pred, gt) depth pair.
DEPTH_METRIC_NAMES = frozenset(
    {"absrel", "sqrel", "rmse", "rmse_log", "silog", "delta_1", "delta_2", "delta_3"}
)

# Depth scale-alignment modes (first-class AlignmentSpec.mode values).
DEPTH_ALIGNMENT_MODES = ("none", "scale_median", "scale_least_squares", "scale_affine")
DEPTH_ALIGNMENT_GRANULARITIES = ("per_frame", "per_sequence", "per_scene")

# Metrics whose value carries the metre unit (sqrel = mean(m² / m) = m).
_METRIC_UNITS = {"rmse": "m", "sqrel": "m"}


# --- validation and masking ------------------------------------------------------


def validate_depth_pair(
    pred: np.ndarray, gt: np.ndarray, *, frame: str
) -> tuple[np.ndarray, np.ndarray]:
    """Validate that pred/gt are same-shape 2D depth arrays; return float64 views."""
    pred_arr = np.asarray(pred, dtype=np.float64)
    gt_arr = np.asarray(gt, dtype=np.float64)
    for role, arr in (("pred", pred_arr), ("gt", gt_arr)):
        if arr.ndim != 2:
            raise InvalidDepthError(
                f"{role} depth for frame '{frame}' must be a 2D (H, W) array; "
                f"got shape {arr.shape}."
            )
        if arr.size == 0:
            raise InvalidDepthError(f"{role} depth for frame '{frame}' is empty.")
    if pred_arr.shape != gt_arr.shape:
        raise InvalidDepthError(
            f"pred and gt depth shapes differ for frame '{frame}': "
            f"pred {pred_arr.shape} vs gt {gt_arr.shape}. Resize is not performed by "
            f"eval3r; supply matching resolutions."
        )
    return pred_arr, gt_arr


@dataclass(frozen=True)
class DepthValidity:
    """Boolean valid-pixel mask plus an explicit breakdown of what was masked."""

    mask: np.ndarray
    n_pixels_total: int
    n_pixels_valid: int
    n_gt_nonfinite: int
    n_gt_invalid_value: int
    n_gt_nonpositive: int
    n_pred_nonfinite: int
    n_pred_nonpositive: int

    @property
    def valid_fraction(self) -> float:
        return self.n_pixels_valid / self.n_pixels_total

    def as_metadata(self) -> dict[str, int]:
        return {
            "n_gt_nonfinite": self.n_gt_nonfinite,
            "n_gt_invalid_value": self.n_gt_invalid_value,
            "n_gt_nonpositive": self.n_gt_nonpositive,
            "n_pred_nonfinite": self.n_pred_nonfinite,
            "n_pred_nonpositive": self.n_pred_nonpositive,
        }


def compute_depth_validity(
    pred: np.ndarray,
    gt: np.ndarray,
    masking: MaskingSpec,
    *,
    frame: str,
    extra_mask: np.ndarray | None = None,
) -> DepthValidity:
    """Build the valid-pixel mask per the protocol's masking spec.

    Always masked: non-finite pixels in either array, and GT pixels equal to any
    protocol-listed ``invalid_depth_values`` sentinel (e.g. 0, 65535). When
    ``ignore_invalid_depth`` is true (the default), non-positive GT *and pred*
    pixels are also masked — ratio and log metrics are undefined there; the counts
    stay visible in the breakdown so an all-zero prediction cannot silently score
    well (it fails with an empty mask instead). ``extra_mask`` intersects an
    externally supplied validity mask (e.g. ``load_mask``).
    """
    gt_finite = np.isfinite(gt)
    pred_finite = np.isfinite(pred)
    mask = gt_finite & pred_finite

    n_gt_invalid_value = 0
    for sentinel in masking.invalid_depth_values:
        hit = mask & (gt == sentinel)
        n_gt_invalid_value += int(np.count_nonzero(hit))
        mask &= ~hit

    n_gt_nonpositive = 0
    n_pred_nonpositive = 0
    if masking.ignore_invalid_depth:
        gt_bad = mask & (gt <= 0)
        n_gt_nonpositive = int(np.count_nonzero(gt_bad))
        mask &= ~gt_bad
        pred_bad = mask & (pred <= 0)
        n_pred_nonpositive = int(np.count_nonzero(pred_bad))
        mask &= ~pred_bad

    if extra_mask is not None:
        extra = np.asarray(extra_mask, dtype=bool)
        if extra.shape != mask.shape:
            raise InvalidDepthError(
                f"external validity mask shape {extra.shape} does not match depth shape "
                f"{mask.shape} for frame '{frame}'."
            )
        mask &= extra

    validity = DepthValidity(
        mask=mask,
        n_pixels_total=int(mask.size),
        n_pixels_valid=int(np.count_nonzero(mask)),
        n_gt_nonfinite=int(np.count_nonzero(~gt_finite)),
        n_gt_invalid_value=n_gt_invalid_value,
        n_gt_nonpositive=n_gt_nonpositive,
        n_pred_nonfinite=int(np.count_nonzero(~pred_finite)),
        n_pred_nonpositive=n_pred_nonpositive,
    )
    if validity.n_pixels_valid == 0:
        raise InvalidDepthError(
            f"no valid pixels remain for frame '{frame}' after masking "
            f"(total {validity.n_pixels_total}, breakdown {validity.as_metadata()}). "
            f"Check depth units, invalid_depth_values, and that pred/gt overlap."
        )
    return validity


# --- scale alignment ---------------------------------------------------------------


@dataclass(frozen=True)
class DepthScaleAlignment:
    """The scale/shift applied to the prediction, with provenance for the writer."""

    mode: str
    granularity: str
    scale: float
    shift: float
    n_pixels_used: int
    frame_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "granularity": self.granularity,
            "scale": self.scale,
            "shift": self.shift,
            "n_pixels_used": self.n_pixels_used,
            "frame_id": self.frame_id,
        }


def estimate_depth_scale(
    pred_valid: np.ndarray,
    gt_valid: np.ndarray,
    mode: str,
    *,
    context: str,
) -> tuple[float, float]:
    """Estimate (scale, shift) mapping pred toward gt over the valid pixels.

    ``scale_median`` and ``scale_least_squares`` return shift 0. ``context`` names
    the frame/sequence for error messages.
    """
    if mode == "none":
        return 1.0, 0.0
    if pred_valid.size == 0:
        raise InvalidDepthError(f"cannot estimate depth scale over zero valid pixels ({context}).")
    if mode == "scale_median":
        return float(np.median(gt_valid / pred_valid)), 0.0
    if mode == "scale_least_squares":
        denom = float(np.sum(pred_valid * pred_valid))
        if denom == 0.0:
            raise InvalidDepthError(
                f"least-squares depth scale is undefined: prediction is zero over all "
                f"valid pixels ({context})."
            )
        return float(np.sum(pred_valid * gt_valid) / denom), 0.0
    if mode == "scale_affine":
        # Least squares for gt ≈ scale * pred + shift.
        var = float(np.var(pred_valid))
        if var == 0.0:
            raise InvalidDepthError(
                f"affine depth alignment is degenerate: prediction is constant over all "
                f"valid pixels ({context}); scale and shift cannot be separated."
            )
        scale = float(np.cov(pred_valid, gt_valid, bias=True)[0, 1] / var)
        shift = float(np.mean(gt_valid) - scale * np.mean(pred_valid))
        return scale, shift
    raise MetricError(
        f"alignment mode '{mode}' is not a depth scale-alignment mode "
        f"(known: {', '.join(DEPTH_ALIGNMENT_MODES)})."
    )


def check_depth_alignment_spec(alignment: AlignmentSpec) -> None:
    """Refuse alignment specs the depth path cannot honor, explicitly."""
    if alignment.mode not in DEPTH_ALIGNMENT_MODES:
        raise MetricError(
            f"alignment mode '{alignment.mode}' is not supported for depth evaluation. "
            f"Depth protocols must use one of: {', '.join(DEPTH_ALIGNMENT_MODES)}."
        )
    if alignment.mode != "none" and alignment.granularity not in DEPTH_ALIGNMENT_GRANULARITIES:
        raise MetricError(
            f"alignment granularity '{alignment.granularity}' is not supported for depth "
            f"evaluation. Use one of: {', '.join(DEPTH_ALIGNMENT_GRANULARITIES)}."
        )


# --- formulas ----------------------------------------------------------------------


def _require_positive(values: np.ndarray, *, metric: str) -> None:
    # Masking already removed non-positive pixels; alignment (negative scale/shift)
    # can reintroduce them, which makes log/ratio metrics undefined.
    if np.any(values <= 0):
        raise MetricError(
            f"metric '{metric}' requires strictly positive depths, but the aligned "
            f"prediction contains non-positive values (alignment produced a negative "
            f"scale or shift). Inspect the alignment record for this frame."
        )


def _depth_metric_value(spec: MetricSpec, pred: np.ndarray, gt: np.ndarray) -> float:
    name = spec.name
    if name == "absrel":
        return float(np.mean(np.abs(pred - gt) / gt))
    if name == "sqrel":
        return float(np.mean((pred - gt) ** 2 / gt))
    if name == "rmse":
        return float(np.sqrt(np.mean((pred - gt) ** 2)))
    if name == "rmse_log":
        _require_positive(pred, metric=name)
        return float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2)))
    if name == "silog":
        _require_positive(pred, metric=name)
        d = np.log(pred) - np.log(gt)
        return float(np.sqrt(max(np.mean(d**2) - np.mean(d) ** 2, 0.0)))
    if name in ("delta_1", "delta_2", "delta_3"):
        if spec.threshold is None:
            raise MetricError(
                f"metric '{name}' requires an explicit threshold in its MetricSpec "
                f"(e.g. 1.25 / 1.5625 / 1.953125); δ thresholds are never defaulted."
            )
        _require_positive(pred, metric=name)
        ratio = np.maximum(pred / gt, gt / pred)
        return float(np.count_nonzero(ratio < spec.threshold) / ratio.size)
    raise MetricError(
        f"metric '{name}' is not a depth metric "
        f"(known: {', '.join(sorted(DEPTH_METRIC_NAMES))})."
    )


def evaluate_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    validity: DepthValidity,
    specs: list[MetricSpec],
    *,
    alignment: DepthScaleAlignment,
    protocol: str,
    protocol_hash: str,
    backend_name: str | None = None,
    scene_id: str | None = None,
    frame_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> list[MetricResult]:
    """Compute the depth metrics in ``specs`` for one already-aligned frame.

    ``pred`` must already have the alignment applied; ``alignment`` is recorded in
    every result's metadata (mode + granularity + scale/shift) so results are
    never presented without their scale-handling context.
    """
    pred_valid = pred[validity.mask]
    gt_valid = gt[validity.mask]
    metadata: dict[str, Any] = {
        "alignment_mode": alignment.mode,
        "alignment_granularity": alignment.granularity,
        "alignment_scale": alignment.scale,
        "alignment_shift": alignment.shift,
        "mask_breakdown": validity.as_metadata(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    results: list[MetricResult] = []
    for spec in specs:
        value = _depth_metric_value(spec, pred_valid, gt_valid)
        results.append(
            MetricResult(
                name=spec.name,
                value=value,
                unit=_METRIC_UNITS.get(spec.name),
                threshold=spec.threshold,
                statistic=spec.statistic,
                scene_id=scene_id,
                frame_id=frame_id,
                protocol=protocol,
                protocol_hash=protocol_hash,
                backend=backend_name,
                n_pixels_valid=validity.n_pixels_valid,
                valid_fraction=validity.valid_fraction,
                metadata=dict(metadata),
            )
        )
    return results
