"""Pose/trajectory metrics (``.agent/metrics.md`` "Pose metrics").

The numeric work (association, alignment, ATE/RPE) is delegated to the
``trajectory`` backend (evo); this module owns the protocol-facing layer:
validating the alignment spec, requiring explicit statistics and RPE delta
parameters, normalizing backend result dicts into :class:`MetricResult`s, and
the ``alignment_scale_error`` diagnostic.

Metric names (result keys):

```text
ate                    absolute trajectory error (APE, translation part)    metres
rpe_translation        relative pose error, translation part               metres
rpe_rotation           relative pose error, rotation angle                 degrees
alignment_scale_error  |ln s| of the estimated alignment scale             unitless
```

Nothing is defaulted silently: an ``ate``/``rpe_*`` spec without an explicit
``statistic`` fails, an ``rpe_*`` spec without explicit ``delta``/``delta_unit``
parameters fails, and the alignment mode must be one of the first-class
trajectory modes. ATE under Sim3 alignment is not comparable to ATE under SE3,
so mode + estimated scale appear in every result's metadata.
"""

from __future__ import annotations

import math
from typing import Any

from eval3r.core.errors import AlignmentError, MetricError
from eval3r.core.result import MetricResult
from eval3r.core.schema import AlignmentSpec, MetricSpec

POSE_METRIC_NAMES = frozenset(
    {"ate", "rpe_translation", "rpe_rotation", "alignment_scale_error"}
)

# First-class AlignmentSpec.mode values for trajectories. "sim3"/"se3" CLI
# shorthands are normalized to these by the runner before reaching this layer.
POSE_ALIGNMENT_MODES = ("none", "trajectory_se3", "trajectory_sim3")


def check_pose_alignment_spec(alignment: AlignmentSpec) -> None:
    """Fail loudly when a protocol's alignment spec is not a trajectory spec."""
    if alignment.mode not in POSE_ALIGNMENT_MODES:
        raise AlignmentError(
            f"alignment mode '{alignment.mode}' is not a trajectory alignment mode; "
            f"pose protocols must use one of: {', '.join(POSE_ALIGNMENT_MODES)}."
        )
    if alignment.mode != "none" and alignment.estimate_on != "trajectory":
        raise AlignmentError(
            f"trajectory alignment '{alignment.mode}' requires estimate_on: "
            f"trajectory (got '{alignment.estimate_on}')."
        )
    if alignment.granularity != "per_scene":
        raise AlignmentError(
            f"trajectory alignment is estimated once per trajectory pair "
            f"(granularity: per_scene); got '{alignment.granularity}'."
        )


def alignment_scale_error(scale: float) -> float:
    """``|ln s|`` — the diagnostic for how far the alignment scale is from 1."""
    if not math.isfinite(scale) or scale <= 0.0:
        raise MetricError(
            f"alignment_scale_error requires a positive finite alignment scale; "
            f"got {scale!r}."
        )
    return abs(math.log(scale))


def require_rpe_parameters(spec: MetricSpec) -> tuple[float, str, bool]:
    """Read explicit ``delta`` / ``delta_unit`` (and ``all_pairs``) from a spec."""
    delta = spec.parameters.get("delta")
    delta_unit = spec.parameters.get("delta_unit")
    if delta is None or delta_unit is None:
        raise MetricError(
            f"metric '{spec.name}' requires explicit 'delta' and 'delta_unit' in "
            f"its parameters (e.g. delta: 1, delta_unit: frames); RPE deltas are "
            f"never defaulted. Got parameters: {spec.parameters!r}."
        )
    return float(delta), str(delta_unit), bool(spec.parameters.get("all_pairs", False))


def _require_statistic(spec: MetricSpec) -> str:
    if spec.statistic is None:
        raise MetricError(
            f"metric '{spec.name}' requires an explicit statistic (e.g. rmse, "
            f"mean, median); statistics are never defaulted for pose metrics."
        )
    return spec.statistic


def pose_metric_result(
    spec: MetricSpec,
    backend_result: dict[str, Any],
    *,
    protocol: str,
    protocol_hash: str,
    backend_name: str,
    scene_id: str,
    extra_metadata: dict[str, Any] | None = None,
) -> MetricResult:
    """Normalize one evo backend result dict into a :class:`MetricResult`.

    For ``ate``/``rpe_*`` the value is the spec's explicit statistic over the
    backend's error series; for ``alignment_scale_error`` it is ``|ln s|`` of the
    backend's estimated alignment scale.
    """
    if spec.name not in POSE_METRIC_NAMES:
        raise MetricError(
            f"unsupported pose metric '{spec.name}'; supported metrics: "
            f"{', '.join(sorted(POSE_METRIC_NAMES))}."
        )

    alignment: dict[str, Any] = backend_result["alignment"]
    metadata: dict[str, Any] = {
        "alignment_mode": alignment["mode"],
        "alignment_scale": alignment["scale"],
        "association": backend_result["association"],
        "n_associated": backend_result["n_associated"],
        "n_dropped_pred": backend_result["n_dropped_pred"],
        "n_dropped_gt": backend_result["n_dropped_gt"],
        **(extra_metadata or {}),
    }

    statistic: str | None
    if spec.name == "alignment_scale_error":
        value = alignment_scale_error(float(alignment["scale"]))
        unit = None
        statistic = None
    else:
        statistic = _require_statistic(spec)
        stats: dict[str, float] = backend_result["stats"]
        if statistic not in stats:
            raise MetricError(
                f"statistic '{statistic}' for metric '{spec.name}' is not produced "
                f"by the {backend_name} backend; available statistics: "
                f"{', '.join(sorted(stats))}."
            )
        value = float(stats[statistic])
        unit = backend_result["unit"]
        metadata["statistics"] = stats
        metadata["pose_relation"] = backend_result["pose_relation"]
        if backend_result["metric"] == "rpe":
            metadata["delta"] = backend_result["delta"]
            metadata["delta_unit"] = backend_result["delta_unit"]
            metadata["all_pairs"] = backend_result["all_pairs"]

    return MetricResult(
        name=spec.name,
        value=value,
        unit=unit,
        statistic=statistic,
        scene_id=scene_id,
        protocol=protocol,
        protocol_hash=protocol_hash,
        backend=backend_name,
        n_points_pred=backend_result["n_pred_poses"],
        n_points_gt=backend_result["n_gt_poses"],
        metadata=metadata,
    )
