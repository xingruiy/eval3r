"""Prediction adaptation resolution.

The protocol hash represents the scientific contract, including the allowed
adaptation envelope. This module resolves one prediction's declared provenance
and optional compact override into an effective, non-hashed adaptation record and
an evaluation protocol copy with the effective alignment mode.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from eval3r.core.errors import AlignmentError
from eval3r.core.manifest import PredictionManifest
from eval3r.core.protocol import EvalProtocol
from eval3r.core.schema import E3RModel, Reconstruction
from eval3r.core.types import AlignmentMode, ScaleType, SourcePoseFormat, WorldAxes


class AdaptationOverride(E3RModel):
    direction: str | None = None
    axes: WorldAxes | None = None
    world_frame: WorldAxes | None = None
    scale: ScaleType | None = None
    alignment_mode: AlignmentMode | None = None
    unit: str | None = None


class AdaptationRecord(E3RModel):
    pose_convention: SourcePoseFormat | None = None
    world_frame: WorldAxes | None = None
    unit: str | None = None
    scale: ScaleType | None = None
    alignment: AlignmentMode
    reason: str
    source: str
    within_envelope: bool
    transformed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


_DIRECTION_ALIASES = {
    "cw": "cam_to_world",
    "c2w": "cam_to_world",
    "cam_to_world": "cam_to_world",
    "wc": "world_to_cam",
    "w2c": "world_to_cam",
    "world_to_cam": "world_to_cam",
}
_AXES_ALIASES: dict[str, WorldAxes] = {
    "opencv": "opencv",
    "cv": "opencv",
    "opengl": "opengl",
    "gl": "opengl",
}
_SCALE_ALIASES: dict[str, ScaleType] = {
    "metric": "metric",
    "relative": "relative",
    "unknown": "unknown",
}
_ALIGNMENT_ALIASES: dict[str, AlignmentMode] = {
    "none": "none",
    "se3": "se3",
    "rigid": "se3",
    "sim3": "sim3",
    "trajectory_se3": "trajectory_se3",
    "traj_se3": "trajectory_se3",
    "trajectory_sim3": "trajectory_sim3",
    "traj_sim3": "trajectory_sim3",
    "scale_median": "scale_median",
    "median": "scale_median",
    "scale_ls": "scale_least_squares",
    "scale_least_squares": "scale_least_squares",
    "scale_affine": "scale_affine",
}
_UNIT_ALIASES = {"m": "m", "cm": "cm", "mm": "mm", "um": "um"}


def _valid_vocabulary() -> str:
    vocab = sorted(
        set(_DIRECTION_ALIASES)
        | set(_AXES_ALIASES)
        | set(_SCALE_ALIASES)
        | set(_ALIGNMENT_ALIASES)
        | set(_UNIT_ALIASES)
    )
    return ", ".join(vocab)


def parse_adaptation_tokens(text: str | None) -> AdaptationOverride | None:
    """Parse an order-independent compact adaptation string.

    Examples: ``cw@opencv@sim3``, ``sim3@gl@w2c``, ``relative@scale_ls``.
    Unknown tokens and duplicate tokens for the same adaptation axis raise with
    the full vocabulary so CLI users can fix typos without hunting docs.
    """
    if text is None or text == "":
        return None

    values: dict[str, Any] = {}
    for raw in text.split("@"):
        token = raw.strip().lower()
        if not token:
            continue
        axis: str
        value: Any
        if token in _DIRECTION_ALIASES:
            axis, value = "direction", _DIRECTION_ALIASES[token]
        elif token in _AXES_ALIASES:
            axis, value = "axes", _AXES_ALIASES[token]
        elif token in _SCALE_ALIASES:
            axis, value = "scale", _SCALE_ALIASES[token]
        elif token in _ALIGNMENT_ALIASES:
            axis, value = "alignment_mode", _ALIGNMENT_ALIASES[token]
        elif token in _UNIT_ALIASES:
            axis, value = "unit", _UNIT_ALIASES[token]
        else:
            raise AlignmentError(
                f"unknown adaptation token '{raw}' in '{text}'. Valid tokens: "
                f"{_valid_vocabulary()}."
            )
        if axis in values:
            raise AlignmentError(
                f"duplicate adaptation token for {axis}: '{raw}' in '{text}'. "
                f"Valid tokens: {_valid_vocabulary()}."
            )
        values[axis] = value
    return AdaptationOverride(**values)


def adaptation_override_from_legacy(
    *,
    adapt: str | None = None,
    align: str | None = None,
    pred_world_frame: WorldAxes | None = None,
    pred_pose_format: SourcePoseFormat | None = None,
    unit: str | None = None,
    scale: ScaleType | None = None,
) -> AdaptationOverride | None:
    """Build a single override object from the compact grammar and legacy flags."""
    override = parse_adaptation_tokens(adapt) or AdaptationOverride()
    if align is not None:
        parsed = parse_adaptation_tokens(align)
        if parsed is not None and parsed.alignment_mode is not None:
            override.alignment_mode = parsed.alignment_mode
        else:
            raise AlignmentError(
                f"unknown alignment adaptation '{align}'. Valid tokens: "
                f"{_valid_vocabulary()}."
            )
    if pred_world_frame is not None:
        override.world_frame = pred_world_frame
    if pred_pose_format is not None:
        direction, axes = _split_pose_format(pred_pose_format)
        override.direction = direction
        override.axes = axes
    if unit is not None:
        override.unit = unit
    if scale is not None:
        override.scale = scale
    if any(
        getattr(override, field) is not None
        for field in ("direction", "axes", "world_frame", "scale", "alignment_mode", "unit")
    ):
        return override
    return None


def resolve_adaptation(
    protocol: EvalProtocol,
    provenance: PredictionManifest | Reconstruction | None,
    override: AdaptationOverride | None,
) -> tuple[EvalProtocol, AdaptationRecord]:
    """Resolve one run's adaptation without changing protocol identity."""
    proto = protocol.model_copy(deep=True)
    pred_scale = _coalesce(
        override.scale if override else None, _get(provenance, "scale"), "metric"
    )
    pred_unit = _coalesce(override.unit if override else None, _get(provenance, "unit"), "m")
    world_frame = _resolve_world_frame(protocol, provenance, override)
    pose_fmt = _resolve_pose_format(protocol, provenance, override)

    allowed = list(proto.alignment.allowed_modes) or [proto.alignment.mode]
    desired = _desired_alignment(proto, pred_scale, override, allowed)
    _check_envelope(proto, desired, pred_scale, allowed)

    old_mode = proto.alignment.mode
    if desired != proto.alignment.mode:
        proto.alignment.mode = desired
        _sync_alignment_shape(proto, desired)

    reason = _reason(old_mode, desired, pred_scale, override)
    source = _source(provenance, override)
    record = AdaptationRecord(
        pose_convention=pose_fmt,
        world_frame=world_frame,
        unit=pred_unit,
        scale=pred_scale,
        alignment=desired,
        reason=reason,
        source=source,
        within_envelope=True,
        transformed=_transformed(protocol, pose_fmt, world_frame, pred_unit, old_mode, desired),
        metadata={
            "allowed_modes": allowed,
            "scale_resolution": proto.alignment.scale_resolution,
        },
    )
    return proto, record


def _get(obj: Any, name: str) -> Any:
    return getattr(obj, name, None) if obj is not None else None


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _resolve_world_frame(
    protocol: EvalProtocol,
    provenance: PredictionManifest | Reconstruction | None,
    override: AdaptationOverride | None,
) -> WorldAxes | None:
    if protocol.prediction_modality in {"mesh", "pointcloud", "pointmap"}:
        return _coalesce(
            override.world_frame if override else None,
            override.axes if override else None,
            _get(provenance, "world_frame"),
            "opencv",
        )
    return _coalesce(override.world_frame if override else None, _get(provenance, "world_frame"))


def _resolve_pose_format(
    protocol: EvalProtocol,
    provenance: PredictionManifest | Reconstruction | None,
    override: AdaptationOverride | None,
) -> SourcePoseFormat | None:
    if protocol.prediction_modality != "camera_trajectory":
        return _get(provenance, "source_pose_format")
    if override and (override.direction or override.axes):
        direction = override.direction or "cam_to_world"
        axes = override.axes or "opencv"
        return f"{direction}_{axes}"  # type: ignore[return-value]
    return _coalesce(_get(provenance, "source_pose_format"), "tum")


def _split_pose_format(fmt: SourcePoseFormat) -> tuple[str | None, WorldAxes | None]:
    if fmt.startswith("cam_to_world_"):
        direction = "cam_to_world"
        suffix = fmt.removeprefix("cam_to_world_")
    elif fmt.startswith("world_to_cam_"):
        direction = "world_to_cam"
        suffix = fmt.removeprefix("world_to_cam_")
    else:
        return None, None
    if suffix in {"opencv", "opengl"}:
        return direction, suffix  # type: ignore[return-value]
    return direction, None


def _desired_alignment(
    proto: EvalProtocol,
    scale: ScaleType,
    override: AdaptationOverride | None,
    allowed: list[AlignmentMode],
) -> AlignmentMode:
    if override is not None and override.alignment_mode is not None:
        if (
            override.alignment_mode != proto.alignment.mode
            and not proto.alignment.allow_override
        ):
            raise AlignmentError(
                f"protocol '{proto.name}' pins alignment mode "
                f"'{proto.alignment.mode}' and disallows legacy alignment overrides "
                f"(allow_override: false); refusing adaptation alignment "
                f"'{override.alignment_mode}'."
            )
        return override.alignment_mode
    if scale not in {"relative", "unknown"}:
        return proto.alignment.mode
    if proto.alignment.scale_resolution == "forbidden":
        return proto.alignment.mode
    modality = proto.prediction_modality
    if modality == "camera_trajectory":
        return _first_allowed(allowed, ["trajectory_sim3", "sim3"], proto.alignment.mode)
    if modality in {"single_depth", "depth_sequence"}:
        return _first_allowed(
            allowed, ["scale_median", "scale_least_squares", "scale_affine"], proto.alignment.mode
        )
    return _first_allowed(allowed, ["sim3", "trajectory_sim3"], proto.alignment.mode)


def _first_allowed(
    allowed: list[AlignmentMode], candidates: list[AlignmentMode], fallback: AlignmentMode
) -> AlignmentMode:
    for candidate in candidates:
        if candidate in allowed:
            return candidate
    return fallback


def _check_envelope(
    proto: EvalProtocol,
    desired: AlignmentMode,
    scale: ScaleType,
    allowed: list[AlignmentMode],
) -> None:
    if desired not in allowed:
        raise AlignmentError(
            f"prediction adaptation requested alignment '{desired}', but protocol "
            f"'{proto.name}' allows only {allowed}. Use a protocol whose "
            f"alignment.allowed_modes includes '{desired}', or remove the adaptation override."
        )
    if scale in {"relative", "unknown"} and proto.alignment.scale_resolution == "forbidden":
        raise AlignmentError(
            f"prediction declares scale='{scale}', but protocol '{proto.name}' forbids "
            f"scale adaptation (scale_resolution: forbidden; allowed_modes: {allowed}). "
            f"Use a metric prediction, or choose a protocol that permits scale resolution "
            f"such as sim3 / trajectory_sim3 / depth scale alignment."
        )


def _sync_alignment_shape(proto: EvalProtocol, mode: AlignmentMode) -> None:
    if mode == "none":
        proto.alignment.estimate_on = "none"
        proto.alignment.solver = "none"
    elif mode in {"trajectory_se3", "trajectory_sim3"}:
        proto.alignment.estimate_on = "trajectory"
        proto.alignment.solver = "evo"
    elif mode.startswith("scale_"):
        proto.alignment.estimate_on = "depth"
        proto.alignment.solver = "none"
    elif mode in {"se3", "sim3"} and proto.alignment.estimate_on == "none":
        proto.alignment.estimate_on = "pointcloud"
        proto.alignment.solver = "umeyama"


def _reason(
    old_mode: AlignmentMode,
    desired: AlignmentMode,
    scale: ScaleType,
    override: AdaptationOverride | None,
) -> str:
    if override and override.alignment_mode is not None:
        return "override"
    if scale in {"relative", "unknown"} and desired != old_mode:
        return "auto_scale_resolution"
    return "passthrough"


def _source(
    provenance: PredictionManifest | Reconstruction | None,
    override: AdaptationOverride | None,
) -> str:
    if override is not None:
        return "override+provenance" if provenance is not None else "override"
    return "provenance" if provenance is not None else "default"


def _transformed(
    protocol: EvalProtocol,
    pose_fmt: SourcePoseFormat | None,
    world_frame: WorldAxes | None,
    unit: str | None,
    old_mode: AlignmentMode,
    desired: AlignmentMode,
) -> bool:
    if desired != old_mode:
        return True
    if unit not in (None, "m"):
        return True
    if world_frame not in (None, "opencv"):
        return True
    return bool(
        protocol.prediction_modality == "camera_trajectory"
        and pose_fmt not in (None, "tum", "cam_to_world_opencv")
    )
