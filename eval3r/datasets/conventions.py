"""Map recorded ``SourcePoseFormat`` strings to verified :class:`PoseConvention`s.

A prediction manifest / reconstruction / ground-truth spec records *what convention a
method emitted* as a ``SourcePoseFormat`` string (``eval3r/core/types.py``). This
module is the single source of truth that turns that string into the concrete
axis/direction :class:`PoseConvention` the transformer needs, so the mapping lives in
exactly one place rather than being re-derived at each call site.

Only unambiguous formats are mapped. ``unknown`` and formats whose base handedness has
not yet been verified against a primary parser (CO3D ``frame_annotations``, Tanks and
Temples ``.log``) raise :class:`PoseConventionError` rather than silently coercing a
guessed handedness — a wrong guess would score in the wrong frame. Verify such a format
against its primary parser and add it here, or convert it inside the adapter and declare
a base format, before it can drive a transform (CLAUDE.md dataset-adapter rules).
"""

from __future__ import annotations

from eval3r.core.errors import PoseConventionError
from eval3r.core.pose_convention import (
    INTERNAL_POSE_CONVENTION,
    PoseConvention,
)
from eval3r.core.types import NormalizedConvention, SourcePoseFormat

# SourcePoseFormat -> PoseConvention. COLMAP and MVSNet store world-to-camera
# extrinsics in OpenCV camera axes (verified against the c2w = inv(w2c) recovery in
# eval3r/backends/camera_pycolmap.py); KITTI-360 cam0->world is a c2w in OpenCV axes;
# TUM trajectories are c2w OpenCV by construction (evo convention).
_FORMAT_TO_CONVENTION: dict[SourcePoseFormat, PoseConvention] = {
    "cam_to_world_opencv": PoseConvention(axes="opencv", direction="cam_to_world"),
    "cam_to_world_opengl": PoseConvention(axes="opengl", direction="cam_to_world"),
    "world_to_cam_opencv": PoseConvention(axes="opencv", direction="world_to_cam"),
    "world_to_cam_opengl": PoseConvention(axes="opengl", direction="world_to_cam"),
    "world_to_cam_colmap": PoseConvention(axes="opencv", direction="world_to_cam"),
    "world_to_cam_mvsnet": PoseConvention(axes="opencv", direction="world_to_cam"),
    "kitti360_cam0_to_world": PoseConvention(axes="opencv", direction="cam_to_world"),
    "tum": PoseConvention(axes="opencv", direction="cam_to_world"),
}

# Formats deliberately not mapped: an unverified base handedness must raise, never be
# guessed. The value is the reason surfaced to the user.
_UNMAPPED: dict[SourcePoseFormat, str] = {
    "unknown": (
        "the source pose format is 'unknown', so its axis handedness and direction "
        "cannot be determined"
    ),
    "co3d_frame_annotations": (
        "CO3D frame_annotations handedness has not been verified against the primary "
        "CO3D parser in eval3r"
    ),
    "tanks_temples_log": (
        "the Tanks and Temples .log trajectory convention has not been verified "
        "against the primary parser in eval3r"
    ),
}


def convention_for(fmt: SourcePoseFormat) -> PoseConvention:
    """Return the verified :class:`PoseConvention` for a source pose format.

    Raises :class:`PoseConventionError` for ``unknown`` and any format whose base
    convention is not yet verified (CO3D, Tanks and Temples ``.log``), telling the user
    to convert in the adapter or declare a verified base format.
    """
    if fmt in _FORMAT_TO_CONVENTION:
        return _FORMAT_TO_CONVENTION[fmt]
    reason = _UNMAPPED.get(fmt)
    if reason is not None:
        raise PoseConventionError(
            f"cannot resolve a pose convention for source_pose_format '{fmt}': {reason}. "
            f"Convert the poses to a verified base format inside the adapter, or declare "
            f"the correct source_pose_format (one of: "
            f"{', '.join(sorted(_FORMAT_TO_CONVENTION))})."
        )
    raise PoseConventionError(
        f"source_pose_format '{fmt}' is not a known convention; expected one of "
        f"{', '.join(sorted(_FORMAT_TO_CONVENTION) + sorted(_UNMAPPED))}."
    )


def normalized_convention_target(nc: NormalizedConvention) -> PoseConvention:
    """Return the target :class:`PoseConvention` a normalized convention resolves to.

    eval3r has exactly one internal convention (``cam_to_world_opencv_meters``), whose
    pose part is :data:`INTERNAL_POSE_CONVENTION` (units are handled separately by the
    normalize stage). The argument is validated so a future normalized convention cannot
    silently fall through to the internal one.
    """
    if nc == "cam_to_world_opencv_meters":
        return INTERNAL_POSE_CONVENTION
    raise PoseConventionError(
        f"no pose-convention target is defined for normalized_convention '{nc}'; the "
        f"only internal convention is 'cam_to_world_opencv_meters'."
    )
