"""Alignment dispatch — explicit modes only, never silent."""

from __future__ import annotations

from typing import Literal

import numpy as np

from eval3r.align.icp import icp
from eval3r.align.similarity import AlignResult, umeyama
from eval3r.utils.errors import AlignmentError
from eval3r.utils.typing import Points

AlignMode = Literal["none", "scale", "se3", "sim3", "icp"]


def align(
    source: Points,
    target: Points,
    *,
    mode: AlignMode = "none",
    correspondences: bool = False,
) -> AlignResult:
    """Estimate an alignment that maps ``source`` onto ``target``.

    With ``correspondences=True`` the two arrays are assumed to be
    point-to-point matched (Umeyama). Otherwise ICP is used after the
    closed-form initialisation in ``mode``.
    """
    if mode == "none":
        return AlignResult(scale=1.0, rotation=np.eye(3), translation=np.zeros(3), mode="none")
    if correspondences:
        if mode in ("scale", "se3", "sim3"):
            return umeyama(source, target, mode=mode)
        raise AlignmentError(
            f"align mode '{mode}' is not valid with correspondences=True; "
            f"use 'scale', 'se3', or 'sim3'."
        )
    if mode == "icp":
        return icp(source, target)
    if mode in ("se3", "sim3"):
        return icp(source, target, estimate_scale=(mode == "sim3"))
    if mode == "scale":
        # crude isotropic-scale-only fit: ratio of bbox extents.
        src = np.asarray(source, dtype=np.float64)
        tgt = np.asarray(target, dtype=np.float64)
        s_ext = float(np.linalg.norm(src.max(0) - src.min(0)))
        t_ext = float(np.linalg.norm(tgt.max(0) - tgt.min(0)))
        if s_ext == 0:
            raise AlignmentError("scale alignment: source has zero extent")
        s = t_ext / s_ext
        return AlignResult(scale=s, rotation=np.eye(3), translation=np.zeros(3), mode="scale")
    raise AlignmentError(f"Unknown alignment mode: {mode!r}")


__all__ = ["align", "AlignMode", "AlignResult", "umeyama", "icp"]
