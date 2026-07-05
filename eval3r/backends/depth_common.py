"""Shared depth-IO helpers used by the imageio and OpenCV depth backends.

Both backends must produce the same contract from ``load_depth``: a 2D float64
array in **metres**. ``depth_unit`` is the multiplier from stored values to metres
(e.g. 0.001 for 16-bit millimetre PNGs). It is required for integer-typed sources,
where units are never self-describing; float sources default to already-metric
(unit 1.0) unless a unit is given (``.agent/backends.md`` depth IO rules).

Invalid values (zeros, sentinels like 65535, NaN/Inf) are *not* masked here — they
are passed through to the protocol-controlled masking in ``eval3r.metrics.depth``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eval3r.core.errors import InvalidDepthError

# Array-file suffixes both backends delegate straight to numpy.
NUMPY_SUFFIXES = (".npy",)


def load_npy_depth(path: Path) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise InvalidDepthError(f"failed to load depth array from '{path}': {exc}") from exc


def to_metric_depth(raw: np.ndarray, path: Path, depth_unit: float | None) -> np.ndarray:
    """Squeeze/validate a raw decoded array and convert it to metres."""
    arr = np.asarray(raw)
    # Tolerate a trailing singleton channel axis (some encoders emit (H, W, 1)).
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise InvalidDepthError(
            f"depth file '{path}' decodes to shape {arr.shape}; depth must be a "
            f"single-channel 2D image. Multi-channel images are not depth maps."
        )
    if arr.size == 0:
        raise InvalidDepthError(f"depth file '{path}' decodes to an empty array.")

    if np.issubdtype(arr.dtype, np.integer):
        if depth_unit is None:
            raise InvalidDepthError(
                f"depth file '{path}' stores integer values ({arr.dtype}), whose unit is "
                f"not self-describing. Pass an explicit depth_unit (metres per stored "
                f"unit, e.g. 0.001 for millimetre PNGs)."
            )
        return arr.astype(np.float64) * depth_unit
    if np.issubdtype(arr.dtype, np.floating):
        out = arr.astype(np.float64)
        return out if depth_unit is None else out * depth_unit
    raise InvalidDepthError(
        f"depth file '{path}' has unsupported dtype {arr.dtype}; expected an integer "
        f"or floating-point single-channel array."
    )


def to_bool_mask(raw: np.ndarray, path: Path) -> np.ndarray:
    """Decode a mask image/array to a 2D bool array (nonzero = valid)."""
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise InvalidDepthError(
            f"mask file '{path}' decodes to shape {arr.shape}; masks must be "
            f"single-channel 2D images."
        )
    return arr != 0
