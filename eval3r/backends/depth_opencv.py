"""OpenCV depth IO backend.

Same ``load_depth`` contract as the imageio backend (2D float64 metres, shared
``depth_unit`` rules from :mod:`eval3r.backends.depth_common`), but decodes via
``cv2.imread(..., IMREAD_UNCHANGED)``, which additionally reads PFM files —
the common float depth interchange format imageio does not handle. ``.npy``
arrays load via numpy, as in the imageio backend.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from eval3r.backends.depth_common import (
    NUMPY_SUFFIXES,
    load_npy_depth,
    to_bool_mask,
    to_metric_depth,
)
from eval3r.core.errors import InvalidDepthError
from eval3r.core.registry import BackendInfo


class OpenCVDepthBackend:
    """Depth/mask image loading via OpenCV (PNG, PFM, ...; plus .npy via numpy)."""

    name = "opencv"

    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            kind="depth_io",
            name=self.name,
            library="opencv-python",
            version=cv2.__version__,
            approximate=False,
        )

    def _read(self, path: Path, *, role: str) -> np.ndarray:
        path = Path(path)
        if not path.is_file():
            raise InvalidDepthError(f"{role} file does not exist: {path}.")
        if path.suffix.lower() in NUMPY_SUFFIXES:
            return load_npy_depth(path)
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise InvalidDepthError(
                f"OpenCV failed to decode {role} file '{path}' "
                f"(cv2.imread returned None); the format may be unsupported or the "
                f"file corrupt."
            )
        return raw

    def load_depth(self, path: Path, depth_unit: float | None = None) -> np.ndarray:
        return to_metric_depth(self._read(path, role="depth"), Path(path), depth_unit)

    def load_mask(self, path: Path) -> np.ndarray:
        return to_bool_mask(self._read(path, role="mask"), Path(path))
