"""imageio depth IO backend (default ``depth_io`` backend).

Loads depth from image files (16-bit PNG, TIFF, ...) via ``imageio.v3`` and from
``.npy`` arrays via numpy, returning 2D float64 metres. ``depth_unit`` handling
and the integer-requires-explicit-unit rule live in
:mod:`eval3r.backends.depth_common`, shared with the OpenCV backend. Invalid
values are passed through untouched for the protocol's masking stage.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from eval3r.backends.depth_common import (
    NUMPY_SUFFIXES,
    load_npy_depth,
    to_bool_mask,
    to_metric_depth,
)
from eval3r.core.errors import InvalidDepthError
from eval3r.core.registry import BackendInfo


class ImageioDepthBackend:
    """Depth/mask image loading via imageio (plus .npy via numpy)."""

    name = "imageio"

    def backend_info(self) -> BackendInfo:
        import imageio

        return BackendInfo(
            kind="depth_io",
            name=self.name,
            library="imageio",
            version=getattr(imageio, "__version__", "unknown"),
            approximate=False,
        )

    def _read(self, path: Path, *, role: str) -> np.ndarray:
        path = Path(path)
        if not path.is_file():
            raise InvalidDepthError(f"{role} file does not exist: {path}.")
        if path.suffix.lower() in NUMPY_SUFFIXES:
            return load_npy_depth(path)
        try:
            return iio.imread(path)
        except Exception as exc:
            raise InvalidDepthError(
                f"imageio failed to read {role} file '{path}': {exc}"
            ) from exc

    def load_depth(self, path: Path, depth_unit: float | None = None) -> np.ndarray:
        return to_metric_depth(self._read(path, role="depth"), Path(path), depth_unit)

    def load_mask(self, path: Path) -> np.ndarray:
        return to_bool_mask(self._read(path, role="mask"), Path(path))
