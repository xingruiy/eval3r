"""Task 014 unit tests: depth IO backends (imageio default, OpenCV for PFM).

Both backends share one contract: ``load_depth`` returns 2D float64 **metres**,
``depth_unit`` is required for integer files (units are never self-describing),
and invalid values pass through untouched for the protocol-controlled masking.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import pytest

from eval3r.backends.depth_imageio import ImageioDepthBackend
from eval3r.backends.depth_opencv import OpenCVDepthBackend
from eval3r.core.errors import InvalidDepthError

BACKENDS = [ImageioDepthBackend(), OpenCVDepthBackend()]
IDS = [b.name for b in BACKENDS]


def _write_png16(path: Path, arr: np.ndarray) -> None:
    iio.imwrite(path, arr.astype(np.uint16))


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_integer_png_with_unit_converts_to_metres(backend, tmp_path: Path) -> None:
    mm = np.array([[1000, 2500], [0, 65535]], dtype=np.uint16)
    path = tmp_path / "depth.png"
    _write_png16(path, mm)
    depth = backend.load_depth(path, 0.001)
    assert depth.dtype == np.float64
    # Invalid sentinels (0, 65535) pass through scaled, NOT masked here.
    np.testing.assert_allclose(depth, mm.astype(np.float64) * 0.001)


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_integer_depth_without_unit_fails_explicitly(backend, tmp_path: Path) -> None:
    path = tmp_path / "depth.png"
    _write_png16(path, np.ones((2, 2), dtype=np.uint16))
    with pytest.raises(InvalidDepthError, match="not self-describing.*depth_unit"):
        backend.load_depth(path, None)


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_float_npy_defaults_to_metres(backend, tmp_path: Path) -> None:
    metres = np.array([[0.5, 1.5], [2.0, 3.0]])
    path = tmp_path / "depth.npy"
    np.save(path, metres)
    np.testing.assert_allclose(backend.load_depth(path, None), metres)
    # An explicit unit still applies to float sources.
    np.testing.assert_allclose(backend.load_depth(path, 2.0), 2.0 * metres)


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_integer_npy_requires_unit(backend, tmp_path: Path) -> None:
    path = tmp_path / "depth.npy"
    np.save(path, np.ones((2, 2), dtype=np.int32))
    with pytest.raises(InvalidDepthError, match="depth_unit"):
        backend.load_depth(path, None)
    np.testing.assert_allclose(backend.load_depth(path, 0.001), 0.001 * np.ones((2, 2)))


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_missing_file_fails_with_path(backend, tmp_path: Path) -> None:
    with pytest.raises(InvalidDepthError, match="does not exist"):
        backend.load_depth(tmp_path / "nope.png", 0.001)


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_multichannel_image_rejected(backend, tmp_path: Path) -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    path = tmp_path / "rgb.png"
    iio.imwrite(path, rgb)
    with pytest.raises(InvalidDepthError, match="single-channel"):
        backend.load_depth(path, 0.001)


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_load_mask_nonzero_is_valid(backend, tmp_path: Path) -> None:
    mask_img = np.array([[0, 255], [1, 0]], dtype=np.uint8)
    path = tmp_path / "mask.png"
    iio.imwrite(path, mask_img)
    mask = backend.load_mask(path)
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask, mask_img != 0)


def test_opencv_reads_pfm_float_depth(tmp_path: Path) -> None:
    # PFM is the reason the OpenCV backend exists; imageio does not decode it.
    depth = (np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0) / 7.0
    path = tmp_path / "depth.pfm"
    assert cv2.imwrite(str(path), depth)
    loaded = OpenCVDepthBackend().load_depth(path, None)
    np.testing.assert_allclose(loaded, depth, rtol=1e-6)


def test_opencv_unreadable_file_fails_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "garbage.png"
    path.write_bytes(b"not an image")
    with pytest.raises(InvalidDepthError, match="failed to decode"):
        OpenCVDepthBackend().load_depth(path, 0.001)


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_backend_info_records_library_version(backend) -> None:
    info = backend.backend_info()
    assert info.kind == "depth_io"
    assert info.name == backend.name
    assert info.version not in ("", None)
    assert info.approximate is False
