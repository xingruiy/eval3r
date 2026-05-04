from __future__ import annotations

import numpy as np
import pytest

from eval3r.align import align
from eval3r.align.similarity import umeyama


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def test_umeyama_identity(rng) -> None:
    src = rng.normal(size=(50, 3))
    res = umeyama(src, src, mode="se3")
    assert np.allclose(res.rotation, np.eye(3), atol=1e-10)
    assert np.allclose(res.translation, 0, atol=1e-10)
    assert res.scale == pytest.approx(1.0)


def test_umeyama_se3_round_trip(rng) -> None:
    src = rng.normal(size=(64, 3))
    R = _random_rotation(rng)
    t = rng.normal(size=3) * 5
    tgt = src @ R.T + t
    res = umeyama(src, tgt, mode="se3")
    err = np.linalg.norm(res.transform(src) - tgt, axis=1).max()
    assert err < 1e-8


def test_umeyama_sim3_round_trip(rng) -> None:
    src = rng.normal(size=(96, 3))
    R = _random_rotation(rng)
    s = 2.5
    t = rng.normal(size=3)
    tgt = s * (src @ R.T) + t
    res = umeyama(src, tgt, mode="sim3")
    assert res.scale == pytest.approx(s, rel=1e-6)
    err = np.linalg.norm(res.transform(src) - tgt, axis=1).max()
    assert err < 1e-6


def test_align_none() -> None:
    pts = np.random.default_rng(0).normal(size=(20, 3))
    res = align(pts, pts, mode="none")
    assert res.scale == 1.0
    assert np.allclose(res.rotation, np.eye(3))
    assert np.allclose(res.translation, 0)
