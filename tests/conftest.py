"""Shared test fixtures: synthetic point clouds and a tiny mesh."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def cube_mesh() -> tuple[np.ndarray, np.ndarray]:
    """A unit cube centred at the origin: 8 verts, 12 triangles."""
    v = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return v, f


@pytest.fixture
def gaussian_cloud(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(1024, 3)).astype(np.float64)
