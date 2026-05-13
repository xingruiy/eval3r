from __future__ import annotations

import numpy as np
import pytest

from eval3r.io.geometry import (
    load_mesh,
    load_point_cloud,
    save_mesh_ply,
    save_point_cloud_ply,
)
from eval3r.utils.errors import EmptyGeometryError, NaNGeometryError


def test_save_load_point_cloud_round_trip(tmp_path, gaussian_cloud) -> None:
    out = tmp_path / "pc.ply"
    save_point_cloud_ply(out, gaussian_cloud)
    loaded = load_point_cloud(out)
    assert loaded.points.shape == gaussian_cloud.shape
    assert np.allclose(np.sort(loaded.points, axis=0), np.sort(gaussian_cloud, axis=0))


def test_save_load_mesh_round_trip(tmp_path, cube_mesh) -> None:
    v, f = cube_mesh
    out = tmp_path / "mesh.ply"
    save_mesh_ply(out, v, f)
    loaded = load_mesh(out)
    assert loaded.vertices.shape == v.shape
    assert loaded.faces.shape == f.shape


def test_save_rejects_nan(tmp_path) -> None:
    bad = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    with pytest.raises(NaNGeometryError):
        save_point_cloud_ply(tmp_path / "bad.ply", bad)


def test_save_rejects_empty(tmp_path) -> None:
    with pytest.raises(EmptyGeometryError):
        save_point_cloud_ply(tmp_path / "empty.ply", np.zeros((0, 3)))
