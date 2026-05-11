"""Tests for ReplicaAdapter loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval3r.datasets.base import Asset
from eval3r.datasets.replica import ReplicaAdapter
from eval3r.io.geometry import MeshData, PointCloudData
from eval3r.io.trajectory import Trajectory
from eval3r.utils.errors import MissingArtifactError, NotSupportedError
from tests.helpers._fake_replica import make_replica_root


@pytest.fixture
def adapter(tmp_path: Path) -> ReplicaAdapter:
    split = make_replica_root(tmp_path, ["room_0"])
    return ReplicaAdapter(tmp_path, split=split, validate_on_init=False)


def test_list_scenes_uses_split_file(adapter: ReplicaAdapter) -> None:
    assert adapter.list_scenes() == ["room_0"]


def test_list_scenes_auto_discovers(tmp_path: Path) -> None:
    make_replica_root(tmp_path, ["office_0", "room_0"])
    ds = ReplicaAdapter(tmp_path, validate_on_init=False)
    assert ds.list_scenes() == ["office_0", "room_0"]


def test_load_mesh_returns_cube(adapter: ReplicaAdapter) -> None:
    mesh = adapter.load_mesh("room_0")
    assert isinstance(mesh, MeshData)
    assert mesh.vertices.shape == (8, 3)
    assert mesh.faces.shape == (12, 3)


def test_load_point_cloud_from_mesh(adapter: ReplicaAdapter) -> None:
    pc = adapter.load_point_cloud("room_0")
    assert isinstance(pc, PointCloudData)
    assert pc.points.shape == (8, 3)


def test_load_depth(adapter: ReplicaAdapter) -> None:
    d = adapter.load_depth("room_0", 0)
    assert d.shape == (4, 4)
    assert d.dtype == np.float32
    # 1000 / 1000 = 1.0
    assert np.allclose(d, 1.0)


def test_load_color(adapter: ReplicaAdapter) -> None:
    c = adapter.load_color("room_0", 0)
    assert c.shape == (4, 4, 3)


def test_load_poses(adapter: ReplicaAdapter) -> None:
    traj = adapter.load_poses("room_0")
    assert isinstance(traj, Trajectory)
    assert traj.poses.shape == (2, 4, 4)
    assert traj.convention == "T_wc"


def test_missing_mesh_raises(tmp_path: Path) -> None:
    (tmp_path / "empty_scene").mkdir()
    ds = ReplicaAdapter(tmp_path, validate_on_init=False)
    with pytest.raises(MissingArtifactError):
        ds.load_mesh("empty_scene")


def test_supported_assets(adapter: ReplicaAdapter) -> None:
    assets = adapter.supported_assets
    assert "mesh" in {a.value for a in assets}
    assert "point_cloud" in {a.value for a in assets}


def test_validate_passes(adapter: ReplicaAdapter) -> None:
    report = adapter.validate(scenes=1)
    assert report.ok


def test_asset_path(adapter: ReplicaAdapter) -> None:
    p = adapter.asset_path("room_0", Asset.MESH)
    assert p.name == "mesh.ply"
