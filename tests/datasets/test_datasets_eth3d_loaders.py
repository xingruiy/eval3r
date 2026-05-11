"""Tests for ETH3DAdapter loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval3r.datasets.base import Asset
from eval3r.datasets.eth3d import ETH3DAdapter
from eval3r.io.geometry import MeshData, PointCloudData
from eval3r.io.trajectory import Trajectory
from eval3r.utils.errors import MissingArtifactError, NotSupportedError
from tests.helpers._fake_eth3d import make_eth3d_root


@pytest.fixture
def adapter(tmp_path: Path) -> ETH3DAdapter:
    split = make_eth3d_root(tmp_path, ["courtyard"])
    return ETH3DAdapter(tmp_path, split=split, track="dslr", validate_on_init=False)


def test_list_scenes_uses_split(adapter: ETH3DAdapter) -> None:
    assert adapter.list_scenes() == ["courtyard"]


def test_list_scenes_auto_discovers(tmp_path: Path) -> None:
    make_eth3d_root(tmp_path, ["courtyard", "facade"])
    ds = ETH3DAdapter(tmp_path, track="dslr", validate_on_init=False)
    assert set(ds.list_scenes()) == {"courtyard", "facade"}


def test_load_mesh_returns_cube(adapter: ETH3DAdapter) -> None:
    mesh = adapter.load_mesh("courtyard")
    assert isinstance(mesh, MeshData)
    assert mesh.vertices.shape == (8, 3)


def test_load_point_cloud_from_mesh(adapter: ETH3DAdapter) -> None:
    pc = adapter.load_point_cloud("courtyard")
    assert isinstance(pc, PointCloudData)
    assert pc.points.shape == (8, 3)


def test_load_intrinsics_colmap(adapter: ETH3DAdapter) -> None:
    K = adapter.load_intrinsics("courtyard")
    assert K.shape == (3, 3)
    assert K[0, 0] == 500.0
    np.testing.assert_allclose(adapter.load_intrinsics_depth("courtyard"), K)
    np.testing.assert_allclose(adapter.load_intrinsics_color("courtyard"), K)


def test_load_poses_colmap(adapter: ETH3DAdapter) -> None:
    traj = adapter.load_poses("courtyard")
    assert isinstance(traj, Trajectory)
    assert traj.poses.shape == (2, 4, 4)
    assert traj.convention == "T_cw"


def test_supported_assets(adapter: ETH3DAdapter) -> None:
    assets = {a.value for a in adapter.supported_assets}
    assert "mesh" in assets
    assert "point_cloud" in assets
    assert "intrinsics_depth" in assets
    assert "intrinsics_color" in assets
    assert "poses" in assets


def test_missing_mesh_raises(tmp_path: Path) -> None:
    (tmp_path / "dslr" / "empty").mkdir(parents=True)
    ds = ETH3DAdapter(tmp_path, track="dslr", validate_on_init=False)
    with pytest.raises(MissingArtifactError):
        ds.load_mesh("empty")


def test_validate_passes(adapter: ETH3DAdapter) -> None:
    report = adapter.validate(scenes=1)
    assert report.ok


def test_asset_path(adapter: ETH3DAdapter) -> None:
    p = adapter.asset_path("courtyard", Asset.MESH)
    assert p.name == "scan.ply"
