"""Tests for TanksTemplesAdapter loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval3r.datasets.base import Asset
from eval3r.datasets.tanks_temples import TanksTemplesAdapter
from eval3r.io.geometry import PointCloudData
from eval3r.io.trajectory import Trajectory
from eval3r.utils.errors import MissingArtifactError, NotSupportedError
from tests._fake_tanks import make_tanks_root


@pytest.fixture
def adapter(tmp_path: Path) -> TanksTemplesAdapter:
    split = make_tanks_root(tmp_path, ["Barn", "Church"])
    return TanksTemplesAdapter(tmp_path, split=split, validate_on_init=False)


def test_list_scenes_uses_split(adapter: TanksTemplesAdapter) -> None:
    assert adapter.list_scenes() == ["Barn", "Church"]


def test_list_scenes_with_subset(tmp_path: Path) -> None:
    make_tanks_root(tmp_path, ["Barn", "Church"])
    ds = TanksTemplesAdapter(tmp_path, subset="training", validate_on_init=False)
    # Training subset has known scenes.
    assert "Barn" in ds.list_scenes()
    assert "Church" in ds.list_scenes()


def test_list_scenes_auto_discovers(tmp_path: Path) -> None:
    make_tanks_root(tmp_path, ["Barn", "Church"])
    ds = TanksTemplesAdapter(tmp_path, validate_on_init=False)
    assert set(ds.list_scenes()) == {"Barn", "Church"}


def test_load_point_cloud(adapter: TanksTemplesAdapter) -> None:
    pc = adapter.load_point_cloud("Barn")
    assert isinstance(pc, PointCloudData)
    assert pc.points.shape == (3, 3)


def test_load_intrinsics(adapter: TanksTemplesAdapter) -> None:
    K = adapter.load_intrinsics("Barn")
    assert K.shape == (3, 3)
    np.testing.assert_allclose(adapter.load_intrinsics_depth("Barn"), K)
    np.testing.assert_allclose(adapter.load_intrinsics_color("Barn"), K)


def test_load_poses(adapter: TanksTemplesAdapter) -> None:
    traj = adapter.load_poses("Barn")
    assert isinstance(traj, Trajectory)
    assert traj.poses.shape == (2, 4, 4)
    assert traj.convention == "T_wc"


def test_no_mesh_support(adapter: TanksTemplesAdapter) -> None:
    with pytest.raises(NotSupportedError):
        adapter.load_mesh("Barn")


def test_supported_assets(adapter: TanksTemplesAdapter) -> None:
    assets = {a.value for a in adapter.supported_assets}
    assert "point_cloud" in assets
    assert "mesh" not in assets
    assert "intrinsics_depth" in assets
    assert "intrinsics_color" in assets


def test_missing_point_cloud_raises(tmp_path: Path) -> None:
    (tmp_path / "EmptyScene").mkdir()
    ds = TanksTemplesAdapter(tmp_path, validate_on_init=False)
    with pytest.raises(MissingArtifactError):
        ds.load_point_cloud("EmptyScene")


def test_validate_passes(adapter: TanksTemplesAdapter) -> None:
    report = adapter.validate(scenes=1)
    assert report.ok




def test_asset_path_color_uses_1_based_6_digit_ids(adapter: TanksTemplesAdapter) -> None:
    p = adapter.asset_path("Barn", Asset.COLOR, frame=0)
    assert p.name == "000001.jpg"

def test_asset_path(adapter: TanksTemplesAdapter) -> None:
    p = adapter.asset_path("Barn", Asset.POINT_CLOUD)
    assert p.name == "Barn.ply"


def test_empty_pose_log_raises(tmp_path: Path) -> None:
    make_tanks_root(tmp_path, ["Barn"])
    (tmp_path / "Barn" / "Barn_COLMAP_SfM.log").write_text("   \n\n")
    ds = TanksTemplesAdapter(tmp_path, validate_on_init=False)
    with pytest.raises(MissingArtifactError, match="pose log is empty"):
        ds.load_poses("Barn")
