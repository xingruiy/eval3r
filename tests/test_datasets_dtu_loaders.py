"""Tests for DTUAdapter loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval3r.datasets.base import Asset
from eval3r.datasets.dtu import DTUAdapter
from eval3r.io.geometry import PointCloudData
from eval3r.io.trajectory import Trajectory
from eval3r.utils.errors import MissingArtifactError, NotSupportedError
from tests._fake_dtu import make_dtu_root


@pytest.fixture
def adapter(tmp_path: Path) -> DTUAdapter:
    split = make_dtu_root(tmp_path, [1, 4])
    return DTUAdapter(tmp_path, split=split, validate_on_init=False)


def test_list_scenes_uses_split(adapter: DTUAdapter) -> None:
    assert adapter.list_scenes() == ["1", "4"]


def test_list_scenes_defaults_to_eval_subset(tmp_path: Path) -> None:
    ds = DTUAdapter(tmp_path, eval_scans=[1, 4], validate_on_init=False)
    assert ds.list_scenes() == ["1", "4"]


def test_load_point_cloud(adapter: DTUAdapter) -> None:
    pc = adapter.load_point_cloud("1")
    assert isinstance(pc, PointCloudData)
    assert pc.points.shape == (3, 3)


def test_load_poses_decomposes_projection(adapter: DTUAdapter) -> None:
    traj = adapter.load_poses("1")
    assert isinstance(traj, Trajectory)
    assert traj.poses.shape == (1, 4, 4)
    assert traj.convention == "T_cw"


def test_load_intrinsics(adapter: DTUAdapter) -> None:
    K = adapter.load_intrinsics("1")
    assert K.shape == (3, 3)
    assert K[2, 2] == 1.0


def test_no_mesh_support(adapter: DTUAdapter) -> None:
    with pytest.raises(NotSupportedError):
        adapter.load_mesh("1")


def test_supported_assets(adapter: DTUAdapter) -> None:
    assets = {a.value for a in adapter.supported_assets}
    assert "point_cloud" in assets
    assert "mesh" not in assets


def test_missing_point_cloud_raises(tmp_path: Path) -> None:
    (tmp_path / "scans" / "scan99").mkdir(parents=True)
    (tmp_path / "Cameras" / "scan99").mkdir(parents=True)
    ds = DTUAdapter(tmp_path, eval_scans=[99], validate_on_init=False)
    with pytest.raises(MissingArtifactError):
        ds.load_point_cloud("99")


def test_validate_passes(adapter: DTUAdapter) -> None:
    report = adapter.validate(scenes=1)
    assert report.ok


def test_asset_path(adapter: DTUAdapter) -> None:
    p = adapter.asset_path("1", Asset.POINT_CLOUD)
    assert p.name == "points.ply"
