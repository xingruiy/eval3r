"""Tests for TumRGBDAdapter loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval3r.datasets.base import Asset
from eval3r.datasets.tum_rgbd import TumRGBDAdapter
from eval3r.io.trajectory import Trajectory
from eval3r.utils.errors import MissingArtifactError, NotSupportedError
from tests._fake_tum import make_tum_root


@pytest.fixture
def adapter(tmp_path: Path) -> TumRGBDAdapter:
    split = make_tum_root(tmp_path, ["fr1_desk"])
    return TumRGBDAdapter(tmp_path, split=split, validate_on_init=False)


def test_list_scenes_uses_split(adapter: TumRGBDAdapter) -> None:
    assert adapter.list_scenes() == ["fr1_desk"]


def test_list_scenes_auto_discovers(tmp_path: Path) -> None:
    make_tum_root(tmp_path, ["fr1_desk", "fr2_xyz"])
    ds = TumRGBDAdapter(tmp_path, validate_on_init=False)
    assert ds.list_scenes() == ["fr1_desk", "fr2_xyz"]


def test_load_depth_converts_scale(adapter: TumRGBDAdapter) -> None:
    d = adapter.load_depth("fr1_desk", 0)
    assert d.shape == (4, 4)
    assert d.dtype == np.float32
    # 5000 / 5000 = 1.0
    assert np.allclose(d, 1.0)


def test_load_depth_custom_scale(tmp_path: Path) -> None:
    split = make_tum_root(tmp_path, ["seq_a"])
    ds = TumRGBDAdapter(tmp_path, split=split, depth_scale=1000.0, validate_on_init=False)
    d = ds.load_depth("seq_a", 0)
    # 5000 / 1000 = 5.0
    assert np.allclose(d, 5.0)


def test_load_color(adapter: TumRGBDAdapter) -> None:
    c = adapter.load_color("fr1_desk", 0)
    assert c.shape == (4, 4, 3)


def test_load_poses_tum(adapter: TumRGBDAdapter) -> None:
    traj = adapter.load_poses("fr1_desk")
    assert isinstance(traj, Trajectory)
    assert traj.poses.shape == (2, 4, 4)
    assert traj.convention == "T_wc"


def test_load_intrinsics_fr1(adapter: TumRGBDAdapter) -> None:
    K = adapter.load_intrinsics("fr1_desk")
    assert K.shape == (3, 3)
    assert K[0, 0] == pytest.approx(517.3)


def test_load_intrinsics_fr3(tmp_path: Path) -> None:
    split = make_tum_root(tmp_path, ["fr3_office"])
    ds = TumRGBDAdapter(tmp_path, split=split, validate_on_init=False)
    K = ds.load_intrinsics("fr3_office")
    assert K[0, 0] == pytest.approx(535.4)


def test_load_intrinsics_custom(tmp_path: Path) -> None:
    split = make_tum_root(tmp_path, ["custom_seq"])
    ds = TumRGBDAdapter(
        tmp_path,
        split=split,
        intrinsics_fx=600.0,
        intrinsics_cx=400.0,
        intrinsics_cy=300.0,
        validate_on_init=False,
    )
    K = ds.load_intrinsics("custom_seq")
    assert K.shape == (3, 3)
    assert K[0, 0] == 600.0
    assert K[0, 2] == 400.0
    assert K[1, 2] == 300.0


def test_intrinsics_unknown_raises(tmp_path: Path) -> None:
    split = make_tum_root(tmp_path, ["unknown_seq"])
    ds = TumRGBDAdapter(tmp_path, split=split, validate_on_init=False)
    with pytest.raises(MissingArtifactError):
        ds.load_intrinsics("unknown_seq")


def test_no_mesh_support(adapter: TumRGBDAdapter) -> None:
    with pytest.raises(NotSupportedError):
        adapter.load_mesh("fr1_desk")
    with pytest.raises(NotSupportedError):
        adapter.load_point_cloud("fr1_desk")


def test_supported_assets(adapter: TumRGBDAdapter) -> None:
    assets = {a.value for a in adapter.supported_assets}
    assert "depth" in assets
    assert "color" in assets
    assert "poses" in assets
    assert "intrinsics" in assets
    assert "mesh" not in assets


def test_validate_passes(adapter: TumRGBDAdapter) -> None:
    report = adapter.validate(scenes=1)
    assert report.ok


def test_asset_path(adapter: TumRGBDAdapter) -> None:
    p = adapter.asset_path("fr1_desk", Asset.POSES)
    assert p.name == "groundtruth.txt"
