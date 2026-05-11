"""Tests for TanksTemplesAdapter loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval3r.datasets.base import Asset
from eval3r.datasets.tanks_temples import TanksTemplesAdapter
from eval3r.io.crop import CropVolume
from eval3r.io.geometry import PointCloudData
from eval3r.io.trajectory import Trajectory
from eval3r.utils.errors import MissingArtifactError, NotSupportedError
from tests.helpers._fake_tanks import _DEFAULT_ALIGNMENT, make_tanks_root, make_tanks_scene


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
    np.testing.assert_allclose(
        K,
        np.array(
            [[2.8, 0.0, 2.0], [0.0, 2.8, 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(adapter.load_intrinsics_depth("Barn"), K)
    np.testing.assert_allclose(adapter.load_intrinsics_color("Barn"), K)


def test_load_poses(adapter: TanksTemplesAdapter) -> None:
    traj = adapter.load_poses("Barn")
    assert isinstance(traj, Trajectory)
    assert traj.poses.shape == (2, 4, 4)
    assert traj.convention == "T_wc"
    # The fixture writes _trans.txt = +10 m in X (see _DEFAULT_ALIGNMENT) and
    # raw T_wc translations [i*0.01, 0, 0]. After SfM→laser alignment the
    # camera centers should be the raw centers shifted by +10 m in X.
    expected = np.array([[10.0, 0.0, 0.0], [10.01, 0.0, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(traj.poses[:, :3, 3], expected)


def test_load_poses_disabled_alignment(tmp_path: Path) -> None:
    split = make_tanks_root(tmp_path, ["Barn"])
    ds = TanksTemplesAdapter(
        tmp_path, split=split, alignment_filename="", validate_on_init=False
    )
    traj = ds.load_poses("Barn")
    raw = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(traj.poses[:, :3, 3], raw)


def test_load_poses_missing_alignment_raises(tmp_path: Path) -> None:
    make_tanks_scene(tmp_path, "Barn", alignment="skip")
    ds = TanksTemplesAdapter(tmp_path, validate_on_init=False)
    with pytest.raises(MissingArtifactError, match="alignment"):
        ds.load_poses("Barn")


def test_load_poses_malformed_alignment_raises(tmp_path: Path) -> None:
    make_tanks_scene(tmp_path, "Barn")
    # Overwrite with a 3x3 matrix to force the shape check.
    (tmp_path / "Barn" / "Barn_trans.txt").write_text(
        "1 0 0\n0 1 0\n0 0 1\n"
    )
    ds = TanksTemplesAdapter(tmp_path, validate_on_init=False)
    with pytest.raises(MissingArtifactError, match="expected 4x4"):
        ds.load_poses("Barn")


def test_load_poses_uses_default_alignment_matrix() -> None:
    # Sanity check that the fixture's documented translation is what we expect.
    np.testing.assert_allclose(_DEFAULT_ALIGNMENT[:3, 3], [10.0, 0.0, 0.0])


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


def test_asset_path(adapter: TanksTemplesAdapter) -> None:
    p = adapter.asset_path("Barn", Asset.POINT_CLOUD)
    assert p.name == "Barn.ply"


def test_empty_pose_log_raises(tmp_path: Path) -> None:
    make_tanks_root(tmp_path, ["Barn"])
    (tmp_path / "Barn" / "Barn_COLMAP_SfM.log").write_text("   \n\n")
    ds = TanksTemplesAdapter(tmp_path, validate_on_init=False)
    with pytest.raises(MissingArtifactError, match="pose log is empty"):
        ds.load_poses("Barn")


def test_load_crop_volume(adapter: TanksTemplesAdapter) -> None:
    vol = adapter.load_crop_volume("Barn")
    assert isinstance(vol, CropVolume)
    assert vol.orthogonal_axis == 1  # default fixture uses Y
    assert vol.axis_min == 0.0
    assert vol.axis_max == 1.0
    assert vol.polygon_2d.shape == (4, 2)


def test_load_crop_volume_disabled(tmp_path: Path) -> None:
    split = make_tanks_root(tmp_path, ["Barn"])
    ds = TanksTemplesAdapter(
        tmp_path, split=split, crop_filename="", validate_on_init=False
    )
    with pytest.raises(NotSupportedError):
        ds.load_crop_volume("Barn")


def test_load_crop_volume_missing(tmp_path: Path) -> None:
    make_tanks_scene(tmp_path, "Barn", crop="skip")
    ds = TanksTemplesAdapter(tmp_path, validate_on_init=False)
    with pytest.raises(MissingArtifactError, match="crop_filename"):
        ds.load_crop_volume("Barn")


def test_other_adapter_no_crop_volume() -> None:
    # The default DatasetAdapter base says no crop. Use the GenericAdapter
    # (or any non-T&T adapter) to confirm the soft-default raises.
    from eval3r.datasets.generic import GenericAdapter

    ds = GenericAdapter.__new__(GenericAdapter)  # bypass __init__ — only need the method
    ds.name = "generic"  # type: ignore[attr-defined]
    with pytest.raises(NotSupportedError):
        ds.load_crop_volume("anything")


def test_load_thresholds_training_scenes(adapter: TanksTemplesAdapter) -> None:
    # Spot-check several scene-specific τ values.
    assert adapter.load_thresholds("Barn") == (0.01,)
    assert adapter.load_thresholds("Caterpillar") == (0.005,)
    assert adapter.load_thresholds("Ignatius") == (0.003,)
    assert adapter.load_thresholds("Courthouse") == (0.025,)


def test_load_thresholds_unknown_scene_raises(adapter: TanksTemplesAdapter) -> None:
    # Advanced subset has no published τ.
    with pytest.raises(NotSupportedError, match="published"):
        adapter.load_thresholds("Auditorium")


def test_other_adapter_no_thresholds() -> None:
    from eval3r.datasets.generic import GenericAdapter

    ds = GenericAdapter.__new__(GenericAdapter)
    ds.name = "generic"  # type: ignore[attr-defined]
    with pytest.raises(NotSupportedError):
        ds.load_thresholds("anything")
