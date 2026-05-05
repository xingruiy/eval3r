from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval3r.datasets import ScanNetAdapter
from eval3r.utils.errors import MissingArtifactError

from ._fake_scannet import make_scannet_root


@pytest.fixture
def adapter(tmp_path: Path) -> ScanNetAdapter:
    split = make_scannet_root(tmp_path, ["scene_a"])
    return ScanNetAdapter(tmp_path, split=split, validate_on_init=False)


def test_load_mesh_returns_cube(adapter: ScanNetAdapter) -> None:
    mesh = adapter.load_mesh("scene_a")
    assert mesh.vertices.shape == (8, 3)
    assert mesh.faces.shape == (12, 3)


def test_load_point_cloud_falls_back_to_mesh_vertices(adapter: ScanNetAdapter) -> None:
    pc = adapter.load_point_cloud("scene_a")
    assert pc.points.shape == (8, 3)


def test_load_depth_converts_mm_to_metres(adapter: ScanNetAdapter) -> None:
    d = adapter.load_depth("scene_a", 0)
    # fixture writes 1000 mm everywhere; expect 1.0 m.
    assert d.shape == (4, 4)
    assert np.allclose(d, 1.0)


def test_load_intrinsics_returns_3x3(adapter: ScanNetAdapter) -> None:
    K = adapter.load_intrinsics("scene_a")
    assert K.shape == (3, 3)
    assert K[0, 0] == pytest.approx(500.0)


def test_load_poses_returns_trajectory(adapter: ScanNetAdapter) -> None:
    traj = adapter.load_poses("scene_a")
    assert traj.poses.shape == (2, 4, 4)
    assert traj.convention == "T_wc"
    # Pose 1 has translation x=0.01 per fixture.
    assert traj.poses[1, 0, 3] == pytest.approx(0.01)


def test_list_scenes_uses_split_file(adapter: ScanNetAdapter) -> None:
    assert adapter.list_scenes() == ["scene_a"]


def test_split_none_auto_discovers_scans(tmp_path: Path) -> None:
    """Without a split, the adapter enumerates `<root>/scans/*` subdirectories."""
    make_scannet_root(tmp_path, ["scene_a", "scene_b"])
    ds = ScanNetAdapter(tmp_path, validate_on_init=False)
    assert ds.list_scenes() == ["scene_a", "scene_b"]


def test_unknown_string_split_is_treated_as_path(tmp_path: Path) -> None:
    """No built-in split names — a string that isn't a path should error clearly."""
    (tmp_path / "scans").mkdir()
    with pytest.raises(MissingArtifactError) as exc:
        ScanNetAdapter(tmp_path, split="test", validate_on_init=False)
    assert "does not bundle" in str(exc.value)
