from __future__ import annotations

from pathlib import Path

import pytest

from eval3r.datasets import ScanNetAdapter
from eval3r.utils.errors import MissingArtifactError

from ._fake_scannet import make_scannet_root


def test_default_layout_validates(tmp_path: Path) -> None:
    split = make_scannet_root(tmp_path, ["scene_a", "scene_b"])
    ds = ScanNetAdapter(tmp_path, split=split)  # validates first scene on init
    report = ds.validate(scenes=2)
    assert report.ok, report.errors


def test_renamed_color_subdir_raises_with_hint(tmp_path: Path) -> None:
    split = make_scannet_root(tmp_path, ["scene_a"], color_subdir="images")
    # Default subdir is "color", which doesn't exist on disk.
    ds = ScanNetAdapter(tmp_path, split=split, validate_on_init=False)
    with pytest.raises(MissingArtifactError) as exc:
        ds.load_color("scene_a", 0)
    msg = str(exc.value)
    assert "color_subdir" in msg
    assert "color not found" in msg


def test_color_subdir_override_recovers(tmp_path: Path) -> None:
    split = make_scannet_root(tmp_path, ["scene_a"], color_subdir="images")
    ds = ScanNetAdapter(tmp_path, split=split, color_subdir="images", validate_on_init=False)
    img = ds.load_color("scene_a", 0)
    assert img.shape == (4, 4, 3)


def test_validate_reports_missing_mesh(tmp_path: Path) -> None:
    split = make_scannet_root(tmp_path, ["scene_a"])
    # Remove mesh
    (tmp_path / "scans" / "scene_a" / "scene_a_vh_clean_2.ply").unlink()
    with pytest.raises(MissingArtifactError):
        ScanNetAdapter(tmp_path, split=split)  # validate_on_init triggers
