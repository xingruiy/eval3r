from __future__ import annotations

import pytest

from eval3r.datasets import (
    Asset,
    DatasetAdapter,
    get_dataset,
    list_datasets,
    register_dataset,
)
from eval3r.utils.errors import NotSupportedError


def test_registry_contains_all_datasets() -> None:
    assert list_datasets() == [
        "dtu", "eth3d", "generic", "replica", "scannet", "tanks_temples", "tum_rgbd",
    ]
    for name in list_datasets():
        cls = get_dataset(name)
        assert cls.name == name


def test_unknown_dataset_raises() -> None:
    with pytest.raises(KeyError):
        get_dataset("not_a_dataset")


def test_register_validates_name() -> None:
    class _NoName(DatasetAdapter):
        def list_scenes(self, split=None):
            return []

    with pytest.raises(ValueError):
        register_dataset(_NoName)


def test_default_loaders_raise_not_supported() -> None:
    class _Empty(DatasetAdapter):
        name = "_empty_"

        def list_scenes(self, split=None):
            return []

    ds = _Empty()
    for fn in (
        lambda: ds.load_mesh("x"),
        lambda: ds.load_point_cloud("x"),
        lambda: ds.load_depth("x", 0),
        lambda: ds.load_color("x", 0),
        lambda: ds.load_intrinsics("x"),
        lambda: ds.load_intrinsics_depth("x"),
        lambda: ds.load_intrinsics_color("x"),
        lambda: ds.load_poses("x"),
    ):
        with pytest.raises(NotSupportedError):
            fn()
    assert ds.supports(Asset.MESH) is False


def test_validate_fails_when_no_scenes() -> None:
    class _Empty(DatasetAdapter):
        name = "_empty_"

        def list_scenes(self, split=None):
            return []

    report = _Empty().validate()
    assert report.ok is False
    assert any(name == "list_scenes" and ok is False for name, ok, _ in report.checks)
    assert "0 scenes found" in report.errors[0]


def test_validate_no_scenes_reason_includes_layout_hint(tmp_path) -> None:
    class _Empty(DatasetAdapter):
        name = "_empty_"
        expected_layout = "<root>/scans/<scene_id>/mesh.ply"

        def __init__(self, root):
            self.root = root
            self._split = None

        def list_scenes(self, split=None):
            return []

    report = _Empty(tmp_path).validate()
    assert "auto-discovery returned no scene directories" in report.errors[0]
    assert "expected layout example: <root>/scans/<scene_id>/mesh.ply" in report.errors[0]


def test_validate_with_scenes_zero_does_not_trigger_no_scene_failure() -> None:
    class _OneScene(DatasetAdapter):
        name = "_one_scene_"

        def list_scenes(self, split=None):
            return ["scene-1"]

    report = _OneScene().validate(scenes=0)
    assert report.ok is True
    assert report.errors == []
    assert report.checks[0] == ("list_scenes", True, "0 scenes")
