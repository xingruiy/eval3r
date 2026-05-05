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


def test_registry_contains_scannet() -> None:
    assert "scannet" in list_datasets()
    cls = get_dataset("scannet")
    assert cls.name == "scannet"


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
        lambda: ds.load_poses("x"),
    ):
        with pytest.raises(NotSupportedError):
            fn()
    assert ds.supports(Asset.MESH) is False
