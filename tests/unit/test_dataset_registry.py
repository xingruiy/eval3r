"""Task 008 dataset-registry + custom-adapter unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval3r.core.errors import DatasetError, UnknownDatasetError
from eval3r.datasets import CustomAdapter, default_registry
from eval3r.datasets.base import file_fingerprint, read_split_file
from eval3r.datasets.registry import DatasetRegistry
from eval3r.protocols import load_protocol

DATA = Path(__file__).resolve().parents[1] / "fixtures" / "benchmark" / "dataset_root"
PREDS = Path(__file__).resolve().parents[1] / "fixtures" / "benchmark" / "preds"


def test_default_registry_has_custom() -> None:
    assert "custom" in default_registry().available()


def test_unknown_dataset_raises_with_names() -> None:
    reg = DatasetRegistry()
    reg.register("custom", CustomAdapter.factory)
    with pytest.raises(UnknownDatasetError) as exc:
        reg.create("nope")
    assert "custom" in str(exc.value)


def test_custom_factory_requires_root() -> None:
    with pytest.raises(DatasetError) as exc:
        CustomAdapter.factory(None)
    assert "--root" in str(exc.value)


def test_custom_iter_scenes_reads_split() -> None:
    adapter = CustomAdapter(DATA)
    assert adapter.iter_scenes("pair") == ["scene_a", "scene_b"]
    assert adapter.iter_scenes("tiny") == ["scene_a", "scene_b", "scene_missing"]


def test_custom_missing_split_raises() -> None:
    with pytest.raises(DatasetError) as exc:
        CustomAdapter(DATA).iter_scenes("does_not_exist")
    assert "does_not_exist" in str(exc.value)


def test_custom_load_scene_populates_gt_and_fingerprint() -> None:
    adapter = CustomAdapter(DATA)
    scene = adapter.load_scene("scene_a")
    assert scene.gt_pointcloud is not None and scene.gt_pointcloud.is_file()
    # GT is user-supplied: provenance/independence must not claim it is independent.
    assert scene.ground_truth.provenance == "unknown"
    assert scene.ground_truth.independence == "unknown"
    assert scene.ground_truth.fingerprint is not None
    assert scene.ground_truth.fingerprint.startswith("sha256:")


def test_custom_resolves_inferred_prediction() -> None:
    adapter = CustomAdapter(DATA)
    recon = adapter.resolve_prediction(PREDS, "scene_a", None)
    assert recon.path.name == "scene_a.ply"
    assert recon.modality == "pointcloud"


def test_custom_missing_prediction_raises() -> None:
    adapter = CustomAdapter(DATA)
    with pytest.raises(DatasetError) as exc:
        adapter.resolve_prediction(PREDS, "scene_missing", None)
    assert "scene_missing" in str(exc.value)


def test_custom_local_evaluation_supported() -> None:
    proto = load_protocol("single_geometry")
    assert CustomAdapter(DATA).local_evaluation("tiny", proto).status == "supported"


def test_split_reader_ignores_comments_and_blanks() -> None:
    assert read_split_file(DATA / "splits" / "tiny.txt") == [
        "scene_a", "scene_b", "scene_missing"
    ]


def test_file_fingerprint_none_for_missing() -> None:
    assert file_fingerprint(DATA / "gt" / "nope.ply") is None
