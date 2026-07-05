"""Task 012 unit tests: Tanks and Temples adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval3r.core.errors import DatasetError
from eval3r.datasets import TanksAndTemplesAdapter, default_registry
from eval3r.datasets.tanks_temples import TRAINING_SCENES

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "tanks_temples_tiny" / "dataset_root"
PREDS = Path(__file__).resolve().parents[1] / "fixtures" / "tanks_temples_tiny" / "preds"


def _adapter() -> TanksAndTemplesAdapter:
    return TanksAndTemplesAdapter(ROOT)


def test_registered_in_dataset_registry() -> None:
    assert "tanks_temples" in default_registry().available()
    adapter = default_registry().create("tanks_temples", ROOT)
    assert adapter.name == "tanks_temples"


def test_factory_requires_root() -> None:
    with pytest.raises(DatasetError, match="requires the dataset root"):
        TanksAndTemplesAdapter.factory(None)


def test_iter_training_scenes_present_only() -> None:
    # Only Barn exists in the fixture; the official list has 7 scenes.
    assert _adapter().iter_scenes("training") == ["Barn"]
    assert "Barn" in TRAINING_SCENES


def test_iter_scenes_unknown_split() -> None:
    with pytest.raises(DatasetError, match="unknown Tanks and Temples split"):
        _adapter().iter_scenes("nonsense")


def test_official_artifacts_resolves_five_files() -> None:
    art = _adapter().official_artifacts("Barn")
    assert art["dataset_dir"] == ROOT / "Barn"
    assert art["gt_pointcloud"].name == "Barn.ply"
    assert art["crop"].name == "Barn.json"
    assert art["alignment"].name == "Barn_trans.txt"
    assert art["trajectory_log"].name == "Barn_COLMAP_SfM.log"
    for key in ("gt_pointcloud", "crop", "alignment", "trajectory_log"):
        assert art[key].is_file()


def test_official_artifacts_missing_scene_errors() -> None:
    with pytest.raises(DatasetError, match="directory is missing"):
        _adapter().official_artifacts("Truck")


def test_gt_provenance_is_independent_laser_scan() -> None:
    gt = _adapter().load_ground_truth("Barn", protocol=None)  # type: ignore[arg-type]
    assert gt.modality == "pointcloud"
    assert gt.provenance == "laser_scan"
    assert gt.independence == "independent"
    assert gt.density == "dense_surface"
    assert gt.unit == "m"
    assert gt.source_pose_format == "tanks_temples_log"


def test_capabilities() -> None:
    caps = _adapter().capabilities
    assert caps.independent_gt is True
    assert caps.official_local_eval is True
    assert caps.official_local_eval_method == "official_script_wrapper"
    assert caps.server_only_eval is False  # per-split server-only via local_evaluation()


def test_local_evaluation_training_supported_intermediate_server_only() -> None:
    adapter = _adapter()
    assert adapter.local_evaluation("training", None).status == "supported"  # type: ignore[arg-type]
    inter = adapter.local_evaluation("intermediate", None)  # type: ignore[arg-type]
    assert inter.status == "server_only"
    assert inter.official_server_required is True
    assert adapter.local_evaluation("advanced", None).status == "server_only"  # type: ignore[arg-type]


def test_gt_fingerprint_joint_over_gt_crop_trans() -> None:
    fp = _adapter().gt_fingerprint("Barn", None)  # type: ignore[arg-type]
    assert fp is not None and fp.startswith("sha256:")
    # Deterministic for identical inputs.
    assert fp == _adapter().gt_fingerprint("Barn", None)  # type: ignore[arg-type]


def test_load_scene_records_official_artifact_paths() -> None:
    scene = _adapter().load_scene("Barn")
    assert scene.gt_pointcloud is not None and scene.gt_pointcloud.name == "Barn.ply"
    assert scene.gt_trajectory is not None and scene.gt_trajectory.name == "Barn_COLMAP_SfM.log"
    assert scene.masks["crop"].name == "Barn.json"
    assert scene.masks["alignment"].name == "Barn_trans.txt"
    assert scene.metadata["source_pose_format"] == "tanks_temples_log"


def test_resolve_prediction_default_layout() -> None:
    recon = _adapter().resolve_prediction(PREDS, "Barn", None)
    assert recon.path == PREDS / "Barn.ply"
    assert recon.modality == "pointcloud"
    assert recon.unit == "m"


def test_resolve_prediction_missing_errors() -> None:
    with pytest.raises(DatasetError, match="prediction for scene 'Barn' not found"):
        _adapter().resolve_prediction(ROOT / "does_not_exist", "Barn", None)
