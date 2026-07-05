"""Task 013 unit tests: ETH3D adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval3r.core.errors import DatasetError
from eval3r.datasets import Eth3dAdapter, default_registry
from eval3r.datasets.eth3d import TEST_SCENES, TRAINING_SCENES, parse_scan_mlp

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "eth3d_tiny"
ROOT = FIX / "dataset_root"
PREDS = FIX / "preds"


def _adapter() -> Eth3dAdapter:
    return Eth3dAdapter(ROOT)


def test_registered_in_dataset_registry() -> None:
    assert "eth3d" in default_registry().available()
    adapter = default_registry().create("eth3d", ROOT)
    assert adapter.name == "eth3d"


def test_factory_requires_root() -> None:
    with pytest.raises(DatasetError, match="requires the dataset root"):
        Eth3dAdapter.factory(None)


def test_iter_training_scenes_present_only() -> None:
    # Only courtyard exists in the fixture; the official high-res list has 13 scenes.
    assert _adapter().iter_scenes("training") == ["courtyard"]
    assert "courtyard" in TRAINING_SCENES
    assert len(TRAINING_SCENES) == 13


def test_iter_test_scenes_full_official_list() -> None:
    # The server-only split is listed for coverage/reporting, never evaluated locally.
    assert _adapter().iter_scenes("test") == list(TEST_SCENES)
    assert len(TEST_SCENES) == 12


def test_iter_scenes_unknown_split() -> None:
    with pytest.raises(DatasetError, match="unknown ETH3D split"):
        _adapter().iter_scenes("validation")


def test_official_artifacts_resolves_mlp_and_scans() -> None:
    art = _adapter().official_artifacts("courtyard")
    assert art["scan_mlp"] == ROOT / "courtyard" / "dslr_scan_eval" / "scan_alignment.mlp"
    scans = art["scan_paths"]
    assert isinstance(scans, list) and [p.name for p in scans] == ["scan1.ply"]
    assert all(p.is_file() for p in scans)


def test_official_artifacts_missing_scene_errors() -> None:
    with pytest.raises(DatasetError, match="directory is missing"):
        _adapter().official_artifacts("pipes")


def test_official_artifacts_missing_scan_ply_errors(tmp_path: Path) -> None:
    scene = tmp_path / "courtyard" / "dslr_scan_eval"
    scene.mkdir(parents=True)
    mlp = ROOT / "courtyard" / "dslr_scan_eval" / "scan_alignment.mlp"
    (scene / "scan_alignment.mlp").write_text(mlp.read_text())
    with pytest.raises(DatasetError, match="references scan 'scan1.ply' but it does not exist"):
        Eth3dAdapter(tmp_path).official_artifacts("courtyard")


def test_parse_scan_mlp_empty_project_errors(tmp_path: Path) -> None:
    empty = tmp_path / "scan_alignment.mlp"
    empty.write_text("<MeshLabProject><MeshGroup></MeshGroup></MeshLabProject>")
    with pytest.raises(DatasetError, match="references no scan meshes"):
        parse_scan_mlp(empty)


def test_gt_provenance_is_independent_laser_scan() -> None:
    gt = _adapter().load_ground_truth("courtyard", protocol=None)  # type: ignore[arg-type]
    assert gt.modality == "pointcloud"
    assert gt.provenance == "laser_scan"
    assert gt.independence == "independent"
    assert gt.density == "dense_surface"
    assert gt.unit == "m"
    assert gt.source_pose_format == "world_to_cam_colmap"
    assert gt.path is not None and gt.path.name == "scan_alignment.mlp"


def test_capabilities() -> None:
    caps = _adapter().capabilities
    assert caps.dense_geometry is True
    assert caps.independent_gt is True
    assert caps.official_local_eval is True
    assert caps.official_local_eval_method == "official_script_wrapper"
    assert caps.server_only_eval is False  # per-split server-only via local_evaluation()


def test_local_evaluation_training_supported_test_server_only() -> None:
    adapter = _adapter()
    assert adapter.local_evaluation("training", None).status == "supported"  # type: ignore[arg-type]
    test = adapter.local_evaluation("test", None)  # type: ignore[arg-type]
    assert test.status == "server_only"
    assert test.official_server_required is True
    assert test.public_gt_available is False


def test_gt_fingerprint_joint_over_mlp_and_scans() -> None:
    fp = _adapter().gt_fingerprint("courtyard", None)  # type: ignore[arg-type]
    assert fp is not None and fp.startswith("sha256:")
    assert fp == _adapter().gt_fingerprint("courtyard", None)  # type: ignore[arg-type]


def test_gt_fingerprint_missing_scene_is_none() -> None:
    assert _adapter().gt_fingerprint("pipes", None) is None  # type: ignore[arg-type]


def test_load_scene_records_cameras_and_non_pinhole_limitation() -> None:
    scene = _adapter().load_scene("courtyard")
    assert scene.variant == "training_public_gt"
    assert scene.gt_pointcloud is not None and scene.gt_pointcloud.name == "scan_alignment.mlp"
    assert scene.metadata["source_pose_format"] == "world_to_cam_colmap"
    # Camera models come from the COLMAP text calibration via pycolmap.
    assert scene.metadata["camera_models"] == {"1": "PINHOLE", "2": "THIN_PRISM_FISHEYE"}
    assert scene.metadata["n_images"] == 2
    # The fixture has a non-pinhole camera: the limitation is recorded, not hidden.
    notes = scene.metadata["camera_model_limitations"]
    assert any("THIN_PRISM_FISHEYE" in note for note in notes)


def test_load_scene_without_calibration_records_limitation(tmp_path: Path) -> None:
    src = ROOT / "courtyard" / "dslr_scan_eval"
    dst = tmp_path / "courtyard" / "dslr_scan_eval"
    dst.mkdir(parents=True)
    for name in ("scan_alignment.mlp", "scan1.ply"):
        (dst / name).write_text((src / name).read_text())
    scene = Eth3dAdapter(tmp_path).load_scene("courtyard")
    assert scene.camera_paths is None
    assert any("calibration directory" in n for n in scene.metadata["camera_model_limitations"])


def test_resolve_prediction_default_layout() -> None:
    recon = _adapter().resolve_prediction(PREDS, "courtyard", None)
    assert recon.path == PREDS / "courtyard.ply"
    assert recon.modality == "pointcloud"
    assert recon.unit == "m"
    assert recon.scale == "metric"


def test_resolve_prediction_missing_errors() -> None:
    with pytest.raises(DatasetError, match="prediction for scene 'courtyard' not found"):
        _adapter().resolve_prediction(ROOT / "does_not_exist", "courtyard", None)
