"""Task 011 ScanNet-adapter unit tests: discovery, GT provenance, pose/depth, trajectory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval3r.core.errors import DatasetError
from eval3r.datasets import ScanNetAdapter, default_registry
from eval3r.protocols.registry import load_protocol

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "scannet_tiny" / "dataset_root"
PREDS = Path(__file__).resolve().parents[1] / "fixtures" / "scannet_tiny" / "preds"
VAL_SCENE = "scene0000_00"
TEST_SCENE = "scene0100_00"


def _adapter() -> ScanNetAdapter:
    return ScanNetAdapter(ROOT)


def test_registered_in_default_registry() -> None:
    assert "scannet" in default_registry().available()


def test_factory_requires_root() -> None:
    with pytest.raises(DatasetError) as exc:
        ScanNetAdapter.factory(None)
    assert "--root" in str(exc.value)


def test_iter_scenes_reads_split_files() -> None:
    assert _adapter().iter_scenes("val") == [VAL_SCENE]
    assert _adapter().iter_scenes("test") == [TEST_SCENE]


def test_missing_split_names_paths() -> None:
    with pytest.raises(DatasetError) as exc:
        _adapter().iter_scenes("nope")
    assert "nope.txt" in str(exc.value)


def test_load_scene_records_reconstruction_derived_gt() -> None:
    scene = _adapter().load_scene(VAL_SCENE)
    gt = scene.ground_truth
    assert gt.modality == "mesh"
    assert gt.provenance == "reconstructed"
    assert gt.independence == "reconstruction_derived"  # never 'independent'
    assert gt.density == "dense_surface"
    assert gt.unit == "m"
    assert gt.source_pose_format == "cam_to_world_opencv"
    assert scene.gt_mesh is not None and scene.gt_mesh.name.endswith("_vh_clean_2.ply")


def test_depth_unit_and_pose_format_recorded() -> None:
    scene = _adapter().load_scene(VAL_SCENE)
    assert scene.metadata["depth_unit"] == 0.001  # 16-bit millimetre depth
    assert scene.metadata["source_pose_format"] == "cam_to_world_opencv"


def test_capabilities_are_honest() -> None:
    caps = _adapter().capabilities
    assert caps.independent_gt is False  # reconstruction-derived, not laser scan
    assert caps.official_local_eval is False  # eval3r_native, no official ScanNet benchmark
    assert caps.requires_external_renderer is True  # test visibility culling
    assert caps.supports_full_scene_geometry is True


def test_layer_convention_from_protocol_variant() -> None:
    ad = _adapter()
    single = load_protocol("scannet_single_layer_geometry_5cm")
    double = load_protocol("scannet_double_layer_geometry_5cm")
    assert "layer_convention=single_layer" in ad.load_ground_truth(VAL_SCENE, single).notes
    assert "layer_convention=double_layer" in ad.load_ground_truth(VAL_SCENE, double).notes


def test_gt_fingerprint_is_content_hash() -> None:
    fp = _adapter().gt_fingerprint(VAL_SCENE, None)  # type: ignore[arg-type]
    assert fp is not None and fp.startswith("sha256:")


def test_resolve_prediction_infers_scene_ply() -> None:
    recon = _adapter().resolve_prediction(PREDS, VAL_SCENE, None)
    assert recon.modality == "mesh"
    assert recon.unit == "m"
    assert recon.path is not None and recon.path.name == f"{VAL_SCENE}.ply"


def test_resolve_prediction_missing_is_explicit() -> None:
    with pytest.raises(DatasetError) as exc:
        _adapter().resolve_prediction(PREDS, "scene9999_99", None)
    assert "scene9999_99" in str(exc.value)


def test_load_trajectory_filters_non_finite_poses() -> None:
    # The val fixture includes one inf (lost-tracking) pose that must be dropped.
    traj = _adapter().load_trajectory(VAL_SCENE)
    assert traj.poses.shape == (4, 4, 4)  # 4 finite poses, one inf dropped
    assert np.isfinite(traj.poses).all()
    assert traj.intrinsics.shape == (3, 3)
    assert (traj.width, traj.height) == (128, 96)
    assert traj.fingerprint is not None and traj.fingerprint.startswith("sha256:")


def test_local_evaluation_supported_but_native() -> None:
    local = _adapter().local_evaluation("val", load_protocol("scannet_single_layer_geometry_5cm"))
    assert local.status == "supported"
    assert "not official ScanNet numbers" in local.reason
