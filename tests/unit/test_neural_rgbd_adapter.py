"""Task 020 Neural-RGBD geometry-adapter unit tests.

Discovery, culled/source variant selection, GT provenance (synthetic-exact/independent),
prediction resolution, and honest capabilities. Geometry only — no depth/pose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval3r.core.errors import DatasetError
from eval3r.datasets import NeuralRGBDAdapter, default_registry
from eval3r.protocols import load_protocol

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "neural_rgbd_tiny" / "official"
PREDS = Path(__file__).resolve().parents[1] / "fixtures" / "neural_rgbd_tiny" / "preds"
SCENE = "breakfast_room"

CULLED = load_protocol("neural_rgbd_geometry_culled")
SOURCE = load_protocol("neural_rgbd_geometry_source")


def _adapter() -> NeuralRGBDAdapter:
    return NeuralRGBDAdapter(ROOT)


def test_registered_in_default_registry() -> None:
    assert "neural_rgbd" in default_registry().available()


def test_factory_requires_root() -> None:
    with pytest.raises(DatasetError) as exc:
        NeuralRGBDAdapter.factory(None)
    assert "--root" in str(exc.value)
    assert "gt_mesh_culled.ply" in str(exc.value)


def test_iter_scenes_all_split_enumerates_mesh_dirs() -> None:
    assert _adapter().iter_scenes("all") == ["breakfast_room", "kitchen"]


def test_unknown_split_without_file_is_explicit() -> None:
    with pytest.raises(DatasetError) as exc:
        _adapter().iter_scenes("nope")
    assert "nope.txt" in str(exc.value) and "split 'all'" in str(exc.value)


def test_default_variant_is_culled() -> None:
    scene = _adapter().load_scene(SCENE)
    assert scene.metadata["mesh_variant"] == "culled"
    assert scene.gt_mesh is not None and scene.gt_mesh.name == "gt_mesh_culled.ply"


def test_variant_selection_resolves_different_meshes() -> None:
    ad = _adapter()
    culled = ad.load_ground_truth(SCENE, CULLED)
    source = ad.load_ground_truth(SCENE, SOURCE)
    assert culled.path is not None and culled.path.name == "gt_mesh_culled.ply"
    assert source.path is not None and source.path.name == "gt_mesh.ply"
    assert culled.path != source.path
    assert "mesh_variant=culled" in culled.notes
    assert "mesh_variant=source" in source.notes


def test_bind_protocol_switches_scene_gt_mesh() -> None:
    ad = _adapter()
    ad.bind_protocol(SOURCE)
    scene = ad.load_scene(SCENE)
    assert scene.metadata["mesh_variant"] == "source"
    assert scene.gt_mesh is not None and scene.gt_mesh.name == "gt_mesh.ply"


def test_unknown_variant_is_explicit() -> None:
    bad = load_protocol("single_geometry")  # variant does not name culled/source
    with pytest.raises(DatasetError) as exc:
        _adapter().load_ground_truth(SCENE, bad)
    assert "neural_rgbd_geometry_culled" in str(exc.value)
    assert "neural_rgbd_geometry_source" in str(exc.value)


def test_gt_is_synthetic_exact_and_independent() -> None:
    gt = _adapter().load_ground_truth(SCENE, CULLED)
    assert gt.modality == "mesh"
    assert gt.provenance == "synthetic_exact"
    assert gt.independence == "independent"  # exact artist mesh, not reconstruction-derived
    assert gt.density == "dense_surface"
    assert gt.unit == "m"
    assert gt.source_pose_format == "cam_to_world_opengl"
    assert any("OpenGL" in n for n in gt.notes)


def test_capabilities_are_honest() -> None:
    caps = _adapter().capabilities
    assert caps.independent_gt is True  # exact synthetic GT
    assert caps.official_local_eval is False  # eval3r_native, no official benchmark
    assert caps.requires_external_renderer is False  # geometry is mesh-to-mesh
    assert caps.supports_full_scene_geometry is True


def test_gt_fingerprint_is_content_hash() -> None:
    fp = _adapter().gt_fingerprint(SCENE, CULLED)
    assert fp is not None and fp.startswith("sha256:")


def test_resolve_prediction_infers_scene_ply() -> None:
    recon = _adapter().resolve_prediction(PREDS, SCENE, None)
    assert recon.modality == "mesh"
    assert recon.unit == "m"
    assert recon.path is not None and recon.path.name == f"{SCENE}.ply"


def test_resolve_prediction_missing_is_explicit() -> None:
    with pytest.raises(DatasetError) as exc:
        _adapter().resolve_prediction(PREDS, "no_such_scene", None)
    assert "no_such_scene" in str(exc.value)


def test_local_evaluation_supported_but_native() -> None:
    local = _adapter().local_evaluation("all", CULLED)
    assert local.status == "supported"
    assert "not official Neural-RGBD numbers" in local.reason
