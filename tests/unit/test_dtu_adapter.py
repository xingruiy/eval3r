"""Task 009 DTU-adapter unit tests: discovery, mm GT, masks, filename resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval3r.core.errors import DatasetError
from eval3r.datasets import DTUAdapter, default_registry

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "dtu_tiny" / "dataset_root"
PREDS = Path(__file__).resolve().parents[1] / "fixtures" / "dtu_tiny" / "preds"


def _adapter() -> DTUAdapter:
    return DTUAdapter(ROOT)


def test_registered_in_default_registry() -> None:
    assert "dtu" in default_registry().available()


def test_factory_requires_root() -> None:
    with pytest.raises(DatasetError) as exc:
        DTUAdapter.factory(None)
    assert "--root" in str(exc.value)


def test_iter_scenes_reads_scan_ids() -> None:
    assert _adapter().iter_scenes("test") == ["1", "4"]
    assert _adapter().iter_scenes("one") == ["1"]


def test_scan_id_accepts_scan_prefix() -> None:
    a = _adapter()
    assert a._scan_int("scan1") == 1
    assert a._scan_int("1") == 1
    assert a._scan_int("scan004") == 4


def test_gt_is_independent_laser_scan_in_mm() -> None:
    gt = _adapter().load_scene("1").ground_truth
    assert gt.modality == "pointcloud"
    assert gt.provenance == "laser_scan"
    assert gt.independence == "independent"
    assert gt.density == "dense_surface"
    assert gt.unit == "mm"
    assert gt.fingerprint is not None and gt.fingerprint.startswith("sha256:")


def test_capabilities_do_not_claim_official_yet() -> None:
    caps = _adapter().capabilities
    assert caps.dense_geometry is True
    assert caps.independent_gt is True
    assert caps.supports_object_centric_geometry is True
    assert caps.supports_full_scene_geometry is False
    # Official-like DTU evaluation is task 010; must not be claimed here.
    assert caps.official_local_eval is False
    assert caps.official_local_eval_method == "none"


def test_masks_and_plane_availability_recorded() -> None:
    a = _adapter()
    s1 = a.load_scene("1")
    assert set(s1.masks) == {"obs_mask", "plane"}
    assert s1.metadata["plane_available"] is True
    assert s1.metadata["native_unit"] == "mm"

    s4 = a.load_scene("4")  # scan 4 has no Plane file
    assert set(s4.masks) == {"obs_mask"}
    assert s4.metadata["plane_available"] is False
    assert s4.metadata["plane_path"] is None


def test_missing_gt_raises() -> None:
    with pytest.raises(DatasetError) as exc:
        _adapter().load_scene("99")
    assert "scan 99" in str(exc.value)


def test_resolve_prediction_honours_light_suffix() -> None:
    recon = _adapter().resolve_prediction(PREDS, "1", None)
    assert recon.path.name == "mvsnet001_l3.ply"
    assert recon.modality == "pointcloud"
    assert recon.unit == "mm"
    assert recon.metadata["light_condition"] == 3
    assert recon.metadata["method"] == "mvsnet"


def test_resolve_missing_prediction_names_light_convention() -> None:
    with pytest.raises(DatasetError) as exc:
        _adapter().resolve_prediction(PREDS, "9", None)
    assert "l3" in str(exc.value)
    assert "009_l3.ply" in str(exc.value)


def test_gt_fingerprint_differs_with_and_without_plane() -> None:
    a = _adapter()
    fp1 = a.gt_fingerprint("1", None)  # GT + ObsMask + Plane
    fp4 = a.gt_fingerprint("4", None)  # GT + ObsMask only
    assert fp1 is not None and fp4 is not None
    assert fp1 != fp4


def test_local_evaluation_supported() -> None:
    assert _adapter().local_evaluation("test", None).status == "supported"
