"""Task 003 protocol tests: built-ins load/validate, registry, and hash regression.

The expected-hash table pins the canonical hash of every built-in protocol so a
behavior-changing edit to a protocol YAML is caught in CI.
"""
# ruff: noqa: E501  -- the expected-hash table has unavoidably long string literals.

from __future__ import annotations

import pytest

from eval3r.core.errors import ProtocolNotFoundError, ProtocolValidationError
from eval3r.protocols import (
    list_protocols,
    load_protocol,
    load_protocol_text,
    protocol_hash,
)

EXPECTED_BUILTINS = {
    "single_geometry",
    "single_depth",
    "single_pose",
    "dtu_native_pointcloud",
    "scannet_single_layer_geometry_5cm",
    "scannet_double_layer_geometry_5cm",
    "scannet_test_single_layer_geometry_5cm",
    "tanks_temples_training_official",
    "tanks_temples_intermediate_server",
    "eth3d_training_official",
    "neural_rgbd_geometry_culled",
    "neural_rgbd_geometry_source",
}

# Regression table: name -> canonical protocol hash. Update deliberately (and bump
# the protocol version) when a protocol's evaluation behavior changes.
EXPECTED_HASHES = {
    "dtu_native_pointcloud": "sha256:3e8f93f951c017dfbbc6149f5c80e35de46535680e967e2e9fe3e2afce0baaad",
    "eth3d_training_official": "sha256:de47a1b7d31b9d31757be0540612d9ba030adaf605637182fc19d2811fff76b3",
    "scannet_double_layer_geometry_5cm": "sha256:3298b78a23584d6a7c49d1341708559cf81d4add16f2295051bdcc2e113ffff6",
    "scannet_single_layer_geometry_5cm": "sha256:dc03fe863debdbdfd6d60fb0c541679652224418c3ea791bdf1aee43e2c18f7d",
    "scannet_test_single_layer_geometry_5cm": "sha256:a36c8acf23d9b4f03a0100698ddedbc286f9fd7fbb29e1d15f8f9686e7f87037",
    "neural_rgbd_geometry_culled": "sha256:d0f02285af30d856169587df8db866749955e461332957ed3676fca710671dc0",
    "neural_rgbd_geometry_source": "sha256:9389e5ef6004ca044b325b7931b81bef7dbc14c75b1962455669e5deb275ac7d",
    "single_depth": "sha256:74ada6a72e74ff4171277c8eec732844d2322ad3cebfca1d3b7d937d27b9aea7",
    "single_geometry": "sha256:a4a36e043762436ce117e0a94755e613feca4e5610ebdf4bf72d36b25dfcc1f6",
    "single_pose": "sha256:c36bfd6e367d4604252c99dbaff551555652c946c93472a067b9a90af048b266",
    "tanks_temples_intermediate_server": "sha256:8b6d39fa9bc8da8ac826fd4ba0cf65b6ebd131a47ad785387675a18689a97485",
    "tanks_temples_training_official": "sha256:cfb79c04616c6083ef6bbe05a90fbba8973f3f871ce051fb4e944205eb591e82",
}


def test_all_expected_builtins_present() -> None:
    assert set(list_protocols()) == EXPECTED_BUILTINS


@pytest.mark.parametrize("name", sorted(EXPECTED_BUILTINS))
def test_builtin_loads_and_validates(name: str) -> None:
    proto = load_protocol(name)
    assert proto.name == name
    assert proto.schema_version == 1


@pytest.mark.parametrize("name", sorted(EXPECTED_HASHES))
def test_builtin_hash_regression(name: str) -> None:
    assert protocol_hash(load_protocol(name)) == EXPECTED_HASHES[name]


def test_server_only_protocol_declares_no_local_metrics() -> None:
    proto = load_protocol("tanks_temples_intermediate_server")
    assert proto.fidelity == "server"
    assert proto.local_evaluation.status == "server_only"
    assert proto.metrics == []


def test_single_depth_uses_scale_median_and_valid_depth() -> None:
    proto = load_protocol("single_depth")
    assert proto.alignment.mode == "scale_median"
    assert proto.alignment.granularity == "per_frame"
    assert proto.masking.valid_region.method == "valid_depth"


def test_single_depth_delta_names_disambiguate_thresholds() -> None:
    # Aggregation keys metrics by name, so the three delta thresholds must have
    # distinct names with explicit thresholds (task 014).
    proto = load_protocol("single_depth")
    deltas = {m.name: m.threshold for m in proto.metrics if m.name.startswith("delta")}
    assert deltas == {"delta_1": 1.25, "delta_2": 1.5625, "delta_3": 1.953125}


def test_single_pose_pins_rpe_deltas_and_association() -> None:
    # RPE at delta=1 frame is not comparable to delta=1 second, so the deltas are
    # explicit protocol parameters, never backend defaults (task 015).
    proto = load_protocol("single_pose")
    assert proto.alignment.mode == "trajectory_sim3"
    assert proto.alignment.solver == "evo"
    assert proto.alignment.parameters["associate_max_diff"] == 0.01
    rpe = {m.name: m.parameters for m in proto.metrics if m.name.startswith("rpe_")}
    assert rpe == {
        "rpe_translation": {"delta": 1, "delta_unit": "frames", "all_pairs": False},
        "rpe_rotation": {"delta": 1, "delta_unit": "frames", "all_pairs": False},
    }
    names = [m.name for m in proto.metrics]
    assert names == ["ate", "rpe_translation", "rpe_rotation", "alignment_scale_error"]


def test_unknown_protocol_raises_not_found() -> None:
    with pytest.raises(ProtocolNotFoundError):
        load_protocol("does_not_exist")


def test_invalid_protocol_text_reports_field() -> None:
    bad = "schema_version: 1\nname: broken\n"  # missing many required fields
    with pytest.raises(ProtocolValidationError) as exc:
        load_protocol_text(bad, source="<bad>")
    # The error names the source and lists failing fields.
    assert "<bad>" in str(exc.value)
