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
    "dtu_official_like_pointcloud",
    "scannet_single_layer_geometry_5cm",
    "scannet_double_layer_geometry_5cm",
    "scannet_test_single_layer_geometry_5cm",
    "tanks_temples_training_official",
    "tanks_temples_intermediate_server_only",
    "eth3d_training_official",
    "neural_rgbd_geometry_culled",
    "neural_rgbd_geometry_source",
}

# Regression table: name -> canonical protocol hash. Update deliberately (and bump
# the protocol version) when a protocol's evaluation behavior changes.
EXPECTED_HASHES = {
    "dtu_official_like_pointcloud": "sha256:af601b9340b833f75b01359653b502903ace11783ed1f46dd5cb08989db9ddb0",
    "eth3d_training_official": "sha256:b98c4ffa41e3153cd8a2ce930804ef66797b55c945568f4bc9c7b01c1261b3e6",
    "scannet_double_layer_geometry_5cm": "sha256:0a97c4341e6e464685289efdf27467c400031ba27353dc4327c2e58ac6b1e5da",
    "scannet_single_layer_geometry_5cm": "sha256:0d14eb271693d3989958f6a5b5b043caa29f6ea59d1ee0c7b3f727e0b90f3047",
    "scannet_test_single_layer_geometry_5cm": "sha256:1116492e7f58e6721e5bd29df1dd0c200aa1c8bbfdb0fd1b0c5f7f14884f00dc",
    "neural_rgbd_geometry_culled": "sha256:51068ee74de7f5ae716020a70d40e4c32277cc68d6aa4632b044093fd6d965c3",
    "neural_rgbd_geometry_source": "sha256:71d9e57d2db54793f1689256e22127c4b44e9ee215dc721bdbc8865c8fdb42bc",
    "single_depth": "sha256:49cad750df8e34ceaaf8656147c60512676734828841c728974a91bb90f7adff",
    "single_geometry": "sha256:4be36cde991da73ee1b083d9c786364b827f70ecd7966efd8096ec28438ecfc3",
    "single_pose": "sha256:7b865d6fe0362175ddbda0592abfb4aaf238c5f6018959a4e467145d534f632b",
    "tanks_temples_intermediate_server_only": "sha256:0ba663f77dc6c2c4f1ce80c7c7b0d39a46c04f555b7132ff1544755933291b78",
    "tanks_temples_training_official": "sha256:253f271c69bf81ec57a26e467460178e60f933cd4f4143d30f7848df69144e9f",
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
    proto = load_protocol("tanks_temples_intermediate_server_only")
    assert proto.fidelity == "server_only"
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
