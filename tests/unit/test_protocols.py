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
}

# Regression table: name -> canonical protocol hash. Update deliberately (and bump
# the protocol version) when a protocol's evaluation behavior changes.
EXPECTED_HASHES = {
    "dtu_official_like_pointcloud": "sha256:192ffe1c471960ca5ea829fe397381205e5de3af88bb3502c3628b2cdca20ee2",
    "eth3d_training_official": "sha256:67a7f8a0198fde3f9cdf3d93e8defe829e6aa21acac22830f0ba057cab737087",
    "scannet_double_layer_geometry_5cm": "sha256:5a7f57763e3d6227fd23e538666d082eec63c27ca8ae10211317f15c0401e24a",
    "scannet_single_layer_geometry_5cm": "sha256:d8fa19896f75b3cae8a37f3273cc10bb4547beb9a5c7b874949e02c96fce9646",
    "scannet_test_single_layer_geometry_5cm": "sha256:864a238ad143ac48031457eb26d945c0f1ea4e75a80d438aa6b1428b80356bea",
    "single_depth": "sha256:5af749bbea61dd7be60978780438d595cffa6013e01547ded556246a6850c938",
    "single_geometry": "sha256:7168bb180bf6f9c1feed8b75b40d74191cb6a57c814cc26c2b0933120121a781",
    "single_pose": "sha256:796eb35aebaca3c1b219993ab770b29daec8723157e23b3eb5f7b51f7286cf0e",
    "tanks_temples_intermediate_server_only": "sha256:bb115c4291cfac50c8d06be7af824ea1ab54b9aad3d0367b973188eb2a0d1333",
    "tanks_temples_training_official": "sha256:e9c5d162a1b6f3cc3f972461ffee7b98593adbee725cf0ca419b96428d726e20",
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
