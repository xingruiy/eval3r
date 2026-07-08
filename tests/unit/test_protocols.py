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
    "dtu_official_like_pointcloud": "sha256:251123ac1c0a674d6309baac50f24f07727fc9b1229d743c703be7c74daecb3f",
    "eth3d_training_official": "sha256:de47a1b7d31b9d31757be0540612d9ba030adaf605637182fc19d2811fff76b3",
    "scannet_double_layer_geometry_5cm": "sha256:ebede3780da161a3771e4c47cf4786d6b3d8ee39e7e99b231310f8d6886f14ce",
    "scannet_single_layer_geometry_5cm": "sha256:39befa72ba1eeb40e98295077a7d0ee098596e5df12357aaea758dffb12b3c18",
    "scannet_test_single_layer_geometry_5cm": "sha256:4cdd952ba7cfbd24b9e81b5f2ea68bad4c330b3b0e9c5a5cbb1f301c9c8fbf88",
    "neural_rgbd_geometry_culled": "sha256:7abc708264d033f081e3d7e86f2d4398e6eaf84d77cb155d30c81c5d0cb85910",
    "neural_rgbd_geometry_source": "sha256:e955be6c7bb99b45128b441dec82b9aca11bc7c28d20700fdb215c6b4f5a0f11",
    "single_depth": "sha256:1ceffc7e9cd1f77ef071619b7ce0c058a41707ceecce6e520522c29c9eeb056e",
    "single_geometry": "sha256:ac14364fc693fef0e4bdd50cf7e6ad80fc219569f0efb6f93a6b7f4d4a61c942",
    "single_pose": "sha256:36570e9b2a113191a3b7afd64ca90bf573042500dc21e27a2bd40adccfb2f568",
    "tanks_temples_intermediate_server_only": "sha256:223c263c11d8b251846e779185425caab42b2cf87a42b104b16b21b4efa49048",
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
