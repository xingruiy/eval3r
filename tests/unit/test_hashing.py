"""Task 003 hashing tests: canonical protocol hash stability and sensitivity.

The hash must be stable across YAML formatting/comment/notes changes and must
change when a field that affects evaluation behavior (threshold, sampling count,
masking method, aggregation, ...) changes. The non-semantic exclusion list is
asserted explicitly here.
"""

from __future__ import annotations

from eval3r.core.hashing import (
    NON_SEMANTIC_RECURSIVE_KEYS,
    NON_SEMANTIC_TOPLEVEL_KEYS,
    canonical_protocol_payload,
    protocol_hash,
)
from eval3r.protocols import load_protocol_data


def _base_data() -> dict:
    return {
        "schema_version": 1,
        "protocol_version": "0.1.0",
        "name": "hash_fixture",
        "fidelity": "native",
        "dataset": {"dataset": "custom", "variant": "v"},
        "prediction_modality": "pointcloud",
        "ground_truth": {
            "modality": "pointcloud",
            "provenance": "laser_scan",
            "independence": "independent",
            "density": "dense_surface",
        },
        "local_evaluation": {"status": "supported"},
        "alignment": {"mode": "none"},
        "confidence": {"policy": "none"},
        "masking": {"pred_culling": {"method": "none"}},
        "sampling": {
            "pred": {"method": "uniform_points", "n_points": 200_000},
            "gt": {"method": "all_points"},
        },
        "metrics": [{"name": "fscore", "threshold": 0.05}],
        "aggregation": {"per_scene": True},
        "failure_policy": {"policy": "abort"},
        "reporting": {"formats": ["json", "csv"]},
        "backend_preferences": {"nearest_neighbor": "scipy"},
    }


def _hash(data: dict) -> str:
    return protocol_hash(load_protocol_data(data, source="<test>"))


def test_exclusion_list_is_explicit() -> None:
    assert NON_SEMANTIC_RECURSIVE_KEYS == frozenset({"notes", "reason"})
    assert NON_SEMANTIC_TOPLEVEL_KEYS == frozenset({"reporting"})


def test_hash_is_deterministic() -> None:
    assert _hash(_base_data()) == _hash(_base_data())


def test_hash_ignores_notes_and_reason() -> None:
    base = _hash(_base_data())
    d = _base_data()
    d["notes"] = ["a human note that does not change evaluation"]
    d["dataset"]["notes"] = ["nested note"]
    d["local_evaluation"]["reason"] = "some prose reason"
    assert _hash(d) == base


def test_hash_ignores_reporting_preferences() -> None:
    base = _hash(_base_data())
    d = _base_data()
    d["reporting"] = {"formats": ["json", "csv", "markdown", "latex"], "save_colored_errors": True}
    assert _hash(d) == base


def test_hash_ignores_key_order_and_whitespace() -> None:
    base = _hash(_base_data())
    d = dict(reversed(list(_base_data().items())))
    assert _hash(d) == base


def test_hash_changes_on_threshold() -> None:
    d = _base_data()
    d["metrics"][0]["threshold"] = 0.02
    assert _hash(d) != _hash(_base_data())


def test_hash_changes_on_sampling_count() -> None:
    d = _base_data()
    d["sampling"]["pred"]["n_points"] = 100_000
    assert _hash(d) != _hash(_base_data())


def test_hash_changes_on_masking_method() -> None:
    d = _base_data()
    d["masking"]["pred_culling"]["method"] = "obs_mask"
    assert _hash(d) != _hash(_base_data())


def test_hash_changes_on_aggregation() -> None:
    d = _base_data()
    d["aggregation"]["weighted_mean"] = True
    assert _hash(d) != _hash(_base_data())


def test_payload_has_no_excluded_keys() -> None:
    payload = canonical_protocol_payload(load_protocol_data(_base_data(), source="<test>"))
    assert "reporting" not in payload

    def _walk(obj: object) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert k not in NON_SEMANTIC_RECURSIVE_KEYS, k
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(payload)
