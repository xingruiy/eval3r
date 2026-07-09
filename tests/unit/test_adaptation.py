"""Prediction adaptation resolver tests."""

from __future__ import annotations

import pytest

from eval3r.core.adaptation import (
    AdaptationOverride,
    parse_adaptation_tokens,
    resolve_adaptation,
)
from eval3r.core.errors import AlignmentError
from eval3r.core.hashing import protocol_hash
from eval3r.core.manifest import PredictionManifest
from eval3r.datasets.conventions import convention_for
from eval3r.protocols import load_protocol


def _manifest(*, scale: str = "metric", pose: str = "cam_to_world_opencv") -> PredictionManifest:
    return PredictionManifest.model_validate(
        {
            "method": "m",
            "dataset": {"dataset": "custom"},
            "prediction_modality": "pointcloud",
            "coordinate_frame": "world",
            "source_pose_format": pose,
            "world_frame": "opencv",
            "scale": scale,
            "unit": "m",
            "scenes": {"s": {"pointcloud": "s.ply"}},
        }
    )


def test_token_order_independence() -> None:
    a = parse_adaptation_tokens("cw@opencv@sim3")
    b = parse_adaptation_tokens("sim3@opencv@cw")
    assert a == b
    assert a == AdaptationOverride(
        direction="cam_to_world", axes="opencv", alignment_mode="sim3"
    )


def test_alias_parsing() -> None:
    override = parse_adaptation_tokens("w2c@gl@scale_ls@relative@mm")
    assert override == AdaptationOverride(
        direction="world_to_cam",
        axes="opengl",
        scale="relative",
        alignment_mode="scale_least_squares",
        unit="mm",
    )


def test_unknown_and_duplicate_tokens_report_vocabulary() -> None:
    with pytest.raises(AlignmentError) as unknown:
        parse_adaptation_tokens("cw@bogus")
    assert "Valid tokens:" in str(unknown.value)
    assert "opencv" in str(unknown.value)

    with pytest.raises(AlignmentError) as duplicate:
        parse_adaptation_tokens("cw@w2c")
    assert "duplicate adaptation token for direction" in str(duplicate.value)
    assert "Valid tokens:" in str(duplicate.value)


def test_relative_scale_is_recorded_without_auto_alignment() -> None:
    proto = load_protocol("single_geometry")
    before = protocol_hash(proto)
    run_proto, record = resolve_adaptation(proto, _manifest(scale="relative"), None)
    assert protocol_hash(proto) == before
    assert record.scale == "relative"
    assert record.alignment == "none"
    assert record.reason == "passthrough"
    assert run_proto.alignment.mode == "none"


def test_relative_scale_is_not_refused_by_native_protocol() -> None:
    proto = load_protocol("dtu_native_pointcloud")
    run_proto, record = resolve_adaptation(proto, _manifest(scale="relative"), None)
    assert record.scale == "relative"
    assert record.alignment == proto.alignment.mode
    assert run_proto.alignment.mode == proto.alignment.mode


def test_convention_composition_matches_convention_for() -> None:
    proto = load_protocol("single_pose")
    run_proto, record = resolve_adaptation(
        proto, None, parse_adaptation_tokens("wc@opengl@trajectory_sim3")
    )
    assert run_proto.alignment.mode == "trajectory_sim3"
    assert record.pose_convention == "world_to_cam_opengl"
    assert convention_for(record.pose_convention) == convention_for("world_to_cam_opengl")
