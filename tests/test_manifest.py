from __future__ import annotations

from eval3r._version import FORMAT_VERSION, __version__
from eval3r.prediction.manifest import (
    CoordinateSystem,
    Manifest,
    PoseConvention,
    Unit,
)


def test_manifest_defaults() -> None:
    m = Manifest(scene_id="s1", dataset="scannet", method="m1")
    assert m.eval3r_version == __version__
    assert m.format_version == FORMAT_VERSION
    assert m.unit is Unit.UNSPECIFIED
    assert m.coordinate_system is CoordinateSystem.UNSPECIFIED
    assert m.pose_convention is PoseConvention.UNSPECIFIED


def test_manifest_round_trip() -> None:
    m = Manifest(
        scene_id="s",
        dataset="d",
        method="m",
        unit="m",
        coordinate_system="colmap",
        pose_convention="T_wc",
    )
    payload = m.model_dump_json()
    m2 = Manifest.model_validate_json(payload)
    assert m == m2
