"""Tests for the SelectionPolygonVolume crop module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval3r.io.crop import CropVolume, crop_points_inside, load_crop_volume_json
from eval3r.utils.errors import MissingArtifactError


def _square_volume_y(tmp_path: Path) -> Path:
    """Unit square in XZ at Y in [0, 1]; orthogonal_axis = Y."""
    payload = {
        "class_name": "SelectionPolygonVolume",
        "orthogonal_axis": "Y",
        "axis_min": 0.0,
        "axis_max": 1.0,
        "bounding_polygon": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        "version_major": 1,
        "version_minor": 0,
    }
    p = tmp_path / "vol.json"
    p.write_text(json.dumps(payload))
    return p


def test_load_round_trip(tmp_path: Path) -> None:
    p = _square_volume_y(tmp_path)
    vol = load_crop_volume_json(p)
    assert isinstance(vol, CropVolume)
    assert vol.orthogonal_axis == 1  # Y
    assert vol.axis_min == 0.0
    assert vol.axis_max == 1.0
    assert vol.polygon_2d.shape == (4, 2)
    np.testing.assert_array_equal(
        vol.polygon_2d,
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    )


def test_inside_outside_band(tmp_path: Path) -> None:
    vol = load_crop_volume_json(_square_volume_y(tmp_path))
    points = np.array(
        [
            [0.5, 0.5, 0.5],   # inside
            [0.5, -0.1, 0.5],  # below band
            [0.5, 1.1, 0.5],   # above band
            [-0.1, 0.5, 0.5],  # left of polygon (X)
            [0.5, 0.5, 1.1],   # outside polygon (Z)
        ],
        dtype=np.float64,
    )
    mask = crop_points_inside(vol, points)
    np.testing.assert_array_equal(mask, [True, False, False, False, False])


def test_boundary_inclusive(tmp_path: Path) -> None:
    vol = load_crop_volume_json(_square_volume_y(tmp_path))
    points = np.array(
        [
            [0.0, 0.0, 0.0],   # on a polygon vertex AND at axis_min
            [1.0, 1.0, 1.0],   # opposite vertex AND axis_max
            [0.0, 0.5, 0.5],   # on a polygon edge
        ],
        dtype=np.float64,
    )
    mask = crop_points_inside(vol, points)
    assert mask.all(), f"expected all boundary points inclusive, got {mask}"


def test_non_convex_polygon_l_shape(tmp_path: Path) -> None:
    # L-shape in XZ at Y in [0, 1]:
    #   (0,0)-(2,0)-(2,1)-(1,1)-(1,2)-(0,2)
    payload = {
        "class_name": "SelectionPolygonVolume",
        "orthogonal_axis": "Y",
        "axis_min": 0.0,
        "axis_max": 1.0,
        "bounding_polygon": [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
    }
    p = tmp_path / "L.json"
    p.write_text(json.dumps(payload))
    vol = load_crop_volume_json(p)
    points = np.array(
        [
            [0.5, 0.5, 0.5],   # inside lower arm
            [0.5, 0.5, 1.5],   # inside upper arm
            [1.5, 0.5, 1.5],   # in the L's notch — should be OUTSIDE
            [1.5, 0.5, 0.5],   # inside lower-right arm
        ],
        dtype=np.float64,
    )
    mask = crop_points_inside(vol, points)
    np.testing.assert_array_equal(mask, [True, True, False, True])


@pytest.mark.parametrize("axis_label,axis_idx", [("X", 0), ("Y", 1), ("Z", 2)])
def test_each_orthogonal_axis(tmp_path: Path, axis_label: str, axis_idx: int) -> None:
    # Unit square in the two non-orthogonal axes at the orthogonal axis
    # in [0, 1]. Polygon vertices have an arbitrary (ignored) coord on
    # the orthogonal axis to confirm we drop it.
    polygon_3d: list[list[float]] = []
    for u, v in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]:
        vert = [0.0, 0.0, 0.0]
        # Fill the two non-orthogonal axes:
        non_orth = [i for i in (0, 1, 2) if i != axis_idx]
        vert[non_orth[0]] = u
        vert[non_orth[1]] = v
        # Junk value on the orthogonal axis (must be ignored):
        vert[axis_idx] = 99.0
        polygon_3d.append(vert)
    payload = {
        "class_name": "SelectionPolygonVolume",
        "orthogonal_axis": axis_label,
        "axis_min": 0.0,
        "axis_max": 1.0,
        "bounding_polygon": polygon_3d,
    }
    p = tmp_path / "ax.json"
    p.write_text(json.dumps(payload))
    vol = load_crop_volume_json(p)

    pt_in = [0.5, 0.5, 0.5]  # Always inside since the unit cube fits all axes
    pt_out = [0.5, 0.5, 0.5]
    pt_out[axis_idx] = 2.0   # Outside the band along the orthogonal axis
    mask = crop_points_inside(vol, np.array([pt_in, pt_out], dtype=np.float64))
    np.testing.assert_array_equal(mask, [True, False])


def test_schema_wrong_class_name(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"class_name": "SomethingElse", "orthogonal_axis": "Y"}))
    with pytest.raises(MissingArtifactError, match="SelectionPolygonVolume"):
        load_crop_volume_json(p)


def test_schema_bad_axis(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({
        "class_name": "SelectionPolygonVolume",
        "orthogonal_axis": "W",
        "axis_min": 0.0, "axis_max": 1.0,
        "bounding_polygon": [[0, 0, 0], [1, 0, 0], [1, 0, 1]],
    }))
    with pytest.raises(MissingArtifactError, match="orthogonal_axis"):
        load_crop_volume_json(p)


def test_schema_bad_polygon_shape(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({
        "class_name": "SelectionPolygonVolume",
        "orthogonal_axis": "Y",
        "axis_min": 0.0, "axis_max": 1.0,
        "bounding_polygon": [[0, 0], [1, 0], [1, 1]],  # 2D, must be 3D
    }))
    with pytest.raises(MissingArtifactError, match=r"shape \(N, 3\)"):
        load_crop_volume_json(p)


def test_schema_too_few_vertices(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({
        "class_name": "SelectionPolygonVolume",
        "orthogonal_axis": "Y",
        "axis_min": 0.0, "axis_max": 1.0,
        "bounding_polygon": [[0, 0, 0], [1, 0, 0]],
    }))
    with pytest.raises(MissingArtifactError, match=">= 3"):
        load_crop_volume_json(p)


def test_schema_axis_max_below_min(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({
        "class_name": "SelectionPolygonVolume",
        "orthogonal_axis": "Y",
        "axis_min": 1.0, "axis_max": 0.0,
        "bounding_polygon": [[0, 0, 0], [1, 0, 0], [1, 0, 1]],
    }))
    with pytest.raises(MissingArtifactError, match="axis_max"):
        load_crop_volume_json(p)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MissingArtifactError, match="not found"):
        load_crop_volume_json(tmp_path / "nope.json")


def test_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(MissingArtifactError, match="valid JSON"):
        load_crop_volume_json(p)


def test_empty_points(tmp_path: Path) -> None:
    vol = load_crop_volume_json(_square_volume_y(tmp_path))
    mask = crop_points_inside(vol, np.zeros((0, 3)))
    assert mask.shape == (0,)
    assert mask.dtype == bool


def test_points_wrong_shape(tmp_path: Path) -> None:
    vol = load_crop_volume_json(_square_volume_y(tmp_path))
    with pytest.raises(ValueError):
        crop_points_inside(vol, np.zeros((5, 2)))
