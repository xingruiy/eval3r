"""Pipeline normalize-stage tests: unit conversion to metres."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval3r.core.errors import DatasetError
from eval3r.pipeline.stages.load import LoadedGeometry
from eval3r.pipeline.stages.normalize import (
    normalize_to_meters,
    scale_matrix,
    unit_to_meters,
)


def _pc(points) -> LoadedGeometry:
    return LoadedGeometry(kind="pointcloud", path=Path("mem"), points=np.asarray(points, float))


def test_unit_to_meters_known_units() -> None:
    assert unit_to_meters("m") == 1.0
    assert unit_to_meters("mm") == 0.001
    assert unit_to_meters("cm") == 0.01
    assert unit_to_meters("Millimeters") == 0.001
    assert unit_to_meters(None) == 1.0


def test_unit_to_meters_unknown_raises() -> None:
    with pytest.raises(DatasetError) as exc:
        unit_to_meters("furlong")
    assert "furlong" in str(exc.value)


def test_normalize_millimeters_scales_points() -> None:
    out = normalize_to_meters(_pc([[100.0, 0, 0], [0, 200.0, 0]]), "mm")
    assert np.allclose(out.points, [[0.1, 0, 0], [0, 0.2, 0]])


def test_normalize_meters_is_noop_identity() -> None:
    g = _pc([[1.0, 2.0, 3.0]])
    # already metric: same object returned (no copy, no scaling)
    assert normalize_to_meters(g, "m") is g
    assert normalize_to_meters(g, None) is g


def test_scale_matrix_is_uniform() -> None:
    m = scale_matrix(0.001)
    assert m[0, 0] == m[1, 1] == m[2, 2] == 0.001
    assert m[3, 3] == 1.0
