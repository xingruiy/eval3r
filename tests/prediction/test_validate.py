from __future__ import annotations

import json
import warnings

from eval3r import PredictionWriter
from eval3r.manifest.manifest import MANIFEST_FILENAME
from eval3r.manifest.validate import validate_prediction


def test_validate_clean(tmp_path, gaussian_cloud) -> None:
    out = tmp_path / "pred"
    with PredictionWriter(
        out, scene_id="s", dataset="d", method="m", unit="m",
        coordinate_system="opengl", pose_convention="T_wc",
    ) as w:
        w.save_point_cloud(gaussian_cloud)
    report = validate_prediction(out)
    assert report.ok, report.errors


def test_validate_detects_missing_file(tmp_path, gaussian_cloud) -> None:
    out = tmp_path / "pred"
    with PredictionWriter(
        out, scene_id="s", dataset="d", method="m", unit="m",
        coordinate_system="opengl", pose_convention="T_wc",
    ) as w:
        w.save_point_cloud(gaussian_cloud)
    (out / "geometry" / "pred_points.ply").unlink()
    report = validate_prediction(out)
    assert not report.ok


def test_validate_warns_on_unspecified(tmp_path, gaussian_cloud) -> None:
    out = tmp_path / "pred"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with PredictionWriter(out, scene_id="s", dataset="d", method="m") as w:
            w.save_point_cloud(gaussian_cloud)
    report = validate_prediction(out)
    assert report.ok
    assert any("unit" in w for w in report.warnings)


def test_validate_handles_bad_manifest(tmp_path) -> None:
    out = tmp_path / "pred"
    out.mkdir()
    (out / MANIFEST_FILENAME).write_text(json.dumps({"oops": True}))
    report = validate_prediction(out)
    assert not report.ok
