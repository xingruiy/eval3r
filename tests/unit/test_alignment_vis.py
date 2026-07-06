"""Task 018 unit tests: mandatory alignment visualization artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from eval3r.backends.pointcloud_plyfile import PlyfilePointCloudBackend
from eval3r.pipeline.stages.align import AlignmentVisData
from eval3r.reports.alignment_vis import (
    GT_COLOR,
    PRED_COLOR,
    write_alignment_vis_outputs,
    write_alignment_visualization,
)

PC = PlyfilePointCloudBackend()


def _vis(scene_id: str = "scene1", n: int = 50) -> AlignmentVisData:
    rng = np.random.default_rng(0)
    before = rng.random((n, 3))
    matrix = np.eye(4)
    matrix[:3, 3] = [1.0, 0.0, 0.0]
    return AlignmentVisData(
        scene_id=scene_id,
        pred_before=before,
        pred_after=before + np.array([1.0, 0.0, 0.0]),
        gt=rng.random((n + 10, 3)),
        alignment={
            "scene_id": scene_id,
            "mode": "se3",
            "solver": "icp",
            "estimate_on": "pointcloud",
            "matrix": matrix.tolist(),
            "scale": 1.0,
            "residual_rmse": 0.01,
            "fitness": 0.95,
            "n_correspondences": n,
            "parameters": {},
        },
        subsample_seed=0,
        max_points=100_000,
    )


def _read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertex = PlyData.read(str(path))["vertex"].data
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
    colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=-1)
    return points, colors


def test_writes_overlays_png_and_manifest(tmp_path: Path) -> None:
    vis = _vis()
    record = write_alignment_visualization(vis, tmp_path, pointcloud_backend=PC)

    before, colors = _read_ply(tmp_path / record["before_ply"])
    # Merged overlay: pred first (red), then gt (blue), fixed colors recorded.
    assert before.shape[0] == 50 + 60
    np.testing.assert_array_equal(colors[0], PRED_COLOR)
    np.testing.assert_array_equal(colors[-1], GT_COLOR)
    assert record["pred_color"] == list(PRED_COLOR)
    assert record["gt_color"] == list(GT_COLOR)

    after, _ = _read_ply(tmp_path / record["after_ply"])
    # The 'after' overlay carries the transformed prediction.
    np.testing.assert_allclose(
        after[:50], vis.pred_after, atol=1e-6  # f4 PLY precision
    )

    png = tmp_path / record["projections_png"]
    assert png.is_file() and png.stat().st_size > 0
    # The transform shown is recorded verbatim in the manifest.
    assert record["alignment"]["matrix"] == vis.alignment["matrix"]
    assert record["subsample_seed"] == 0


def test_run_directory_outputs_land_in_debug_with_scene_prefix(tmp_path: Path) -> None:
    captures = [_vis("sceneA"), _vis("sceneB")]
    records = write_alignment_vis_outputs(captures, tmp_path, pointcloud_backend=PC)
    assert len(records) == 2
    debug = tmp_path / "debug"
    assert (debug / "sceneA_alignment_before.ply").is_file()
    assert (debug / "sceneB_alignment_after.ply").is_file()
    assert (debug / "sceneA_alignment_projections.png").is_file()
    manifest = json.loads((debug / "alignment_vis.json").read_text(encoding="utf-8"))
    assert [r["scene_id"] for r in manifest["alignment_visualizations"]] == [
        "sceneA",
        "sceneB",
    ]


def test_no_captures_writes_nothing(tmp_path: Path) -> None:
    assert write_alignment_vis_outputs([], tmp_path, pointcloud_backend=PC) == []
    assert not (tmp_path / "debug").exists()
