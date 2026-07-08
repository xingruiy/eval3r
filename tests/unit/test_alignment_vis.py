"""Task 018 unit tests: mandatory alignment visualization artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from eval3r.backends.pointcloud_plyfile import PlyfilePointCloudBackend
from eval3r.pipeline.stages.align import AlignmentVisData, TrajectoryAlignmentVisData
from eval3r.reports.alignment_vis import (
    GT_COLOR,
    PRED_COLOR,
    write_alignment_vis_outputs,
    write_alignment_visualization,
    write_trajectory_alignment_vis_outputs,
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
    assert (debug / "scenes" / "sceneA" / "alignment_before.ply").is_file()
    assert (debug / "scenes" / "sceneB" / "alignment_after.ply").is_file()
    assert (debug / "scenes" / "sceneA" / "alignment_projections.png").is_file()
    manifest = json.loads((debug / "alignment_vis.json").read_text(encoding="utf-8"))
    assert [r["scene_id"] for r in manifest["alignment_visualizations"]] == [
        "sceneA",
        "sceneB",
    ]
    assert manifest["alignment_visualizations"][0]["before_ply"] == (
        "scenes/sceneA/alignment_before.ply"
    )
    index = json.loads((debug / "debug_index.json").read_text(encoding="utf-8"))
    assert "alignment_visualizations" in index["debug_artifacts"]


def test_no_captures_writes_nothing(tmp_path: Path) -> None:
    assert write_alignment_vis_outputs([], tmp_path, pointcloud_backend=PC) == []
    assert not (tmp_path / "debug").exists()


def _trajectory_vis(scene_id: str = "traj") -> TrajectoryAlignmentVisData:
    pred_before = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float
    )
    pred_after = pred_before + np.array([0.5, 0.0, 0.0])
    gt = pred_after.copy()
    matrix = np.eye(4)
    matrix[:3, 3] = [0.5, 0.0, 0.0]
    return TrajectoryAlignmentVisData(
        scene_id=scene_id,
        pred_before=pred_before,
        pred_after=pred_after,
        gt=gt,
        alignment={
            "scene_id": scene_id,
            "mode": "trajectory_se3",
            "solver": "evo",
            "estimate_on": "trajectory",
            "matrix": matrix.tolist(),
            "scale": 1.0,
            "rotation": np.eye(3).tolist(),
            "translation": [0.5, 0.0, 0.0],
            "residual_rmse": 0.0,
            "n_correspondences": 3,
        },
        association={
            "policy": "nearest_timestamp",
            "associate_max_diff": 0.01,
            "offset": 0.0,
        },
        n_pred_poses=3,
        n_gt_poses=4,
        n_associated=3,
        n_dropped_pred=0,
        n_dropped_gt=1,
    )


def test_writes_trajectory_alignment_artifacts_and_indexes(tmp_path: Path) -> None:
    records = write_trajectory_alignment_vis_outputs(
        [_trajectory_vis("sceneT")], tmp_path, pointcloud_backend=PC
    )
    assert len(records) == 1
    debug = tmp_path / "debug"
    scene_dir = debug / "scenes" / "sceneT"
    assert (scene_dir / "trajectory_alignment_before.ply").is_file()
    assert (scene_dir / "trajectory_alignment_after.ply").is_file()
    assert (scene_dir / "trajectory_alignment_projections.png").is_file()
    per_scene = json.loads(
        (scene_dir / "trajectory_alignment.json").read_text(encoding="utf-8")
    )
    assert per_scene["counts"]["n_associated"] == 3
    assert per_scene["association"]["associate_max_diff"] == 0.01

    manifest = json.loads((debug / "trajectory_alignment_vis.json").read_text())
    record = manifest["trajectory_alignment_visualizations"][0]
    assert record["scene_id"] == "sceneT"
    assert record["before_ply"] == "scenes/sceneT/trajectory_alignment_before.ply"
    assert record["n_dropped_gt"] == 1
    index = json.loads((debug / "debug_index.json").read_text())
    assert "trajectory_alignment_visualizations" in index["debug_artifacts"]


def test_no_trajectory_captures_writes_nothing(tmp_path: Path) -> None:
    assert write_trajectory_alignment_vis_outputs([], tmp_path, pointcloud_backend=PC) == []
    assert not (tmp_path / "debug").exists()
