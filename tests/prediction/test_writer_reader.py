from __future__ import annotations

import warnings

import numpy as np
import pytest

from eval3r import PredictionReader, PredictionWriter
from eval3r.utils.errors import EvalAssumptionWarning


def test_writer_reader_round_trip(tmp_path, gaussian_cloud, cube_mesh) -> None:
    v, f = cube_mesh
    out = tmp_path / "pred"
    with PredictionWriter(
        out,
        scene_id="scene_test",
        dataset="synthetic",
        method="unit_test",
        unit="m",
        coordinate_system="opengl",
        pose_convention="T_wc",
    ) as w:
        w.save_point_cloud(gaussian_cloud)
        w.save_mesh(v, f)
        w.save_metadata({"checkpoint": "ckpt.pth"})

    reader = PredictionReader(out)
    assert reader.manifest.scene_id == "scene_test"
    assert reader.manifest.metadata["checkpoint"] == "ckpt.pth"
    assert reader.points.points.shape == gaussian_cloud.shape
    assert reader.mesh.faces.shape == f.shape


def test_writer_warns_on_unspecified(tmp_path, gaussian_cloud) -> None:
    out = tmp_path / "pred"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with PredictionWriter(
            out, scene_id="s", dataset="d", method="m"
        ) as w:
            w.save_point_cloud(gaussian_cloud)
    assumption = [c for c in caught if issubclass(c.category, EvalAssumptionWarning)]
    assert assumption
    msg = str(assumption[0].message)
    for tag in ("unit", "coordinate_system", "pose_convention"):
        assert tag in msg


def test_writer_refuses_overwrite(tmp_path, gaussian_cloud) -> None:
    out = tmp_path / "pred"
    with PredictionWriter(out, scene_id="s", dataset="d", method="m", unit="m",
                          coordinate_system="opengl", pose_convention="T_wc") as w:
        w.save_point_cloud(gaussian_cloud)
    with pytest.raises(FileExistsError):
        PredictionWriter(out, scene_id="s", dataset="d", method="m", unit="m",
                         coordinate_system="opengl", pose_convention="T_wc")


def test_writer_refuses_nonempty_dir_without_manifest(tmp_path) -> None:
    out = tmp_path / "pred"
    out.mkdir()
    (out / "leftover.txt").write_text("stale")

    with pytest.raises(FileExistsError, match="non-empty"):
        PredictionWriter(
            out,
            scene_id="s",
            dataset="d",
            method="m",
            unit="m",
            coordinate_system="opengl",
            pose_convention="T_wc",
        )


def test_writer_overwrite_cleans_dir_without_manifest(tmp_path, gaussian_cloud) -> None:
    out = tmp_path / "pred"
    out.mkdir()
    stale = out / "leftover.txt"
    stale.write_text("stale")

    with PredictionWriter(
        out,
        scene_id="s",
        dataset="d",
        method="m",
        unit="m",
        coordinate_system="opengl",
        pose_convention="T_wc",
        overwrite=True,
    ) as w:
        w.save_point_cloud(gaussian_cloud)

    assert not stale.exists()


def test_reader_detects_corruption(tmp_path, gaussian_cloud) -> None:
    out = tmp_path / "pred"
    with PredictionWriter(out, scene_id="s", dataset="d", method="m", unit="m",
                          coordinate_system="opengl", pose_convention="T_wc") as w:
        w.save_point_cloud(gaussian_cloud)
    pc_path = out / "geometry" / "pred_points.ply"
    data = pc_path.read_bytes()
    pc_path.write_bytes(data + b"\x00")
    reader = PredictionReader(out)
    from eval3r.utils.errors import CorruptedArtifactError

    with pytest.raises(CorruptedArtifactError):
        _ = reader.points


def test_writer_save_poses(tmp_path) -> None:
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, :3, 3] = np.arange(12, dtype=np.float64).reshape(4, 3)
    out = tmp_path / "pred"
    with PredictionWriter(
        out, scene_id="s", dataset="d", method="m", unit="m",
        coordinate_system="opengl", pose_convention="T_wc",
    ) as w:
        w.save_poses(poses, timestamps=np.linspace(0, 1, 4))
    reader = PredictionReader(out)
    assert reader.poses.poses.shape == poses.shape
    assert np.allclose(reader.poses.poses[:, :3, 3], poses[:, :3, 3], atol=1e-6)
