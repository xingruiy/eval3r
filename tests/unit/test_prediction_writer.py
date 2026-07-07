"""Task 019 unit tests: PredictionWriter and the eval3r-native layout rules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from eval3r.backends.pointcloud_plyfile import PlyfilePointCloudBackend
from eval3r.backends.trajectory_evo import EvoTrajectoryBackend
from eval3r.core.errors import PredictionLayoutError
from eval3r.core.manifest import PredictionManifest
from eval3r.core.schema import DatasetVariant
from eval3r.predictions import PredictionWriter
from eval3r.predictions.layout import LAYOUT_NAME, required_field

PC = PlyfilePointCloudBackend()


def make_writer(root: Path, **overrides: object) -> PredictionWriter:
    kwargs: dict = dict(
        method="unit_method",
        dataset="fixture",
        split="pair",
        prediction_modality="pointcloud",
        scale="metric",
        coordinate_frame="world",
    )
    kwargs.update(overrides)
    return PredictionWriter(root, **kwargs)


@pytest.fixture
def trajectory_rows() -> np.ndarray:
    rng = np.random.default_rng(7)
    n = 6
    return np.concatenate(
        [
            (np.arange(n) * 0.1)[:, None],
            rng.random((n, 3)),
            np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        ],
        axis=1,
    )


def test_writer_arrays_round_trip_and_manifest_validates(
    tmp_path: Path, trajectory_rows: np.ndarray
) -> None:
    root = tmp_path / "preds"
    rng = np.random.default_rng(0)
    points = rng.random((120, 3))
    pointmap = rng.random((4, 5, 3)).astype(np.float32)
    pointmap[0, 0] = np.nan  # invalid pixels are legal in pointmaps
    conf = rng.random(120).astype(np.float32)

    writer = make_writer(root, confidence={"present": True})
    entry = writer.add_scene(
        "scene_a",
        pointcloud=points,
        pointmap=pointmap,
        trajectory=trajectory_rows,
        confidence=conf,
        metadata={"note": "unit"},
    )
    manifest = writer.finalize()

    # Canonical relative paths.
    assert entry.pointcloud == Path("scene_a/pointcloud.ply")
    assert entry.pointmap == Path("scene_a/pointmap.npy")
    assert entry.trajectory == Path("scene_a/trajectory_tum.txt")
    assert entry.confidence == Path("scene_a/confidence.npy")

    # Written files round-trip through eval3r's own loaders.
    np.testing.assert_allclose(PC.load_pointcloud(root / entry.pointcloud), points, atol=1e-6)
    np.testing.assert_allclose(np.load(root / entry.pointmap), pointmap)
    traj = EvoTrajectoryBackend().load_trajectory(root / entry.trajectory)
    assert traj.num_poses == trajectory_rows.shape[0]
    np.testing.assert_allclose(traj.positions_xyz, trajectory_rows[:, 1:4], atol=1e-8)
    np.testing.assert_allclose(np.load(root / entry.confidence), conf)

    # manifest.yaml re-validates as a PredictionManifest with layout provenance.
    data = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    parsed = PredictionManifest.model_validate(data)
    assert parsed.method == "unit_method"
    assert parsed.dataset.split == "pair"
    assert parsed.metadata["layout"] == LAYOUT_NAME
    from eval3r import __version__

    assert parsed.metadata["eval3r_version"] == __version__
    assert manifest.scenes["scene_a"].metadata["note"] == "unit"

    # Every written field carries a sha256 fingerprint.
    fingerprints = parsed.scenes["scene_a"].metadata["fingerprints"]
    assert set(fingerprints) == {"pointcloud", "pointmap", "trajectory", "confidence"}
    assert all(v.startswith("sha256:") for v in fingerprints.values())


def test_writer_records_and_reader_picks_up_conventions(tmp_path: Path) -> None:
    root = tmp_path / "preds"
    rng = np.random.default_rng(1)
    writer = make_writer(
        root,
        source_pose_format="cam_to_world_opengl",
        world_frame="opengl",
    )
    writer.add_scene("scene_a", pointcloud=rng.random((50, 3)))
    manifest = writer.finalize()

    # The convention fields are recorded on the writer's manifest...
    assert manifest.source_pose_format == "cam_to_world_opengl"
    assert manifest.world_frame == "opengl"
    assert manifest.normalized_convention == "cam_to_world_opencv_meters"

    # ...survive serialization and are auto-picked-up by the reader.
    data = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    assert data["world_frame"] == "opengl"
    parsed = PredictionManifest.model_validate(data)
    assert parsed.world_frame == "opengl"
    assert parsed.source_pose_format == "cam_to_world_opengl"


def test_writer_world_frame_defaults_to_opencv(tmp_path: Path) -> None:
    root = tmp_path / "preds"
    writer = make_writer(root)
    writer.add_scene("scene_a", pointcloud=np.random.default_rng(2).random((10, 3)))
    manifest = writer.finalize()
    assert manifest.world_frame == "opencv"


def test_writer_copies_files_keeping_suffix(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    PC.save_pointcloud(np.random.default_rng(1).random((50, 3)), source / "recon.ply")
    (source / "cams.json").write_text("{}", encoding="utf-8")
    (source / "model.obj").write_text("v 0 0 0\n", encoding="utf-8")

    root = tmp_path / "preds"
    writer = make_writer(root, prediction_modality="mesh")
    entry = writer.add_scene(
        "scene_a",
        mesh=source / "model.obj",
        pointcloud=source / "recon.ply",
        camera_file=source / "cams.json",
    )
    writer.finalize()

    assert entry.mesh == Path("scene_a/mesh.obj")  # canonical stem, source suffix
    assert entry.pointcloud == Path("scene_a/pointcloud.ply")
    assert entry.camera_file == Path("scene_a/cameras.json")
    assert (root / "scene_a" / "mesh.obj").read_text(encoding="utf-8") == "v 0 0 0\n"


def test_writer_copies_depth_sequence_directory(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(3):
        np.save(frames / f"{i:06d}.npy", np.full((4, 4), float(i)))

    root = tmp_path / "preds"
    writer = make_writer(root, prediction_modality="depth_sequence", depth_unit=1.0)
    entry = writer.add_scene("scene_a", depth_dir=frames)
    manifest = writer.finalize()

    assert entry.depth_dir == Path("scene_a/depth")
    assert sorted(p.name for p in (root / entry.depth_dir).iterdir()) == [
        "000000.npy", "000001.npy", "000002.npy",
    ]
    assert manifest.scenes["scene_a"].metadata["fingerprints"]["depth_dir"].startswith("sha256:")


def test_context_manager_finalizes_only_on_success(tmp_path: Path) -> None:
    root = tmp_path / "ok"
    with make_writer(root) as writer:
        writer.add_scene("scene_a", pointcloud=np.random.default_rng(2).random((10, 3)))
    assert (root / "manifest.yaml").is_file()

    failed_root = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="boom"):
        with make_writer(failed_root) as writer:
            writer.add_scene("scene_a", pointcloud=np.random.default_rng(2).random((10, 3)))
            raise RuntimeError("boom")
    # A half-written directory never gets a manifest.
    assert not (failed_root / "manifest.yaml").exists()


def test_writer_refusals(tmp_path: Path) -> None:
    points = np.random.default_rng(3).random((10, 3))

    # Modality without a canonical layout.
    with pytest.raises(PredictionLayoutError, match="colmap_reconstruction"):
        make_writer(tmp_path / "a", prediction_modality="colmap_reconstruction")
    assert required_field("mesh") == "mesh"

    # DatasetVariant and variant/split strings together are ambiguous.
    with pytest.raises(PredictionLayoutError, match="ambiguous"):
        make_writer(
            tmp_path / "b",
            dataset=DatasetVariant(dataset="fixture"),
            split="pair",
        )

    writer = make_writer(tmp_path / "preds")
    # Missing the modality-required field.
    with pytest.raises(PredictionLayoutError, match="requires the 'pointcloud' entry"):
        writer.add_scene("scene_a", trajectory=np.zeros((2, 8)))
    # Scene ids must be plain directory names.
    with pytest.raises(PredictionLayoutError, match="path separators"):
        writer.add_scene("a/b", pointcloud=points)
    # Missing source file names scene, field, and path.
    with pytest.raises(PredictionLayoutError, match="scene 'scene_a' field 'pointcloud'"):
        writer.add_scene("scene_a", pointcloud=tmp_path / "nope.ply")
    # Copy-only fields refuse arrays (eval3r builds no mesh geometry).
    with pytest.raises(PredictionLayoutError, match="copy-only"):
        writer.add_scene("scene_a", pointcloud=points, mesh=np.zeros((3, 3)))  # type: ignore[arg-type]
    # Confidence files require confidence.present in the manifest.
    with pytest.raises(PredictionLayoutError, match="confidence.present = false"):
        writer.add_scene("scene_a", pointcloud=points, confidence=np.ones(10, dtype=np.float32))
    # Finalize without scenes is not a valid export.
    with pytest.raises(PredictionLayoutError, match="no scenes"):
        writer.finalize()

    # Duplicate scenes are refused.
    writer.add_scene("scene_a", pointcloud=points)
    with pytest.raises(PredictionLayoutError, match="already added"):
        writer.add_scene("scene_a", pointcloud=points)

    writer.finalize()
    # No writes after finalize, no double finalize, no overwriting an export.
    with pytest.raises(PredictionLayoutError, match="already finalized"):
        writer.add_scene("scene_b", pointcloud=points)
    with pytest.raises(PredictionLayoutError, match="already finalized"):
        writer.finalize()
    with pytest.raises(PredictionLayoutError, match="never overwrites"):
        make_writer(tmp_path / "preds")


def test_writer_rejects_malformed_arrays(tmp_path: Path) -> None:
    writer = make_writer(tmp_path / "preds", confidence={"present": True})
    with pytest.raises(PredictionLayoutError, match=r"\(N, 3\) array"):
        writer.add_scene("s", pointcloud=np.zeros((4, 2)))
    with pytest.raises(PredictionLayoutError, match="NaN or Inf"):
        writer.add_scene("s", pointcloud=np.full((4, 3), np.nan))
    with pytest.raises(PredictionLayoutError, match=r"\(N, 8\)"):
        writer.add_scene("s", pointcloud=np.ones((4, 3)), trajectory=np.zeros((5, 7)))
    with pytest.raises(PredictionLayoutError, match="strictly increasing"):
        rows = np.zeros((3, 8))
        rows[:, 7] = 1.0  # identity quaternion, but all timestamps equal
        writer.add_scene("s", pointcloud=np.ones((4, 3)), trajectory=rows)
    with pytest.raises(PredictionLayoutError, match=r"\(H, W, 3\)"):
        writer.add_scene("s", pointcloud=np.ones((4, 3)), pointmap=np.zeros((4, 4)))
    with pytest.raises(PredictionLayoutError, match="finite float array"):
        writer.add_scene(
            "s", pointcloud=np.ones((4, 3)), confidence=np.array([np.inf], dtype=np.float32)
        )
    # add_scene is atomic: after any refusal the same scene can be re-added.
    entry = writer.add_scene("s", pointcloud=np.ones((4, 3)))
    assert entry.pointcloud == Path("s/pointcloud.ply")
    writer.finalize()
