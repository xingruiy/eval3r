"""Task 019 unit tests: read_prediction_dir / check_prediction_dir resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from eval3r.core.errors import PredictionLayoutError
from eval3r.predictions import PredictionWriter, check_prediction_dir, read_prediction_dir


@pytest.fixture
def written_dir(tmp_path: Path) -> Path:
    root = tmp_path / "preds"
    rng = np.random.default_rng(11)
    with PredictionWriter(
        root,
        method="reader_unit",
        dataset="fixture",
        split="pair",
        prediction_modality="pointcloud",
        scale="metric",
        coordinate_frame="world",
    ) as writer:
        for scene_id in ("scene_a", "scene_b"):
            writer.add_scene(scene_id, pointcloud=rng.random((30, 3)))
    return root


def test_reader_resolves_absolute_paths(written_dir: Path) -> None:
    check = read_prediction_dir(written_dir, verify=True)
    assert check.ok
    assert set(check.scenes) == {"scene_a", "scene_b"}
    for scene_id, fields in check.scenes.items():
        path = fields["pointcloud"]
        assert path.is_absolute()
        assert path == written_dir / scene_id / "pointcloud.ply"
        assert path.is_file()
    assert check.manifest.method == "reader_unit"


def test_reader_missing_file_names_scene_field_and_path(written_dir: Path) -> None:
    missing = written_dir / "scene_b" / "pointcloud.ply"
    missing.unlink()
    with pytest.raises(PredictionLayoutError) as excinfo:
        read_prediction_dir(written_dir)
    message = str(excinfo.value)
    assert "scene 'scene_b'" in message
    assert "field 'pointcloud'" in message
    assert str(missing) in message
    # The intact scene is not reported.
    assert "scene 'scene_a'" not in message


def test_verify_reports_every_mismatch_not_just_the_first(written_dir: Path) -> None:
    # Tamper with both scenes: verification must list both, with both digests.
    for scene_id in ("scene_a", "scene_b"):
        (written_dir / scene_id / "pointcloud.ply").write_bytes(b"tampered")
    with pytest.raises(PredictionLayoutError) as excinfo:
        read_prediction_dir(written_dir, verify=True)
    message = str(excinfo.value)
    assert "scene 'scene_a'" in message and "scene 'scene_b'" in message
    assert message.count("fingerprint mismatch") == 2
    assert "recorded sha256:" in message and "actual sha256:" in message

    # Without verification the files still resolve (existence only).
    assert check_prediction_dir(written_dir, verify=False).ok


def test_verify_flags_missing_fingerprint_records(written_dir: Path) -> None:
    # A hand-edited manifest without fingerprints cannot pass verification vacuously.
    manifest_path = written_dir / "manifest.yaml"
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["scenes"]["scene_a"]["metadata"] = {}
    manifest_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    check = check_prediction_dir(written_dir, verify=True)
    assert list(check.issues) == ["scene_a"]
    assert "no fingerprint recorded" in check.issues["scene_a"][0]
    assert check_prediction_dir(written_dir, verify=False).ok


def test_reader_wraps_manifest_problems(tmp_path: Path, written_dir: Path) -> None:
    with pytest.raises(PredictionLayoutError, match="is not a directory"):
        read_prediction_dir(tmp_path / "nowhere")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(PredictionLayoutError, match="no manifest.yaml"):
        read_prediction_dir(empty)

    manifest_path = written_dir / "manifest.yaml"
    manifest_path.write_text("method: [unclosed", encoding="utf-8")
    with pytest.raises(PredictionLayoutError, match="not valid YAML"):
        read_prediction_dir(written_dir)

    # Schema-invalid manifests are wrapped with the source path, not raw Pydantic.
    manifest_path.write_text(
        yaml.safe_dump({"method": "x", "scenes": {}}), encoding="utf-8"
    )
    with pytest.raises(PredictionLayoutError, match="schema"):
        read_prediction_dir(written_dir)


def test_verify_detects_directory_field_changes(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(2):
        np.save(frames / f"{i:06d}.npy", np.full((2, 2), float(i)))

    root = tmp_path / "preds"
    with PredictionWriter(
        root,
        method="reader_unit",
        dataset="fixture",
        prediction_modality="depth_sequence",
        scale="metric",
        coordinate_frame="world",
        depth_unit=1.0,
    ) as writer:
        writer.add_scene("scene_a", depth_dir=frames)

    assert read_prediction_dir(root, verify=True).ok
    # Editing a single frame changes the joint directory fingerprint.
    np.save(root / "scene_a" / "depth" / "000001.npy", np.full((2, 2), 99.0))
    with pytest.raises(PredictionLayoutError, match="field 'depth_dir'.*fingerprint mismatch"):
        read_prediction_dir(root, verify=True)
