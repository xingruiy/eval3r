"""Task 014 integration tests: the single-file / sequence depth path end-to-end.

Covers the Python API and the ``e3r metric depth`` CLI on tiny synthetic depth
files with analytic expectations, and checks the run directory records the depth
unit, mask counts, and alignment mode + granularity (``.agent/tasks/014``).
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from typer.testing import CliRunner

from eval3r import evaluate_depth
from eval3r.cli.main import app
from eval3r.core.errors import SceneEvaluationError
from eval3r.reports.json import read_run_result_json

runner = CliRunner()

GT = np.array([[1.0, 2.0], [3.0, 4.0]])


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    """GT as a 16-bit millimetre PNG, prediction = 2 x gt as a float .npy in metres."""
    gt_path = tmp_path / "gt.png"
    iio.imwrite(gt_path, (GT * 1000).astype(np.uint16))
    pred_path = tmp_path / "pred.npy"
    np.save(pred_path, 2.0 * GT)
    return pred_path, gt_path


def test_median_scale_alignment_zeroes_constant_scale_error(tmp_path: Path) -> None:
    pred, gt = _write_pair(tmp_path)
    result = evaluate_depth(pred, gt, gt_depth_unit=0.001, out_dir=tmp_path / "run")
    m = result.metrics
    assert m["absrel"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["silog"] == pytest.approx(0.0, abs=1e-12)
    assert m["delta_1"] == 1.0 and m["delta_2"] == 1.0 and m["delta_3"] == 1.0
    assert result.n_scenes_evaluated == 1
    assert not result.failed_scenes


def test_align_none_override_exposes_scale_error(tmp_path: Path) -> None:
    # pred = 2 x gt without alignment: absrel = 1.0, delta ratio 2 -> only delta_3 passes.
    pred, gt = _write_pair(tmp_path)
    run = evaluate_depth(
        pred, gt, gt_depth_unit=0.001, align="none", return_run=True
    )
    m = run.result.metrics
    assert m["absrel"] == pytest.approx(1.0)
    assert m["delta_1"] == 0.0
    assert m["delta_2"] == 0.0
    assert m["delta_3"] == 0.0  # ratio 2.0 >= 1.953125
    assert m["rmse_log"] == pytest.approx(np.log(2.0))
    assert run.overrides["align"] == "none"
    # The override changes evaluation behavior, so the hash differs from the builtin.
    from eval3r.protocols import load_protocol, protocol_hash

    assert run.protocol_hash != protocol_hash(load_protocol("single_depth"))


def test_run_directory_records_units_masks_and_alignment(tmp_path: Path) -> None:
    pred, gt = _write_pair(tmp_path)
    out = tmp_path / "run"
    result = evaluate_depth(pred, gt, gt_depth_unit=0.001, out_dir=out)

    for name in (
        "results.json", "results.csv", "per_scene.csv", "failures.json",
        "environment.json", "backend_versions.json", "logs.txt",
        "protocol.yaml", "config.yaml", "alignment_transforms.json",
    ):
        assert (out / name).is_file(), f"missing {name}"

    written = read_run_result_json(out / "results.json")
    assert written.protocol == "single_depth"
    assert written.alignment.mode == "scale_median"
    assert written.alignment.granularity == "per_frame"
    assert written.backend_versions["depth_io"]["name"] == "imageio"
    assert written.metadata["depth_unit_gt"] == 0.001
    assert written.metadata["depth_unit_pred"] == 1.0
    assert written.metadata["n_frames"] == 1

    # Per-metric metadata carries mode + granularity + mask breakdown (never a bare value).
    scene_metrics = [m for m in written.per_scene_metrics if m.frame_id is None]
    assert {m.name for m in scene_metrics} == {
        "absrel", "sqrel", "rmse", "rmse_log", "silog", "delta_1", "delta_2", "delta_3"
    }
    frame_metrics = [m for m in written.per_scene_metrics if m.frame_id is not None]
    assert frame_metrics and frame_metrics[0].metadata["alignment_mode"] == "scale_median"
    assert frame_metrics[0].metadata["alignment_granularity"] == "per_frame"
    assert frame_metrics[0].n_pixels_valid == 4
    assert frame_metrics[0].valid_fraction == 1.0
    assert "mask_breakdown" in frame_metrics[0].metadata

    transforms = json.loads((out / "alignment_transforms.json").read_text())
    (record,) = transforms["alignment_transforms"]
    assert record["mode"] == "scale_median"
    assert record["scale"] == pytest.approx(0.5)
    assert result.metrics["absrel"] == pytest.approx(0.0)


def _write_sequence(tmp_path: Path) -> tuple[Path, Path]:
    """Two frames with different scale errors: frame0 pred = 2 x gt, frame1 pred = 4 x gt."""
    pred_dir = tmp_path / "pred_seq"
    gt_dir = tmp_path / "gt_seq"
    pred_dir.mkdir()
    gt_dir.mkdir()
    for i, factor in enumerate((2.0, 4.0)):
        np.save(pred_dir / f"{i:04d}.npy", factor * GT)
        np.save(gt_dir / f"{i:04d}.npy", GT)
    return pred_dir, gt_dir


def test_sequence_per_frame_alignment_zeroes_both_frames(tmp_path: Path) -> None:
    pred_dir, gt_dir = _write_sequence(tmp_path)
    run = evaluate_depth(pred_dir, gt_dir, return_run=True)
    assert run.result.metrics["absrel"] == pytest.approx(0.0)
    assert run.n_frames == 2
    assert run.protocol.prediction_modality == "depth_sequence"
    # One alignment record per frame, each with its own recovered scale.
    scales = sorted(r["scale"] for r in run.alignment_records)
    assert scales == pytest.approx([0.25, 0.5])
    # Per-frame results present for both frames, plus the per-scene aggregates.
    frame_ids = {m.frame_id for m in run.result.per_scene_metrics}
    assert frame_ids == {None, "0000", "0001"}
    scene = next(m for m in run.result.per_scene_metrics if m.frame_id is None)
    assert scene.metadata["n_frames"] == 2
    assert scene.metadata["aggregation"] == "per_frame_mean"


def test_sequence_per_sequence_alignment_cannot_fix_mixed_scales(tmp_path: Path) -> None:
    pred_dir, gt_dir = _write_sequence(tmp_path)
    run = evaluate_depth(
        pred_dir, gt_dir, align_granularity="per_sequence", return_run=True
    )
    # A single pooled scale cannot zero two different per-frame scale errors.
    assert run.result.metrics["absrel"] > 0.1
    (record,) = run.alignment_records
    assert record["granularity"] == "per_sequence"
    assert record["frame_id"] is None
    assert record["n_pixels_used"] == 8


def test_sequence_frame_mismatch_fails_explicitly(tmp_path: Path) -> None:
    pred_dir, gt_dir = _write_sequence(tmp_path)
    (gt_dir / "0001.npy").unlink()
    with pytest.raises(SceneEvaluationError, match="matched by filename stem"):
        evaluate_depth(pred_dir, gt_dir)


def test_all_invalid_gt_aborts_with_mask_breakdown(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.png"
    iio.imwrite(gt_path, np.zeros((2, 2), dtype=np.uint16))  # all pixels invalid (0)
    pred_path = tmp_path / "pred.npy"
    np.save(pred_path, GT)
    with pytest.raises(SceneEvaluationError, match="no valid pixels remain"):
        evaluate_depth(pred_path, gt_path, gt_depth_unit=0.001)


def test_integer_gt_without_unit_aborts_explicitly(tmp_path: Path) -> None:
    pred, gt = _write_pair(tmp_path)
    with pytest.raises(SceneEvaluationError, match="depth_unit"):
        evaluate_depth(pred, gt)  # gt is a 16-bit PNG; unit is required


def test_cli_depth_writes_run_directory(tmp_path: Path) -> None:
    pred, gt = _write_pair(tmp_path)
    out = tmp_path / "cli_run"
    result = runner.invoke(
        app,
        [
            "metric", "depth", str(pred), "--gt", str(gt),
            "--gt-depth-unit", "0.001",
            "--align", "scale_median", "--align-granularity", "per_frame",
            "--out", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "results.json").is_file()
    written = read_run_result_json(out / "results.json")
    assert written.metrics["absrel"] == pytest.approx(0.0)
    # The CLI echoes the resolved configuration, not just an exit code.
    assert "scale_median" in result.output
    assert "single_depth" in result.output


def test_cli_rejects_unknown_align_mode(tmp_path: Path) -> None:
    pred, gt = _write_pair(tmp_path)
    result = runner.invoke(
        app, ["metric", "depth", str(pred), "--gt", str(gt), "--align", "icp"]
    )
    assert result.exit_code == 2
    assert "invalid --align" in result.output


def test_cli_failure_prints_reason(tmp_path: Path) -> None:
    pred, gt = _write_pair(tmp_path)
    result = runner.invoke(
        app, ["metric", "depth", str(pred), "--gt", str(gt)]  # missing --gt-depth-unit
    )
    assert result.exit_code == 1
    assert "depth_unit" in result.output
