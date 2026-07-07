"""Task 015 integration tests: the single-trajectory pose path end-to-end.

Covers the Python API and the ``e3r metric pose`` CLI on tiny synthetic TUM
trajectories with analytic expectations, and checks the run directory records the
association counts, alignment mode, and estimated Sim3 scale
(``.agent/tasks/015``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from eval3r import evaluate_pose
from eval3r.cli.main import app
from eval3r.core.errors import AlignmentError, SceneEvaluationError
from eval3r.reports.json import read_run_result_json

runner = CliRunner()


def _positions(n: int = 20) -> np.ndarray:
    t = np.arange(n, dtype=float) * 0.1
    return np.stack([np.cos(t), np.sin(t), 0.1 * t], axis=1)


def _write_tum(path: Path, positions: np.ndarray) -> Path:
    ts = np.arange(len(positions), dtype=float) * 0.1
    lines = [
        f"{ts[i]:.6f} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} 0 0 0 1"
        for i, p in enumerate(positions)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_pair(tmp_path: Path, scale: float = 2.0) -> tuple[Path, Path]:
    """GT helix; prediction = scale x gt + constant offset (a known Sim3)."""
    pos = _positions()
    gt = _write_tum(tmp_path / "gt_tum.txt", pos)
    pred = _write_tum(tmp_path / "pred_tum.txt", scale * pos + np.array([1.0, 2.0, 3.0]))
    return pred, gt


def test_sim3_default_recovers_scale_and_zeroes_errors(tmp_path: Path) -> None:
    pred, gt = _write_pair(tmp_path)
    result = evaluate_pose(pred, gt, out_dir=tmp_path / "run")
    m = result.metrics
    assert m["ate"] == pytest.approx(0.0, abs=1e-9)
    assert m["rpe_translation"] == pytest.approx(0.0, abs=1e-9)
    assert m["rpe_rotation"] == pytest.approx(0.0, abs=1e-9)
    assert m["alignment_scale_error"] == pytest.approx(math.log(2.0), abs=1e-9)
    assert result.n_scenes_evaluated == 1
    assert not result.failed_scenes


def test_align_none_override_exposes_offset_and_changes_hash(tmp_path: Path) -> None:
    pos = _positions()
    gt = _write_tum(tmp_path / "gt.txt", pos)
    pred = _write_tum(tmp_path / "pred.txt", pos + np.array([3.0, 0.0, 4.0]))
    run = evaluate_pose(pred, gt, align="none", return_run=True)
    m = run.result.metrics
    assert m["ate"] == pytest.approx(5.0, abs=1e-9)  # |(3,0,4)| = 5, unaligned
    assert m["rpe_translation"] == pytest.approx(0.0, abs=1e-12)
    assert m["alignment_scale_error"] == 0.0
    assert run.overrides["align"] == "none"
    # Alignment adaptation is recorded outside protocol identity.
    from eval3r.protocols import load_protocol, protocol_hash

    assert run.protocol_hash == protocol_hash(load_protocol("single_pose"))
    assert run.result.adaptation is not None
    assert run.result.adaptation.alignment == "none"


def test_se3_shorthand_normalizes_and_keeps_scale_one(tmp_path: Path) -> None:
    pred, gt = _write_pair(tmp_path)  # pred = 2 x gt + offset
    run = evaluate_pose(pred, gt, align="se3", return_run=True)
    assert run.protocol.alignment.mode == "trajectory_se3"
    assert run.overrides["align"] == "trajectory_se3"
    (record,) = run.alignment_records
    assert record["scale"] == 1.0
    assert run.result.metrics["ate"] > 0.1  # SE3 cannot correct the 2x scale
    assert run.result.metrics["alignment_scale_error"] == 0.0


def test_run_directory_records_association_and_alignment(tmp_path: Path) -> None:
    pred, gt = _write_pair(tmp_path)
    out = tmp_path / "run"
    evaluate_pose(pred, gt, out_dir=out)

    for name in (
        "results.json", "results.csv", "per_scene.csv", "failures.json",
        "environment.json", "backend_versions.json", "logs.txt",
        "protocol.yaml", "config.yaml", "alignment_transforms.json",
    ):
        assert (out / name).is_file(), f"missing {name}"

    written = read_run_result_json(out / "results.json")
    assert written.protocol == "single_pose"
    assert written.alignment.mode == "trajectory_sim3"
    assert written.alignment.solver == "evo"
    assert written.backend_versions["trajectory"]["name"] == "evo"
    assert written.backend_versions["trajectory"]["library"] == "evo"
    assert written.metadata["n_associated"] == 20
    assert written.metadata["n_dropped_pred"] == 0
    assert written.metadata["n_dropped_gt"] == 0
    assert written.metadata["association"]["associate_max_diff"] == 0.01

    # Per-metric metadata carries mode + scale + association (never a bare value).
    by_name = {m.name: m for m in written.per_scene_metrics}
    assert set(by_name) == {"ate", "rpe_translation", "rpe_rotation", "alignment_scale_error"}
    ate = by_name["ate"]
    assert ate.statistic == "rmse" and ate.unit == "m"
    assert ate.n_points_pred == 20 and ate.n_points_gt == 20
    assert ate.metadata["alignment_mode"] == "trajectory_sim3"
    assert ate.metadata["alignment_scale"] == pytest.approx(0.5, abs=1e-9)
    rpe_rot = by_name["rpe_rotation"]
    assert rpe_rot.unit == "deg"
    assert rpe_rot.metadata["delta"] == 1.0
    assert rpe_rot.metadata["delta_unit"] == "frames"
    assert rpe_rot.metadata["all_pairs"] is False

    transforms = json.loads((out / "alignment_transforms.json").read_text())
    (record,) = transforms["alignment_transforms"]
    assert record["mode"] == "trajectory_sim3"
    assert record["scale"] == pytest.approx(0.5, abs=1e-9)
    assert record["scale_error"] == pytest.approx(math.log(2.0), abs=1e-9)
    assert record["n_poses_used"] == 20
    assert len(record["rotation"]) == 3 and len(record["translation"]) == 3


def test_missing_prediction_file_aborts_explicitly(tmp_path: Path) -> None:
    gt = _write_tum(tmp_path / "gt.txt", _positions(5))
    with pytest.raises(SceneEvaluationError, match="does not exist"):
        evaluate_pose(tmp_path / "nope.txt", gt)


def test_no_timestamp_overlap_aborts_with_reason(tmp_path: Path) -> None:
    pos = _positions(5)
    gt = _write_tum(tmp_path / "gt.txt", pos)
    pred = tmp_path / "pred.txt"
    lines = [
        f"{100.0 + i * 0.1:.6f} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} 0 0 0 1"
        for i, p in enumerate(pos)
    ]
    pred.write_text("\n".join(lines) + "\n")
    with pytest.raises(SceneEvaluationError, match="associate_max_diff"):
        evaluate_pose(pred, gt)


def test_pinned_alignment_protocol_refuses_override(tmp_path: Path) -> None:
    # A protocol that pins its alignment (allow_override: false) must refuse
    # --align (the Sim3-on-metric-scale guard).
    import yaml

    from eval3r.protocols import load_protocol_text

    proto_yaml = (
        Path(__file__).resolve().parents[2]
        / "eval3r" / "protocols" / "builtin" / "single_pose.yaml"
    )
    data = yaml.safe_load(proto_yaml.read_text())
    data["alignment"]["mode"] = "trajectory_se3"
    data["alignment"]["allow_override"] = False
    pinned = tmp_path / "pinned_pose.yaml"
    pinned.write_text(yaml.safe_dump(data))
    load_protocol_text(pinned.read_text(), source=str(pinned))  # sanity: valid

    pred, gt = _write_pair(tmp_path)
    with pytest.raises(AlignmentError, match="allow_override"):
        evaluate_pose(pred, gt, align="sim3", protocol=str(pinned))


def test_cli_pose_writes_run_directory(tmp_path: Path) -> None:
    pred, gt = _write_pair(tmp_path)
    out = tmp_path / "cli_run"
    result = runner.invoke(
        app,
        [
            "metric", "pose", str(pred), "--gt", str(gt),
            "--backend", "evo", "--align", "sim3",
            "--out", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "results.json").is_file()
    written = read_run_result_json(out / "results.json")
    assert written.metrics["ate"] == pytest.approx(0.0, abs=1e-9)
    assert written.metrics["alignment_scale_error"] == pytest.approx(math.log(2.0), abs=1e-9)
    # The CLI echoes the resolved configuration, not just an exit code.
    assert "single_pose" in result.output
    assert "trajectory_sim3" in result.output


def test_cli_rejects_unknown_align_mode(tmp_path: Path) -> None:
    pred, gt = _write_pair(tmp_path)
    result = runner.invoke(
        app, ["metric", "pose", str(pred), "--gt", str(gt), "--align", "icp"]
    )
    assert result.exit_code == 2
    assert "invalid --align" in result.output


def test_cli_failure_prints_reason(tmp_path: Path) -> None:
    pred, _ = _write_pair(tmp_path)
    result = runner.invoke(
        app, ["metric", "pose", str(pred), "--gt", str(tmp_path / "missing.txt")]
    )
    assert result.exit_code == 1
    assert "does not exist" in result.output
