"""Task 018 integration: `e3r align` end-to-end + automatic pipeline overlays.

Drives the real CLI in-process on analytically constructed Sim3/SE3 pairs, checks
that every mandatory artifact is written (overlay PLYs, projection PNG,
``pred_aligned.ply``, ``alignment.json``), and that a protocol-driven ICP alignment
inside an evaluation run lands its overlays in the run directory's ``debug/``
automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from eval3r.api import align_geometries, evaluate_geometry
from eval3r.backends.pointcloud_plyfile import PlyfilePointCloudBackend
from eval3r.cli.main import app
from eval3r.core.errors import AlignmentError
from eval3r.pipeline.runner import GeometryRunOutput
from eval3r.protocols import load_protocol

runner = CliRunner()
PC = PlyfilePointCloudBackend()


def _rotation_z(degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def sim3_pair(tmp_path: Path) -> dict[str, Path]:
    """pred.ply / gt.ply with gt = 0.7 * R(10°) @ pred + t exactly.

    The cloud is anisotropic (a 4 x 2 x 0.5 box) so the optimum is well-determined:
    a near-uniform cube is almost self-similar under rotation, and ICP from a
    rotation-free centroid init can settle into a locally-stuck registration —
    exactly the failure mode the mandatory visualization exists to reveal, but not
    what this recovery test is about.
    """
    rng = np.random.default_rng(1)
    gt = rng.random((1500, 3)) * np.array([4.0, 2.0, 0.5])
    rot = _rotation_z(10.0)
    scale, t = 0.7, np.array([1.0, -0.5, 0.25])
    pred = ((gt - t) / scale) @ rot  # gt == scale * pred @ rot.T + t
    np.testing.assert_allclose(scale * pred @ rot.T + t, gt, atol=1e-12)
    PC.save_pointcloud(pred, tmp_path / "pred.ply")
    PC.save_pointcloud(gt, tmp_path / "gt.ply")
    return {"pred": tmp_path / "pred.ply", "gt": tmp_path / "gt.ply"}


def test_cli_icp_sim3_end_to_end(sim3_pair: dict[str, Path], tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "align", str(sim3_pair["pred"]), "--gt", str(sim3_pair["gt"]),
            "--mode", "sim3", "--max-corr-dist", "auto", "--out", str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    # Every mandatory artifact exists.
    for name in (
        "alignment.json",
        "pred_aligned.ply",
        "alignment_before.ply",
        "alignment_after.ply",
        "alignment_projections.png",
        "alignment_vis.json",
    ):
        assert (out_dir / name).is_file(), name

    record = json.loads((out_dir / "alignment.json").read_text(encoding="utf-8"))
    alignment = record["alignment"]
    assert alignment["scale"] == pytest.approx(0.7, abs=0.01)
    assert alignment["solver"] == "icp"
    # The auto-resolved correspondence distance is recorded, not silent.
    assert alignment["parameters"]["max_corr_dist_auto"] is True
    assert alignment["parameters"]["max_correspondence_distance"] > 0
    assert record["backend_versions"]["registration"]["name"] == "open3d"
    # The resolved config is echoed, including the auto rule.
    assert "auto" in result.output and "scale" in result.output


def test_cli_trajectory_solver_end_to_end(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    rot = _rotation_z(30.0)
    t = np.array([2.0, -1.0, 0.5])

    positions = np.cumsum(rng.normal(scale=0.1, size=(30, 3)), axis=0)
    for name, pos in (("pred_traj.txt", positions), ("gt_traj.txt", positions @ rot.T + t)):
        (tmp_path / name).write_text(
            "\n".join(
                f"{i * 0.1:.3f} {p[0]} {p[1]} {p[2]} 0 0 0 1" for i, p in enumerate(pos)
            )
            + "\n",
            encoding="utf-8",
        )
    cloud = rng.random((300, 3))
    PC.save_pointcloud(cloud, tmp_path / "pred.ply")
    PC.save_pointcloud(cloud @ rot.T + t, tmp_path / "gt.ply")

    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "align", str(tmp_path / "pred.ply"), "--gt", str(tmp_path / "gt.ply"),
            "--solver", "trajectory",
            "--pred-trajectory", str(tmp_path / "pred_traj.txt"),
            "--gt-trajectory", str(tmp_path / "gt_traj.txt"),
            "--associate-max-diff", "0.01",
            "--out", str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    record = json.loads((out_dir / "alignment.json").read_text(encoding="utf-8"))
    alignment = record["alignment"]
    assert alignment["estimate_on"] == "trajectory"
    assert alignment["n_correspondences"] == 30
    np.testing.assert_allclose(np.asarray(alignment["matrix"])[:3, 3], t, atol=1e-9)
    # The propagated prediction matches the GT cloud (f4 PLY precision).
    from plyfile import PlyData

    vertex = PlyData.read(str(out_dir / "pred_aligned.ply"))["vertex"].data
    aligned = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
    np.testing.assert_allclose(aligned, cloud @ rot.T + t, atol=1e-5)


def test_cli_refuses_bad_mode_and_missing_max_corr_dist(
    sim3_pair: dict[str, Path], tmp_path: Path
) -> None:
    bad_mode = runner.invoke(
        app, ["align", str(sim3_pair["pred"]), "--gt", str(sim3_pair["gt"]), "--mode", "icp"]
    )
    assert bad_mode.exit_code == 2

    missing_mcd = runner.invoke(
        app, ["align", str(sim3_pair["pred"]), "--gt", str(sim3_pair["gt"]),
              "--out", str(tmp_path / "o")]
    )
    assert missing_mcd.exit_code == 1
    assert "max_corr_dist" in missing_mcd.output


def test_api_trajectory_solver_requires_all_inputs(
    sim3_pair: dict[str, Path], tmp_path: Path
) -> None:
    with pytest.raises(AlignmentError, match="pred_trajectory, gt_trajectory, associate_max_diff"):
        align_geometries(
            sim3_pair["pred"], sim3_pair["gt"],
            out_dir=tmp_path / "o", solver="trajectory",
        )


def test_pipeline_run_writes_alignment_overlays_to_debug(tmp_path: Path) -> None:
    # A protocol-driven ICP alignment inside a normal evaluation run must land its
    # before/after overlays in the run directory's debug/ automatically.
    rng = np.random.default_rng(3)
    gt = rng.random((800, 3))
    rot = _rotation_z(10.0)
    t = np.array([0.5, 0.2, -0.1])
    pred = (gt - t) @ rot  # gt == pred @ rot.T + t (SE3, no scale)
    PC.save_pointcloud(pred, tmp_path / "pred.ply")
    PC.save_pointcloud(gt, tmp_path / "gt.ply")

    proto = load_protocol("single_geometry").model_copy(deep=True)
    proto.alignment.mode = "se3"
    proto.alignment.solver = "icp"
    proto.alignment.estimate_on = "pointcloud"
    proto.alignment.parameters = {"max_correspondence_distance": 0.2}
    proto_path = tmp_path / "proto.yaml"
    proto_path.write_text(
        yaml.safe_dump(proto.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )

    out_dir = tmp_path / "run"
    run = evaluate_geometry(
        tmp_path / "pred.ply", tmp_path / "gt.ply",
        protocol=str(proto_path), out_dir=out_dir, return_run=True,
    )
    assert isinstance(run, GeometryRunOutput)
    assert not run.result.failed_scenes

    # Aligned metrics: pred and gt coincide after ICP.
    assert run.result.metrics["accuracy"] == pytest.approx(0.0, abs=1e-6)

    debug = out_dir / "debug"
    assert (debug / "pred_alignment_before.ply").is_file()
    assert (debug / "pred_alignment_after.ply").is_file()
    assert (debug / "pred_alignment_projections.png").is_file()
    manifest = json.loads((debug / "alignment_vis.json").read_text(encoding="utf-8"))
    assert manifest["alignment_visualizations"][0]["alignment"]["solver"] == "icp"

    # The registration backend version is recorded with the run.
    assert run.result.backend_versions["registration"]["name"] == "open3d"
    # And the applied transform is in alignment_transforms.json.
    transforms = json.loads(
        (out_dir / "alignment_transforms.json").read_text(encoding="utf-8")
    )
    assert transforms["alignment_transforms"][0]["solver"] == "icp"
