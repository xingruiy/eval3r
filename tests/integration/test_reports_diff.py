"""Task 016 integration tests: reports, debug outputs, and `e3r diff` end-to-end.

Acceptance per `.agent/plan.md` "Reporting and diffing":
``e3r diff runs/method_a runs/method_b``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import eval3r
from eval3r import diff_runs, evaluate_geometry
from eval3r.backends.pointcloud_plyfile import PlyfilePointCloudBackend
from eval3r.cli.main import app
from eval3r.core.errors import RunComparisonError

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "geom"
runner = CliRunner()


def _full_reporting_protocol(tmp_path: Path) -> Path:
    """single_geometry copy with all report formats and debug outputs enabled."""
    src = Path(eval3r.__file__).parent / "protocols" / "builtin" / "single_geometry.yaml"
    proto = yaml.safe_load(src.read_text())
    proto["reporting"]["formats"] = ["json", "csv", "markdown", "latex", "html"]
    proto["reporting"]["save_colored_errors"] = True
    proto["reporting"]["save_distance_histogram"] = True
    path = tmp_path / "single_geometry_full_reporting.yaml"
    path.write_text(yaml.safe_dump(proto, sort_keys=False))
    return path


def _run(tmp_path: Path, out: str, *, threshold: float = 0.05, method: str = "method") -> Path:
    out_dir = tmp_path / out
    evaluate_geometry(
        FIX / "pred.ply", FIX / "gt.ply",
        threshold=threshold, method=method,
        protocol=str(_full_reporting_protocol(tmp_path)),
        out_dir=out_dir,
    )
    return out_dir


def test_run_directory_contains_reports_and_debug_outputs(tmp_path: Path) -> None:
    out = _run(tmp_path, "run")
    for name in ("results.md", "results.tex", "report.html"):
        assert (out / name).is_file(), f"missing {name}"

    # full coverage: no banner in any format
    assert "PARTIAL COVERAGE" not in (out / "results.md").read_text()
    html = (out / "report.html").read_text()
    assert "single_geometry" in html and "sha256:" in html

    # debug outputs: colored PLY point count matches the scored prediction points,
    # histogram is a real PNG, and the manifest records the parameters
    debug = out / "debug"
    manifest = json.loads((debug / "debug_outputs.json").read_text())
    record = manifest["debug_outputs"][0]
    ply = debug / record["error_colored_ply"]
    png = debug / record["histogram_png"]
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    points = PlyfilePointCloudBackend().load_pointcloud(ply)
    assert points.shape[0] == record["n_points_pred"]
    assert record["colormap"] == "turbo"
    assert "colormap_vmax" in record


def test_api_diff_runs_strict_and_loose(tmp_path: Path) -> None:
    dir_a = _run(tmp_path, "run_a", method="method_a")
    dir_b = _run(tmp_path, "run_b", method="method_b")
    diff = diff_runs(dir_a, dir_b)
    assert diff.strict is True
    assert diff.warnings == []
    assert {m.name for m in diff.metrics} >= {"accuracy", "completeness", "fscore"}
    assert all(m.delta == 0.0 for m in diff.metrics)

    # a different threshold changes the protocol hash -> strict refusal, loose ok
    dir_c = _run(tmp_path, "run_c", threshold=0.02, method="method_c")
    with pytest.raises(RunComparisonError, match="protocol hashes"):
        diff_runs(dir_a, dir_c)
    loose = diff_runs(dir_a, dir_c, loose=True)
    assert loose.strict is False
    assert "protocol_hash" in {w.field for w in loose.warnings}


def test_cli_diff_acceptance(tmp_path: Path) -> None:
    dir_a = _run(tmp_path, "method_a", method="method_a")
    dir_b = _run(tmp_path, "method_b", method="method_b")

    # acceptance command: e3r diff runs/method_a runs/method_b
    result = runner.invoke(app, ["diff", str(dir_a), str(dir_b)])
    assert result.exit_code == 0, result.output
    assert "no comparability warnings" in result.output
    assert "aggregate metrics" in result.output
    assert "method_a" in result.output and "method_b" in result.output


def test_cli_diff_refuses_hash_mismatch_and_allows_loose(tmp_path: Path) -> None:
    dir_a = _run(tmp_path, "run_a", method="method_a")
    dir_c = _run(tmp_path, "run_c", threshold=0.02, method="method_c")

    refused = runner.invoke(app, ["diff", str(dir_a), str(dir_c)])
    assert refused.exit_code == 1
    assert "diff refused" in refused.output
    assert "protocol hashes" in refused.output
    assert "--loose" in refused.output

    loose = runner.invoke(app, ["diff", str(dir_a), str(dir_c), "--loose"])
    assert loose.exit_code == 0, loose.output
    assert "NON-STRICT" in loose.output
    assert "protocol_hash" in loose.output


def test_cli_diff_shows_partial_coverage_banner(tmp_path: Path) -> None:
    # fabricate a partial-coverage run next to a full one by editing results.json
    dir_a = _run(tmp_path, "run_a", method="method_a")
    dir_b = _run(tmp_path, "run_b", method="method_b")
    data = json.loads((dir_b / "results.json").read_text())
    data["n_scenes_expected"] = 3
    data["failed_scenes"] = [
        {
            "scene_id": "scene_x",
            "stage": "load",
            "reason": "prediction file missing",
            "recoverable": False,
        }
    ]
    (dir_b / "results.json").write_text(json.dumps(data))

    result = runner.invoke(app, ["diff", str(dir_a), str(dir_b)])
    assert result.exit_code == 0, result.output
    assert "PARTIAL COVERAGE" in result.output
    assert "scene_coverage" in result.output  # comparability warning fired
