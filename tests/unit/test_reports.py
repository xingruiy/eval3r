"""Task 016 tests: Markdown/LaTeX/HTML reports and debug outputs.

Partial coverage must be visible in every report format (string assertions per
``.agent/reproducibility.md``), report headers must surface the protocol identity
and the policies in effect, and debug outputs must match the scored points.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from eval3r.backends.pointcloud_plyfile import PlyfilePointCloudBackend
from eval3r.core.schema import ReportingSpec
from eval3r.metrics.geometry import compute_directional_distances
from eval3r.reports import (
    build_report_data,
    render_html,
    render_latex,
    render_markdown,
    write_geometry_debug_outputs,
    write_run_directory,
)

RENDERERS = {
    "markdown": render_markdown,
    "latex": render_latex,
    "html": render_html,
}


# --- report data ----------------------------------------------------------------


def test_report_data_marks_partial_coverage(make_run_result) -> None:
    data = build_report_data(make_run_result())
    assert data.coverage.expected == 2
    assert data.coverage.evaluated == 1
    assert data.coverage.failed == 1
    assert data.coverage.partial
    assert data.banner is not None and "PARTIAL COVERAGE" in data.banner
    assert "skip_and_flag" in data.banner


def test_report_data_full_coverage_has_no_banner(make_run_result) -> None:
    data = build_report_data(
        make_run_result(n_scenes_evaluated=2, n_scenes_expected=2, failed_scenes=[])
    )
    assert not data.coverage.partial
    assert data.banner is None


def test_report_header_surfaces_protocol_and_policies(make_run_result) -> None:
    header = dict(build_report_data(make_run_result()).header)
    assert header["protocol hash"] == "sha256:aaa"
    assert header["fidelity"] == "eval3r_native"
    assert "provenance=laser_scan" in header["ground truth"]
    assert "independence=independent" in header["ground truth"]
    assert header["local evaluation"] == "supported"
    assert "mode=none" in header["alignment"]
    assert header["confidence policy"] == "none"
    assert "n=1000" in header["sampling"]
    assert header["failure policy"] == "skip_and_flag"


def test_report_data_failed_scene_row_carries_reason(make_run_result) -> None:
    data = build_report_data(make_run_result())
    rows = {row[0]: row for row in data.per_scene_rows}
    assert rows["scan001"][1] == "ok"
    assert rows["scan002"][1] == "failed (load)"
    assert "prediction file missing" in rows["scan002"][-1]
    assert data.failures == [
        ("scan002", "load", "prediction file missing: preds/scan002.ply")
    ]


# --- every format shows coverage, failures, and the banner ------------------------


@pytest.mark.parametrize("fmt", sorted(RENDERERS))
def test_partial_coverage_visible_in_every_format(make_run_result, fmt: str) -> None:
    text = RENDERERS[fmt](make_run_result())
    assert "PARTIAL COVERAGE" in text
    assert "1 of 2 scenes" in text
    # underscores are latex-escaped, so match the format's own spelling
    policy = r"skip\_and\_flag" if fmt == "latex" else "skip_and_flag"
    assert policy in text
    assert "scan002" in text
    # the reason string contains no format-special characters, so it must appear
    # verbatim in every format
    assert "prediction file missing" in text


@pytest.mark.parametrize("fmt", sorted(RENDERERS))
def test_full_coverage_has_no_banner_in_any_format(make_run_result, fmt: str) -> None:
    text = RENDERERS[fmt](
        make_run_result(n_scenes_evaluated=2, n_scenes_expected=2, failed_scenes=[])
    )
    assert "PARTIAL COVERAGE" not in text


@pytest.mark.parametrize("fmt", sorted(RENDERERS))
def test_header_and_metrics_in_every_format(make_run_result, fmt: str) -> None:
    text = RENDERERS[fmt](make_run_result())
    assert "sha256:aaa" in text
    assert "accuracy" in text and "0.03" in text
    assert "fscore" in text and "0.61" in text


def test_latex_escapes_special_characters(make_run_result) -> None:
    result = make_run_result(method="method_50%_a&b")
    text = render_latex(result)
    assert r"method\_50\%\_a\&b" in text
    assert "method_50%_a&b" not in text


def test_html_is_escaped_and_self_contained(make_run_result) -> None:
    result = make_run_result(method="<script>alert(1)</script>")
    text = render_html(result)
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text
    assert "<style>" in text  # inline CSS; no external assets


# --- run directory format selection ----------------------------------------------


def test_run_directory_writes_requested_formats(make_run_result, tmp_path) -> None:
    out = write_run_directory(
        make_run_result(), tmp_path / "run", formats=["json", "csv", "markdown", "latex", "html"]
    )
    assert (out / "results.md").is_file()
    assert (out / "results.tex").is_file()
    assert (out / "report.html").is_file()
    assert "PARTIAL COVERAGE" in (out / "results.md").read_text()


def test_run_directory_omits_unrequested_formats(make_run_result, tmp_path) -> None:
    out = write_run_directory(make_run_result(), tmp_path / "run", formats=["json", "csv"])
    assert not (out / "results.md").exists()
    assert not (out / "results.tex").exists()
    assert not (out / "report.html").exists()


# --- debug outputs ----------------------------------------------------------------


class _ExactNN:
    name = "scipy"

    def nearest_distances(self, query: np.ndarray, reference: np.ndarray) -> np.ndarray:
        from scipy.spatial import cKDTree

        return cKDTree(reference).query(query)[0]


def _distances(n: int = 50):
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(n, 3))
    pred = gt + 0.01
    return compute_directional_distances(pred, gt, _ExactNN())


def test_colored_ply_point_count_matches_input(tmp_path) -> None:
    dist = _distances(50)
    reporting = ReportingSpec(save_colored_errors=True, save_distance_histogram=False)
    backend = PlyfilePointCloudBackend()
    records = write_geometry_debug_outputs(
        [("scene0", dist)], tmp_path, reporting=reporting, pointcloud_backend=backend
    )
    ply = tmp_path / "debug" / "scene0_error.ply"
    assert ply.is_file()
    points = backend.load_pointcloud(ply)
    assert points.shape == (50, 3)
    assert records[0]["colormap"] == "turbo"
    assert records[0]["colormap_vmax"] == pytest.approx(float(dist.pred_to_gt.max()))


def test_histogram_written_as_png_with_manifest(tmp_path) -> None:
    dist = _distances(30)
    reporting = ReportingSpec(save_colored_errors=False, save_distance_histogram=True)
    write_geometry_debug_outputs(
        [("scene0", dist)],
        tmp_path,
        reporting=reporting,
        pointcloud_backend=PlyfilePointCloudBackend(),
    )
    png = tmp_path / "debug" / "scene0_histogram.png"
    assert png.is_file()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    manifest = json.loads((tmp_path / "debug" / "debug_outputs.json").read_text())
    record = manifest["debug_outputs"][0]
    assert record["scene_id"] == "scene0"
    assert record["histogram_png"] == "scene0_histogram.png"
    assert record["n_points_pred"] == 30
    assert not (tmp_path / "debug" / "scene0_error.ply").exists()


def test_debug_outputs_noop_when_not_requested(tmp_path) -> None:
    dist = _distances(10)
    reporting = ReportingSpec()  # both debug outputs default off
    records = write_geometry_debug_outputs(
        [("scene0", dist)],
        tmp_path,
        reporting=reporting,
        pointcloud_backend=PlyfilePointCloudBackend(),
    )
    assert records == []
    assert not (tmp_path / "debug").exists()
