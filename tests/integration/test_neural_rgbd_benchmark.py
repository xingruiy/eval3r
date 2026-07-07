"""Task 020 integration: Neural-RGBD geometry benchmark (culled and source mesh variants).

Mesh-to-mesh geometry only — no renderer/GL needed. The prediction mesh matches the culled
(observed) region, so the culled protocol scores near-perfectly while the source/uncropped
protocol penalizes the unobserved region — proving the variant selection actually changes the
GT the benchmark scores against.
"""

from __future__ import annotations

from pathlib import Path

from eval3r import run_benchmark
from eval3r.reports.json import read_run_result_json

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "neural_rgbd_tiny" / "official"
PREDS = Path(__file__).resolve().parents[1] / "fixtures" / "neural_rgbd_tiny" / "preds"


def test_culled_geometry_benchmark(tmp_path: Path) -> None:
    run = run_benchmark(
        PREDS, dataset="neural_rgbd", split="all",
        protocol="neural_rgbd_geometry_culled",
        root=ROOT, out_dir=tmp_path / "run", return_run=True,
    )
    r = run.result
    assert r.n_scenes_evaluated == 2 and not r.failed_scenes
    assert r.fidelity == "eval3r_native"
    for metric in ("accuracy", "completeness", "chamfer", "precision", "recall", "fscore"):
        assert metric in r.metrics
    # prediction matches the culled/observed region -> near-perfect F-score.
    assert r.metrics["fscore"] > 0.95
    # no eval-time culling; culled_fraction recorded as 0.
    assert r.metrics["culled_fraction"] == 0.0
    assert r.backend_versions["pointcloud"]["name"] == "open3d"

    written = read_run_result_json(tmp_path / "run" / "results.json")
    assert written.n_scenes_evaluated == 2


def test_source_variant_penalizes_unobserved_region(tmp_path: Path) -> None:
    culled = run_benchmark(
        PREDS, dataset="neural_rgbd", split="all",
        protocol="neural_rgbd_geometry_culled",
        root=ROOT, out_dir=tmp_path / "culled", return_run=True,
    )
    source = run_benchmark(
        PREDS, dataset="neural_rgbd", split="all",
        protocol="neural_rgbd_geometry_source",
        root=ROOT, out_dir=tmp_path / "source", return_run=True,
    )
    # Scoring the same prediction against the full uncropped mesh lowers recall/F-score,
    # because the prediction only covers the observed (culled) region.
    assert source.result.metrics["recall"] < culled.result.metrics["recall"]
    assert source.result.metrics["fscore"] < culled.result.metrics["fscore"]
