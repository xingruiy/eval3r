"""Task 011 integration: ScanNet val (no cull) and test (visibility cull) benchmarks.

The test-split visibility cull renders with pyrender and needs a headless GL context;
that test skips cleanly (with a reason) when none is available, like the MATLAB path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval3r import run_benchmark
from eval3r.reports.json import read_run_result_json

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "scannet_tiny" / "dataset_root"
PREDS = Path(__file__).resolve().parents[1] / "fixtures" / "scannet_tiny" / "preds"


def _gl_available() -> tuple[bool, str]:
    try:
        import pyrender

        r = pyrender.OffscreenRenderer(16, 16)
        r.delete()
        return True, ""
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"no headless GL context for pyrender: {exc}"


_GL_OK, _GL_REASON = _gl_available()


def test_val_split_runs_without_culling(tmp_path: Path) -> None:
    run = run_benchmark(
        PREDS, dataset="scannet", split="val",
        protocol="scannet_single_layer_geometry_5cm",
        root=ROOT, out_dir=tmp_path / "run", return_run=True,
    )
    r = run.result
    assert r.n_scenes_evaluated == 1 and not r.failed_scenes
    assert r.fidelity == "eval3r_native"
    # val does NOT cull; culled_fraction is recorded as 0.
    assert r.metrics["culled_fraction"] == 0.0
    # the far, unobserved box in the prediction is NOT removed, so it hurts precision.
    assert r.metrics["precision"] < 0.95
    # protocol's declared open3d point-cloud backend is recorded.
    assert r.backend_versions["pointcloud"]["name"] == "open3d"
    assert "visibility" not in r.backend_versions

    # run directory written with per-scene coverage
    written = read_run_result_json(tmp_path / "run" / "results.json")
    assert written.n_scenes_evaluated == 1


@pytest.mark.skipif(not _GL_OK, reason=_GL_REASON)
def test_test_split_visibility_culls_prediction(tmp_path: Path) -> None:
    run = run_benchmark(
        PREDS, dataset="scannet", split="test",
        protocol="scannet_test_single_layer_geometry_5cm",
        root=ROOT, out_dir=tmp_path / "run", return_run=True,
    )
    r = run.result
    assert r.n_scenes_evaluated == 1 and not r.failed_scenes
    # the far box (~half the prediction vertices) is culled away.
    assert r.metrics["culled_fraction"] > 0.2
    # after culling only the observed slab remains -> high precision.
    assert r.metrics["precision"] > 0.95
    # the visibility backend is recorded.
    assert r.backend_versions["visibility"]["name"] == "render_tsdf"

    cf = next(m for m in r.per_scene_metrics if m.name == "culled_fraction")
    assert cf.metadata["method"] == "render_tsdf_trim"
    assert "pyrender" in cf.metadata["renderer"]
    assert cf.metadata["trajectory_fingerprint"] is not None


@pytest.mark.skipif(not _GL_OK, reason=_GL_REASON)
def test_culling_changes_precision_vs_val(tmp_path: Path) -> None:
    val = run_benchmark(
        PREDS, dataset="scannet", split="val",
        protocol="scannet_single_layer_geometry_5cm",
        root=ROOT, out_dir=tmp_path / "val", return_run=True,
    )
    test = run_benchmark(
        PREDS, dataset="scannet", split="test",
        protocol="scannet_test_single_layer_geometry_5cm",
        root=ROOT, out_dir=tmp_path / "test", return_run=True,
    )
    # Culling the far hallucination lifts precision on the same prediction.
    assert test.result.metrics["precision"] > val.result.metrics["precision"]
