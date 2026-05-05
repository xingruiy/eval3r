from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval3r import PredictionWriter
from eval3r.benchmark import BenchmarkConfig, run_benchmark
from eval3r.datasets import ScanNetAdapter
from eval3r.io.geometry import save_mesh_ply
from eval3r.prediction import PredictionLocator
from tests._fake_scannet import CUBE_FACES, CUBE_VERTS, make_scannet_root


def _make_dataset(tmp_path: Path) -> tuple[ScanNetAdapter, Path]:
    split = make_scannet_root(tmp_path / "ds", ["s1", "s2", "s3"])
    return ScanNetAdapter(tmp_path / "ds", split=split, validate_on_init=False), split


def _make_preds_root(tmp_path: Path) -> Path:
    preds = tmp_path / "preds"
    rng = np.random.default_rng(0)
    # s1: manifest format, perfect
    with PredictionWriter(
        preds / "s1", scene_id="s1", dataset="d", method="m",
        unit="m", coordinate_system="opengl", pose_convention="T_wc",
    ) as w:
        w.save_mesh(CUBE_VERTS, CUBE_FACES)
    # s2: raw mesh.ply, slightly perturbed
    save_mesh_ply(
        preds / "s2" / "mesh.ply",
        CUBE_VERTS + rng.normal(scale=0.005, size=CUBE_VERTS.shape),
        CUBE_FACES,
    )
    # s3: missing on purpose
    return preds


def test_run_benchmark_workers_1(tmp_path: Path) -> None:
    ds, _ = _make_dataset(tmp_path)
    preds = _make_preds_root(tmp_path)
    cfg = BenchmarkConfig(samples=4096, seed=0, workers=1, thresholds=(0.05,))
    result = run_benchmark(ds, preds, config=cfg, progress=False)
    assert result.coverage["n_total"] == 3
    assert result.coverage["n_evaluated"] == 2
    assert result.coverage["n_missing_pred"] == 1
    statuses = {o.scene_id: o.status for o in result.scenes}
    assert statuses == {"s1": "ok", "s2": "ok", "s3": "missing_pred"}
    assert result.summary["chamfer"]["n"] == 2
    # summary_all includes the missing scene with the configured defaults.
    assert result.summary_all["chamfer"]["n"] == 3
    assert result.summary_all["f@0.05"]["n"] == 3
    # Defaults: distance=1.0, fscore=0.0.
    expected_chamfer_all = (
        result.summary["chamfer"]["mean"] * 2 + cfg.missing_distance_default
    ) / 3
    assert result.summary_all["chamfer"]["mean"] == pytest.approx(expected_chamfer_all)
    expected_f_all = (result.summary["f@0.05"]["mean"] * 2 + cfg.missing_fscore_default) / 3
    assert result.summary_all["f@0.05"]["mean"] == pytest.approx(expected_f_all)


def test_summary_all_respects_overrides(tmp_path: Path) -> None:
    ds, _ = _make_dataset(tmp_path)
    preds = _make_preds_root(tmp_path)
    cfg = BenchmarkConfig(
        samples=4096, seed=0, workers=1, thresholds=(0.05,),
        missing_distance_default=2.5, missing_fscore_default=0.1,
    )
    result = run_benchmark(ds, preds, config=cfg, progress=False)
    expected = (result.summary["chamfer"]["mean"] * 2 + 2.5) / 3
    assert result.summary_all["chamfer"]["mean"] == pytest.approx(expected)
    expected_f = (result.summary["f@0.05"]["mean"] * 2 + 0.1) / 3
    assert result.summary_all["f@0.05"]["mean"] == pytest.approx(expected_f)


def test_summary_all_when_zero_succeed(tmp_path: Path) -> None:
    ds, _ = _make_dataset(tmp_path)
    # Empty preds root: every scene is missing.
    cfg = BenchmarkConfig(samples=4096, seed=0, workers=1, thresholds=(0.05,))
    result = run_benchmark(ds, tmp_path / "no_preds", config=cfg, progress=False)
    assert result.coverage["n_evaluated"] == 0
    # summary is empty stats; summary_all reflects pure defaults.
    assert result.summary["chamfer"]["n"] == 0
    assert result.summary_all["chamfer"]["n"] == 3
    assert result.summary_all["chamfer"]["mean"] == pytest.approx(cfg.missing_distance_default)
    assert result.summary_all["f@0.05"]["mean"] == pytest.approx(cfg.missing_fscore_default)


def test_run_benchmark_workers_2_matches_workers_1(tmp_path: Path) -> None:
    ds, _ = _make_dataset(tmp_path)
    preds = _make_preds_root(tmp_path)
    cfg1 = BenchmarkConfig(samples=4096, seed=0, workers=1, thresholds=(0.05,))
    cfg2 = BenchmarkConfig(samples=4096, seed=0, workers=2, thresholds=(0.05,))
    r1 = run_benchmark(ds, preds, config=cfg1, progress=False)
    r2 = run_benchmark(ds, preds, config=cfg2, progress=False)
    assert r1.coverage == r2.coverage
    assert r1.summary["chamfer"]["mean"] == pytest.approx(r2.summary["chamfer"]["mean"])
    assert r1.summary["chamfer"]["std"] == pytest.approx(r2.summary["chamfer"]["std"])


def test_corrupted_pred_marked_failed(tmp_path: Path) -> None:
    ds, _ = _make_dataset(tmp_path)
    preds = _make_preds_root(tmp_path)
    # Corrupt s2's mesh.
    (preds / "s2" / "mesh.ply").write_bytes(b"not a ply")
    cfg = BenchmarkConfig(samples=2048, seed=0, workers=1)
    result = run_benchmark(ds, preds, config=cfg, progress=False)
    statuses = {o.scene_id: o.status for o in result.scenes}
    assert statuses["s2"] == "failed"
    # The other scenes still complete.
    assert statuses["s1"] == "ok"


def test_fail_on_missing_raises(tmp_path: Path) -> None:
    ds, _ = _make_dataset(tmp_path)
    preds = _make_preds_root(tmp_path)
    cfg = BenchmarkConfig(samples=2048, seed=0, workers=1, fail_on_missing=True)
    with pytest.raises(Exception):
        run_benchmark(ds, preds, config=cfg, progress=False)


def test_extra_geometry_pattern(tmp_path: Path) -> None:
    ds, _ = _make_dataset(tmp_path)
    preds = tmp_path / "preds"
    save_mesh_ply(preds / "s1" / "out" / "final.ply", CUBE_VERTS, CUBE_FACES)
    save_mesh_ply(preds / "s2" / "out" / "final.ply", CUBE_VERTS, CUBE_FACES)
    save_mesh_ply(preds / "s3" / "out" / "final.ply", CUBE_VERTS, CUBE_FACES)
    locator = PredictionLocator(preds_root=preds, extra_patterns=("out/final.ply",))
    result = run_benchmark(
        ds, preds, locator=locator,
        config=BenchmarkConfig(samples=2048, seed=0, workers=1),
        progress=False,
    )
    assert result.coverage["n_evaluated"] == 3
