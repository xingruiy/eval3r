"""Tests for ScanNetBenchmark (and shared BaseBenchmark infrastructure)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from eval3r import PredictionWriter
from eval3r.benchmark.scannet import ScanNetBenchmark, ScanNetBenchmarkConfig
from eval3r.io.geometry import save_mesh_ply
from eval3r.manifest import PredictionLocator
from tests.helpers._fake_scannet import CUBE_FACES, CUBE_VERTS, make_scannet_root


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


def _make_benchmark(tmp_path: Path, *, workers: int = 1, **kw):
    split = make_scannet_root(tmp_path / "ds", ["s1", "s2", "s3"])
    cfg = ScanNetBenchmarkConfig(
        samples=4096, seed=0, workers=workers,
        metrics=["chamfer", "accuracy", "completeness", "fscore@0.05"],
        **kw,
    )
    preds = _make_preds_root(tmp_path)
    bm = ScanNetBenchmark(gt_root=tmp_path / "ds", pred_root=preds, cfg=cfg)
    return bm, split, preds


def test_run_workers_1(tmp_path: Path) -> None:
    bm, split, _ = _make_benchmark(tmp_path, workers=1)
    result = bm.run(split=str(split))
    assert result.coverage["n_total"] == 3
    assert result.coverage["n_evaluated"] == 2
    assert result.coverage["n_missing_pred"] == 1
    statuses = {o.scene_id: o.status for o in result.scenes}
    assert statuses == {"s1": "ok", "s2": "ok", "s3": "missing_pred"}
    assert result.summary["chamfer"]["n"] == 2
    assert result.summary_all["chamfer"]["n"] == 3


def test_summary_all_respects_overrides(tmp_path: Path) -> None:
    split = make_scannet_root(tmp_path / "ds", ["s1", "s2", "s3"])
    cfg = ScanNetBenchmarkConfig(
        samples=4096, seed=0, workers=1,
        metrics=["chamfer", "accuracy", "completeness", "fscore@0.05"],
        missing_distance_default=2.5,
        missing_fscore_default=0.1,
    )
    preds = _make_preds_root(tmp_path)
    result = ScanNetBenchmark(gt_root=tmp_path / "ds", pred_root=preds, cfg=cfg).run(
        split=str(split)
    )
    expected_chamfer = (result.summary["chamfer"]["mean"] * 2 + 2.5) / 3
    assert result.summary_all["chamfer"]["mean"] == pytest.approx(expected_chamfer)
    expected_f = (result.summary["fscore@0.05_f"]["mean"] * 2 + 0.1) / 3
    assert result.summary_all["fscore@0.05_f"]["mean"] == pytest.approx(expected_f)


def test_summary_all_when_zero_succeed(tmp_path: Path) -> None:
    split = make_scannet_root(tmp_path / "ds", ["s1", "s2", "s3"])
    cfg = ScanNetBenchmarkConfig(samples=4096, seed=0, workers=1, metrics=["chamfer", "accuracy", "completeness", "fscore@0.05"])
    result = ScanNetBenchmark(
        gt_root=tmp_path / "ds",
        pred_root=tmp_path / "no_preds",
        cfg=cfg,
    ).run(split=str(split))
    assert result.coverage["n_evaluated"] == 0
    assert result.summary_all["chamfer"]["mean"] == pytest.approx(cfg.missing_distance_default)
    assert result.summary_all["fscore@0.05_f"]["mean"] == pytest.approx(cfg.missing_fscore_default)


def test_workers_2_matches_workers_1(tmp_path: Path) -> None:
    split = make_scannet_root(tmp_path / "ds", ["s1", "s2", "s3"])
    preds = _make_preds_root(tmp_path)

    def _run(workers):
        cfg = ScanNetBenchmarkConfig(
            samples=4096, seed=0, workers=workers, metrics=["chamfer", "accuracy", "completeness", "fscore@0.05"]
        )
        return ScanNetBenchmark(gt_root=tmp_path / "ds", pred_root=preds, cfg=cfg).run(
            split=str(split)
        )

    r1 = _run(1)
    r2 = _run(2)
    assert r1.coverage == r2.coverage
    assert r1.summary["chamfer"]["mean"] == pytest.approx(r2.summary["chamfer"]["mean"])


def test_fail_on_missing_raises(tmp_path: Path) -> None:
    split = make_scannet_root(tmp_path / "ds", ["s1", "s2", "s3"])
    preds = _make_preds_root(tmp_path)
    cfg = ScanNetBenchmarkConfig(
        samples=2048, seed=0, workers=1, fail_on_missing=True, metrics=["chamfer"]
    )
    with pytest.raises(Exception):
        ScanNetBenchmark(gt_root=tmp_path / "ds", pred_root=preds, cfg=cfg).run(
            split=str(split)
        )


def test_corrupted_pred_marked_failed(tmp_path: Path) -> None:
    split = make_scannet_root(tmp_path / "ds", ["s1", "s2", "s3"])
    preds = _make_preds_root(tmp_path)
    (preds / "s2" / "mesh.ply").write_bytes(b"not a ply")
    cfg = ScanNetBenchmarkConfig(samples=2048, seed=0, workers=1, metrics=["chamfer"])
    result = ScanNetBenchmark(gt_root=tmp_path / "ds", pred_root=preds, cfg=cfg).run(
        split=str(split)
    )
    statuses = {o.scene_id: o.status for o in result.scenes}
    assert statuses["s2"] == "failed"
    assert statuses["s1"] == "ok"


def test_extra_geometry_pattern(tmp_path: Path) -> None:
    split = make_scannet_root(tmp_path / "ds", ["s1", "s2", "s3"])
    preds = tmp_path / "preds"
    for sid in ("s1", "s2", "s3"):
        save_mesh_ply(preds / sid / "out" / "final.ply", CUBE_VERTS, CUBE_FACES)
    locator = PredictionLocator(preds_root=preds, extra_patterns=("out/final.ply",))
    cfg = ScanNetBenchmarkConfig(samples=2048, seed=0, workers=1, metrics=["chamfer"])
    result = ScanNetBenchmark(gt_root=tmp_path / "ds", pred_root=preds, cfg=cfg).run(
        split=str(split), locator=locator
    )
    assert result.coverage["n_evaluated"] == 3


def test_config_stored_in_result(tmp_path: Path) -> None:
    split = make_scannet_root(tmp_path / "ds", ["s1"])
    preds = _make_preds_root(tmp_path)
    cfg = ScanNetBenchmarkConfig(
        samples=2048, seed=99, workers=1, metrics=["chamfer"],
        mask_dir="/tmp/masks",
    )
    result = ScanNetBenchmark(gt_root=tmp_path / "ds", pred_root=preds, cfg=cfg).run(
        split=str(split)
    )
    assert result.config["samples"] == 2048
    assert result.config["seed"] == 99
    assert result.config["mask_dir"] == "/tmp/masks"
