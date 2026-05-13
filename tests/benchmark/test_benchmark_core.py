from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from eval3r import PredictionWriter
from eval3r.benchmark import BenchmarkConfig, run_benchmark
from eval3r.benchmark import core as benchmark_core
from eval3r.datasets import ScanNetAdapter
from eval3r.io.geometry import save_mesh_ply
from eval3r.prediction import PredictionLocator
from tests.helpers._fake_scannet import CUBE_FACES, CUBE_VERTS, make_scannet_root


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


def test_abrupt_worker_exit_marked_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds, _ = _make_dataset(tmp_path)
    preds = _make_preds_root(tmp_path)
    original_evaluate_one = benchmark_core._evaluate_one

    def crash_one_scene(*args, **kwargs):
        if args[0] == "s2":
            os._exit(70)
        return original_evaluate_one(*args, **kwargs)

    monkeypatch.setattr(benchmark_core, "_evaluate_one", crash_one_scene)

    cfg = BenchmarkConfig(samples=2048, seed=0, workers=2, thresholds=(0.05,))
    result = run_benchmark(ds, preds, config=cfg, progress=False)

    statuses = {o.scene_id: o.status for o in result.scenes}
    assert statuses == {"s1": "ok", "s2": "failed", "s3": "missing_pred"}
    assert result.coverage["n_evaluated"] == 1
    assert result.coverage["n_failed"] == 1
    failed = next(o for o in result.scenes if o.scene_id == "s2")
    assert failed.error is not None
    assert "Worker process exited abruptly with exit code 70" in failed.error


def test_traj_alignment_defers_gt_pose_loading_until_worker(tmp_path: Path) -> None:
    class _NoMissingPredPoseLoadScanNet(ScanNetAdapter):
        def load_poses(self, scene_id: str):
            if scene_id == "s3":
                raise AssertionError("missing-pred scene should not load GT poses")
            return super().load_poses(scene_id)

    split = make_scannet_root(tmp_path / "ds", ["s1", "s2", "s3"])
    ds = _NoMissingPredPoseLoadScanNet(
        tmp_path / "ds", split=split, validate_on_init=False
    )
    preds = _make_preds_root(tmp_path)
    cfg = BenchmarkConfig(
        samples=2048,
        seed=0,
        workers=1,
        thresholds=(0.05,),
        align="traj_se3",
    )

    result = run_benchmark(ds, preds, config=cfg, progress=False)

    statuses = {o.scene_id: o.status for o in result.scenes}
    assert statuses["s3"] == "missing_pred"


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


class _CropAwareScanNet(ScanNetAdapter):
    """ScanNet adapter that exposes a half-cube crop volume."""

    def load_crop_volume(self, scene_id: str):
        from eval3r.filtering.polygon import PolygonFilter

        # Half-cube along +X: keep only x in [0, 0.5], y/z in [-0.5, 0.5].
        return PolygonFilter(
            polygon_2d=np.array(
                [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
                dtype=np.float64,
            ),
            axis_min=0.0,
            axis_max=0.5,
            orthogonal_axis=0,  # X
        )


def test_crop_to_eval_region_changes_metrics(tmp_path: Path) -> None:
    split = make_scannet_root(tmp_path / "ds", ["s1", "s2"])
    ds = _CropAwareScanNet(tmp_path / "ds", split=split, validate_on_init=False)
    preds = _make_preds_root(tmp_path)

    # With crop enabled (default True in adapter that supports it), only
    # half-cube prediction points survive into the metric. Without crop,
    # all prediction points contribute.
    cfg_off = BenchmarkConfig(
        samples=4096, seed=0, workers=1, thresholds=(0.05,),
        crop_to_eval_region=False,
    )
    cfg_on = BenchmarkConfig(
        samples=4096, seed=0, workers=1, thresholds=(0.05,),
        crop_to_eval_region=True,
    )
    r_off = run_benchmark(ds, preds, config=cfg_off, progress=False)
    r_on = run_benchmark(ds, preds, config=cfg_on, progress=False)

    assert r_off.config["crop_to_eval_region"] is False
    assert r_on.config["crop_to_eval_region"] is True
    # Cropping the pred should noticeably move the chamfer metric (no
    # equality required — just "different by more than numerical noise").
    assert r_on.summary["chamfer"]["mean"] != pytest.approx(
        r_off.summary["chamfer"]["mean"], abs=1e-6
    )


def test_crop_applied_after_alignment(tmp_path: Path) -> None:
    # Regression: the crop volume lives in the GT/laser frame. The
    # prediction may still be in its own frame when it enters the
    # benchmark loop, so the crop must be applied AFTER alignment.
    #
    # Setup: shift the GT cube to x in [9.5, 10.5] while leaving the
    # prediction at the standard cube x in [-0.5, 0.5]. SE(3) alignment
    # should recover a +10 translation. The crop volume only keeps
    # x in [9.5, 10.5], i.e. it's defined in the GT frame.
    split = make_scannet_root(tmp_path / "ds", ["s1"])

    # Overwrite the GT mesh with the +10-translated cube.
    shifted = CUBE_VERTS + np.array([10.0, 0.0, 0.0])
    save_mesh_ply(
        tmp_path / "ds" / "scans" / "s1" / "s1_vh_clean_2.ply",
        shifted,
        CUBE_FACES,
    )

    class _ShiftedCropAdapter(ScanNetAdapter):
        def load_crop_volume(self, scene_id: str):
            from eval3r.filtering.polygon import PolygonFilter

            # Polygon in YZ around the GT-frame cube. Margin keeps
            # alignment numerical noise inside without making the crop
            # cover the un-aligned pred (which sits at x in [-0.5, 0.5]).
            return PolygonFilter(
                polygon_2d=np.array(
                    [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
                    dtype=np.float64,
                ),
                axis_min=9.0,
                axis_max=11.0,
                orthogonal_axis=0,  # X
            )

    ds = _ShiftedCropAdapter(tmp_path / "ds", split=split, validate_on_init=False)

    # Prediction is at the original (un-shifted) cube.
    preds = tmp_path / "preds"
    save_mesh_ply(preds / "s1" / "mesh.ply", CUBE_VERTS, CUBE_FACES)

    cfg = BenchmarkConfig(
        samples=4096,
        seed=0,
        workers=1,
        thresholds=(0.05,),
        align="se3",
        crop_to_eval_region=True,
    )
    result = run_benchmark(ds, preds, config=cfg, progress=False)

    # Scene must succeed: post-alignment pred lives at x in [9.5, 10.5],
    # which is fully inside the crop volume. Under the previous (buggy)
    # implementation the crop ran on the un-aligned pred at x in
    # [-0.5, 0.5] and rejected every point — the scene would have failed.
    assert result.coverage["n_evaluated"] == 1
    statuses = {o.scene_id: o.status for o in result.scenes}
    assert statuses == {"s1": "ok"}
    # Cross-check that running with crop=False on this same setup
    # yields a comparable metric — i.e. the crop kept (essentially)
    # the same sampled points, which only happens when the crop runs
    # on the aligned cloud.
    cfg_no_crop = BenchmarkConfig(
        samples=4096, seed=0, workers=1, thresholds=(0.05,),
        align="se3", crop_to_eval_region=False,
    )
    r_no_crop = run_benchmark(ds, preds, config=cfg_no_crop, progress=False)
    assert result.summary["chamfer"]["mean"] == pytest.approx(
        r_no_crop.summary["chamfer"]["mean"], rel=0.1
    )


def test_crop_disabled_when_adapter_unsupported(tmp_path: Path) -> None:
    # Plain ScanNetAdapter does not implement load_crop_volume; even with
    # crop_to_eval_region=True the run must not error and must match the
    # baseline (no-crop) numbers from cfg_off.
    ds, _ = _make_dataset(tmp_path)
    preds = _make_preds_root(tmp_path)
    cfg_on = BenchmarkConfig(
        samples=4096, seed=0, workers=1, thresholds=(0.05,),
        crop_to_eval_region=True,
    )
    r_on = run_benchmark(ds, preds, config=cfg_on, progress=False)

    cfg_off = BenchmarkConfig(
        samples=4096, seed=0, workers=1, thresholds=(0.05,),
        crop_to_eval_region=False,
    )
    r_off = run_benchmark(ds, preds, config=cfg_off, progress=False)

    # Adapter doesn't ship a crop volume → flag is a no-op.
    assert r_on.summary["chamfer"]["mean"] == pytest.approx(
        r_off.summary["chamfer"]["mean"]
    )


class _PerSceneTauScanNet(ScanNetAdapter):
    """Adapter exposing different τ per scene (mimics T&T scene-tau)."""

    _TAU = {"s1": 0.01, "s2": 0.05, "s3": 0.05}

    def load_thresholds(self, scene_id: str):
        if scene_id not in self._TAU:
            from eval3r.utils.errors import NotSupportedError as _N

            raise _N(f"no τ for {scene_id}")
        return (self._TAU[scene_id],)


def test_use_dataset_thresholds_pools_under_canonical_keys(tmp_path: Path) -> None:
    # When the adapter exposes heterogeneous τ across scenes, the
    # aggregate must collapse the per-τ buckets into canonical
    # f / precision / recall columns so each scene contributes to the
    # same list at its own τ — matching the official T&T protocol.
    split = make_scannet_root(tmp_path / "ds", ["s1", "s2", "s3"])
    ds = _PerSceneTauScanNet(tmp_path / "ds", split=split, validate_on_init=False)
    preds = _make_preds_root(tmp_path)

    cfg = BenchmarkConfig(
        samples=4096, seed=0, workers=1, thresholds=(0.05,),
        use_dataset_thresholds=True,
    )
    result = run_benchmark(ds, preds, config=cfg, progress=False)

    assert result.config["use_dataset_thresholds"] is True
    assert result.config["thresholds_pooled"] is True
    # Canonical keys present; per-τ keys absent.
    assert "f" in result.summary
    assert "precision" in result.summary
    assert "recall" in result.summary
    assert "f@0.05" not in result.summary
    assert "f@0.01" not in result.summary
    # n equals the number of evaluated scenes (s3 is missing pred).
    assert result.summary["f"]["n"] == result.coverage["n_evaluated"] == 2


def test_threshold_multiplier_scales_dataset_thresholds(tmp_path: Path) -> None:
    # With dataset thresholds + multiplier=2.0, every scene's per-scene
    # τ is doubled; we confirm by reading back the fscore dict keys.
    split = make_scannet_root(tmp_path / "ds", ["s1", "s2"])
    ds = _PerSceneTauScanNet(tmp_path / "ds", split=split, validate_on_init=False)
    preds = _make_preds_root(tmp_path)

    cfg = BenchmarkConfig(
        samples=4096, seed=0, workers=1, thresholds=(0.05,),
        use_dataset_thresholds=True,
        threshold_multiplier=2.0,
    )
    result = run_benchmark(ds, preds, config=cfg, progress=False)

    assert result.config["threshold_multiplier"] == 2.0
    by_id = {o.scene_id: o for o in result.scenes if o.status == "ok"}
    # _PerSceneTauScanNet τ: s1=0.01, s2=0.05. After ×2: 0.02, 0.10.
    assert list(by_id["s1"].result.fscore.keys()) == [pytest.approx(0.02)]
    assert list(by_id["s2"].result.fscore.keys()) == [pytest.approx(0.10)]


def test_threshold_multiplier_scales_global_fallback(tmp_path: Path) -> None:
    # Without dataset thresholds, the multiplier scales cfg.thresholds
    # and the aggregator emits f@<scaled> columns.
    ds, _ = _make_dataset(tmp_path)
    preds = _make_preds_root(tmp_path)

    cfg = BenchmarkConfig(
        samples=4096, seed=0, workers=1, thresholds=(0.05,),
        use_dataset_thresholds=False,
        threshold_multiplier=3.0,
    )
    result = run_benchmark(ds, preds, config=cfg, progress=False)

    assert result.config["threshold_multiplier"] == 3.0
    # Effective threshold = 0.05 × 3.0 (matched as a float to avoid
    # float-precision artefacts in the column-name string).
    expected_key = f"f@{0.05 * 3.0}"
    assert expected_key in result.summary
    assert "f@0.05" not in result.summary


def test_threshold_multiplier_invalid_raises(tmp_path: Path) -> None:
    ds, _ = _make_dataset(tmp_path)
    preds = _make_preds_root(tmp_path)
    cfg = BenchmarkConfig(
        samples=2048, seed=0, workers=1, thresholds=(0.05,),
        threshold_multiplier=0.0,
    )
    with pytest.raises(ValueError, match="threshold_multiplier"):
        run_benchmark(ds, preds, config=cfg, progress=False)


def test_no_dataset_thresholds_keeps_per_tau_columns(tmp_path: Path) -> None:
    # With the toggle off the legacy f@<thr> schema is preserved.
    split = make_scannet_root(tmp_path / "ds", ["s1", "s2", "s3"])
    ds = _PerSceneTauScanNet(tmp_path / "ds", split=split, validate_on_init=False)
    preds = _make_preds_root(tmp_path)

    cfg = BenchmarkConfig(
        samples=4096, seed=0, workers=1, thresholds=(0.05,),
        use_dataset_thresholds=False,
    )
    result = run_benchmark(ds, preds, config=cfg, progress=False)

    assert result.config["use_dataset_thresholds"] is False
    assert result.config["thresholds_pooled"] is False
    assert "f@0.05" in result.summary
    assert "f" not in result.summary
