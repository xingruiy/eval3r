"""Shared base classes and utilities for all dataset benchmarks."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal

import numpy as np

from eval3r.io.geometry import (
    MeshData,
    PointCloudData,
    load_mesh,
    load_point_cloud,
)
from eval3r.manifest.discovery import PredictionLocator, ResolvedPrediction
from eval3r.pipeline import PipelineResult
from eval3r.utils.errors import MissingArtifactError
from eval3r.utils.logging import get_logger
from eval3r.utils.typing import PathLike

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# BenchmarkConfig — shared base
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Shared configuration for all dataset benchmarks."""

    sampler: str = "area"
    aligner: str = "none"
    metrics: list[str] = field(
        default_factory=lambda: ["chamfer", "fscore@0.05"]
    )
    samples: int = 200_000
    seed: int = 42
    workers: int = field(
        default_factory=lambda: min(8, os.cpu_count() or 1)
    )
    fail_on_missing: bool = False
    missing_distance_default: float = 1.0
    missing_fscore_default: float = 0.0
    verbose: bool = False
    debug_plot: bool = False
    # Trajectory alignment — only used when aligner starts with "traj_"
    pred_pose_dir: str | None = None
    pred_pose_file: str = "{scene_id}.txt"
    pred_pose_convention: str = "unspecified"


# ---------------------------------------------------------------------------
# SceneOutcome / BenchmarkResult
# ---------------------------------------------------------------------------

SceneStatus = Literal["ok", "missing_pred", "missing_gt", "failed"]


@dataclass
class SceneOutcome:
    scene_id: str
    status: SceneStatus
    result: PipelineResult | None = None
    error: str | None = None
    pred_path: Path | None = None
    gt_path: Path | None = None


@dataclass
class BenchmarkResult:
    dataset: str
    split: str
    scenes: list[SceneOutcome]
    summary: dict[str, dict[str, float]]
    """Mean / median / std / n over **successful** scenes only."""
    summary_all: dict[str, dict[str, float]]
    """Mean / median / std / n over **all** scenes; missing get defaults."""
    coverage: dict[str, int]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "scenes": [
                {
                    "scene_id": o.scene_id,
                    "status": o.status,
                    "pred_path": str(o.pred_path) if o.pred_path else None,
                    "gt_path": str(o.gt_path) if o.gt_path else None,
                    "error": o.error,
                    "result": o.result.to_dict() if o.result else None,
                }
                for o in self.scenes
            ],
            "summary": self.summary,
            "summary_all": self.summary_all,
            "coverage": self.coverage,
            "config": self.config,
        }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "median": float(np.median(arr)) if len(arr) else float("nan"),
        "std": float(arr.std()) if len(arr) else float("nan"),
        "n": int(len(arr)),
    }


def collect_values(outcomes: list[SceneOutcome]) -> dict[str, list[float]]:
    """Return per-metric value lists from ``status == "ok"`` outcomes."""
    buckets: dict[str, list[float]] = {}
    for o in outcomes:
        if o.status != "ok" or o.result is None:
            continue
        for k, v in o.result.values.items():
            if isinstance(v, tuple) and len(v) == 3:
                buckets.setdefault(f"{k}_f", []).append(float(v[0]))
                buckets.setdefault(f"{k}_precision", []).append(float(v[1]))
                buckets.setdefault(f"{k}_recall", []).append(float(v[2]))
            else:
                buckets.setdefault(k, []).append(float(v))
    return buckets


def aggregate(outcomes: list[SceneOutcome]) -> dict[str, dict[str, float]]:
    """Mean / median / std / n over **successful** outcomes only."""
    return {name: _stats(vs) for name, vs in collect_values(outcomes).items()}


def _metric_cfg_to_keys(metric_names: list[str]) -> list[tuple[str, bool]]:
    """Expand metric config strings to (summary_key, is_distance) pairs."""
    _DISTANCE = {"chamfer", "accuracy", "completeness"}
    out: list[tuple[str, bool]] = []
    for m in metric_names:
        if m in _DISTANCE:
            out.append((m, True))
        elif m.startswith("fscore@"):
            out.append((f"{m}_f", False))
            out.append((f"{m}_precision", False))
            out.append((f"{m}_recall", False))
    return out


def aggregate_all(
    outcomes: list[SceneOutcome],
    *,
    n_total: int,
    distance_default: float = 1.0,
    fscore_default: float = 0.0,
    seed_metric_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Mean / median / std / n over all scenes; missing get a default.

    ``seed_metric_names`` pre-seeds keys from the configured metrics list so
    that summary_all is populated even when zero scenes succeeded.
    """
    _DISTANCE_KEYS = frozenset({"chamfer", "accuracy", "completeness"})
    values = collect_values(outcomes)

    # Pre-seed with configured metrics (ensures keys exist even with 0 successes)
    if seed_metric_names:
        for key, is_dist in _metric_cfg_to_keys(seed_metric_names):
            values.setdefault(key, [])

    padded: dict[str, list[float]] = {}
    for name, vs in values.items():
        is_distance = name in _DISTANCE_KEYS
        default = distance_default if is_distance else fscore_default
        n_missing = max(0, n_total - len(vs))
        padded[name] = list(vs) + [float(default)] * n_missing
    return {name: _stats(vs) for name, vs in padded.items()}


# ---------------------------------------------------------------------------
# Pred-loading shared helpers
# ---------------------------------------------------------------------------


def _pred_descriptor(rp: ResolvedPrediction | None) -> dict[str, Any] | None:
    if rp is None:
        return None
    return {"kind": rp["kind"], "path": str(rp["path"])}


def _load_pred_geometry(
    pred_desc: dict[str, Any],
    scene_id: str,
) -> MeshData | PointCloudData:
    kind = pred_desc["kind"]
    path = Path(pred_desc["path"])
    if kind == "manifest":
        from eval3r.manifest.reader import PredictionReader

        reader = PredictionReader(path, verify_hashes=False)
        try:
            return reader.mesh
        except MissingArtifactError:
            return reader.points
    if kind == "mesh_file":
        return load_mesh(path)
    return load_point_cloud(path)


def _resolve_mask_pattern(mask_dir: str, pattern: str, scene_id: str) -> Path:
    rel = Path(pattern.format(scene_id=scene_id))
    if rel.is_absolute():
        raise ValueError("Mask path patterns must be relative to mask_dir.")
    return Path(mask_dir) / rel


# ---------------------------------------------------------------------------
# Work directory
# ---------------------------------------------------------------------------


def _make_work_dir(dataset_name: str) -> Path:
    ts = str(time.time()).encode()
    h = hashlib.sha1(ts).hexdigest()[:8]
    work_dir = Path(".eval3r") / "benchmarks" / f"{dataset_name}-{h}"
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

BenchmarkJob = tuple[str, dict[str, Any] | None]


def _worker_process(
    result_queue: mp.Queue,
    job: BenchmarkJob,
    fn: Callable,
) -> None:
    result_queue.put(fn(job))


def _abrupt_failure(job: BenchmarkJob, exitcode: int | None) -> SceneOutcome:
    return SceneOutcome(
        scene_id=job[0],
        status="failed",
        error=f"Worker process exited abruptly with exit code {exitcode}.",
        pred_path=Path(job[1]["path"]) if job[1] else None,
    )


def _run_jobs_parallel(
    jobs: list[BenchmarkJob],
    fn: Callable[[BenchmarkJob], SceneOutcome],
    *,
    workers: int,
) -> list[SceneOutcome]:
    ctx = mp.get_context()
    active: list[tuple[mp.Process, mp.Queue, BenchmarkJob]] = []
    outcomes: list[SceneOutcome] = []
    next_job = 0
    max_workers = max(1, workers)

    def _start_more() -> None:
        nonlocal next_job
        while next_job < len(jobs) and len(active) < max_workers:
            job = jobs[next_job]
            next_job += 1
            q: mp.Queue = ctx.Queue(maxsize=1)
            proc = ctx.Process(target=_worker_process, args=(q, job, fn))
            proc.start()
            active.append((proc, q, job))

    def _finish(proc: mp.Process, q: mp.Queue, job: BenchmarkJob, outcome: SceneOutcome) -> None:
        proc.join()
        q.close()
        q.join_thread()
        active.remove((proc, q, job))
        outcomes.append(outcome)
        _log.info("benchmark: %s -> %s", outcome.scene_id, outcome.status)
        if outcome.status == "failed" and outcome.error:
            for line in outcome.error.rstrip().split("\n"):
                _log.warning("benchmark:   %s", line)
        _start_more()

    try:
        _start_more()
        while active:
            made_progress = False
            for proc, q, job in list(active):
                try:
                    outcome = q.get_nowait()
                except queue.Empty:
                    outcome = None

                if outcome is not None:
                    _finish(proc, q, job, outcome)
                    made_progress = True
                    continue

                if not proc.is_alive():
                    proc.join()
                    try:
                        outcome = q.get(timeout=0.2)
                    except queue.Empty:
                        outcome = _abrupt_failure(job, proc.exitcode)
                    q.close()
                    q.join_thread()
                    active.remove((proc, q, job))
                    outcomes.append(outcome)
                    _log.info("benchmark: %s -> %s", outcome.scene_id, outcome.status)
                    _start_more()
                    made_progress = True

            if not made_progress and active:
                time.sleep(0.05)
    except BaseException:
        for proc, q, _job in active:
            if proc.is_alive():
                proc.terminate()
            proc.join()
            q.close()
            q.join_thread()
        raise

    return outcomes


def _build_result(
    dataset_name: str,
    split: str | None,
    outcomes: list[SceneOutcome],
    scenes: list[str],
    cfg: BenchmarkConfig,
) -> BenchmarkResult:
    outcomes_by_id = {o.scene_id: o for o in outcomes}
    ordered = [outcomes_by_id[sid] for sid in scenes if sid in outcomes_by_id]
    coverage = {
        "n_total": len(scenes),
        "n_evaluated": sum(1 for o in ordered if o.status == "ok"),
        "n_missing_pred": sum(1 for o in ordered if o.status == "missing_pred"),
        "n_missing_gt": sum(1 for o in ordered if o.status == "missing_gt"),
        "n_failed": sum(1 for o in ordered if o.status == "failed"),
    }
    return BenchmarkResult(
        dataset=dataset_name,
        split=split or "auto",
        scenes=ordered,
        summary=aggregate(ordered),
        summary_all=aggregate_all(
            ordered,
            n_total=len(scenes),
            distance_default=cfg.missing_distance_default,
            fscore_default=cfg.missing_fscore_default,
            seed_metric_names=cfg.metrics,
        ),
        coverage=coverage,
        config=dataclasses.asdict(cfg),
    )


# ---------------------------------------------------------------------------
# BaseBenchmark
# ---------------------------------------------------------------------------


class BaseBenchmark(ABC):
    dataset_name: ClassVar[str]

    def __init__(
        self,
        gt_root: PathLike,
        pred_root: PathLike,
        *,
        cfg: BenchmarkConfig | None = None,
    ) -> None:
        self.gt_root = Path(gt_root)
        self.pred_root = Path(pred_root)
        self.cfg = cfg or self._default_config()

    @abstractmethod
    def _default_config(self) -> BenchmarkConfig: ...

    @abstractmethod
    def _list_scenes(self, split: str | None) -> list[str]: ...

    @property
    @abstractmethod
    def _worker(self) -> Callable: ...

    def run(
        self,
        split: str | None = None,
        *,
        locator: PredictionLocator | None = None,
    ) -> BenchmarkResult:
        work_dir = _make_work_dir(self.dataset_name)
        scenes = self._list_scenes(split)
        loc = locator or PredictionLocator(preds_root=self.pred_root)
        jobs: list[BenchmarkJob] = [
            (sid, _pred_descriptor(loc.resolve(sid))) for sid in scenes
        ]

        if self.cfg.fail_on_missing:
            for sid, desc in jobs:
                if desc is None:
                    raise MissingArtifactError(
                        f"No prediction found for scene {sid!r} under {loc.preds_root}"
                    )

        fn = partial(self._worker, gt_root=self.gt_root, cfg=self.cfg, work_dir=work_dir)

        if self.cfg.workers <= 1 or len(jobs) <= 1:
            outcomes: list[SceneOutcome] = []
            for job in jobs:
                o = fn(job)
                outcomes.append(o)
                _log.info("benchmark: %s -> %s", o.scene_id, o.status)
                if o.status == "failed" and o.error:
                    for line in o.error.rstrip().split("\n"):
                        _log.warning("benchmark:   %s", line)
        else:
            outcomes = _run_jobs_parallel(jobs, fn, workers=self.cfg.workers)

        result = _build_result(self.dataset_name, split, outcomes, scenes, self.cfg)
        (work_dir / "results.json").write_text(json.dumps(result.to_dict(), indent=2))
        return result


__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "BaseBenchmark",
    "SceneOutcome",
    "SceneStatus",
    "aggregate",
    "aggregate_all",
    "collect_values",
    "_pred_descriptor",
    "_load_pred_geometry",
    "_resolve_mask_pattern",
    "_make_work_dir",
    "_run_jobs_parallel",
    "_build_result",
]
