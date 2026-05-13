"""Tests for the aggregate / collect_values helpers in benchmark.base."""

from __future__ import annotations

from eval3r.benchmark.base import SceneOutcome, aggregate, collect_values
from eval3r.pipeline import PipelineResult


def _pipeline_result(**values) -> PipelineResult:
    return PipelineResult(
        values=values,
        n_samples=1000,
        n_visible=1000,
        n_total=1000,
        align_mode="none",
        align_scale=1.0,
    )


def _scene(scene_id: str, **values) -> SceneOutcome:
    return SceneOutcome(
        scene_id=scene_id,
        status="ok",
        result=_pipeline_result(**values),
    )


def test_collect_values_float_metrics() -> None:
    outcomes = [
        _scene("s1", chamfer=0.01, accuracy=0.02),
        _scene("s2", chamfer=0.03, accuracy=0.04),
    ]
    out = collect_values(outcomes)
    assert out["chamfer"] == [0.01, 0.03]
    assert out["accuracy"] == [0.02, 0.04]


def test_collect_values_tuple_metrics() -> None:
    outcomes = [
        _scene("s1", **{"fscore@0.05": (0.8, 0.85, 0.75)}),
        _scene("s2", **{"fscore@0.05": (0.6, 0.7, 0.55)}),
    ]
    out = collect_values(outcomes)
    assert out["fscore@0.05_f"] == [0.8, 0.6]
    assert out["fscore@0.05_precision"] == [0.85, 0.7]
    assert out["fscore@0.05_recall"] == [0.75, 0.55]
    assert "fscore@0.05" not in out


def test_collect_values_skips_non_ok() -> None:
    outcomes = [
        _scene("s1", chamfer=0.01),
        SceneOutcome(scene_id="s2", status="missing_pred"),
        SceneOutcome(scene_id="s3", status="failed", error="oops"),
    ]
    out = collect_values(outcomes)
    assert out["chamfer"] == [0.01]


def test_aggregate_mean() -> None:
    outcomes = [
        _scene("s1", chamfer=0.01),
        _scene("s2", chamfer=0.03),
    ]
    summary = aggregate(outcomes)
    assert summary["chamfer"]["mean"] == 0.02
    assert summary["chamfer"]["n"] == 2


def test_aggregate_empty() -> None:
    summary = aggregate([])
    assert summary == {}
