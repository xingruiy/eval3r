"""Tests for the aggregate / collect_values helpers."""

from __future__ import annotations

from eval3r.benchmark.aggregate import aggregate, collect_values
from eval3r.benchmark.core import SceneOutcome
from eval3r.metrics.metric3d import EvalResult3D


def _scene(scene_id: str, tau: float, f_value: float) -> SceneOutcome:
    return SceneOutcome(
        scene_id=scene_id,
        status="ok",
        result=EvalResult3D(
            chamfer=0.0,
            chamfer_variant="l1_mean_bidirectional",
            accuracy=0.0,
            completeness=0.0,
            fscore={tau: {"f": f_value, "precision": f_value, "recall": f_value}},
        ),
    )


def test_collect_values_default_keeps_per_tau_columns() -> None:
    outcomes = [
        _scene("s1", 0.01, 0.6),
        _scene("s2", 0.05, 0.8),
    ]
    out = collect_values(outcomes, thresholds=(0.05,))
    # Default (pooled=False) emits one column per τ — and seeds 0.05 with
    # exactly the scene that used 0.05.
    assert out["f@0.01"] == [0.6]
    assert out["f@0.05"] == [0.8]
    assert "f" not in out


def test_collect_values_pooled_flattens_into_canonical() -> None:
    outcomes = [
        _scene("s1", 0.01, 0.6),
        _scene("s2", 0.05, 0.8),
    ]
    out = collect_values(outcomes, pooled=True)
    # Each scene contributes its own scene-τ f-score to the same list,
    # regardless of which τ produced it. Order = sorted by τ.
    assert out["f"] == [0.6, 0.8]
    assert out["precision"] == [0.6, 0.8]
    assert out["recall"] == [0.6, 0.8]
    # No per-τ columns when pooled.
    assert "f@0.01" not in out
    assert "f@0.05" not in out


def test_aggregate_pooled_mean_is_unweighted() -> None:
    outcomes = [
        _scene("s1", 0.01, 0.6),
        _scene("s2", 0.05, 0.8),
    ]
    summary = aggregate(outcomes, pooled=True)
    # Mean across scenes at their own τ — no τ weighting.
    assert summary["f"]["mean"] == 0.7
    assert summary["f"]["n"] == 2
