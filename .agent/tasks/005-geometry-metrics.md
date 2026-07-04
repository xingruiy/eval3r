# 005 — Geometry metrics

## Goal

Point-set geometry metrics (`accuracy`, `completeness`, `chamfer`, `precision@τ`,
`recall@τ`, `fscore@τ`, distance percentiles, coverage, diagnostics) implemented in
`metrics/geometry.py` with analytically verified synthetic tests.

## Scope

- Input contract per `.agent/metrics.md`: Nx3/Mx3 validation, NaN/Inf filtering, empty-array
  behavior (clear structured failure or protocol-defined worst score), meters internally,
  `n_points_pred` / `n_points_gt` recorded.
- Formulas exactly as pinned in `.agent/metrics.md`; every statistic (mean/median/percentile),
  Chamfer reduction (sum/mean), threshold, clamp, and F-score aggregation order comes from
  `MetricSpec` — no defaults inside metric functions.
- `fscore = 0` (not NaN) when `precision + recall == 0`.
- Optional: normal consistency (requires normals passed in), clamped accuracy/completeness.
- Diagnostics: valid_fraction, culled_fraction fields on `MetricResult`.
- Aggregation primitives in `metrics/aggregation.py`: per_scene_then_mean, global,
  per_frame_then_scene_then_mean; mean/median/weighted-mean across scenes.
- NN distances go through the `nearest_neighbor` backend from task 004 — no KD-tree code
  inside metrics/.

## Out of Scope

- Mesh loading/sampling (backend, task 004) and pipeline orchestration (task 007).
- Depth and pose metrics (tasks 014, 015).

## Relevant Files

- `.agent/metrics.md` — "Geometry metrics", "Aggregation", "Failure behavior", "Tests"
- `.agent/schema.md` — `MetricSpec`, `MetricResult`, `AggregationSpec`
- `.agent/plan.md` — "Metrics → Geometry metrics", "Testing strategy → Synthetic geometry tests"

## Plan

1. Implement distance computation + statistic/reduction application driven by `MetricSpec`.
2. Implement threshold metrics and F-score with explicit aggregation order.
3. Implement aggregation primitives.
4. Synthetic tests with analytic expectations: identical clouds (all zeros), translated
   clouds (known distance), scaled clouds, partial overlap, single point, plane, cube and
   sphere sampled surfaces; Chamfer sum vs mean; mean vs median; precision/recall/F-score
   edge cases; empty inputs; NaN/Inf filtering; F-score aggregation order.

## Findings

(record during implementation)

## Decisions

(record during implementation)

## Verification

```bash
pytest tests/unit/test_geometry_metrics*.py tests/unit/test_aggregation*.py
```

Each formula test's expected value is derived analytically in the test, not from a prior run.

## Status

todo
