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

- `metrics/geometry.py`: `clean_points` (shape validation, non-finite row filtering, empty →
  `InvalidGeometryError`), `compute_directional_distances` (pred→gt and gt→pred via the
  `nearest_neighbor` backend; no KD-tree in metrics/), and `evaluate_geometry_metrics` that
  dispatches `MetricSpec` → `MetricResult`. Every statistic/reduction/threshold/clamp/percentile
  is read from the spec — no defaults in the metric functions. `GEOMETRY_METRIC_NAMES` =
  {accuracy, completeness, chamfer, precision, recall, fscore, coverage}.
- Formulas per `.agent/metrics.md`: accuracy = stat(pred→gt), completeness = stat(gt→pred),
  chamfer = stat(each) combined by explicit sum|mean reduction, precision/recall/coverage =
  strict `< τ` fraction, `fscore = 2PR/(P+R)` with **0 (not NaN) when P+R==0**. Distance metrics
  carry `unit="m"`; ratios carry `unit=None`. `n_points_pred/gt` and `valid_fraction` recorded.
- `metrics/aggregation.py`: `mean/median/weighted_mean`, `per_scene_then_mean`, and
  `global_fscore` (pools `PrecisionRecallCounts` across scenes). A test demonstrates
  per-scene-then-mean vs global give different F-scores (aggregation order is not free).
- Synthetic analytic tests (`test_geometry_metrics.py` 20, `test_aggregation.py` 8): identical
  clouds → zeros/perfect; known distances for mean≠median; Chamfer sum(3.0) vs mean(1.5);
  percentile; clamp; precision/recall partial; fscore=0 on no matches; strict-`<` boundary;
  empty/shape/NaN handling with valid_fraction; unknown-name / missing-statistic /
  missing-threshold / missing-reduction errors.

## Decisions

- Empty (or all-non-finite) input is a structured `InvalidGeometryError` raised by the metric
  layer; the runner (task 007) applies the protocol failure policy (abort / skip_and_flag /
  score_worst). Metric code never silently returns a worst score.
- `fraction_within` uses strict `<` τ (matches "distance < τ" in `.agent/metrics.md`).
- `chamfer` requires an explicit `reduction` (sum|mean) — `reduction=None` raises rather than
  guessing, per the "no hidden choices" rule.
- **Normal consistency deferred**: it needs nearest-neighbor *indices*, which the `NNBackend`
  interface (task 004) does not expose, and no built-in protocol uses it. Adding it later will
  extend the NN interface deliberately rather than pre-emptively. `culled_fraction` /
  `valid_fraction` as standalone metric names are pipeline diagnostics (task 007/011), not
  computed from point sets here.

## Verification

```bash
pytest tests/unit/test_geometry_metrics*.py tests/unit/test_aggregation*.py
```

Each formula test's expected value is derived analytically in the test, not from a prior run.

Outcomes (feature/repo-foundation):

```text
pytest            -> 112 passed (adds 28 geometry + aggregation tests)
ruff check .      -> All checks passed!
mypy eval3r       -> Success: no issues found in 83 source files
```

## Status

done
