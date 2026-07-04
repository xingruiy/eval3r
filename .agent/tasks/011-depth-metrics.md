# 011 — Depth metrics

## Goal

Depth evaluation (`AbsRel`, `SqRel`, `RMSE`, `RMSE-log`, δ thresholds, scale-invariant
error) works for single frames and depth sequences with explicit units, masking, and
first-class scale alignment; `e3r metric depth` runs end-to-end.

## Scope

- `metrics/depth.py`: formulas per `.agent/metrics.md`; explicit depth unit required for
  integer depth; invalid values masked (zero/negative invalid unless protocol says
  otherwise); valid pixel count + fraction recorded.
- Scale alignment as first-class `AlignmentSpec` modes: `scale_median`,
  `scale_least_squares`, `scale_affine`; granularity `per_frame` / `per_sequence` /
  `per_scene`. Mode + granularity must appear in metric metadata and report headers —
  never via free-form parameters.
- `backends/depth_imageio.py` (+ optional `depth_opencv.py`) for the `depth_io` registry
  kind: PNG/PFM/npy loading with `depth_unit` handling; invalid values passed to masking.
- Per-frame then per-scene aggregation for depth sequences; depth sequences are never
  converted into scene reconstructions (hard boundary).
- `single_depth` built-in protocol executes through the task-006 runner (same stages,
  depth-shaped load/mask/metric).
- CLI: `e3r metric depth pred.png --gt gt.png --depth-unit 0.001 --align scale_median
  --align-granularity per_frame`.
- δ threshold naming in results must disambiguate 1.25 / 1.25² / 1.25³.

## Out of Scope

- Dataset-specific depth protocols (Hypersim/7-Scenes/etc. — backlog adapters).
- Confidence-map handling beyond the protocol `ConfidenceSpec` plumbing.

## Relevant Files

- `.agent/metrics.md` — "Depth metrics"
- `.agent/schema.md` — `AlignmentSpec` (scale modes note)
- `.agent/protocols.md` — `single_depth` template
- `.agent/plan.md` — "Depth metrics" milestone, CLI depth example

## Plan

1. Implement formulas + masking + unit handling with synthetic analytic tests.
2. Implement the three scale-alignment estimators with known-answer tests (e.g. pred = 2×gt
   → median scale 0.5 recovers zero error).
3. Depth IO backend; sequence aggregation.
4. Wire runner + CLI; metadata/report-header checks.

## Findings

(record during implementation)

## Decisions

(record during implementation; e.g. affine alignment solver details)

## Verification

```bash
pytest tests/unit/test_depth*.py
e3r metric depth pred.png --gt gt.png --align scale_median  # on fixture images
```

Acceptance per `.agent/plan.md` milestone; results record unit, mask counts, alignment mode
and granularity.

## Status

todo
