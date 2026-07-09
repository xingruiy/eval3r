# 014 — Depth metrics

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
- Per-frame then per-scene aggregation for depth sequences; any conversion into scene
  geometry belongs in an explicit pipeline/backend workflow, not hidden depth-metric
  behavior.
- `single_depth` built-in protocol executes through the task-007 runner (same stages,
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
- `eval3r/metrics/depth.py`, `eval3r/pipeline/depth_runner.py`,
  `eval3r/backends/depth_{common,imageio,opencv}.py`, `eval3r/cli/metric.py`

## Plan

1. Implement formulas + masking + unit handling with synthetic analytic tests.
2. Implement the three scale-alignment estimators with known-answer tests (e.g. pred = 2×gt
   → median scale 0.5 recovers zero error).
3. Depth IO backend; sequence aggregation.
4. Wire runner + CLI; metadata/report-header checks.

## Findings

- **δ metric names had to change.** The old `single_depth.yaml` declared three metrics all
  named `delta` with different thresholds — but aggregation and `RunResult.metrics` key by
  metric name, so the three values would collide. Renamed to `delta_1` / `delta_2` /
  `delta_3` (thresholds still explicit: 1.25 / 1.5625 / 1.953125, strict `<`), the same
  pattern task 013 used for the per-tolerance ETH3D metrics. Also added `silog` (the task
  goal lists scale-invariant error but the old protocol omitted it).
- **PFM needs OpenCV.** imageio does not decode PFM (a common float-depth interchange
  format); `cv2.imread(IMREAD_UNCHANGED)` reads and writes it (verified by roundtrip in
  the test). Hence two `depth_io` backends with one shared contract, `imageio` the default.
- **Non-positive prediction pixels are masked, with counts recorded.** absrel/sqrel divide
  by gt; rmse_log/silog/δ take log or ratio of pred — undefined at pred ≤ 0. Community
  eval code often silently clamps pred to a min depth; eval3r instead masks those pixels
  and records `n_pred_nonpositive` (plus the full mask breakdown) in metric metadata, so a
  method emitting garbage cannot silently benefit — an all-invalid prediction fails with
  an explicit empty-mask error naming the breakdown. If *alignment* (negative scale/shift)
  reintroduces non-positive values, log metrics raise rather than clamp.
- **Pre-existing test-env quirk, not from this change:** with `FORCE_COLOR` set (as in
  some non-interactive shells), rich emits ANSI codes into `CliRunner` captures and
  `test_cli_benchmark_validate_reports_missing`'s substring assert fails (reproduced at
  HEAD 41f29c7 in a clean worktree). The suite passes without `FORCE_COLOR`; not fixed
  here (out of scope).
- `test_smoke.py::test_stub_command_fails_loudly_with_reason` pointed at `metric depth`
  as the canonical stub; retargeted to `metric pose` (task 015).

## Decisions

- `metrics/depth.py`: closed metric-name set `absrel, sqrel, rmse, rmse_log, silog,
  delta_1, delta_2, delta_3`. Formulas: absrel = mean(|p−g|/g); sqrel = mean((p−g)²/g)
  (unit **metres**: m²/m); rmse = √mean((p−g)²) (metres); rmse_log = √mean((ln p − ln g)²);
  silog = √(mean(d²) − mean(d)²), d = ln p − ln g — Eigen et al. scale-invariant log
  error, reported **unscaled** (not ×100; documented in the protocol notes and metrics.md);
  delta_k = fraction of max(p/g, g/p) < τ (strict). δ specs without a threshold fail —
  thresholds are never defaulted.
- Masking per `MaskingSpec`: non-finite (either side) and `invalid_depth_values`
  sentinels always masked; `ignore_invalid_depth: true` additionally masks non-positive
  gt and pred. `ignore_invalid_depth: false` is the documented "protocol says otherwise"
  escape (only sentinels + non-finite masked). Every frame records n_pixels_valid,
  valid_fraction, and the five-way mask breakdown.
- Scale alignment estimators (`estimate_depth_scale`): scale_median s = median(g/p);
  scale_least_squares s = Σpg/Σp²; scale_affine closed-form least squares for
  g ≈ s·p + t (bias-corrected covariance / variance). Degenerate inputs (constant pred
  for affine, zero pred for LSQ) fail explicitly. Granularity: `per_frame` = one estimate
  per frame; `per_sequence`/`per_scene` = one estimate over all frames' pooled valid
  pixels (identical pooling on this single-scene path; recorded under the requested
  granularity). Mode/granularity/scale/shift appear in every MetricResult's metadata and
  in `alignment_transforms.json`.
- `pipeline/depth_runner.py` mirrors the task-007 geometry runner (stages resolve → load
  → mask → align → metric → aggregate, same failure policies / SceneFailure / RunResult
  assembly) rather than bolting depth onto the geometry-typed stage functions. Sequences:
  pred dir + gt dir matched by filename stem (duplicate or unmatched stems fail with the
  lists named); modality auto-set to `single_depth` / `depth_sequence` and recorded.
  Per-frame MetricResults (frame_id set) and per-scene aggregates (mean over frames,
  `aggregation: per_frame_mean`, n_frames in metadata) are both written to
  `per_scene_metrics`.
- Depth IO: `depth_io` registry kind with `imageio` (default; images via imageio.v3,
  .npy via numpy) and `opencv` (adds PFM); shared unit logic in `backends/depth_common.py`.
  `load_depth` returns 2D float64 **metres**; integer sources require an explicit
  `depth_unit` (fail otherwise), float sources default to metric with an optional unit
  multiplier; multi-channel images rejected; invalid values pass through to masking.
  `load_mask` returns bool (nonzero = valid).
- `single_depth.yaml` → protocol_version 0.2.0 (new hash pinned in test_protocols.py):
  delta renames + silog, `backend_preferences: {depth_io: imageio}`,
  `save_alignment_transforms: true` (estimated scales are result-affecting), notes on
  δ naming / masking / silog convention. Docs updated in the same change
  (protocols.md, metrics.md, backends.md).
- API `evaluate_depth(...)` (+ lazy re-export in `eval3r/__init__`), CLI
  `e3r metric depth` with `--depth-unit` (pred) and `--gt-depth-unit` separately (pred
  and GT commonly differ, e.g. float npy vs mm PNG), `--align`, `--align-granularity`;
  overrides recorded and re-hashed. CLI echoes resolved protocol+hash, units, alignment,
  masking, failure policy, per-metric valid-pixel counts, and each alignment record.

## Verification

```bash
ruff check .   # All checks passed!
mypy eval3r    # Success: no issues found in 92 source files
EVAL3R_ETH3D_TOOL=~/xingrui_ws/tools/multi-view-evaluation/build/ETH3DMultiViewEvaluation \
  pytest -q    # 339 passed, 3 skipped (skips = TnT real-data tests only)
mkdocs build   # OK
# analytic checks (tests/unit/test_depth_metrics.py, 23 tests): pred = 1.1×gt →
#   absrel exactly 0.1, rmse_log = ln 1.1; silog = 0 for any constant scale;
#   pred = 2×gt → median/LSQ scale 0.5; pred = 2×gt+3 → affine (0.5, −1.5);
#   δ strictness at ratio exactly 1.25; empty-mask/degenerate failures.
# depth IO (tests/unit/test_depth_io.py, 18 tests): 16-bit mm PNG ×0.001 → metres,
#   integer-without-unit fails, PFM roundtrip via OpenCV, mask loading.
# end-to-end (tests/integration/test_single_depth_metric.py, 11 tests): API + CLI,
#   sequence per_frame vs per_sequence granularity (recovered scales 0.5/0.25 vs one
#   pooled scale), run-directory completeness, frame-mismatch/abort failures.
# acceptance CLI run (in-process python -m eval3r.cli.main, PATH e3r is foreign):
#   metric depth pred.npy --gt gt.png --gt-depth-unit 0.001 --align scale_median \
#     --align-granularity per_frame
#   → all 8 metrics at analytic values (absrel 0, delta_* 1), scale 0.5 recorded;
#   results.json carries depth units, mask breakdown, alignment mode+granularity,
#   depth_io backend version; alignment_transforms.json has the per-frame record.
```

## Status

done
