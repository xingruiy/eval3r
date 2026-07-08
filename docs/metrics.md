# Metrics

Metric implementations are small, explicit, and controlled entirely by the active
protocol. They never make hidden choices about alignment, scale, masks, thresholds,
sampling, or aggregation — those are protocol fields, hashed and recorded.

## Geometry metrics

Operate on two point sets after the pipeline has applied the protocol's alignment,
masking/culling, and sampling. Meshes are surface-sampled first (deterministic per-scene
seeds, recorded); raw vertices are used only if a protocol explicitly asks for them.

```text
accuracy      = statistic of nearest-neighbor distances pred → gt
completeness  = statistic of nearest-neighbor distances gt → pred
chamfer       = accuracy + completeness, or their mean — the reduction is declared
precision@τ   = fraction of pred points with pred→gt distance < τ
recall@τ      = fraction of gt points with gt→pred distance < τ
fscore@τ      = 2PR / (P + R)      (0 when P + R == 0, never NaN)
```

The statistic (mean/median/percentile), thresholds, Chamfer reduction, and F-score
aggregation order (`per_scene_then_mean` vs global) are always explicit in the protocol.
Every result records `n_points_pred`, `n_points_gt`, `valid_fraction`, and
`culled_fraction`. NaN/Inf points are filtered; empty inputs fail loudly under the
protocol's failure policy.

## Depth metrics

Operate in image space on masked, unit-normalized depth. Metric names are the result
keys:

```text
absrel     mean(|pred - gt| / gt)                              unitless
sqrel      mean((pred - gt)² / gt)                             meters
rmse       sqrt(mean((pred - gt)²))                            meters
rmse_log   sqrt(mean((ln pred - ln gt)²))                      unitless
silog      sqrt(mean(d²) - mean(d)²), d = ln pred - ln gt      unitless (Eigen et al., unscaled)
delta_1/2/3  fraction of pixels with max(pred/gt, gt/pred) < 1.25 / 1.25² / 1.25³
```

Masking is explicit: non-finite pixels, protocol-listed invalid sentinel values (e.g. 0,
65535), and — by default — non-positive pixels are masked, with the per-frame mask
breakdown recorded so an all-invalid prediction fails loudly instead of silently scoring
on a subset.

Scale alignment is a first-class protocol field, never a hidden parameter:

```text
mode:         none | scale_median | scale_least_squares | scale_affine
granularity:  per_frame | per_sequence | per_scene
```

`AbsRel` under per-frame affine alignment is **not** comparable to `AbsRel` under
per-sequence median scaling — mode and granularity appear in every metric's metadata and
in report headers, and every estimated scale/shift is written to the run's alignment
records.

Depth sequences are aggregated per frame and per scene. They are never converted into
meshes, fused point clouds, or TSDF volumes.

## Pose metrics

Delegated to [evo](https://github.com/MichaelGrupp/evo) (association and error math are
never reimplemented). TUM-format trajectories.

```text
ate                    evo APE, translation part                  meters
rpe_translation        evo RPE, translation part, explicit delta  meters
rpe_rotation           evo RPE, rotation angle, explicit delta    degrees
alignment_scale_error  |ln s| of the estimated alignment scale    unitless
```

The statistic (rmse/mean/median/...) is explicit per metric; RPE `delta` and `delta_unit`
(frames/seconds/meters) are never defaulted — RPE at 1 frame is not comparable to RPE at
1 second. Alignment modes are `none` / `trajectory_se3` / `trajectory_sim3` (Umeyama via
evo); users choose the intended alignment/adaptation mode explicitly. Every
result records the association policy and tolerance, associated/dropped pose counts on
both sides, the alignment mode, and the Sim3 scale when used.

## Diagnostics

Reported alongside primary metrics when available, never as substitutes:
`alignment_scale_error`, alignment residuals, valid fractions, culled fractions, scene
coverage, failure coverage. These catch wrong pose conventions, wrong depth scales,
excessive culling, and partial coverage.

## Aggregation

Aggregation stages (per-frame, per-scene, mean/median across scenes, weighted mean) are
explicit in the protocol. When a scene fails under `skip_and_flag`, every aggregate is
visibly labeled as **partial** — see [Failure policy](failure_policy.md).
