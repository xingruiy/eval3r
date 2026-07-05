# eval3r Metrics

Metrics define the numeric quantities reported by `eval3r`. A metric implementation should be small, explicit, and controlled by the active protocol. Metrics must not make hidden choices about alignment, scale, masks, thresholds, sampling, or aggregation.

The library evaluates meshes, point clouds, pointmaps, depth predictions, and trajectories. It does not integrate RGB-D frames, build TSDF volumes, or turn depth sequences into scene reconstructions.

## Metric layer boundaries

The metric layer owns:

```text
formula definitions
input validation
NaN / Inf filtering
threshold handling
statistic and reduction handling
per-frame and per-scene aggregation primitives
metric result construction
```

The metric layer does not own:

```text
dataset file parsing
pose convention normalization
mesh loading
surface sampling implementation
nearest-neighbor backend implementation
trajectory association implementation
official benchmark scripts
```

Those tasks belong to dataset adapters or backend adapters.

## Geometry metrics

Geometry metrics operate on two point sets after any required alignment, masking, culling, and sampling have already been applied by the pipeline.

Input:

```text
pred_points: Nx3
gt_points: Mx3
optional pred_normals: Nx3
optional gt_normals: Mx3
thresholds: list[float]
```

Required validation:

```text
arrays must be rank-2 with shape Nx3 / Mx3
arrays must be finite after NaN / Inf filtering
empty arrays must produce a clear failure or protocol-defined worst score
units must be meters internally
n_points_pred and n_points_gt must be recorded
```

Primary geometry metrics:

```text
accuracy
completeness
chamfer
precision@τ
recall@τ
fscore@τ
distance percentiles
coverage
```

Optional geometry metrics:

```text
normal consistency
clamped accuracy / completeness
valid fraction
culled fraction
```

Definitions:

```text
accuracy      = statistic of nearest-neighbor distances pred → gt
completeness  = statistic of nearest-neighbor distances gt → pred
chamfer       = accuracy + completeness, or (accuracy + completeness) / 2, as declared
precision@τ   = fraction of pred points with pred→gt distance < τ
recall@τ      = fraction of gt points with gt→pred distance < τ
fscore@τ      = 2 * precision * recall / (precision + recall)
```

Protocol-pinned choices:

```text
accuracy statistic: mean | median | percentile
completeness statistic: mean | median | percentile
Chamfer reduction: sum | mean
threshold values
F-score aggregation: per_scene_then_mean | global
clamping distance, if any
```

If `precision + recall == 0`, F-score should be zero, not NaN.

## Mesh metrics

Meshes are evaluated by sampling their surfaces and then applying point-set geometry metrics.

Rules:

```text
mesh surfaces are sampled before distance metrics
raw vertices are not used unless the protocol explicitly requests raw vertices
sample count is protocol-defined
sample seed is protocol-defined and recorded
sampling backend and backend version are recorded
normal consistency requires normal-aware surface sampling
```

Recommended default behavior:

```text
sample method: surface_area
sample count: protocol-specific, commonly 200000 for scene-level protocols
seed: derive per scene from protocol base seed and scene_id
```

Do not perform mesh repair, hole filling, watertight conversion, smoothing, or decimation unless a protocol explicitly declares that preprocessing step.

## Point-cloud metrics

Point-cloud metrics operate directly on loaded points.

Pipeline order:

```text
load point cloud
normalize units and coordinate convention
validate shape
remove NaN / Inf
apply alignment
apply confidence filtering if present and protocol-defined
apply masks / culling
sample or downsample if protocol-defined
compute geometry metrics
record point counts and valid fractions
```

Optional downsampling must be declared in the protocol. A performance-motivated downsampling choice changes the metric and must be recorded.

## Pointmap metrics

Pointmaps are supported only when they can be interpreted as explicit 3D points in a declared coordinate frame.

Supported cases:

```text
single pointmap evaluated against a matching GT pointmap or point cloud
per-frame pointmap evaluated per frame, then aggregated
object-centric pointmap evaluated under a declared object-centric protocol
```

Unsupported by design:

```text
integrating pointmaps into a scene reconstruction inside eval3r
implicit fusion across frames
hidden confidence filtering chosen by the method
```

Pointmap protocols must state:

```text
coordinate frame
source pose format if poses are used
scale type
confidence policy
masking policy
aggregation order
```

## Depth metrics

Depth metrics operate in image space.

Input:

```text
pred_depth
gt_depth
valid_mask
optional confidence
optional intrinsics metadata for validation
```

Primary depth metrics (implemented in `metrics/depth.py`, task 014; metric names
are the result keys — the δ names disambiguate the three thresholds because
aggregation keys metrics by name):

```text
absrel    mean(|pred - gt| / gt)                              unitless
sqrel     mean((pred - gt)^2 / gt)                            metres
rmse      sqrt(mean((pred - gt)^2))                           metres
rmse_log  sqrt(mean((ln pred - ln gt)^2))                     unitless
silog     sqrt(mean(d^2) - mean(d)^2), d = ln pred - ln gt    unitless
delta_1   fraction of pixels with max(pred/gt, gt/pred) < 1.25
delta_2   same at 1.25² = 1.5625
delta_3   same at 1.25³ = 1.953125
```

`silog` is the Eigen et al. scale-invariant log error, reported unscaled (not
multiplied by 100). δ thresholds are strict (`<`) and must still be explicit in
the protocol's MetricSpec; a `delta_*` spec without a threshold fails loudly.

Required handling:

```text
depth unit is explicit
integer depth scales are recorded
invalid depth values are masked
zero or negative depths are invalid unless a protocol says otherwise
valid pixel count is recorded
valid fraction is recorded
```

Masking detail (task 014): non-finite pixels in either array and protocol-listed
`invalid_depth_values` sentinels are always masked. When `ignore_invalid_depth`
is true (the default), non-positive GT **and prediction** pixels are also masked
— ratio and log metrics are undefined there; the per-frame mask breakdown
(`n_gt_nonfinite`, `n_gt_invalid_value`, `n_gt_nonpositive`, `n_pred_nonfinite`,
`n_pred_nonpositive`) is recorded in metric metadata so an all-invalid prediction
fails loudly (empty mask) instead of silently scoring on a subset. Alignment can
reintroduce non-positive predictions (negative scale/shift); log metrics then
raise rather than clamp.

Scale alignment modes (declared as first-class `AlignmentSpec.mode` values in `.agent/schema.md`, never as free-form parameters):

```text
none
scale_median
scale_least_squares
scale_affine
```

Granularity (declared as `AlignmentSpec.granularity`):

```text
per_frame
per_sequence
per_scene
```

Scale alignment mode and granularity must appear in metric metadata and report headers. For example, `AbsRel` under per-frame affine alignment is not comparable to `AbsRel` under per-sequence median scaling.

Estimator definitions (task 014):

```text
scale_median         s = median(gt / pred) over valid pixels; shift 0
scale_least_squares  s = Σ(pred·gt) / Σ(pred²); shift 0
scale_affine         (s, t) minimizing ||s·pred + t − gt||² (closed form)
```

Degenerate inputs (constant prediction for affine, all-zero prediction for least
squares) fail explicitly. On the single-file path, `per_frame` estimates per
frame, while `per_sequence` / `per_scene` pool all frames' valid pixels into one
estimate (identical pooling for a single scene, recorded under the requested
granularity). Every estimated scale/shift is written to the run's alignment
records and each metric's metadata.

Depth sequences are aggregated over frames and scenes. They are not converted into meshes, fused point clouds, or TSDF volumes.

## Pose metrics

Pose metrics delegate trajectory association and metric computation to `evo`
(implemented in `metrics/pose.py` + `backends/trajectory_evo.py`, task 015;
eval3r never reimplements association or pose-error math).

Metric names (the result keys):

```text
ate                    evo APE, translation part                       metres
rpe_translation        evo RPE, translation part, explicit delta       metres
rpe_rotation           evo RPE, rotation angle, explicit delta         degrees
alignment_scale_error  |ln s| of the estimated alignment scale         unitless
```

`ate`/`rpe_*` values are the spec's **explicit** statistic (rmse / mean / median
/ std / min / max / sse) over evo's error series — a spec without a statistic
fails loudly. `rpe_*` specs must pin `delta` and `delta_unit`
(frames / seconds / meters) in their parameters — RPE at delta = 1 frame is not
comparable to RPE at delta = 1 second, so deltas are never defaulted; `all_pairs`
is recorded alongside. Alignment modes are the first-class `AlignmentSpec.mode`
values `none` / `trajectory_se3` / `trajectory_sim3` (Umeyama via evo, estimated
once per trajectory pair — granularity `per_scene`). Trajectory files are TUM
format (`timestamp x y z qx qy qz qw`); further source formats arrive with their
dataset adapters.

Supported metrics:

```text
ATE
RPE
rotation error
translation error
SE3-aligned trajectory error
Sim3-aligned trajectory error
```

Protocol-pinned choices:

```text
pose file format
source pose convention
timestamp association policy
maximum timestamp difference
interpolation policy
alignment mode
scale correction policy
subsequence policy
```

Pose results must record:

```text
number of associated poses
number of dropped poses
alignment mode
Sim3 scale if used
backend name and version
association parameters
```

Recording (task 015): every pose MetricResult carries `alignment_mode`,
`alignment_scale`, the association policy (`nearest_timestamp` +
`associate_max_diff` + `offset`), and the associated/dropped pose counts on both
sides in its metadata (`n_points_pred`/`n_points_gt` hold the raw pose counts).
The full estimated transform (rotation, translation, scale, `|ln s|`) is written
to `alignment_transforms.json`. Association tolerance is an explicit alignment
parameter — the backend refuses to default it.

## Diagnostic metrics

Diagnostics explain why an evaluation behaved a certain way. They should be reported alongside primary metrics when available, but not substituted for them.

Useful diagnostics:

```text
alignment_scale_error = |log(s)| for Sim3 scale
alignment residual RMSE
alignment inlier fraction
valid pixel fraction
valid point fraction
culled fraction
confidence-vs-error summaries
scene coverage
failure coverage
```

Diagnostic metrics are especially useful for detecting:

```text
wrong pose convention
incorrect depth scale
invalid confidence thresholding
excessive visibility culling
partial benchmark coverage
```

## Aggregation

Aggregation must be explicit.

Supported aggregation stages:

```text
per_frame
per_scene
mean across scenes
median across scenes
weighted mean
```

F-score aggregation is particularly important. The protocol must say whether it uses:

```text
per_scene_then_mean
```

or:

```text
global precision / recall followed by one global F-score
```

The default for dataset benchmarks should usually be `per_scene_then_mean`, but it must still be written in the protocol.

Partial coverage must be visible. If a scene fails and the failure policy is `skip_and_flag`, aggregate reports must say that the aggregate is over partial coverage.

## Failure behavior

Metric implementations should not hide invalid inputs.

Failure examples:

```text
empty prediction after masking
empty ground truth after masking
missing required confidence map
missing required depth scale
shape mismatch
unsupported metric / modality pair
backend missing
```

The pipeline applies the protocol's failure policy:

```text
abort
skip_and_flag
score_worst
```

Metric code should raise structured errors with enough context for the runner to write `failures.json`.

## Metric result metadata

Each metric result should record:

```text
name
value
unit
threshold
statistic
reduction
scene_id
frame_id when applicable
protocol name
protocol hash
backend name
n_points_pred
n_points_gt
n_pixels_valid
valid_fraction
culled_fraction
metadata
```

## Tests

Required tests:

```text
accuracy and completeness on identical point clouds
known translated point clouds
Chamfer sum vs mean
mean vs median statistics
precision / recall / F-score edge cases
empty prediction handling
NaN / Inf filtering
threshold behavior
F-score aggregation order
depth invalid-mask behavior
depth scale alignment behavior
pose metric wrapper normalization
failure policy interaction
```

Use synthetic data with analytically checkable outputs. Do not require full datasets in normal CI.
