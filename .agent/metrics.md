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

Primary depth metrics:

```text
AbsRel
SqRel
RMSE
RMSE-log
δ < 1.25
δ < 1.25²
δ < 1.25³
scale-invariant depth error
```

Required handling:

```text
depth unit is explicit
integer depth scales are recorded
invalid depth values are masked
zero or negative depths are invalid unless a protocol says otherwise
valid pixel count is recorded
valid fraction is recorded
```

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

Depth sequences are aggregated over frames and scenes. They are not converted into meshes, fused point clouds, or TSDF volumes.

## Pose metrics

Pose metrics should delegate trajectory association and metric computation to `evo` where practical.

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
