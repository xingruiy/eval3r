# eval3r Protocols

Protocols define how evaluation is performed. They are the main unit of comparability in `eval3r`.

A protocol is not just a list of metrics. It also defines dataset variant, ground-truth provenance, local evaluability, alignment, masking, sampling, confidence handling, metric definitions, aggregation, failure policy, reporting, and backend preferences.

## Protocol rules

```text
Metrics never choose alignment silently.
Sampling seeds are always recorded.
Metric thresholds are always explicit.
Metric statistics are always explicit.
Chamfer reduction is always explicit.
F-score aggregation order is always explicit.
Scale alignment is always explicit.
Pose source and normalized convention are always explicit.
Masking and culling are always explicit.
Confidence filtering is always explicit.
Failure policy is always explicit.
Ground-truth provenance is always recorded.
Local evaluation status is always recorded.
Every result includes the protocol hash.
```

## Fidelity labels

```text
official
  The protocol delegates to an official script or benchmark tool and records the official backend version.

official_like
  The protocol reimplements established dataset semantics locally and should be regression-tested against known numbers.

eval3r_native
  The protocol is defined by eval3r. It may be useful and repeatable, but it does not claim official leaderboard comparability.

server_only
  The split cannot be locally evaluated because GT is withheld or the official server is required.
```

## Built-in protocol set

Initial built-ins:

```text
single_geometry.yaml
single_depth.yaml
single_pose.yaml
dtu_official_like_pointcloud.yaml
scannet_single_layer_geometry_5cm.yaml
scannet_double_layer_geometry_5cm.yaml
tanks_temples_training_official.yaml
eth3d_training_official_like.yaml
```

Later built-ins (added together with their adapters):

```text
hypersim_depth.yaml
seven_scenes_depth.yaml
seven_scenes_pose.yaml
neural_rgbd_depth.yaml
replica_variant_geometry.yaml
co3d_pose_eval3r_native.yaml
kitti360_pose.yaml
waymo_pose.yaml
blendedmvs_depth.yaml
```

## Single-file geometry protocol

This is a simple protocol for comparing two files. It is useful for debugging and quick checks. It does not claim dataset-level fidelity.

```yaml
schema_version: 1
protocol_version: 0.1.0
name: single_geometry
fidelity: eval3r_native

dataset:
  dataset: custom
  variant: single_file
  split: null
  version: null
  fingerprint: null
  notes: []

prediction_modality: pointcloud

ground_truth:
  modality: pointcloud
  provenance: unknown
  independence: unknown
  density: unknown
  path: null
  fingerprint: null
  unit: m
  source_pose_format: unknown
  normalized_convention: cam_to_world_opencv_meters
  local_evaluation_status: supported
  notes: []

local_evaluation:
  status: supported
  reason: local files supplied by user
  public_gt_available: true
  external_assets_required: []
  official_server_required: false

alignment:
  mode: none
  estimate_on: none
  solver: none
  granularity: per_scene
  allow_override: true
  parameters: {}

confidence:
  policy: none
  allow_override: true

masking:
  pred_culling:
    method: none
  gt_culling:
    method: none
  valid_region:
    method: none
  ignore_invalid_depth: true
  invalid_depth_values: []

sampling:
  pred:
    method: all_points
    n_points: null
    seed: derive
  gt:
    method: all_points
    n_points: null
    seed: derive

metrics:
  - name: accuracy
    statistic: mean
  - name: completeness
    statistic: mean
  - name: chamfer
    statistic: mean
    reduction: mean
  - name: precision
    threshold: 0.05
  - name: recall
    threshold: 0.05
  - name: fscore
    threshold: 0.05
    aggregation: per_scene_then_mean

aggregation:
  per_scene: true
  mean: true
  median: true
  weighted_mean: false
  weights: none

failure_policy:
  policy: abort
  worst_values: {}

reporting:
  save_protocol_copy: true
  save_manifest_copy: true
  save_environment: true
  save_alignment_transforms: true
  save_colored_errors: false
  save_distance_histogram: false
  formats: [json, csv]

backend_preferences:
  mesh: trimesh
  pointcloud: plyfile
  nearest_neighbor: scipy

notes:
  - Single-file protocol for local comparison. Not a dataset benchmark.
  - Point-cloud and mesh file IO requires the lightweight 'mesh' extra
    (pip install 'eval3r[mesh]'); the default backends avoid Open3D so the
    quick-check path works without heavy optional dependencies.
```

## DTU official-like point-cloud protocol

DTU's reference is a laser-scanned point cloud. Evaluation is point-cloud based and uses ObsMask and Plane files.

```yaml
schema_version: 1
protocol_version: 0.1.0
name: dtu_official_like_pointcloud
fidelity: official_like

dataset:
  dataset: dtu
  variant: mvs_2014
  split: test
  version: 2014
  fingerprint: null
  notes:
    - Uses DTU laser-scan point clouds and official observability files.

prediction_modality: pointcloud

ground_truth:
  modality: pointcloud
  provenance: laser_scan
  independence: independent
  density: dense_surface
  path: null
  fingerprint: null
  unit: mm
  source_pose_format: world_to_cam_mvsnet
  normalized_convention: cam_to_world_opencv_meters
  local_evaluation_status: supported
  notes:
    - GT files are stlXXX_total.ply.
    - Prediction resolution should support conventional files such as <method>XXX_l3.ply.
    - ObsMaskXXX_10.mat and PlaneXXX.mat define official-like culling.
    - Local evaluation may use a validated Python port, MATLAB script, or another documented backend.

local_evaluation:
  status: supported
  reason: public GT and observability files are available.
  public_gt_available: true
  external_assets_required: []
  official_server_required: false

alignment:
  mode: none
  estimate_on: none
  solver: none
  granularity: per_scene
  allow_override: false
  parameters: {}

confidence:
  policy: none
  allow_override: false

masking:
  pred_culling:
    method: obs_mask
    source: dataset
    tolerance: null
  gt_culling:
    method: obs_mask
    source: dataset
  valid_region:
    method: dataset_official_mask
    source: dataset
  ignore_invalid_depth: true
  invalid_depth_values: []

sampling:
  pred:
    method: voxel_downsample
    voxel_size: 0.0002
    seed: derive
  gt:
    method: all_points
    seed: derive

metrics:
  - name: accuracy
    statistic: mean
    reduction: none
  - name: completeness
    statistic: mean
    reduction: none
  - name: overall
    statistic: mean
    reduction: mean

aggregation:
  per_scene: true
  mean: true
  median: true
  weighted_mean: false
  weights: none

failure_policy:
  policy: abort
  worst_values: {}

reporting:
  save_protocol_copy: true
  save_manifest_copy: true
  save_environment: true
  save_alignment_transforms: false
  save_colored_errors: false
  save_distance_histogram: true
  formats: [json, csv, markdown]

backend_preferences:
  pointcloud: open3d
  nearest_neighbor: scipy
  official_eval: dtu_eval_python

notes:
  - Missing Plane files must be recorded explicitly.
  - Native DTU units are millimeters; eval3r normalizes internal point arrays to meters but records source units.
```

## ScanNet single-layer geometry protocol

This protocol evaluates geometry against the selected ScanNet GT mesh convention with explicit visibility behavior.

```yaml
schema_version: 1
protocol_version: 0.1.0
name: scannet_single_layer_geometry_5cm
fidelity: eval3r_native

dataset:
  dataset: scannet
  variant: scannetv2_val_single_layer
  split: val
  version: scannetv2
  fingerprint: null
  notes:
    - Single-layer mesh convention.

prediction_modality: mesh

ground_truth:
  modality: mesh
  provenance: reconstructed
  independence: reconstruction_derived
  density: dense_surface
  path: null
  fingerprint: null
  unit: m
  source_pose_format: cam_to_world_opencv
  normalized_convention: cam_to_world_opencv_meters
  local_evaluation_status: supported
  notes:
    - GT mesh is BundleFusion-derived, not an independent laser scan.

local_evaluation:
  status: supported
  reason: validation GT mesh is available locally after dataset download.
  public_gt_available: true
  external_assets_required: []
  official_server_required: false

alignment:
  mode: none
  estimate_on: none
  solver: none
  granularity: per_scene
  allow_override: false
  parameters: {}

confidence:
  policy: none
  allow_override: false

masking:
  pred_culling:
    method: gt_visibility
    source: gt
    tolerance: 0.05
  gt_culling:
    method: none
  valid_region:
    method: scene_bounds
    source: dataset
  ignore_invalid_depth: true
  invalid_depth_values: [0]

sampling:
  pred:
    method: surface_area
    n_points: 200000
    seed: derive
  gt:
    method: surface_area
    n_points: 200000
    seed: derive

metrics:
  - name: accuracy
    statistic: mean
  - name: completeness
    statistic: mean
  - name: chamfer
    statistic: mean
    reduction: mean
  - name: precision
    threshold: 0.05
  - name: recall
    threshold: 0.05
  - name: fscore
    threshold: 0.05
    aggregation: per_scene_then_mean
  - name: culled_fraction

aggregation:
  per_scene: true
  mean: true
  median: true
  weighted_mean: false
  weights: none

failure_policy:
  policy: skip_and_flag
  worst_values: {}

reporting:
  save_protocol_copy: true
  save_manifest_copy: true
  save_environment: true
  save_alignment_transforms: false
  save_colored_errors: true
  save_distance_histogram: true
  formats: [json, csv, markdown, latex]

backend_preferences:
  mesh: trimesh
  pointcloud: open3d
  nearest_neighbor: scipy

notes:
  - Protocol must state whether single-layer or double-layer mesh convention is used.
  - Visibility culling is part of this protocol and should be recorded per scene.
  - ScanNet has no official geometry-reconstruction benchmark. This protocol follows the
    community single-layer 5cm F-score convention (TransformerFusion-style evaluation)
    and is therefore eval3r_native, not official_like. Results must not be presented as
    official ScanNet benchmark numbers, and the reference implementation used for any
    regression fixtures must be named in the task notes.
```

## ScanNet double-layer geometry protocol

This follows the same structure as the single-layer protocol but changes the dataset variant and GT convention.

```yaml
schema_version: 1
protocol_version: 0.1.0
name: scannet_double_layer_geometry_5cm
fidelity: eval3r_native

dataset:
  dataset: scannet
  variant: scannetv2_val_double_layer
  split: val
  version: scannetv2
  fingerprint: null
  notes:
    - Double-layer mesh convention.

prediction_modality: mesh
```

The remaining fields should match `scannet_single_layer_geometry_5cm` unless the documented double-layer protocol requires different GT processing.

## Tanks and Temples training official protocol

This protocol should wrap the official evaluation script.

```yaml
schema_version: 1
protocol_version: 0.1.0
name: tanks_temples_training_official
fidelity: official

dataset:
  dataset: tanks_temples
  variant: training_public_gt
  split: training
  version: null
  fingerprint: null
  notes:
    - Local official evaluation is only available for public-GT training scenes.

prediction_modality: pointcloud

ground_truth:
  modality: pointcloud
  provenance: laser_scan
  independence: independent
  density: dense_surface
  path: null
  fingerprint: null
  unit: m
  source_pose_format: tanks_temples_log
  normalized_convention: cam_to_world_opencv_meters
  local_evaluation_status: supported
  notes:
    - Official backend handles crop volumes, transforms, ICP behavior, and per-scene thresholds.

local_evaluation:
  status: supported
  reason: training split GT is public.
  public_gt_available: true
  external_assets_required: []
  official_server_required: false

alignment:
  mode: icp
  estimate_on: pointcloud
  solver: official_backend
  granularity: per_scene
  allow_override: false
  parameters: {}

confidence:
  policy: none
  allow_override: false

masking:
  pred_culling:
    method: dataset_official_mask
    source: dataset
  gt_culling:
    method: dataset_official_mask
    source: dataset
  valid_region:
    method: dataset_official_mask
    source: dataset
  ignore_invalid_depth: true
  invalid_depth_values: []

sampling:
  pred:
    method: all_points
    seed: derive
  gt:
    method: all_points
    seed: derive

metrics:
  - name: precision
    threshold: null
    aggregation: per_scene_then_mean
  - name: recall
    threshold: null
    aggregation: per_scene_then_mean
  - name: fscore
    threshold: null
    aggregation: per_scene_then_mean

aggregation:
  per_scene: true
  mean: true
  median: true
  weighted_mean: false
  weights: none

failure_policy:
  policy: abort
  worst_values: {}

reporting:
  save_protocol_copy: true
  save_manifest_copy: true
  save_environment: true
  save_alignment_transforms: true
  save_colored_errors: false
  save_distance_histogram: false
  formats: [json, csv, markdown]

backend_preferences:
  official_eval: tnt_official

notes:
  - Per-scene thresholds are resolved by the official backend, not hardcoded as a global value.
  - Intermediate and advanced splits are server-only and should use a separate server_only protocol stub.
```

## Tanks and Temples server-only protocol stub

```yaml
schema_version: 1
protocol_version: 0.1.0
name: tanks_temples_intermediate_server_only
fidelity: server_only

dataset:
  dataset: tanks_temples
  variant: intermediate_server
  split: intermediate
  version: null
  fingerprint: null

prediction_modality: pointcloud

ground_truth:
  modality: server_only
  provenance: server_only
  independence: server_only
  density: server_only
  local_evaluation_status: server_only
  notes:
    - GT is withheld. Submit to official benchmark server.

local_evaluation:
  status: server_only
  reason: GT for this split is withheld.
  public_gt_available: false
  external_assets_required: []
  official_server_required: true
```

The CLI should refuse to run local metrics with this protocol.

## ETH3D training official-like protocol

```yaml
schema_version: 1
protocol_version: 0.1.0
name: eth3d_training_official_like
fidelity: official_like

dataset:
  dataset: eth3d
  variant: training_public_gt
  split: training
  version: null
  fingerprint: null
  notes:
    - Uses public training GT and COLMAP text cameras.

prediction_modality: pointcloud

ground_truth:
  modality: pointcloud
  provenance: laser_scan
  independence: independent
  density: dense_surface
  unit: m
  source_pose_format: world_to_cam_colmap
  normalized_convention: cam_to_world_opencv_meters
  local_evaluation_status: supported
  notes:
    - Occlusion-aware GT preparation is part of the dataset tooling.

local_evaluation:
  status: supported
  reason: ETH3D training GT is public.
  public_gt_available: true
  external_assets_required: []
  official_server_required: false

alignment:
  mode: none
  estimate_on: none
  solver: none
  granularity: per_scene
  allow_override: false
  parameters: {}

confidence:
  policy: none
  allow_override: false

masking:
  pred_culling:
    method: scene_bounds
    source: dataset
  gt_culling:
    method: dataset_official_mask
    source: dataset
  valid_region:
    method: dataset_official_mask
    source: dataset
  ignore_invalid_depth: true
  invalid_depth_values: []

sampling:
  pred:
    method: all_points
    seed: derive
  gt:
    method: all_points
    seed: derive

metrics:
  - name: accuracy
    statistic: mean
  - name: completeness
    statistic: mean
  - name: precision
    threshold: 0.02
  - name: recall
    threshold: 0.02
  - name: fscore
    threshold: 0.02
    aggregation: per_scene_then_mean

aggregation:
  per_scene: true
  mean: true
  median: true
  weighted_mean: false
  weights: none

failure_policy:
  policy: skip_and_flag
  worst_values: {}

reporting:
  save_protocol_copy: true
  save_manifest_copy: true
  save_environment: true
  save_alignment_transforms: false
  save_colored_errors: true
  save_distance_histogram: true
  formats: [json, csv, markdown]

backend_preferences:
  camera: pycolmap
  pointcloud: open3d
  nearest_neighbor: scipy

notes:
  - The official ETH3D multi-view-evaluation tool reports accuracy / completeness / F1
    at multiple tolerances (1, 2, 5, 10, 20, 50 cm). The 2 cm headline threshold used
    here, and the official_like fidelity claim, must be regression-validated against
    that tool on a fixture before this protocol is frozen; extend the metric list to
    the official tolerance set if validation shows it is needed for comparability.
```

## Single-depth protocol

```yaml
schema_version: 1
protocol_version: 0.1.0
name: single_depth
fidelity: eval3r_native

dataset:
  dataset: custom
  variant: single_depth
  split: null

prediction_modality: single_depth

ground_truth:
  modality: depth
  provenance: unknown
  independence: unknown
  density: depth_image
  unit: m
  local_evaluation_status: supported

local_evaluation:
  status: supported
  reason: local files supplied by user
  public_gt_available: true
  external_assets_required: []
  official_server_required: false

alignment:
  mode: scale_median
  estimate_on: depth
  solver: none
  granularity: per_frame
  allow_override: true
  parameters: {}

confidence:
  policy: none
  allow_override: true

masking:
  pred_culling:
    method: none
  gt_culling:
    method: none
  valid_region:
    method: valid_depth
  ignore_invalid_depth: true
  invalid_depth_values: [0]

sampling:
  pred:
    method: all_points
    seed: derive
  gt:
    method: all_points
    seed: derive

metrics:
  - name: absrel
  - name: sqrel
  - name: rmse
  - name: rmse_log
  - name: delta
    threshold: 1.25
  - name: delta
    threshold: 1.5625
  - name: delta
    threshold: 1.953125

aggregation:
  per_frame: true
  per_scene: true
  mean: true
  median: true
  weighted_mean: false
  weights: none

failure_policy:
  policy: abort
  worst_values: {}

reporting:
  save_protocol_copy: true
  save_manifest_copy: true
  save_environment: true
  save_alignment_transforms: false
  formats: [json, csv]
```

## Pose protocol

```yaml
schema_version: 1
protocol_version: 0.1.0
name: single_pose
fidelity: eval3r_native

dataset:
  dataset: custom
  variant: single_trajectory
  split: null

prediction_modality: camera_trajectory

ground_truth:
  modality: trajectory
  provenance: unknown
  independence: unknown
  density: trajectory_only
  unit: m
  source_pose_format: tum
  normalized_convention: cam_to_world_opencv_meters
  local_evaluation_status: supported

local_evaluation:
  status: supported
  reason: local trajectory files supplied by user
  public_gt_available: true
  external_assets_required: []
  official_server_required: false

alignment:
  mode: trajectory_sim3
  estimate_on: trajectory
  solver: evo
  granularity: per_scene
  allow_override: true
  parameters:
    associate_max_diff: 0.01

confidence:
  policy: none
  allow_override: false

masking:
  pred_culling:
    method: none
  gt_culling:
    method: none
  valid_region:
    method: none
  ignore_invalid_depth: true
  invalid_depth_values: []

sampling:
  pred:
    method: none
  gt:
    method: none

metrics:
  - name: ate
    statistic: rmse
  - name: rpe_translation
    statistic: rmse
  - name: rpe_rotation
    statistic: rmse
  - name: alignment_scale_error

aggregation:
  per_scene: true
  mean: true
  median: true
  weighted_mean: false
  weights: none

failure_policy:
  policy: abort
  worst_values: {}

reporting:
  save_protocol_copy: true
  save_manifest_copy: true
  save_environment: true
  save_alignment_transforms: true
  formats: [json, csv, markdown]

backend_preferences:
  trajectory: evo
```

## Protocol naming

Use descriptive names:

```text
<dataset>_<variant>_<metric_family>_<important_threshold_or_convention>
```

Examples:

```text
scannet_single_layer_geometry_5cm
scannet_double_layer_geometry_5cm
dtu_official_like_pointcloud
tanks_temples_training_official
eth3d_training_official_like
hypersim_depth
kitti360_pose
```

Avoid names that hide important protocol choices:

```text
bad: scannet_geometry
bad: dtu_eval
bad: replica_benchmark
```

## Override policy

Protocols may permit CLI overrides for quick experiments. Overrides must be recorded and should usually change the run status to non-canonical.

Rules:

```text
official protocols should generally disallow overrides
official_like protocols should disallow overrides unless the changed field is explicitly experimental
eval3r_native protocols may allow overrides
all overrides are written into config.yaml and results.json
```

## Protocol validation checklist

Before adding a built-in protocol, confirm:

```text
dataset variant is named
ground-truth provenance is recorded
local evaluability is recorded
pose source convention is recorded
metric definitions are explicit
thresholds are explicit
alignment is explicit
masking / culling is explicit
confidence policy is explicit
sampling policy is explicit
failure policy is explicit
backend preferences are explicit
expected result files are documented
tiny fixture exists
regression target exists if official_like or official
```
