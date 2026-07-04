# eval3r Schema

This file defines the core schema concepts for `eval3r`. The implementation should use Pydantic models so validation, JSON schema generation, and canonical hashing are available from the beginning.

The schema is intentionally explicit. Dataset names, pose conventions, ground-truth provenance, alignment, masking, confidence filtering, and failure policies should be recorded rather than inferred.

## Internal convention

All adapters should normalize data into a common internal convention before metric computation.

Recommended internal convention:

```text
pose: camera-to-world
camera axes: OpenCV-style, +X right, +Y down, +Z forward
linear unit: meters
point arrays: float64 or float32 Nx3
image origin: top-left for depth and mask arrays
```

Dataset-native conventions are still recorded separately. This prevents losing information about the original files.

## Enumerations

### Prediction modality

```python
PredictionModality = Literal[
    "mesh",
    "pointcloud",
    "pointmap",
    "single_depth",
    "depth_sequence",
    "camera_trajectory",
    "colmap_reconstruction",
]
```

### Ground-truth modality

```python
GroundTruthModality = Literal[
    "mesh",
    "pointcloud",
    "depth",
    "depth_sequence",
    "trajectory",
    "lidar",
    "server_only",
]
```

### Ground-truth provenance

```python
GTProvenance = Literal[
    "laser_scan",
    "synthetic_exact",
    "reconstructed",
    "lidar_sparse",
    "sensor_depth",
    "server_only",
    "unknown",
]
```

### Ground-truth independence

```python
GTIndependence = Literal[
    "independent",
    "reconstruction_derived",
    "sensor_derived",
    "server_only",
    "unknown",
]
```

### Ground-truth density

```python
GTDensity = Literal[
    "dense_surface",
    "sparse_lidar",
    "object_pointcloud",
    "depth_image",
    "trajectory_only",
    "server_only",
    "unknown",
]
```

### Source pose format

```python
SourcePoseFormat = Literal[
    "cam_to_world_opencv",
    "cam_to_world_opengl",
    "world_to_cam_opencv",
    "world_to_cam_opengl",
    "world_to_cam_colmap",
    "world_to_cam_mvsnet",
    "tanks_temples_log",
    "co3d_frame_annotations",
    "kitti360_cam0_to_world",
    "tum",
    "unknown",
]
```

### Normalized convention

```python
NormalizedConvention = Literal[
    "cam_to_world_opencv_meters",
]
```

### Scale type

```python
ScaleType = Literal[
    "metric",
    "relative",
    "unknown",
]
```

### Fidelity

```python
Fidelity = Literal[
    "official",
    "official_like",
    "eval3r_native",
    "server_only",
]
```

### Local evaluation status

```python
LocalEvaluationStatus = Literal[
    "supported",
    "server_only",
    "missing_public_gt",
    "requires_external_assets",
    "requires_external_renderer",
    "unsupported",
]
```

### Official local evaluation method

`official_local_eval` means that eval3r can run the dataset's declared official or official-like protocol locally for the requested split. It does not require every dataset to use the same mechanism. Some datasets use a literal official script wrapper; others use a validated port that matches the official protocol.

```python
OfficialLocalEvalMethod = Literal[
    "none",
    "official_script_wrapper",
    "validated_official_port",
    "matlab_official_script",
    "server_only",
]
```

## Dataset capabilities

```python
class DatasetCapabilities(BaseModel):
    dense_geometry: bool = False
    independent_gt: bool = False
    depth_metric: bool = False
    pose_metric: bool = False
    official_local_eval: bool = False
    official_local_eval_method: OfficialLocalEvalMethod = "none"
    server_only_eval: bool = False
    requires_external_renderer: bool = False
    requires_paid_assets: bool = False
    supports_full_scene_geometry: bool = False
    supports_object_centric_geometry: bool = False
    supports_sparse_lidar_geometry: bool = False
    notes: list[str] = []
```

Adapters expose this so the CLI can reject unsupported protocol / dataset combinations before running. `official_local_eval` tracks local support for the declared protocol on the requested split, not whether the dataset has a leaderboard server. The method field records whether eval3r uses a literal official script, a validated port, the MATLAB script, or no local official path.

## Dataset variant

A dataset name is often not enough. Replica with one rendered trajectory bundle is not the same benchmark as Replica with another. ScanNet single-layer and double-layer mesh conventions are not equivalent.

```python
class DatasetVariant(BaseModel):
    dataset: str
    variant: str | None = None
    split: str | None = None
    version: str | None = None
    fingerprint: str | None = None
    notes: list[str] = []
```

Examples:

```yaml
dataset: scannet
variant: scannetv2_val_single_layer
split: val
version: scannetv2
```

```yaml
dataset: replica
variant: nice_slam_8scene_rendered
split: test
fingerprint: sha256:...
```

## Ground-truth specification

```python
class GroundTruthSpec(BaseModel):
    modality: GroundTruthModality
    provenance: GTProvenance
    independence: GTIndependence
    density: GTDensity
    path: Path | None = None
    fingerprint: str | None = None
    unit: str | None = "m"
    source_pose_format: SourcePoseFormat | None = None
    normalized_convention: NormalizedConvention | None = "cam_to_world_opencv_meters"
    local_evaluation_status: LocalEvaluationStatus = "supported"
    server_url: str | None = None
    notes: list[str] = []
```

### Provenance versus independence

`provenance` records how the reference was produced. `independence` records whether that reference is independent of the evaluated prediction for the purpose of interpreting the score. They often correlate, but they are not the same field.

Examples:

```text
DTU:
  provenance = laser_scan
  independence = independent

ScanNet:
  provenance = reconstructed
  independence = reconstruction_derived

KITTI-360 sparse LiDAR comparison:
  provenance = lidar_sparse
  independence = sensor_derived

A method evaluated against its own exported depth rendered into a diagnostic view:
  provenance = sensor_depth or reconstructed
  independence = unknown or reconstruction_derived
```

Keep both fields because `provenance` describes the data source while `independence` controls the warning shown in reports.

This block is included in both protocols and results.

## Local evaluation specification

```python
class LocalEvaluationSpec(BaseModel):
    status: LocalEvaluationStatus
    reason: str | None = None
    public_gt_available: bool = True
    external_assets_required: list[str] = []
    official_server_required: bool = False
```

Examples:

```yaml
local_evaluation:
  status: supported
  reason: DTU public GT and ObsMask / Plane files are available; eval3r uses a validated official-like local path.
  public_gt_available: true
  external_assets_required: []
  official_server_required: false
```

```yaml
local_evaluation:
  status: server_only
  reason: Tanks and Temples intermediate GT is withheld
  public_gt_available: false
  official_server_required: true
```

## Reconstruction

```python
class Reconstruction(BaseModel):
    path: Path | None = None
    modality: PredictionModality
    coordinate_frame: str
    source_pose_format: SourcePoseFormat = "unknown"
    normalized_convention: NormalizedConvention = "cam_to_world_opencv_meters"
    scale: ScaleType
    unit: str = "m"
    depth_unit: float | None = None
    cameras: Path | None = None
    intrinsics_source: Literal["predicted", "gt", "dataset_default", "unknown"] = "unknown"
    confidence_paths: list[Path] | None = None
    native_confidence_threshold: float | None = None
    metadata: dict = {}
```

Rules:

```text
source_pose_format records what was found on disk
normalized_convention records what the adapter emitted internally
unit is meters unless explicitly declared otherwise
depth_unit is required for integer depth files
confidence metadata is required when confidence is used or self-filtered
```

## Scene data

```python
class SceneData(BaseModel):
    scene_id: str
    dataset: str
    variant: str | None = None
    rgb_paths: list[Path] | None = None
    depth_paths: list[Path] | None = None
    camera_paths: list[Path] | None = None
    gt_mesh: Path | None = None
    gt_pointcloud: Path | None = None
    gt_depth_paths: list[Path] | None = None
    gt_trajectory: Path | None = None
    visibility_source: Path | None = None
    masks: dict[str, Path] = {}
    ground_truth: GroundTruthSpec
    capabilities: DatasetCapabilities | None = None
    metadata: dict = {}
```

## Prediction manifest

```python
class UsesGTSpec(BaseModel):
    pose: bool = False
    intrinsics: bool = False
    scale: bool = False

class ConfidenceManifestSpec(BaseModel):
    present: bool = False
    native_threshold: float | None = None
    self_filtered: bool = False

class ScenePredictionEntry(BaseModel):
    mesh: Path | None = None
    pointcloud: Path | None = None
    pointmap: Path | None = None
    depth: Path | None = None
    depth_dir: Path | None = None
    trajectory: Path | None = None
    camera_file: Path | None = None
    confidence: Path | None = None
    confidence_dir: Path | None = None
    metadata: dict = {}

class PredictionManifest(BaseModel):
    method: str
    version: str | None = None
    dataset: DatasetVariant
    prediction_modality: PredictionModality
    coordinate_frame: str
    source_pose_format: SourcePoseFormat = "unknown"
    normalized_convention: NormalizedConvention = "cam_to_world_opencv_meters"
    scale: ScaleType
    unit: str = "m"
    depth_unit: float | None = None
    uses_gt: UsesGTSpec = UsesGTSpec()
    intrinsics_source: Literal["predicted", "gt", "dataset_default", "unknown"] = "unknown"
    confidence: ConfidenceManifestSpec = ConfidenceManifestSpec()
    scenes: dict[str, ScenePredictionEntry]
    metadata: dict = {}
```

## Alignment schema

```python
class AlignmentSpec(BaseModel):
    mode: Literal[
        "none",
        "se3",
        "sim3",
        "icp",
        "trajectory_se3",
        "trajectory_sim3",
        "scale_median",
        "scale_least_squares",
        "scale_affine",
    ] = "none"
    estimate_on: Literal[
        "none",
        "trajectory",
        "pointcloud",
        "depth",
        "manual",
    ] = "none"
    solver: Literal[
        "none",
        "umeyama",
        "ransac_umeyama",
        "icp",
        "evo",
        "official_backend",
    ] = "none"
    granularity: Literal["global", "per_scene", "per_sequence", "per_frame"] = "per_scene"
    allow_override: bool = False
    parameters: dict = {}
```

The `scale_median`, `scale_least_squares`, and `scale_affine` modes are the depth scale-alignment modes from `.agent/metrics.md`. Depth protocols must declare scale alignment through `mode` and `granularity`, never through free-form `parameters`, so the choice is visible, hashed, and reported.

## Sampling schema

```python
class SamplingSideSpec(BaseModel):
    method: Literal[
        "none",
        "surface_area",
        "uniform_points",
        "voxel_downsample",
        "random_points",
        "all_points",
    ] = "all_points"
    n_points: int | None = None
    seed: int | Literal["derive"] | None = "derive"
    voxel_size: float | None = None
    return_normals: bool = False

class SamplingSpec(BaseModel):
    pred: SamplingSideSpec
    gt: SamplingSideSpec
```

## Masking and culling schema

```python
class CullingSpec(BaseModel):
    method: Literal[
        "none",
        "scene_bounds",
        "visibility_mask",
        "gt_visibility",
        "obs_mask",
        "dataset_official_mask",
        "object_mask",
        "valid_depth",
        "custom",
    ] = "none"
    tolerance: float | None = None
    source: Literal["gt", "pred", "dataset", "custom", "none"] = "none"
    path: Path | None = None
    parameters: dict = {}

class MaskingSpec(BaseModel):
    pred_culling: CullingSpec = CullingSpec()
    gt_culling: CullingSpec = CullingSpec()
    valid_region: CullingSpec = CullingSpec()
    ignore_invalid_depth: bool = True
    invalid_depth_values: list[float] = []
```

The `valid_depth` culling method restricts a depth protocol's `valid_region` to
pixels with valid ground-truth depth (used by the `single_depth` protocol in
`.agent/protocols.md`). It complements the pixel-value masking done by
`ignore_invalid_depth` / `invalid_depth_values`.

## Confidence schema

```python
class ConfidenceSpec(BaseModel):
    policy: Literal[
        "none",
        "method_default",
        "threshold",
        "percentile",
        "top_k_fraction",
    ] = "none"
    threshold: float | None = None
    percentile: float | None = None
    top_k_fraction: float | None = None
    allow_override: bool = False
```

## Metric schema

```python
class MetricSpec(BaseModel):
    name: str
    threshold: float | None = None
    statistic: Literal["mean", "median", "rmse", "percentile"] | None = None
    percentile: float | None = None
    reduction: Literal["sum", "mean", "none"] | None = None
    clamp: float | None = None
    aggregation: Literal[
        "per_scene_then_mean",
        "global",
        "per_frame_then_scene_then_mean",
    ] | None = None
    parameters: dict = {}
```

## Aggregation schema

```python
class AggregationSpec(BaseModel):
    per_frame: bool = False
    per_scene: bool = True
    mean: bool = True
    median: bool = True
    weighted_mean: bool = False
    weights: Literal["none", "n_points", "n_pixels", "scene_weight"] = "none"
```

## Failure policy schema

```python
class FailurePolicySpec(BaseModel):
    policy: Literal["abort", "skip_and_flag", "score_worst"] = "abort"
    worst_values: dict[str, float] = {}
```

## Reporting schema

```python
class ReportingSpec(BaseModel):
    save_protocol_copy: bool = True
    save_manifest_copy: bool = True
    save_environment: bool = True
    save_alignment_transforms: bool = True
    save_colored_errors: bool = False
    save_distance_histogram: bool = False
    formats: list[Literal["json", "csv", "markdown", "latex", "html"]] = ["json", "csv"]
```

## Evaluation protocol

```python
class EvalProtocol(BaseModel):
    schema_version: int
    protocol_version: str
    name: str
    fidelity: Fidelity
    dataset: DatasetVariant
    prediction_modality: PredictionModality
    ground_truth: GroundTruthSpec
    local_evaluation: LocalEvaluationSpec
    alignment: AlignmentSpec
    confidence: ConfidenceSpec
    masking: MaskingSpec
    sampling: SamplingSpec
    metrics: list[MetricSpec]
    aggregation: AggregationSpec
    failure_policy: FailurePolicySpec
    reporting: ReportingSpec
    backend_preferences: dict[str, str] = {}
    notes: list[str] = []
```

`backend_preferences` keys must be backend registry kinds as defined in `.agent/backends.md` (`mesh`, `pointcloud`, `nearest_neighbor`, `registration`, `trajectory`, `camera`, `depth_io`, `official_eval`). Values are backend names registered under that kind.

## Metric result

```python
class MetricResult(BaseModel):
    name: str
    value: float | None
    unit: str | None = None
    threshold: float | None = None
    statistic: str | None = None
    reduction: str | None = None
    scene_id: str | None = None
    frame_id: str | None = None
    protocol: str
    protocol_hash: str
    backend: str | None = None
    n_points_pred: int | None = None
    n_points_gt: int | None = None
    n_pixels_valid: int | None = None
    valid_fraction: float | None = None
    culled_fraction: float | None = None
    metadata: dict = {}
```

## Scene failure

```python
class SceneFailure(BaseModel):
    scene_id: str
    stage: Literal[
        "resolve",
        "load",
        "normalize",
        "align",
        "mask",
        "sample",
        "metric",
        "aggregate",
        "report",
    ]
    reason: str
    traceback: str | None = None
    recoverable: bool = True
```

## Run result

```python
class RunResult(BaseModel):
    schema_version: int
    eval3r_version: str
    method: str | None = None
    method_version: str | None = None
    dataset: DatasetVariant
    split: str | None = None
    protocol: str
    protocol_version: str
    protocol_hash: str
    fidelity: Fidelity
    ground_truth: GroundTruthSpec
    local_evaluation: LocalEvaluationSpec
    n_scenes_expected: int
    n_scenes_evaluated: int
    failed_scenes: list[SceneFailure] = []
    failure_policy: FailurePolicySpec
    metrics: dict[str, float | None]
    metric_definitions: list[MetricSpec] = []
    per_scene_metrics: list[MetricResult] = []
    confidence_policy: ConfidenceSpec
    alignment: AlignmentSpec
    masking: MaskingSpec
    sampling: SamplingSpec
    aggregation: AggregationSpec
    uses_gt: UsesGTSpec | None = None
    backend_versions: dict[str, Any] = {}  # kind -> metadata dict or version string
    environment: dict = {}
    command: str | None = None
    manifest_path: Path | None = None
    protocol_path: Path | None = None
    git_commit: str | None = None
    timestamp: str
    metadata: dict = {}
```

This model carries every field that `.agent/reproducibility.md` requires in `results.json`, including the metric definitions, aggregation, and failure policy actually used, so a result file is self-describing without the original protocol file.

## Canonical hashing

Protocol hashing must be stable across YAML formatting differences.

Process:

```text
parse YAML into EvalProtocol
validate with Pydantic
serialize to JSON with sorted keys
remove comments and formatting-only information
encode as UTF-8
sha256 hash the bytes
prefix with sha256:
```

Changing any field that changes evaluation behavior must change the hash.

## Required result fields

No benchmark result should be considered complete without:

```text
schema_version
protocol name and hash
dataset name, variant, split
ground_truth provenance, independence, density, fingerprint
local evaluation status
prediction modality
alignment spec
masking / culling spec
sampling spec
metric definitions
confidence policy
failure policy
scene coverage
backend versions
command and environment
```
