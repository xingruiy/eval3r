# eval3r Project Plan

## Project goal

`eval3r` is a plain 3D reconstruction evaluation library. Its job is to make evaluation runs explicit, repeatable, and easy to inspect. It should help users compare reconstruction outputs such as meshes, point clouds, depth predictions, and camera trajectories against dataset-specific references under named protocols.

The library should stay focused. It should not become a geometry-processing framework, a dataset conversion warehouse, a SLAM system, a rendering engine, or a reconstruction method implementation.

The central idea is simple:

> `eval3r` loads predictions and references, normalizes dataset conventions, applies an explicit protocol, computes metrics, and records enough metadata for the result to be understood later.

## Scope

The library supports evaluation of these prediction types:

```text
mesh
pointcloud
pointmap
single_depth
depth_sequence
camera_trajectory
colmap_reconstruction
```

Depth sequences are supported for depth metrics and per-frame aggregation. They are not converted into scene reconstructions by the library.

Pointmaps are supported when they can be interpreted as explicit 3D points in a declared coordinate frame. If a method outputs per-frame pointmaps, the protocol must say whether the metric is per-frame, per-view, object-centric, or already scene-frame. The library does not build a scene reconstruction by integrating RGB-D or depth frames.

## Non-goals

The library does not provide:

```text
TSDF integration
RGB-D fusion
volumetric fusion
online mapping
SLAM tracking
mesh repair as a default preprocessing step
learned reconstruction models
large-scale dataset download management
full replacement of official benchmark servers
```

The library may wrap official evaluation scripts where that is the safest way to match a dataset's established protocol. Tanks and Temples is the clearest example.

## Design principles

### Protocols are the source of evaluation meaning

A metric number is not meaningful without knowing the protocol that produced it. The protocol specifies alignment, masking, sampling, thresholds, metric definitions, aggregation, failure handling, and dataset variant.

Metrics must not silently choose alignment, culling, scale handling, thresholds, confidence filtering, or sampling behavior.

### Dataset adapters normalize, not reinterpret

Dataset adapters are responsible for loading dataset files, parsing native conventions, resolving ground truth, and normalizing data into internal eval3r conventions. They should not implement generic geometry algorithms.

For example, a ScanNet adapter knows how to read an exported ScanNet scene, pose files, intrinsics, and GT mesh variants. It does not own nearest-neighbor search or mesh sampling.

### Backends do commodity work

The library delegates common operations to mature tools:

```text
mesh loading and surface sampling       trimesh or Open3D
point cloud loading                     Open3D, plyfile, numpy
nearest-neighbor search                 scipy, Open3D
trajectory metrics                      evo
COLMAP parsing                          pycolmap
camera model parsing beyond pinhole     pycolmap or dataset-specific parser
official benchmark behavior             official script wrapper when appropriate
```

All dependencies are required and always installed; there is no optional-extra system. This is a research-oriented project where inspectability outranks install footprint.

### Results should expose uncertainty and incomparability

Many datasets called “ground truth” are not independent measurements. Some are laser scans, some are synthetic meshes, some are reconstructed meshes, some are sparse LiDAR, and some are server-only. The result schema must record this.

A result against DTU's laser-scanned point cloud and a result against CO3D's COLMAP-derived point cloud are not the same kind of evidence. The library should make that visible rather than hiding it behind one generic “geometry score.”

### Local evaluation support must be honest

Some splits are locally evaluable because GT is public. Other splits can only be evaluated through a benchmark server. Some datasets require paid assets or external rendered trajectory bundles. The dataset adapter and protocol should state this explicitly.

## Architecture

The library has five main layers:

```text
CLI / Python API
        ↓
Evaluation runner
        ↓
Dataset adapter + prediction manifest + protocol
        ↓
Metric layer
        ↓
Backend layer
```

The runner follows a single staged pipeline:

```text
resolve → load → normalize → align → mask/cull → sample → metric → aggregate → report
```

The pipeline stages are configured by the protocol. Geometry, depth, and pose evaluations should reuse the same runner where possible so failure handling, logging, metadata capture, and result writing do not drift across separate implementations.

## Repository layout

Recommended package structure:

```text
eval3r/
  __init__.py

  core/
    types.py
    schema.py
    protocol.py
    manifest.py
    result.py
    ground_truth.py
    pose_convention.py
    local_eval.py
    registry.py
    errors.py
    logging.py
    hashing.py

  datasets/
    __init__.py
    base.py
    capabilities.py
    conventions.py
    variants.py
    scannet.py
    dtu.py
    tanks_temples.py
    eth3d.py
    seven_scenes.py
    neural_rgbd.py
    replica.py
    hypersim.py
    co3d.py
    blendedmvs.py
    kitti360.py
    custom.py

  protocols/
    __init__.py
    registry.py
    loader.py
    builtin/
      single_geometry.yaml
      single_depth.yaml
      single_pose.yaml
      scannet_single_layer_geometry_5cm.yaml
      scannet_double_layer_geometry_5cm.yaml
      dtu_official_like_pointcloud.yaml
      tanks_temples_training_official.yaml
      eth3d_training_official.yaml

  metrics/
    __init__.py
    geometry.py
    depth.py
    pose.py
    diagnostics.py
    aggregation.py

  backends/
    __init__.py
    mesh_trimesh.py
    mesh_open3d.py
    pointcloud_open3d.py
    pointcloud_plyfile.py
    nn_scipy.py
    nn_open3d.py
    registration_open3d.py
    trajectory_evo.py
    camera_pycolmap.py
    depth_imageio.py
    depth_opencv.py
    tnt_official.py
    dtu_eval.py

  pipeline/
    __init__.py
    runner.py
    stages/
      resolve.py
      load.py
      normalize.py
      align.py
      mask.py
      sample.py
      metric.py
      aggregate.py
      report.py

  reports/
    __init__.py
    table.py
    csv.py
    json.py
    markdown.py
    html.py
    latex.py
    plots.py
    diff.py

  cli/
    __init__.py
    main.py
    metric.py
    benchmark.py
    dataset.py
    protocol.py
    diff.py
    inspect.py
    vis.py

tests/
  unit/
  integration/
  fixtures/
  regression/

docs/
  index.md
  install.md
  quickstart.md
  schema.md
  protocols.md
  datasets.md
  metrics.md
  backends.md
  reproducibility.md
  fidelity.md
  confidence.md
  failure_policy.md
  examples/
```

Note: `docs/` is the mkdocs **user documentation** of the package (written in task 017). The **planning documents** (this file and its siblings) live in `.agent/` and remain the source of truth for design decisions; if they and the user docs disagree, fix both in the same change.

## Dataset support strategy

Dataset adapters should be introduced in tiers. The goal is not to support every named dataset immediately. The goal is to support each dataset only when the library can state what the GT means and what evaluation is actually possible.

The tiers below are not mutually exclusive. They separate two questions:

```text
implementation priority: what should be built first?
ground-truth honesty: what does the reference actually represent?
```

This plan intentionally uses an engineering-driven build order. Because fused reconstruction support is out of scope, the first adapters should exercise geometry metrics, official-like point-cloud evaluation, result metadata, and dataset convention normalization with as little extra machinery as possible.

### Core dense geometry implementation targets

These are the most useful early targets for mesh and point-cloud evaluation.

```text
DTU
Tanks and Temples training split
ETH3D training split
ScanNet validation split
```

DTU has laser-scanned GT point clouds and shipped ObsMask / Plane files. It is a strong early adapter because the official-like behavior is well defined and can be regression-tested.

Tanks and Temples should be implemented through an official script wrapper for the public training split. Intermediate and advanced splits are server-only and should be marked as not locally evaluable.

ETH3D should use COLMAP text camera models and pycolmap where needed. Non-pinhole camera cases should not be treated as minimal pinhole without a warning.

ScanNet should support geometry evaluation against the appropriate GT mesh variant. Protocols must distinguish single-layer and double-layer conventions. Visibility culling is protocol-defined and should never be silently enabled. ScanNet is early because it is important for indoor reconstruction evaluation, not because its GT is independently measured.

### Reconstruction-derived GT datasets

These datasets are useful but need careful result labeling because the reference geometry is reconstructed, fused, COLMAP-derived, or rendered from a reconstructed mesh. This is not an indoor/outdoor grouping.

```text
ScanNet
7-Scenes
Neural-RGBD real scenes
CO3D
BlendedMVS
```

7-Scenes has depth, poses, and no official calibration file. Protocols should pin the nominal Kinect intrinsics and record invalid-depth handling.

Neural-RGBD uses OpenGL-style conventions and has released culled meshes for some comparisons. The adapter should record whether a pre-culled mesh or another mesh source is used.

CO3D is object-centric and stores camera information in `frame_annotations.jgz`, not plain pose files. Its per-sequence point cloud is COLMAP-derived, so geometry protocols should be labeled as eval3r-native rather than official-like.

BlendedMVS is a mixed indoor/outdoor and object/scene dataset. It provides rendered depth from reconstructed meshes and MVSNet-style camera files, so it is useful for depth or MVS-style experiments but should not be treated as an independent dense GT benchmark.

### Synthetic controlled datasets

These can be valuable when the exact benchmark variant is pinned.

```text
Replica
Hypersim
Neural-RGBD synthetic scenes
```

Replica is a scene-asset dataset, not a native camera/depth sequence benchmark. Any rendered trajectory/depth benchmark variant must name the release it uses and fingerprint it.

Hypersim provides rich synthetic data, but full mesh geometry requires separate assets. Its depth is Euclidean ray distance, not ordinary z-depth, and asset units must be converted to meters per scene.

### Pose and sparse-LiDAR oriented datasets

These should not be presented as dense reconstruction benchmarks by default.

```text
KITTI-360
```

KITTI-360 is better suited to trajectory metrics and sparse point comparisons. Any point-cloud protocol should clearly state that it compares against sparse LiDAR rather than dense surface GT. KITTI-360's perspective cameras can use a pinhole path, but its fisheye cameras require a fisheye-capable camera backend or must be out of scope for the protocol.

## Dataset capability model

Each adapter should declare capabilities:

```text
dense_geometry
independent_gt
depth_metric
pose_metric
official_local_eval
official_local_eval_method
server_only_eval
requires_external_renderer
requires_paid_assets
supports_full_scene_geometry
supports_object_centric_geometry
supports_sparse_lidar_geometry
```

The CLI should inspect these capabilities before running a benchmark. If a requested protocol is not locally evaluable, the command should fail clearly rather than producing a misleading local result. `official_local_eval` means that the declared official or official-like protocol can run locally for the requested split; `official_local_eval_method` records whether that is through an official script wrapper, a validated port, the MATLAB script, or no local path.

## Built-in adapter implications

### ScanNet

Responsibilities:

```text
resolve official train / val / test scene IDs
read exported color / depth / pose / intrinsic layout
record depth_unit = 0.001 for mm-stored depth
normalize camera-to-world OpenCV-style poses
resolve GT mesh variant
support single-layer and double-layer geometry protocols
support protocol-defined visibility culling when requested
hash the GT mesh and scene list
```

The adapter should not assume one ScanNet protocol is canonical. The protocol name should reveal the mesh-layer convention.

### DTU

Responsibilities:

```text
load stlXXX_total.ply laser-scan GT point cloud
load ObsMaskXXX_10.mat
load PlaneXXX.mat when available
record missing plane files explicitly
handle native millimeter units
resolve conventional prediction filenames such as <method>XXX_l3.ply
wrap or port official-like point-cloud evaluation behavior
record whether the local evaluator is a validated Python port, MATLAB script, or another backend
```

DTU geometry should be point-cloud based. Mesh evaluation on DTU predictions is possible only after sampling or conversion, but the GT reference remains a point cloud.

### Tanks and Temples

Responsibilities:

```text
resolve training-set GT point cloud
resolve COLMAP / .log trajectory files
resolve alignment transform
resolve crop volume JSON
call official evaluation backend
record per-scene thresholds
mark intermediate / advanced as server-only
```

The local official protocol should only be claimed for public GT scenes.

### ETH3D

Responsibilities:

```text
parse COLMAP text cameras
load training-set GT scans / depth when available
respect occlusion-aware GT preparation
use pycolmap for nontrivial camera models
mark test scenes as server-only
```

Minimal pinhole fallback should be allowed only for scenes whose camera model actually fits that assumption.

### 7-Scenes

Responsibilities:

```text
load TrainSplit.txt and TestSplit.txt
load frame color / depth / pose files
mask invalid depth value 65535
record nominal Kinect intrinsics used by the protocol
record that GT geometry is reconstruction-derived
```

Without a pinned GT mesh or TSDF-derived reference release, geometry protocols should remain eval3r-native.

### Neural-RGBD

Responsibilities:

```text
load images, depth variants, focal.txt, poses.txt
normalize OpenGL-style poses
record which depth variant is used
record whether released culled meshes are used
```

The adapter should treat synthetic and real scenes differently in GT provenance.

### Replica

Responsibilities:

```text
load raw scene mesh when used directly
record semantic / glass filtering if applied
require a named rendered trajectory/depth bundle for sequence benchmarks
fingerprint both mesh and trajectory bundle when used
```

Replica has no single native camera trajectory. A benchmark variant must define one. Before treating a NICE-SLAM / iMAP-style rendered release as built-in, verify the exact release, trajectory convention, depth convention, and file layout against its primary repository.

### Hypersim

Responsibilities:

```text
load HDF5 depth / position / normal files
convert Euclidean ray distance to z-depth when needed
apply meters_per_asset_unit
use corrected camera parameters where required
record mesh asset availability
```

Depth-only protocols should be allowed even when mesh assets are unavailable.

### CO3D

Responsibilities:

```text
parse frame_annotations.jgz and sequence_annotations.jgz
load pointcloud.ply and object masks
record COLMAP-derived GT provenance
support pose-focused or object-centric eval3r-native protocols
```

CO3D should not be a core dense geometry benchmark.

### BlendedMVS

Responsibilities:

```text
parse MVSNet-style cam.txt files
invert world-to-camera extrinsics into normalized internal poses
load rendered PFM depths
record invalid or empty rendered depth frames
record reconstruction-derived GT provenance
```

Before freezing fixtures or parser tests, verify the exact MVSNet `cam.txt` block order and extrinsic direction against the primary MVSNet / BlendedMVS parser.

### KITTI-360

Responsibilities:

```text
load calibration and pose files
handle missing pose entries explicitly
support pose metrics first
label sparse LiDAR comparisons as sparse, not dense-surface geometry
limit minimal-pinhole support to perspective cameras unless a fisheye-capable backend is used
```

## Metrics

### Geometry metrics

Input:

```text
pred_points: Nx3
gt_points: Mx3
thresholds: list[float]
```

Supported metrics:

```text
accuracy
completeness
chamfer
precision@τ
recall@τ
fscore@τ
distance percentiles
coverage
optional normal consistency
```

Definitions are pinned by protocol:

```text
accuracy      = statistic NN distance pred → gt
completeness  = statistic NN distance gt → pred
chamfer       = accuracy + completeness, or mean of the two, as declared
precision@τ   = fraction of pred points with distance to gt < τ
recall@τ      = fraction of gt points with distance to pred < τ
fscore@τ      = 2PR / (P + R), with aggregation order declared
```

Important rules:

```text
mean vs median must be explicit
Chamfer sum vs mean must be explicit
thresholds must be explicit
F-score aggregation order must be explicit
surface meshes must be sampled before point-distance metrics
raw mesh vertices are not used unless the protocol explicitly requests that behavior
```

### Mesh metrics

Mesh evaluation is implemented by loading the mesh, sampling the surface, and then running geometry metrics on sampled points.

The protocol controls:

```text
sample count
sample method
sample seed
normal sampling
masking / culling
alignment
```

### Point-cloud metrics

Point-cloud evaluation loads points, validates shape, removes invalid values, applies masks and culling, and computes geometry metrics.

Optional downsampling must be protocol-defined and recorded.

### Depth metrics

Input:

```text
pred_depth
gt_depth
valid_mask
optional confidence
```

Supported metrics:

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

Scale handling must be explicit, declared through the protocol's alignment spec (`scale_median`, `scale_least_squares`, `scale_affine`), never through free-form parameters:

```text
none
scale_median
scale_least_squares
scale_affine
```

Granularity must be explicit:

```text
per_frame
per_sequence
per_scene
```

Depth sequences are aggregated per frame and per scene. They are not converted into a scene reconstruction by eval3r.

### Pose metrics

Pose metrics should delegate to evo where available.

Supported metrics:

```text
ATE
RPE
rotation error
translation error
SE3-aligned trajectory error
Sim3-aligned trajectory error
```

Pose association, timestamp handling, interpolation, and alignment are protocol-defined.

### Diagnostic metrics

Diagnostic metrics can help explain evaluation behavior:

```text
alignment scale error |log(s)|
alignment residual statistics
valid fraction
culled fraction
confidence-vs-error summaries
scene coverage
failure coverage
```

These are not substitutes for primary metrics, but they should appear in results when available.

## Alignment policy

Alignment is always explicit.

Supported modes:

```text
none
se3
sim3
icp
trajectory_se3
trajectory_sim3
```

The protocol also declares where the transform is estimated:

```text
trajectory
pointcloud
depth
manual
none
```

Rules:

```text
ICP never runs by default
Sim3 and other scale-resolving choices are user/prediction adaptation choices, not protocol permission gates
alignment parameters and transforms are saved
alignment backend is saved
alignment residuals are saved when available
```

## Masking and culling policy

Masking and culling are protocol-level decisions.

Supported policies:

```text
none
valid_depth
scene_bounds
object_mask
visibility_mask
obs_mask
dataset_official_mask
custom_mask
```

The dataset adapter resolves official masks and observability data. The metric layer receives already-valid points or pixels.

If a protocol requires a mask and the mask is missing, the scene should fail according to the protocol failure policy. It should not silently fall back to no mask.

## Confidence policy

Predictions may include confidence values or confidence maps. Confidence handling is a protocol decision, not a method-specific hidden threshold.

Supported policies:

```text
none
method_default
threshold
percentile
top_k_fraction
```

If a method self-filters predictions before export, the manifest must declare that. Reports should distinguish uniform protocol filtering from method-native filtering.

## Failure accounting

Every benchmark result must record:

```text
number of expected scenes
number of evaluated scenes
failed scenes
failure reasons
failure policy
whether aggregate metrics are partial
```

Supported failure policies:

```text
abort
skip_and_flag
score_worst
```

Partial aggregates must be labeled visibly in JSON, CSV, Markdown, LaTeX, and HTML reports.

## Prediction manifest

Benchmark predictions should be described by a manifest. For simple single-file evaluation, the manifest can be omitted.

Example:

```yaml
method: example_method
version: 2026-07-04
dataset:
  name: scannet
  variant: scannetv2_val
split: val
prediction_modality: mesh
coordinate_frame: world
source_pose_format: cam_to_world_opencv
normalized_convention: cam_to_world_opencv
scale: metric
uses_gt:
  pose: false
  intrinsics: false
  scale: false
intrinsics_source: predicted
confidence:
  present: false
  native_threshold: null
scenes:
  scene0000_00:
    mesh: scene0000_00/recon.ply
  scene0001_00:
    mesh: scene0001_00/recon.ply
```

Rules:

```text
benchmark manifests should declare modality, scale, coordinate frame, pose convention, and GT usage
method version should be recorded when known
scene entries should be explicit for benchmark runs
manifests are copied into the run directory
```

Task 019 defines the official eval3r-native on-disk layout for these manifests
(`PredictionWriter` / `read_prediction_dir` / `e3r prediction validate|show`):
canonical per-scene filenames, relative paths, and sha256 fingerprints recorded in
the `metadata` dicts. See the layout note in `.agent/schema.md` ("eval3r-native
prediction layout") and `docs/prediction_format.md`.

## Result directory

Each run creates a directory:

```text
runs/
  2026-07-04_153000_scannet_method/
    results.json
    results.csv
    per_scene.csv
    failures.json
    protocol.yaml
    manifest.yaml
    config.yaml
    environment.json
    backend_versions.json
    alignment_transforms.json
    logs.txt
    results.md
    results.tex
    report.html
    debug/
      debug_outputs.json
      scene0000_00_error.ply
      scene0000_00_histogram.png
```

The result must include:

```text
schema version
method
dataset and dataset variant
split
protocol name and hash
fidelity label
ground-truth provenance
local evaluability status
scene counts
failure accounting
metric definitions
alignment policy
masking / culling policy
confidence policy
sampling policy
backend names and versions
command line
python and platform information
git commit when available
```

## CLI

### Single geometry evaluation

```bash
e3r metric geometry pred.ply \
  --gt gt.ply \
  --threshold 0.05 \
  --sample 200000 \
  --input pointcloud \
  --gt-input pointcloud
```

### Mesh evaluation

```bash
e3r metric geometry pred_mesh.ply \
  --gt gt_mesh.ply \
  --input mesh \
  --gt-input mesh \
  --sample 200000 \
  --threshold 0.05
```

### Depth evaluation

```bash
e3r metric depth pred_depth.png \
  --gt gt_depth.png \
  --depth-unit 0.001 \
  --align scale_median \
  --align-granularity per_frame
```

### Pose evaluation

```bash
e3r metric pose pred_tum.txt \
  --gt gt_tum.txt \
  --backend evo \
  --align sim3
```

### Dataset benchmark

```bash
e3r benchmark run preds/ \
  --dataset scannet \
  --split val \
  --protocol scannet_single_layer_geometry_5cm
```

### Dataset inspection

```bash
e3r dataset inspect scannet --root /data/scannet
```

### Protocol inspection

```bash
e3r protocol show scannet_single_layer_geometry_5cm
```

### Prediction validation

```bash
e3r benchmark validate preds/ \
  --dataset scannet \
  --split val \
  --protocol scannet_single_layer_geometry_5cm
```

### Result diffing

```bash
e3r diff runs/method_a runs/method_b
```

`e3r diff` should refuse to compare runs with different protocol hashes unless the user explicitly requests a loose comparison.

### CLI output

This is a research tool, so the CLI must be verbose and self-explaining, using `rich`:

```text
before running, echo the resolved configuration: protocol name and hash, dataset and variant, split, prediction manifest, and the alignment / masking / sampling / confidence / failure policies in effect
echo resolved input and output paths verbatim
show per-scene progress and per-scene outcomes
on failure, print the full reason verbatim (with traceback when available), never just a non-zero exit
partial scene coverage and skipped scenes must be visible, not hidden behind an average
```

## Python API

Basic geometry use:

```python
from eval3r import evaluate_geometry

result = evaluate_geometry(
    pred="pred.ply",
    gt="gt.ply",
    input_type="mesh",
    gt_type="mesh",
    threshold=0.05,
    sample=200_000,
)
print(result)
```

Benchmark use:

```python
from eval3r import run_benchmark

results = run_benchmark(
    pred_root="outputs/method_scannet",
    dataset="scannet",
    split="val",
    protocol="scannet_single_layer_geometry_5cm",
)
```

Protocol loading:

```python
from eval3r import load_protocol

protocol = load_protocol("scannet_single_layer_geometry_5cm")
```

Result comparison:

```python
from eval3r import diff_runs

report = diff_runs("runs/method_a", "runs/method_b")
```

## Dependency policy

All dependencies are required and always installed. There is no optional-extra system.

```toml
dependencies = [
  "numpy",
  "scipy",
  "pandas",
  "pyyaml",
  "typer",
  "rich",
  "pydantic",
  "trimesh",
  "plyfile",
  "open3d",
  "evo",
  "pycolmap",
  "imageio",
  "opencv-python",
  "pyrender",
  "matplotlib",
]
```

`pyrender` powers only the `visibility` backend's offscreen depth rendering for
evaluation-time visibility culling (ScanNet single-/double-layer); it needs a headless GL
context (`PYOPENGL_PLATFORM=egl`).

Rules:

```text
a plain 'pip install eval3r' pulls in every backend; there are no extras
the project does not depend on PyTorch, FAISS, or Waymo tooling (no torch/FAISS NN backend, no Waymo adapter)
MATLAB is the only optional external tool, used solely for the DTU MATLAB official-script path; a missing MATLAB fails with an explicit message and points at the validated Python port
backend names and versions are recorded in result metadata
default backend preferences use trimesh/plyfile and scipy, using Open3D only where a protocol asks for it
```

## Testing strategy

### Unit tests

Test:

```text
metric formulas
Chamfer sum vs mean definitions
mean vs median reductions
precision / recall / F-score edge cases
F-score aggregation order
empty point clouds
NaN / Inf handling
protocol loading and hashing
manifest validation
backend registry
result serialization
failure policy behavior
```

### Synthetic geometry tests

Use small known geometries:

```text
identical point clouds
translated point clouds
scaled point clouds
partially overlapping point clouds
empty prediction
empty ground truth
single point
plane
cube mesh
sphere mesh
```

Expected outputs should be analytically checkable.

### Dataset fixture tests

Use tiny fixtures, not full datasets:

```text
tests/fixtures/scannet_tiny/
tests/fixtures/dtu_tiny/
tests/fixtures/tanks_temples_tiny/
tests/fixtures/eth3d_tiny/
```

### Backend tests

All backends are always installed, so backend tests do not use `pytest.importorskip`. Only the DTU MATLAB external-tool path may skip when MATLAB is absent, and it must skip with an explicit reason.

### Regression tests

For official-like protocols, save expected outputs on tiny fixtures and compare with tolerance.

For DTU and Tanks and Temples, add documented validation against known official or official-like outputs when full data is available locally.

### Determinism tests

Benchmark results should be identical under:

```text
scene-order shuffling
parallel vs serial evaluation
repeated runs with the same protocol hash and seeds
```

## Documentation plan

These are the mkdocs user documentation pages under `docs/` (written in task 017; distinct from the `.agent/` planning documents):

```text
docs/index.md
docs/install.md
docs/quickstart.md
docs/prediction_format.md
docs/schema.md
docs/protocols.md
docs/datasets.md
docs/metrics.md
docs/backends.md
docs/reproducibility.md
docs/fidelity.md
docs/confidence.md
docs/failure_policy.md
docs/examples/scannet.md
docs/examples/dtu.md
docs/examples/tanks_temples.md
docs/examples/eth3d.md
```

README structure:

```text
short description
installation
quick example
supported prediction types
supported dataset adapters
protocol-driven evaluation explanation
backend delegation policy
example benchmark command
example result table
development status
citation section
```

## Versioning policy

Use semantic versioning for the package:

```text
0.1.x  core schema, protocol loader, single-file geometry metrics
0.2.x  stable geometry metrics, mesh sampling, result export
0.3.x  DTU official-like adapter and ScanNet geometry adapter
0.4.x  Tanks and Temples official wrapper and ETH3D training adapter
0.5.x  depth metrics and pose metrics
0.6.x  plugin API and third-party dataset adapters
1.0.0  stable public API and protocol schema
```

Protocol versions are tracked independently:

```yaml
schema_version: 1
protocol_version: 0.1.0
name: scannet_single_layer_geometry_5cm
```

Changing protocol behavior must increment the protocol version and change the protocol hash.

## Milestones

### Repository foundation

Goals:

```text
create package skeleton
add pyproject.toml
add CLI entry point
add basic docs
add CI
add linting and tests
```

Deliverables:

```text
eval3r package imports successfully
e3r --help works
pytest runs
docs build locally
```

Recommended tools:

```text
ruff
pytest
mypy or pyright
mkdocs
GitHub Actions
```

### Core schema and protocol loader

Goals:

```text
implement pydantic schemas
implement protocol YAML loader
implement manifest parser
implement canonical protocol hashing
implement result schema
implement backend registry
```

Deliverables:

```text
EvalProtocol
PredictionManifest
GroundTruthSpec
DatasetCapabilities
MetricResult
RunResult
BackendRegistry
Protocol validation
```

Acceptance:

```bash
e3r protocol show scannet_single_layer_geometry_5cm
```

### Geometry metrics

Goals:

```text
implement point-cloud geometry metrics
implement mesh surface sampling through backend
add scipy nearest-neighbor backend
add Trimesh mesh backend
add result export
```

Deliverables:

```text
accuracy
completeness
Chamfer
precision@τ
recall@τ
F-score@τ
distance percentiles
```

Acceptance:

```bash
e3r metric geometry pred.ply --gt gt.ply --threshold 0.05
```

### Dataset adapter for DTU

Goals:

```text
implement DTU adapter
load laser-scanned GT point clouds
load ObsMask and Plane files
handle millimeter units
wrap or port official-like evaluation behavior
```

Deliverables:

```text
DTUAdapter
dtu_official_like_pointcloud.yaml
DTU benchmark example
small DTU fixture
```

Acceptance:

```bash
e3r benchmark run preds/ \
  --dataset dtu \
  --split test \
  --protocol dtu_official_like_pointcloud
```

### Dataset adapter for ScanNet

Goals:

```text
implement ScanNet adapter
resolve exported scenes
load GT mesh variants
load intrinsics and poses
support single-layer and double-layer protocol labels
support visibility culling when protocol requests it
```

Deliverables:

```text
ScanNetAdapter
scannet_single_layer_geometry_5cm.yaml
scannet_double_layer_geometry_5cm.yaml
benchmark runner integration
per_scene.csv
results.json
```

Acceptance:

```bash
e3r benchmark run preds/ \
  --dataset scannet \
  --split val \
  --protocol scannet_single_layer_geometry_5cm
```

### Tanks and Temples official wrapper

Goals:

```text
implement Tanks and Temples adapter
wrap official evaluation script
resolve crop volume, log trajectory, transform, and GT point cloud
mark server-only splits clearly
```

Deliverables:

```text
TanksAndTemplesAdapter
tnt_official backend
tanks_temples_training_official.yaml
```

Acceptance:

```bash
e3r benchmark run preds/ \
  --dataset tanks_temples \
  --split training \
  --protocol tanks_temples_training_official
```

### ETH3D adapter

Goals:

```text
implement ETH3D training adapter
parse COLMAP text cameras
load public training GT
handle occlusion-aware GT assumptions
record non-pinhole camera limitations
```

Deliverables:

```text
ETH3DAdapter
eth3d_training_official.yaml
pycolmap camera backend path
```

Acceptance:

```bash
e3r benchmark run preds/ \
  --dataset eth3d \
  --split training \
  --protocol eth3d_training_official
```

### Depth metrics

Goals:

```text
add depth IO
add valid-mask handling
add scale alignment modes
add per-frame and per-scene aggregation
```

Deliverables:

```text
AbsRel
SqRel
RMSE
RMSE-log
δ metrics
depth protocol templates
```

Acceptance:

```bash
e3r metric depth pred.png --gt gt.png --align scale_median
```

### Pose metrics

Goals:

```text
add evo backend
add trajectory loading support
add ATE and RPE wrappers
support SE3 and Sim3 alignment
```

Deliverables:

```text
TrajectoryBackend
evo wrapper
ATE
RPE
pose protocol templates
```

Acceptance:

```bash
e3r metric pose pred_tum.txt --gt gt_tum.txt --backend evo --align sim3
```

### Reporting and diffing

Goals:

```text
add markdown, CSV, JSON, LaTeX, and HTML reports
add result diffing
add error-colored point cloud outputs
add failure banners for partial coverage
```

Deliverables:

```text
results.md
results.tex
report.html
e3r diff
error_colored.ply
```

Acceptance:

```bash
e3r diff runs/method_a runs/method_b
```

### Plugin API and public release

Goals:

```text
stabilize dataset plugin API
stabilize backend plugin API
stabilize protocol schema
add examples
publish to PyPI
```

Deliverables:

```text
plugin docs
example custom dataset
example custom protocol
PyPI package
versioned docs
```

Acceptance:

```bash
pip install eval3r
e3r --help
```

## Development workflow

Use:

```text
main      stable branch
dev       active integration branch
feature/* feature branches
```

Every schema, protocol, dataset, or metric change should include tests and documentation updates.

Recommended implementation order:

```text
schemas before adapters
metric definitions before benchmark runners
single-file evaluation before dataset benchmark evaluation
small fixtures before full datasets
official-like adapters only after the relevant protocol is documented
```

## Definition of done

A feature is done only when:

```text
it has schema coverage if it affects inputs or outputs
it has protocol documentation if it changes evaluation behavior
it has tests
it records metadata needed for reproducibility
it fails clearly, stating the reason, when required files or the MATLAB external tool are missing
it does not silently change alignment, masking, culling, scale, or sampling behavior
```
