# eval3r Dataset Adapters

Dataset adapters are responsible for turning dataset-native files into eval3r's internal evaluation objects. They should make dataset conventions explicit, normalize poses and units, expose ground-truth provenance, and declare what evaluation is actually supported.

Adapters should not implement generic geometry algorithms. Mesh loading, surface sampling, nearest-neighbor search, trajectory metrics, and official benchmark behavior should be delegated to backend interfaces.

## Adapter responsibilities

Each adapter should handle:

```text
scene discovery
split discovery
native file layout parsing
camera and pose parsing
unit normalization
ground-truth resolution
ground-truth fingerprinting
visibility / mask / crop metadata resolution
local-evaluation status
capability declaration
prediction resolution by scene ID
```

Each adapter should normalize loaded data to the internal convention defined in `.agent/schema.md`:

```text
pose: camera-to-world
camera axes: OpenCV-style, +X right, +Y down, +Z forward
linear unit: meters
point arrays: Nx3
image origin: top-left
```

The adapter must also record the source convention before normalization. Do not discard whether the dataset stored `world_to_cam_colmap`, `cam_to_world_opengl`, `tanks_temples_log`, `co3d_frame_annotations`, or another native format.

## Adapter interface

Recommended interface:

```python
class DatasetAdapter(Protocol):
    name: str
    capabilities: DatasetCapabilities

    def iter_scenes(self, split: str) -> Iterable[str]: ...

    def load_scene(self, scene_id: str) -> SceneData: ...

    def load_ground_truth(
        self,
        scene_id: str,
        protocol: EvalProtocol,
    ) -> GroundTruthSpec: ...

    def load_visibility_data(
        self,
        scene_id: str,
        protocol: EvalProtocol,
    ) -> Any | None: ...

    def resolve_prediction(
        self,
        pred_root: Path,
        scene_id: str,
        manifest: PredictionManifest | None,
    ) -> Reconstruction: ...

    def gt_fingerprint(
        self,
        scene_id: str,
        protocol: EvalProtocol,
    ) -> str | None: ...

    def default_protocols(self) -> list[EvalProtocol]: ...
```

`load_visibility_data` may return dataset-specific structures, but the metric layer should receive only validated masks, valid points, valid pixels, or crop/culling decisions. Dataset-specific parsing should stay inside `datasets/` or a backend wrapper.

## Capability declaration

Every adapter must declare capabilities:

```python
class DatasetCapabilities(BaseModel):
    dense_geometry: bool = False
    independent_gt: bool = False
    depth_metric: bool = False
    pose_metric: bool = False
    official_local_eval: bool = False
    official_local_eval_method: Literal[
        "none",
        "official_script_wrapper",
        "validated_official_port",
        "matlab_official_script",
        "server_only",
    ] = "none"
    server_only_eval: bool = False
    requires_external_renderer: bool = False
    requires_paid_assets: bool = False
    supports_full_scene_geometry: bool = False
    supports_object_centric_geometry: bool = False
    supports_sparse_lidar_geometry: bool = False
    notes: list[str] = []
```

The CLI should use this information to reject unsupported dataset / protocol combinations before a long run starts. `official_local_eval` means eval3r can execute the declared official or official-like protocol locally for that split. The method field distinguishes a literal official-script wrapper from a validated port or MATLAB script.

Examples:

```text
DTU:
  dense_geometry = true
  independent_gt = true
  official_local_eval = true
  official_local_eval_method = validated_official_port
  supports_full_scene_geometry = false
  supports_object_centric_geometry = true
  local_evaluation_status = supported

Tanks and Temples training:
  dense_geometry = true
  independent_gt = true
  official_local_eval = true
  official_local_eval_method = official_script_wrapper
  server_only_eval = false
  local_evaluation_status = supported

Tanks and Temples intermediate / advanced:
  server_only_eval = true
  official_local_eval = false
  official_local_eval_method = server_only
  local_evaluation_status = server_only

CO3D:
  dense_geometry = false
  independent_gt = false
  pose_metric = true
  supports_object_centric_geometry = true
  official_local_eval = false
  local_evaluation_status = supported

KITTI-360:
  pose_metric = true
  supports_sparse_lidar_geometry = true
  dense_geometry = false
  official_local_eval = false
  local_evaluation_status = supported
```

## Ground-truth fields

Every adapter must fill `GroundTruthSpec` with these fields:

```text
modality
provenance
independence
density
fingerprint
unit
source_pose_format
normalized_convention
local_evaluation_status
notes
```

This is necessary because dataset names do not tell the user what the reference actually is. `provenance` and `independence` are related but not identical: provenance says how the reference was produced; independence says how the score should be interpreted. For example, LiDAR-derived sparse GT is sensor-derived rather than reconstruction-derived, while ScanNet and CO3D are reconstruction-derived; both should trigger different report wording from an independent laser scan.

Examples:

```text
DTU:
  modality: pointcloud
  provenance: laser_scan
  independence: independent
  density: dense_surface

ScanNet:
  modality: mesh
  provenance: reconstructed
  independence: reconstruction_derived
  density: dense_surface

CO3D:
  modality: pointcloud
  provenance: reconstructed
  independence: reconstruction_derived
  density: object_pointcloud

KITTI-360:
  modality: lidar
  provenance: lidar_sparse
  independence: sensor_derived
  density: sparse_lidar
```

## Dataset variants

A dataset name is often not enough. Adapters should expose explicit dataset variants when a benchmark is only defined by extra assumptions, a specific GT release, or a specific rendering trajectory.

Examples:

```text
scannetv2_val_single_layer
scannetv2_val_double_layer
replica_nice_slam_8scene_rendered
hypersim_depth_only
hypersim_mesh_assets
eth3d_training_colmap_text
```

Variant identity should be included in the protocol hash and result metadata.

## Local evaluation status

Adapters must not pretend that server-only or asset-gated benchmarks are locally evaluable.

Use these statuses:

```text
supported
server_only
missing_public_gt
requires_external_assets
requires_external_renderer
unsupported
```

Examples:

```text
DTU: supported when using the validated official-like local evaluator or the MATLAB script
Tanks and Temples training: supported
Tanks and Temples intermediate / advanced: server_only
ETH3D training: supported
ETH3D test: server_only
Hypersim full mesh geometry: requires_external_assets
Replica RGB-D sequence benchmark: requires_external_renderer or requires pinned rendered release
```

## Dataset tiers

The tiers below are not an exclusive partition and are not a single ranking. They combine two axes: implementation priority and ground-truth honesty. A dataset may appear in an early implementation group while also carrying a reconstructed-GT warning. ScanNet is the main example: it is useful to implement early, but its reference mesh is reconstruction-derived.

The build order in `.agent/plan.md` is an intentional engineering roadmap, not a claim that earlier datasets have better GT. With fused reconstruction support removed, the roadmap prioritizes small, testable geometry and official-like point-cloud behavior before broader dataset coverage.

### Core dense geometry implementation targets

These are the main early targets for mesh and point-cloud evaluation.

```text
DTU
Tanks and Temples training split
ETH3D training split
ScanNet validation split
```

DTU and Tanks and Temples have independent measured GT for local public splits. ETH3D has laser-scan GT with official occlusion handling. ScanNet is useful for indoor reconstruction work, but its mesh is reconstruction-derived, so results must be labeled accordingly.

### Reconstruction-derived GT datasets

These datasets are useful, but their reference geometry is reconstructed, fused, COLMAP-derived, or rendered from a reconstructed asset rather than independently measured. This is the real grouping criterion; it is not an indoor/outdoor category.

```text
ScanNet
7-Scenes
Neural-RGBD real scenes
CO3D
BlendedMVS
```

CO3D is object-centric. BlendedMVS is a mixed collection covering architectures, street views, sculptures, and small objects. Their common issue is not scene type; it is that dense geometry scores do not have the same meaning as scores against independent laser-scan or exact synthetic GT.

### Synthetic controlled datasets

```text
Replica
Hypersim
Neural-RGBD synthetic scenes
```

These can be useful when the exact benchmark variant is pinned. Replica has no native trajectory. Hypersim mesh geometry may require external paid assets. Hypersim depth is Euclidean ray distance and must be converted correctly before z-depth metrics.

### Pose and sparse-LiDAR oriented datasets

```text
KITTI-360
```

This should be treated primarily as a pose or sparse point comparison dataset, not a dense-surface reconstruction benchmark.

## Dataset-specific adapter requirements

### ScanNet

Adapter responsibilities:

```text
handle exported .sens layout
load color/depth/pose/intrinsic directories when present
record depth_unit = 0.001 for 16-bit millimeter depth
normalize camera-to-world OpenCV-style exported poses
resolve GT mesh variant
record GT as reconstruction-derived
support single-layer and double-layer protocol variants
resolve visibility-culling inputs when required by protocol
hash GT mesh files used by the protocol
```

Rules:

```text
Do not silently choose single-layer or double-layer evaluation.
Do not silently enable or disable visibility culling.
Do not label ScanNet GT as independent laser-scan ground truth.
```

Split-specific culling (implemented in task 011):

```text
val split runs WITHOUT visibility culling; culled_fraction is recorded as 0.
test split runs gt_visibility culling (community NeuralRecon/TransformerFusion convention).
gt_visibility is realized by the 'visibility' backend: render the prediction depth from the
  GT camera trajectory (pyrender, EGL) and TSDF-integrate (open3d) to recover the observed
  region, then keep prediction vertices within the protocol tolerance of that region.
the adapter supplies the GT trajectory via load_trajectory (cam-to-world OpenCV poses,
  depth intrinsics, image size); non-finite (lost-tracking) poses are dropped.
renderer + TSDF backend, versions, voxel size, trajectory fingerprint, and per-scene
  culled_fraction are recorded (CLAUDE.md evaluation-time visibility-culling exception).
these protocols are eval3r_native; results are never official ScanNet benchmark numbers.
```

### DTU

Adapter responsibilities:

```text
resolve scan IDs
load laser-scanned point cloud from Points/stl
load ObsMask and Plane files for official-like evaluation
record native units as millimeters and normalize to meters when needed
handle missing Plane files explicitly
support point-cloud evaluation, not mesh-GT evaluation
resolve conventional prediction filenames such as <method>XXX_l3.ply, where l3 denotes the all-lights-on evaluation convention
```

Rules:

```text
Do not skip ObsMask / Plane while claiming official-like fidelity.
Do not treat DTU GT as a mesh.
Record Plane availability in scene metadata.
Do not ignore the DTU light-condition suffix when resolving prediction files.
```

### Tanks and Temples

Adapter responsibilities:

```text
resolve public training scenes
resolve GT point cloud, crop file, .log trajectory, alignment transform
call official backend wrapper for official local evaluation
record official backend version
mark intermediate and advanced splits as server-only
record per-scene thresholds from official metadata or backend output
```

Rules:

```text
Do not reimplement official evaluation unless there is a regression-tested reason.
Do not use a global threshold for all scenes.
Do not claim local official evaluation for withheld-GT splits.
```

Implementation (task 012):

```text
TanksAndTemplesAdapter resolves five per-scene artifacts from <root>/<Scene>/:
  <Scene>.ply (laser-scan GT, metres), <Scene>.json (crop volume),
  <Scene>_trans.txt (alignment), <Scene>_COLMAP_SfM.log (.log trajectory),
  <Scene>_mapping_reference.txt.
official_artifacts(scene) validates all four required inputs and returns the scene
  directory (dataset_dir) the official toolbox reads by naming convention.
GT is labelled laser_scan / independent / dense_surface, unit metres.
gt_fingerprint jointly hashes GT point cloud + crop + alignment transform.
Only the training split is locally evaluable; local_evaluation('intermediate') and
  ('advanced') return server_only so the CLI refuses them before any computation.
Official precision/recall/F-score, ICP, cropping, and per-scene dTau are delegated to
  the tnt_official backend (never reimplemented); the per-scene threshold comes from the
  official output and is recorded per scene. Fidelity is 'official'.
The official toolbox is run byte-for-byte unmodified under its pinned open3d==0.9
  interpreter (EVAL3R_TNT_PYTHON), never ported to newer open3d (a result-affecting
  change). No fake toolbox exists: the end-to-end tests drive the real toolbox and skip
  when EVAL3R_TNT_TOOLBOX / EVAL3R_TNT_PYTHON / EVAL3R_TNT_DATA are unset.
Verified end-to-end on real Barn (commit 2a0d1b25, prediction = Barn_COLMAP.ply):
  precision 0.4569 / recall 0.5529 / f-score 0.5003 at dTau 0.01.
```

### ETH3D

Adapter responsibilities:

```text
parse COLMAP text camera files
use pycolmap or a camera-capable backend for nontrivial camera models
resolve training GT and occlusion-aware references
mark test split as server-only
record camera model limitations in scene metadata
```

Rules:

```text
Do not force unsupported non-pinhole cameras into the minimal pinhole fallback.
Do not claim official local evaluation for server-only test scenes.
```

### 7-Scenes

Adapter responsibilities:

```text
load scene sequences and TrainSplit / TestSplit files
load frame color, depth, and pose files
record depth_unit = 0.001
mask invalid depth value 65535
pin nominal Kinect intrinsics in the protocol or dataset variant
record GT surface as reconstruction-derived when a fused surface is used
```

Rules:

```text
Do not infer intrinsics silently.
Do not evaluate invalid depth pixels.
```

### Neural-RGBD

Adapter responsibilities:

```text
load images, depth folders, focal.txt, poses.txt, and GT meshes when available
record native OpenGL-style convention
record single-focal centered-principal-point assumption
separate released culled meshes from uncropped/source meshes as distinct variants
```

Rules:

```text
Do not mix culled and uncropped mesh references under one protocol.
Do not ignore the OpenGL convention warning.
```

### Replica

Adapter responsibilities:

```text
load raw mesh assets
support only pinned rendered trajectory/depth variants for sequence benchmarks
fingerprint trajectory and render bundle, not just mesh
record renderer and pose convention
handle glass/mirror metadata if a protocol excludes those surfaces
```

Rules:

```text
Do not treat raw Replica as having a native RGB-D benchmark trajectory.
Do not compare results across different rendered trajectories as if they are one benchmark.
```

### Hypersim

Adapter responsibilities:

```text
read HDF5 depth and camera metadata
convert Euclidean ray distance to z-depth when required
apply meters_per_asset_unit to positions and camera translations
use corrected camera parameters where required
record mesh-asset availability
```

Rules:

```text
Do not treat depth_meters as ordinary z-depth.
Do not ignore per-scene scale conversion.
Do not offer full mesh geometry without checking mesh asset availability.
```

### CO3D

Adapter responsibilities:

```text
load frame_annotations.jgz and sequence_annotations.jgz
load object masks, depth maps, and pointcloud.ply
record GT as COLMAP-derived / reconstruction-derived
fingerprint pointcloud.ply and frame_annotations.jgz jointly
prefer pose or object-centric protocols over dense full-scene geometry
```

Rules:

```text
Do not assume per-frame pose text files exist.
Do not label CO3D geometry as official dense-surface evaluation.
```

### BlendedMVS

Adapter responsibilities:

```text
parse MVSNet-style cam.txt files
normalize world-to-camera extrinsics
load rendered depth maps
mask invalid or empty frames
record GT as rendered from reconstructed mesh
```

Rules:

```text
Do not treat cam.txt as camera-to-world.
Do not treat rendered depth from reconstructed mesh as independent GT.
```

### KITTI-360

Adapter responsibilities:

```text
load calibration and pose files
handle sparse or missing pose entries explicitly
support pose metrics and sparse-LiDAR comparison protocols
mark sparse-LiDAR protocols as eval3r_native
limit minimal-pinhole support to perspective cameras image_00 / image_01 unless a fisheye-capable backend is used
```

Rules:

```text
Do not label sparse-LiDAR comparison as dense-surface geometry.
Do not interpolate missing poses without recording the policy.
Do not silently treat image_02 / image_03 fisheye cameras as pinhole cameras.
```

## Adapter assumptions requiring direct fixture checks

Some dataset conventions are common in downstream tooling but should still be checked against the primary parser before locking tests or public fixtures.

```text
Replica rendered trajectories:
  Verify the exact NICE-SLAM / iMAP-style rendered release, file layout, trajectory convention, and depth convention before treating it as a built-in variant.

BlendedMVS camera files:
  Verify the MVSNet cam.txt block order and extrinsic direction against the primary MVSNet / BlendedMVS parser before freezing parser tests. The adapter should still normalize to eval3r's internal camera-to-world convention.
```

## Adapter tests

Use small fixtures whenever possible. Normal CI should not require full datasets.

Required tests:

```text
scene discovery
split parsing
pose convention normalization
unit normalization
ground-truth spec construction
capability declaration
local-evaluation status
missing-file errors
variant fingerprinting
mask / crop / visibility metadata resolution
```

Dataset-specific tests should include at least one convention-normalization test. For example, a known world-to-camera COLMAP pose should normalize to the expected camera-to-world internal pose.

## Common adapter mistakes

Avoid:

```text
silently assuming all poses are camera-to-world
silently assuming all cameras are pinhole
silently assuming all depths are z-depth
silently converting millimeters to meters without recording the source unit
treating reconstructed GT as independent GT
running a server-only split locally and reporting unofficial numbers
using raw mesh vertices for geometry metrics without protocol permission
hiding dataset-specific behavior inside generic utilities
```
