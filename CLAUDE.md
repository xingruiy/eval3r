# CLAUDE.md

This file gives operating rules for coding agents working on `eval3r`.

## Project summary

`eval3r` is a plain 3D reconstruction evaluation library. It evaluates meshes, point clouds, depth predictions, and trajectories under explicit dataset-aware protocols.

`eval3r` is a research-oriented project. Correctness, explicitness, and inspectability outrank packaging minimalism and install footprint. Every dependency is a required base dependency and is always installed; there is no optional-extra system to reason about. Prefer surfacing the full truth of what happened (what was resolved, what failed, and why) over terse or convenient output.

The library should focus on:

```text
protocol definitions
dataset adapters
prediction manifests
metric definitions
benchmark orchestration
result schemas
reproducibility records
reports
```

The library should delegate common geometry, camera, trajectory, and IO work to existing libraries where practical.

## Hard boundaries

Do not add support for these **as reconstruction methods**:

```text
TSDF integration
RGB-D fusion
volumetric fusion
online mapping
SLAM tracking
learned reconstruction methods
mesh repair as a default preprocessing step
large proprietary dataset downloads inside the package
```

Depth sequences are supported for depth metrics. They are not converted into scene reconstructions by eval3r.

### Evaluation-time visibility culling exception

Offscreen depth rendering (pyrender) and TSDF volume integration (open3d) are permitted
**solely as an evaluation-time visibility-culling mechanism** — rendering a *prediction's*
depth from the ground-truth camera trajectory and TSDF-trimming the prediction to the
observed region before scoring (the community ScanNet single-/double-layer convention).
This is not reconstruction: no new scene geometry is produced, only the caller's prediction
is masked. Whenever this path runs it must be recorded in result metadata (renderer +
TSDF backend and versions, voxel size, trajectory fingerprint, and per-scene culled
fraction), and it is only ever enabled when the protocol's masking explicitly requests it.
It must never run silently or as a default preprocessing step.

## Start-of-session checklist

Before making changes, read:

```text
.agent/plan.md
.agent/schema.md
.agent/protocols.md
.agent/datasets.md, if working on an adapter
.agent/metrics.md, if working on metrics
.agent/backends.md, if working on backend delegation
.agent/reproducibility.md, if changing result files
```

Also inspect existing tests for the relevant module before editing code.

## Task management

Use `.agent/tasks/` as durable agent working memory.

Create or update:

```text
.agent/tasks/README.md          canonical index of task slices, statuses, ordering, and next task
.agent/tasks/001-short-name.md  one coherent implementation slice
.agent/tasks/002-short-name.md  next coherent implementation slice
```

Completed task files stay in `.agent/tasks/` and are marked `done`. Do not move completed tasks to an archive unless the user explicitly asks for a different policy.

Each task file should include:

```text
Goal
Scope
Out of Scope
Relevant Files
Plan
Findings
Decisions
Verification
Status
```

Use task files as coordination aids, not as project source of truth. Code, tests, and official docs remain authoritative.

Workflow:

```text
create numbered task files before substantial multi-step work
keep each task small enough to finish and verify independently
mark the active task in_progress before editing
record durable findings and decisions, not a full transcript
record verification commands and outcomes
mark verified tasks done
update .agent/tasks/README.md whenever task status or ordering changes
resume interrupted sessions from the first non-done task in README.md
```

Before resuming a task, re-read the actual relevant files because task notes can be stale.

## Source-of-truth files

```text
.agent/plan.md              source of truth for project scope and roadmap
.agent/schema.md            source of truth for model fields and result structure
.agent/protocols.md         source of truth for protocol behavior
.agent/datasets.md          source of truth for dataset adapter behavior
.agent/metrics.md           source of truth for metric definitions and aggregation
.agent/backends.md          source of truth for backend delegation and dependencies
.agent/reproducibility.md   source of truth for result files, hashing, and run metadata
CLAUDE.md                 source of truth for agent workflow rules
```

If code and docs disagree, update both in the same change. Do not silently make behavior drift away from the documented schema or protocol.

## Schema rules

Any change to schema fields must update:

```text
.agent/schema.md
Pydantic models
schema serialization tests
sample JSON/YAML fixtures
result writer or reader if affected
```

Never remove or rename result fields without considering backward compatibility.

If a field affects evaluation behavior, it must be included in canonical protocol hashing.

## Protocol rules

Every protocol must explicitly define:

```text
dataset variant
ground-truth provenance
ground-truth independence
ground-truth density
local evaluation status
prediction modality
alignment
masking and culling
confidence policy
sampling
metrics
aggregation
failure policy
backend preferences
```

Metrics must not silently choose:

```text
alignment
scale handling
thresholds
masking
culling
sampling count
sampling seed
confidence threshold
aggregation order
```

When changing a built-in protocol, update:

```text
.agent/protocols.md
protocol YAML file
protocol version
expected protocol hash in tests
regression fixtures if behavior changed
```

## Dataset adapter rules

Dataset adapters normalize dataset-native files into eval3r's internal convention:

```text
camera-to-world pose
OpenCV-style camera axes
meters
validated point/depth arrays
```

Adapters must record the original source convention. Do not discard it after normalization.

Each adapter must declare capabilities:

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

Do not present a dataset as a dense-surface benchmark when its GT is sparse LiDAR, COLMAP-derived, reconstruction-derived, or server-only. `official_local_eval` means the declared official or official-like protocol can run locally for that split; record whether the method is an official script wrapper, a validated port, MATLAB, or server-only.

## Dataset-specific cautions

### ScanNet

```text
Exported poses are camera-to-world and OpenCV-style for standard SensReader exports.
GT mesh is BundleFusion-derived, not independent laser-scan GT.
Single-layer and double-layer geometry protocols are different.
Visibility culling is protocol-defined and must be recorded.
Depth is 16-bit millimeters; depth_unit = 0.001.
```

### DTU

```text
GT is laser-scanned point cloud, not mesh.
Native units are millimeters.
Official-like evaluation uses ObsMask and Plane files.
Missing Plane files must be recorded explicitly.
Resolve conventional prediction names such as <method>XXX_l3.ply; do not ignore the light-condition suffix.
Do not skip ObsMask/Plane while claiming official-like fidelity.
```

### Tanks and Temples

```text
Wrap the official evaluation script for official local training-set evaluation.
Training split has public GT; intermediate and advanced splits are server-only.
Per-scene thresholds are not a global constant.
Crop volume, .log trajectory, alignment transform, and backend version must be recorded.
```

### ETH3D

```text
Camera data is COLMAP text format.
Use pycolmap for nontrivial camera models.
Training GT is public; test GT is server-only.
Do not use minimal pinhole assumptions for unsupported camera models without a clear warning.
```

### 7-Scenes

```text
No official intrinsics file is provided.
Nominal Kinect intrinsics must be protocol-pinned.
Depth invalid value 65535 must be masked.
GT surface, when used, is reconstruction-derived.
```

### Neural-RGBD

```text
Native convention is OpenGL-style.
focal.txt is a single focal value with assumed centered principal point.
Released culled meshes and uncropped/source meshes are different GT variants.
```

### Replica

```text
Raw Replica has meshes, not a native camera/depth benchmark trajectory.
Any rendered trajectory bundle must be named and fingerprinted.
Renderer convention must be recorded.
Verify any NICE-SLAM / iMAP-style rendered release before making it a built-in variant.
```

### Hypersim

```text
Depth is Euclidean ray distance, not standard z-depth.
meters_per_asset_unit is required for metric scale.
Some camera parameters require corrected projection metadata.
Full mesh geometry may require external paid assets.
```

### CO3D

```text
Camera data is in frame_annotations.jgz.
pointcloud.ply is COLMAP-derived, not independent GT.
Geometry protocols should be eval3r-native and clearly labeled.
Object-centric geometry may be supported, but it is not full-scene dense geometry.
```

### BlendedMVS

```text
Camera files use MVSNet-style cam.txt conventions.
Extrinsics are world-to-camera and must be normalized.
Rendered depth comes from reconstructed meshes and may contain invalid or empty frames.
Verify cam.txt parsing against the primary MVSNet / BlendedMVS parser before freezing fixtures.
```

### KITTI-360

```text
This is primarily a pose / sparse-LiDAR dataset for eval3r.
Do not label sparse LiDAR comparison as dense-surface reconstruction evaluation.
KITTI-360 image_00 / image_01 are perspective; image_02 / image_03 are fisheye and require a capable backend or must be excluded by protocol.
```

## Metric implementation rules

Geometry metrics:

```text
validate shapes
remove NaN and Inf
apply confidence filtering if protocol says so
apply culling / masking if protocol says so
sample only according to protocol
compute pred→gt and gt→pred distances
record n_points_pred, n_points_gt, valid_fraction, culled_fraction
```

Mesh metrics:

```text
sample surfaces before distance metrics
never use raw vertices unless the protocol explicitly asks for raw vertices
record mesh backend and sampling seed
```

Depth metrics:

```text
record depth unit
mask invalid pixels
record scale alignment mode and granularity
never convert a depth sequence into a scene reconstruction inside eval3r
```

Pose metrics:

```text
prefer evo backend
record association policy
record alignment mode
record scale if Sim3 is used
```

## Backend rules

All Python dependencies are required base dependencies and are always installed. There is no optional-extra system. Do not add `pip install 'eval3r[...]'` install hints, `pytest.importorskip` guards, or "install this extra" error paths.

MATLAB is the only optional external tool (it is not a pip package). It is used only for the DTU MATLAB official-script path; whether it was used must be recorded in result metadata.

The project does not depend on:

```text
PyTorch
FAISS
Waymo tooling
```

These were dropped: there is no torch/FAISS nearest-neighbor backend and no Waymo adapter.

Backend names and versions must be written to result metadata.

## Result and reproducibility rules

Every benchmark result must record:

```text
schema_version
protocol name and hash
dataset and variant
split
scene list
prediction manifest
ground-truth provenance and fingerprint
local evaluation status
alignment policy
masking / culling policy
sampling policy
confidence policy
failure policy
metric definitions
backend versions
command line
platform and Python version
git commit when available
```

Partial scene coverage must be visible in every report format.

Do not average only successful scenes without reporting failed scenes and failure reasons.

## Testing rules

Every behavior-changing change needs tests.

Add or update tests for:

```text
schema validation
protocol hashing
metric formulas
aggregation behavior
failure policy behavior
dataset adapter parsing
pose convention normalization
masking / culling behavior
```

Use tiny fixtures. Do not require full datasets in normal CI.

All backends are always importable, so backend tests do not need `pytest.importorskip`. Tests that exercise the MATLAB external-tool path should skip cleanly when MATLAB is absent and say so explicitly.

## Documentation rules

Update docs in the same change when modifying:

```text
schema fields
protocol behavior
metric definitions
dataset adapter behavior
CLI options
result directory structure
backend dependencies
```

Do not rely on comments in code as the only documentation for evaluation behavior.

## Data and credential rules

Do not commit:

```text
.env
API keys
tokens
cookies
downloaded proprietary datasets
licensed data
benchmark server credentials
authentication state
large raw dataset files
```

Keep `.env.example` limited to variable names and comments.

Use small synthetic or public mini-fixtures for tests.

## Git workflow

Use:

```text
main      stable branch
dev       active integration branch
feature/* feature branches
```

Before opening a PR:

```bash
ruff check .
pytest
mypy eval3r  # or pyright, depending on project choice
mkdocs build
```

If a command is not yet configured, do not invent a passing status. State what is missing and add the config if appropriate.

## Implementation order

Prefer this order:

```text
schemas
protocol loader and hashing
single-file geometry metric
result writer
DTU adapter
ScanNet adapter
Tanks and Temples wrapper
ETH3D adapter
depth metrics
pose metrics
reports and diffing
plugin API
```

Do not implement a complex dataset adapter before the schema and protocol behavior it needs are documented.

## Code style

```text
Use small modules with clear interfaces.
Use type hints.
Prefer Pydantic models for external inputs and outputs.
Keep backends behind adapter interfaces.
Keep dataset-specific parsing inside datasets/.
Keep metric formulas inside metrics/.
Do not hide dataset-specific behavior in generic utilities.
```

Error messages should tell the user:

```text
what failed
which scene failed
which file was missing or invalid
which protocol required it
how to fix it when obvious
```

## Error and CLI verbosity rules

This is a research tool. Silent, terse, or lossy output is a defect.

Every handled error must explicitly state the reason:

```text
never swallow an exception or reduce it to a boolean or a bare exit code
say what failed, why it failed, and the concrete inputs involved
name the scene, the file path, the protocol, and the field or check that failed
when a fix is obvious, state it
prefer re-raising with added context over catching and hiding
```

The CLI must give rich, indicative, verbose output using `rich`:

```text
before running, echo the resolved configuration: protocol name and hash, dataset and variant, split, prediction manifest, and the alignment / masking / sampling / confidence / failure policies actually in effect
echo the resolved input and output paths verbatim
show per-scene progress and per-scene outcomes
on failure, print the full reason verbatim (with traceback when available), never just a non-zero exit
partial scene coverage and skipped scenes must be visible, not hidden behind an average
```

## Definition of done

A task is done only when:

```text
code works for the intended path
tests cover the new behavior
docs are updated
schema and protocol implications are handled
result metadata is complete
failure behavior is explicit
errors and CLI output explicitly state reasons and resolved configuration
```
