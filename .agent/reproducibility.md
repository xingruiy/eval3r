# eval3r Reproducibility

A result is useful only when it can be understood, audited, and rerun. `eval3r` should record the protocol, dataset variant, ground-truth meaning, backend choices, failures, and environment for every benchmark run.

No benchmark number should be reported without enough metadata to know what it means.

## Reproducibility principles

```text
A metric number is inseparable from its protocol.
A dataset name is not enough; dataset variant and GT provenance must be recorded.
Partial scene coverage must be visible.
Backend choices and versions must be recorded.
Official-like claims require regression tests or wrapped official behavior.
Server-only splits must not produce local official-looking numbers.
```

## Run directory

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
      debug_index.json
      debug_outputs.json
      alignment_vis.json
      trajectory_alignment_vis.json
      scenes/
        scene0000_00/
          error.ply
          histogram.png
          alignment_before.ply
          alignment_after.ply
          alignment_projections.png
          trajectory_alignment_before.ply
          trajectory_alignment_after.ply
          trajectory_alignment_projections.png
          trajectory_alignment.json
```

Only files actually produced should be present, but `results.json`, `protocol.yaml`, `environment.json`, and `logs.txt` should be standard for every run. `manifest.yaml` should be present for benchmark runs.

`results.md`, `results.tex`, and `report.html` are written only when the protocol's
`reporting.formats` requests `markdown` / `latex` / `html`; every one of them shows the
run's scene coverage, failure reasons, failure policy, and a partial-coverage banner
whenever the aggregates do not cover every expected scene. Heavy per-scene debug artifacts
live under `debug/scenes/<scene_id>/`, while top-level manifests keep runs discoverable:
`debug/debug_outputs.json` for geometry error clouds / histograms,
`debug/alignment_vis.json` for geometry before/after alignment overlays, and
`debug/trajectory_alignment_vis.json` for trajectory alignment overlays. `debug/debug_index.json`
indexes every debug artifact class emitted by the run. Geometry error outputs are written
only when the protocol requests them (`save_colored_errors` / `save_distance_histogram`);
trajectory alignment outputs are written automatically whenever trajectory alignment is
actually estimated (`trajectory_se3` / `trajectory_sim3` pose runs, or geometry alignment
with `estimate_on: trajectory`). Only the eval3r-native geometry path can produce per-point
debug outputs; official-toolbox paths own their distance computation.

## Required result fields

`results.json` must include:

```text
schema_version
method
method_version
dataset
variant
split
protocol
protocol_version
protocol_hash
fidelity
local_evaluation_status
ground_truth
prediction_modality
scene_counts
failure_policy
failed_scenes
metrics
metric_definitions
alignment
masking
confidence
sampling
aggregation
backend_versions
manifest_path
protocol_path
command
platform
python_version
eval3r_version
git_commit when available
timestamp
```

Recommended structure:

```json
{
  "schema_version": 1,
  "method": "example_method",
  "method_version": "2026-07-04",
  "dataset": {
    "dataset": "scannet",
    "variant": "scannetv2_val_single_layer",
    "split": "val",
    "version": "scannetv2",
    "fingerprint": "sha256:..."
  },
  "protocol": {
    "name": "scannet_single_layer_geometry_5cm",
    "version": "0.1.0",
    "hash": "sha256:...",
    "fidelity": "official_like"
  },
  "ground_truth": {
    "modality": "mesh",
    "provenance": "reconstructed",
    "independence": "reconstruction_derived",
    "density": "dense_surface",
    "fingerprint": "sha256:..."
  },
  "local_evaluation": {
    "status": "supported",
    "public_gt_available": true,
    "official_server_required": false
  },
  "scene_counts": {
    "expected": 312,
    "evaluated": 312,
    "failed": 0
  },
  "metrics": {
    "accuracy_mean": 0.031,
    "completeness_mean": 0.047,
    "chamfer_mean": 0.039,
    "fscore@0.05": 0.612
  }
}
```

## Protocol hashing

Canonical protocol hashing should be deterministic:

```text
load YAML
validate into EvalProtocol model
serialize to canonical JSON
sort keys
remove non-semantic formatting differences
encode as UTF-8
sha256 hash
```

The hash should change when evaluation behavior changes. It should not change because of YAML comments or formatting.

Fields that affect protocol hash include:

```text
dataset variant
prediction modality
ground-truth specification
local-evaluation status
alignment
masking / culling
confidence policy
sampling
metrics
aggregation
failure policy
backend preferences when they affect metric semantics
```

Fields that may be excluded if treated as non-semantic:

```text
human notes
formatting
comments
report-only preferences that do not affect metrics
```

The exclusion list must be explicit in code and tests.

## Ground-truth fingerprinting

Ground-truth fingerprints should identify the reference actually used.

Examples:

```text
ScanNet: hash the GT mesh variant used by the protocol
DTU: hash GT point cloud plus ObsMask / Plane files where practical
T&T: record official GT point cloud, crop file, alignment file, and official backend version
ETH3D: hash public training GT and camera files where practical
Replica: hash mesh plus rendered trajectory/depth bundle if the protocol depends on the rendered release
CO3D: hash pointcloud.ply plus frame_annotations.jgz
Hypersim: record scene version and mesh/depth source, plus meters_per_asset_unit metadata
```

For very large files, use a documented fingerprint policy, such as file size + mtime + partial hash + manifest checksum. Prefer strong hashes for small and medium GT files.

## Prediction manifest preservation

Benchmark runs should copy the prediction manifest into the run directory.

The manifest should record:

```text
method
method version
dataset and variant
prediction modality
source pose format
normalized convention
scale type
unit
depth unit when relevant
uses_gt pose / intrinsics / scale
intrinsics source
confidence availability and self-filtering
scene prediction paths
metadata
```

If no manifest is provided for a benchmark run, eval3r may infer simple layouts only when the protocol allows it. Inferred manifests should be written to the run directory and marked as inferred.

## Scene coverage and failures

`failures.json` should include:

```text
scene_id
stage
error_type
message
file path if relevant
protocol requirement that failed
whether scene was skipped or assigned worst score
```

Example:

```json
{
  "failed_scenes": [
    {
      "scene_id": "scan003",
      "stage": "mask",
      "error_type": "MissingPlaneFile",
      "message": "DTU official-like protocol requires Plane003.mat or an explicit missing-plane fallback.",
      "policy_action": "skip_and_flag"
    }
  ]
}
```

Partial coverage must be visible in:

```text
results.json
results.csv
per_scene.csv
Markdown reports
LaTeX reports
HTML reports
CLI summary output
```

Do not average only successful scenes without reporting failed scenes and the failure policy.

## Per-scene CSV

`per_scene.csv` should include enough information to debug suspicious scores:

```text
scene_id
status
failure_reason
accuracy
completeness
chamfer
precision@τ
recall@τ
fscore@τ
n_points_pred
n_points_gt
n_pixels_valid
valid_fraction
culled_fraction
alignment_mode
alignment_scale
backend_nn
backend_mesh
runtime_seconds
```

Columns may be metric-dependent, but core status and coverage columns should be stable.

## Environment metadata

`environment.json` should include only the minimal, non-identifying facts needed to
interpret a run:

```text
eval3r version
python version
python implementation
platform
OS
CPU architecture
resolved command (no surrounding shell/user context)
```

Backend package versions used by the run are recorded separately in
`backend_versions.json`.

Do not record identifying or unnecessary information. In particular, `environment.json`
must NOT contain:

```text
secrets, full environment variables, API tokens, cookies, dataset credentials
working directory or other absolute local paths
git commit or git dirty state
timestamp or timezone
hostname or username
```

Note: a single ordering `timestamp` may still live on the `RunResult` for run
bookkeeping, and `git_commit` remains an optional `RunResult` field a user can set
explicitly, but neither is auto-captured into `environment.json`.

## Backend versions

Every backend used in a run should be recorded:

```text
mesh backend
point-cloud backend
nearest-neighbor backend
registration backend
trajectory backend
camera backend
depth IO backend
official evaluation backend
```

Record:

```text
backend name
library version
parameters that affect results
approximate vs exact behavior
official script version or commit if applicable
```

## Alignment transform records

If alignment is used, save one record per scene or per granularity unit:

```json
{
  "scene_id": "scene0000_00",
  "mode": "sim3",
  "estimate_on": "trajectory",
  "solver": "umeyama",
  "granularity": "per_scene",
  "scale": 1.037,
  "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  "translation": [0, 0, 0],
  "backend": "evo",
  "residual_rmse": 0.012
}
```

ICP records must also include:

```text
init transform
max correspondence distance
iteration count
convergence criteria
fitness
inlier RMSE
```

## Report comparability

Reports should make incomparability visible.

Warn when:

```text
protocol hashes differ
GT provenance differs
local evaluation status differs
scene coverage differs
failure policy differs
alignment mode differs
confidence policy differs
sampling count differs
backend officialness differs
```

`e3r diff` should refuse strict comparisons when protocol hashes differ. A loose comparison mode may be allowed, but it should label the result as non-strict. Prediction adaptation records are reproducibility metadata, not diff comparability triggers.

## Result schema compatibility

Result schemas need versioning.

Rules:

```text
schema_version is required in results.json
readers should handle older schema versions where practical
renaming or removing result fields requires a migration or compatibility note
new optional fields should have defaults
new required fields require a schema-version bump
```

Protocol version and result schema version are separate. Changing a protocol threshold should increment the protocol version and hash, but not necessarily the result schema version.

## Reproducibility tests

Required tests:

```text
protocol hash stable across YAML formatting
protocol hash changes when metric behavior changes
result writer includes required fields
partial coverage appears in reports
failures.json records structured failures
backend versions are recorded
manifest is copied or inferred and saved
alignment transforms are saved when alignment runs
results with different protocol hashes are rejected by strict diff
old result schema fixture can still be read if compatibility is promised
```

## Data and credentials

Do not store in run directories or committed fixtures:

```text
API keys
tokens
cookies
benchmark server credentials
licensed datasets
private dataset download links
large raw datasets
user authentication state
```

Use tiny synthetic fixtures for CI. Store instructions for acquiring full datasets in documentation, not the data itself.
