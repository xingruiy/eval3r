# Datasets

Dataset adapters turn dataset-native files into eval3r's internal evaluation objects.
They normalize conventions (poses, axes, units), resolve and fingerprint ground truth,
declare what evaluation is actually possible, and record what the reference really is.

```bash
e3r dataset list
e3r dataset inspect <name> --root /path/to/dataset [--split <split>]
```

## Ground-truth honesty

Many datasets called "ground truth" are not independent measurements. eval3r records,
for every reference: **modality** (mesh/pointcloud/...), **provenance** (how it was
produced: laser_scan, synthetic_exact, reconstructed, lidar_sparse, ...),
**independence** (whether it is independent of the kind of prediction being scored:
independent, reconstruction_derived, sensor_derived, server_only), and **density**
(dense_surface, sparse_lidar, object_pointcloud, ...).

A score against DTU's laser scan and a score against a reconstruction-derived mesh are
not the same kind of evidence, and reports and `e3r diff` make that visible instead of
hiding it behind one generic "geometry score".

## Capabilities and local evaluability

Every adapter declares capabilities (dense_geometry, independent_gt, depth_metric,
pose_metric, official_local_eval + method, server_only_eval, ...), and the CLI checks
them in preflight: a protocol/split combination that is not locally evaluable is
**refused before any computation** — a server-only split never produces local
official-looking numbers.

## Built-in adapters

### DTU

- **GT**: laser-scanned point clouds (`Points/stl/stlXXX_total.ply`) — independent,
  dense-surface. Native units are **millimeters** (normalized to meters internally, source
  unit recorded).
- **Official-like evaluation**: validated Python port of the official MATLAB evaluation,
  including `ObsMaskXXX_10.mat` and `PlaneXXX.mat` handling. Missing Plane files are
  recorded explicitly, never skipped silently. The MATLAB script itself is an optional
  external path.
- **Predictions**: conventional names like `<method>024_l3.ply`; the light-condition
  suffix is respected.
- Worked example: [DTU](examples/dtu.md).

### ScanNet

- **GT**: BundleFusion-derived meshes — reconstruction-derived, **not** independent laser
  scans; results are labeled accordingly. Depth is 16-bit millimeters (`depth_unit`
  0.001); exported poses are camera-to-world, OpenCV-style.
- **Protocols**: community single-/double-layer 5 cm F-score conventions
  (`eval3r_native`; ScanNet has no official reconstruction benchmark). The val protocol
  runs without visibility culling; the test protocol applies protocol-defined visibility
  culling (prediction depth rendered from the GT trajectory, TSDF-trimmed to the observed
  region) — the renderer, TSDF backend, versions, voxel size, trajectory fingerprint, and
  per-scene culled fraction are all recorded.
- Worked example: [ScanNet](examples/scannet.md).

### Tanks and Temples

- **GT**: laser-scanned point clouds for the **training** split (public); intermediate
  and advanced splits are **server-only** and refused locally.
- **Official evaluation**: the real official python toolbox, run unmodified as a
  subprocess under its pinned `open3d==0.9` interpreter (see
  [install](install.md#tanks-and-temples-official-toolbox)). Crop volumes, `.log`
  trajectories, alignment transforms, ICP, and **per-scene** thresholds are the official
  toolbox's — per-scene thresholds are never a global constant.
- Worked example: [Tanks and Temples](examples/tanks_temples.md).

### ETH3D (high-res DSLR multi-view)

- **GT**: occlusion-aware laser scans (`dslr_scan_eval`) for the **training** split; the
  test split is server-only. Cameras are COLMAP text, parsed with pycolmap; non-pinhole
  camera models are recorded as explicit limitations, never silently approximated as
  pinhole.
- **Official evaluation**: the real `multi-view-evaluation` binary (see
  [install](install.md#eth3d-multi-view-evaluation-binary)) at the official tolerance
  set (1, 2, 5, 10, 20, 50 cm; the 2 cm F1 is the conventional headline number). Its
  voxel-normalized, free-space-aware scoring is never reimplemented.
- Worked example: [ETH3D](examples/eth3d.md).

### custom

Single-file and simple-layout evaluation with user-declared ground truth (the
`single_*` protocols). GT provenance is `unknown` unless you say otherwise in a custom
protocol.

## Planned adapters

7-Scenes, Neural-RGBD, Replica, Hypersim, CO3D, BlendedMVS, and KITTI-360 are
intentionally deferred until the conventions each needs are pinned (see the project
roadmap in the repository). Writing third-party adapters against the internal registry is
possible but **not yet a stable public API** — see [Stability](index.md#stability).
