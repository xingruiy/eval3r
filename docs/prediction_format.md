# Prediction format

## Prediction modalities

```text
mesh                     PLY/OBJ/... loadable by the mesh backend
pointcloud               PLY point cloud
pointmap                 explicit 3D points in a declared coordinate frame
single_depth             one depth image (PNG/TIFF/PFM/.npy)
depth_sequence           a directory of depth frames, matched to GT by filename stem
camera_trajectory        TUM-format text file (timestamp x y z qx qy qz qw)
```

Pointmaps are supported only when they can be interpreted as explicit 3D points in a
declared frame; eval3r never integrates per-frame pointmaps or depth frames into a scene
reconstruction.

## Conventions

Internally, everything is normalized to: camera-to-world poses, OpenCV-style camera axes
(+X right, +Y down, +Z forward), meters, Nx3 float point arrays, top-left image origin.
Dataset adapters do this normalization and **record the source convention** — what the
files on disk actually contained is never discarded.

Predictions for dataset benchmarks are expected in the dataset's ground-truth frame in
meters unless the protocol declares an alignment step. Integer depth files always need an
explicit `depth_unit` (meters per stored unit, e.g. `0.001` for millimeter PNGs) —
eval3r refuses to guess.

## Prediction manifest

Benchmark predictions should be described by a manifest YAML. It declares what the
predictions *are* — modality, scale, coordinate frame, pose convention, GT usage, and
confidence behavior — so the result can be interpreted later.

```yaml
method: example_method
version: 2026-07-04
dataset:
  dataset: scannet
  variant: scannetv2_val_single_layer
split: val
prediction_modality: mesh
coordinate_frame: world
source_pose_format: cam_to_world_opencv
normalized_convention: cam_to_world_opencv_meters
scale: metric              # metric | relative | unknown
uses_gt:
  pose: false
  intrinsics: false
  scale: false
intrinsics_source: predicted
confidence:
  present: false
  native_threshold: null
  self_filtered: false
scenes:
  scene0000_00:
    mesh: scene0000_00/recon.ply
  scene0001_00:
    mesh: scene0001_00/recon.ply
```

Rules:

- `uses_gt` matters: a method that consumed GT poses is not comparable to one that did
  not, and reports surface this.
- If a method self-filters its predictions by confidence before export, the manifest must
  say so (`confidence.self_filtered`) — reports distinguish uniform protocol filtering
  from method-native filtering.
- Scene entries should be explicit for benchmark runs.
- The manifest is copied into the run directory (`manifest.yaml`).

## Manifest resolution

`e3r benchmark run preds/` resolves the manifest in this order:

1. `--manifest path.yaml` if given,
2. `preds/manifest.yaml` if present,
3. otherwise a simple per-scene layout is **inferred**, written to the run directory,
   and marked as inferred.

`e3r benchmark validate` checks preflight and per-scene prediction resolution for every
scene without evaluating anything — use it before long runs.

## Dataset-specific file naming

Adapters resolve dataset-conventional prediction names; see the per-dataset examples:

- **DTU**: conventional files like `<method>024_l3.ply` — the light-condition suffix
  (`l3` = all lights on) is part of the convention and is never ignored.
  ([example](examples/dtu.md))
- **ScanNet**: one mesh per scene ID. ([example](examples/scannet.md))
- **Tanks and Temples**: one point cloud per scene, scored by the official toolbox.
  ([example](examples/tanks_temples.md))
- **ETH3D**: one point cloud per scene in the GT (COLMAP) frame in meters.
  ([example](examples/eth3d.md))
