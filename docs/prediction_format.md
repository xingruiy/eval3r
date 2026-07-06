# Prediction format

## The official eval3r-native layout

The recommended way to hand predictions to eval3r is a **self-contained prediction
directory**: a `manifest.yaml` at the root plus one directory per scene with
canonical filenames. `e3r benchmark run <pred_root> ...` consumes it unchanged
(the manifest is picked up automatically), and the directory is relocatable
because every manifest path is relative to the root.

```text
<pred_root>/
  manifest.yaml                  # PredictionManifest (schema below)
  <scene_id>/
    mesh.ply                     # or mesh.obj, ... — the source file's suffix is kept
    pointcloud.ply
    pointmap.npy                 # (N, 3) or (H, W, 3) float; NaN marks invalid points
    depth.png  |  depth/         # single depth file, or a directory of depth frames
    trajectory_tum.txt           # per-scene camera trajectory (TUM: t x y z qx qy qz qw)
    cameras.json
    confidence.npy | confidence/
```

Each filename maps 1:1 onto a `ScenePredictionEntry` field. The declared
`prediction_modality` decides which entry every scene must have (`mesh` →
`mesh`, `pointcloud` → `pointcloud`, `pointmap` → `pointmap`, `single_depth` →
`depth`, `depth_sequence` → `depth_dir`, `camera_trajectory` → `trajectory`);
everything else is auxiliary and always allowed. `trajectory_tum.txt` is what
protocol-driven trajectory-first alignment (see [Alignment](alignment.md)) reads
for the prediction side.

The writer records provenance in the manifest's open-ended `metadata` dicts (the
schema itself is unchanged): `metadata.layout = "eval3r-native-v1"`,
`metadata.eval3r_version`, and per-scene `metadata.fingerprints` with a
`sha256:` digest per file (directory entries get a joint hash over the sorted
file list), so a prediction directory can be verified long after export.

## Writing predictions: `PredictionWriter`

```python
import numpy as np
from eval3r import PredictionWriter

with PredictionWriter(
    "exports/my_method_scannet",
    method="my_method", version="2026-07-06",
    dataset="scannet", variant="scannetv2_val_single_layer", split="val",
    prediction_modality="mesh", scale="metric", coordinate_frame="world",
    source_pose_format="cam_to_world_opencv", intrinsics_source="predicted",
) as writer:
    for scene_id, outputs in my_method_results.items():
        writer.add_scene(
            scene_id,
            mesh=outputs.mesh_path,                # files are copied in
            trajectory=outputs.poses_tum,          # (N, 8) array -> trajectory_tum.txt
        )
# __exit__ validates everything and writes manifest.yaml
```

`add_scene` accepts source file paths (copied into the layout) and, where eval3r
owns a plain writer, numpy arrays: point clouds ((N, 3) → PLY), pointmaps
((N, 3) or (H, W, 3) → `.npy`), confidence (float array → `.npy`), and
trajectories ((N, 8) `[t x y z qx qy qz qw]` rows → TUM text). Meshes, depth
files, and camera files are **copy-only** — eval3r builds no geometry. The
writer refuses duplicate scenes, entries inconsistent with the declared
modality, missing source files, malformed arrays, and overwriting an existing
export; a failed `add_scene` leaves nothing behind, and a directory that hit an
exception mid-export never gets a manifest.

## Reading and verifying: `read_prediction_dir` / `e3r prediction`

```python
from eval3r import read_prediction_dir

check = read_prediction_dir("exports/my_method_scannet", verify=True)
check.manifest                    # validated PredictionManifest
check.scenes["scene0000_00"]      # {"mesh": <absolute Path>, ...}
```

`read_prediction_dir` resolves every declared path to an absolute path and
raises one error naming **every** missing scene/field/file (never just the
first); `verify=True` additionally re-hashes each file against the recorded
fingerprints. The same checks drive the CLI:

```bash
e3r prediction validate exports/my_method_scannet          # schema + files + fingerprints
e3r prediction validate exports/my_method_scannet --no-verify   # skip re-hashing
e3r prediction show exports/my_method_scannet              # manifest + scene/file table
```

`validate` prints a per-scene outcome table and lists every failure verbatim
(exit 1 on any); `show` summarizes the manifest and the file layout without
hashing.

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
reconstruction. `colmap_reconstruction` has no eval3r-native layout (a COLMAP model is a
directory of dataset-specific files); describe such predictions with a hand-authored
manifest.

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

The manifest declares what the predictions *are* — modality, scale, coordinate frame,
pose convention, GT usage, and confidence behavior — so the result can be interpreted
later. `PredictionWriter` produces it; it can also be authored by hand (fingerprints are
then optional, but `e3r prediction validate` will say so instead of passing vacuously):

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
    mesh: scene0000_00/mesh.ply
    metadata:
      fingerprints:
        mesh: sha256:9f2c...
metadata:
  layout: eval3r-native-v1
  eval3r_version: 0.5.0
```

Rules:

- `uses_gt` matters: a method that consumed GT poses is not comparable to one that did
  not, and reports surface this.
- If a method self-filters its predictions by confidence before export, the manifest must
  say so (`confidence.self_filtered`) — reports distinguish uniform protocol filtering
  from method-native filtering. The writer refuses per-scene confidence files unless
  `confidence.present` is declared.
- Scene entries should be explicit for benchmark runs.
- The manifest is copied into the run directory (`manifest.yaml`).

## Manifest resolution

`e3r benchmark run preds/` resolves the manifest in this order:

1. `--manifest path.yaml` if given,
2. `preds/manifest.yaml` if present (this is how the eval3r-native layout is consumed),
3. otherwise a simple per-scene layout is **inferred**, written to the run directory,
   and marked as inferred.

`e3r benchmark validate` checks preflight and per-scene prediction resolution for every
scene without evaluating anything — use it before long runs. `e3r prediction validate`
checks the prediction directory itself (schema, files, fingerprints) without needing a
dataset.

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

These conventions stay inside the adapters; the eval3r-native layout above is the
dataset-agnostic interchange format.
