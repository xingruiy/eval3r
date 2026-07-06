# Example: ETH3D (high-res DSLR multi-view)

ETH3D training-split GT is an **independent, occlusion-aware laser scan**; scoring is
done by the **real official `multi-view-evaluation` binary** (fidelity `official`). The
test split is **server-only** and refused locally. This adapter covers the high-res DSLR
multi-view benchmark; the low-res multi-camera variant is a different benchmark.

## One-time setup: the official binary

Build the official tool and point eval3r at it — see
[Installation](../install.md#eth3d-multi-view-evaluation-binary):

```bash
export EVAL3R_ETH3D_TOOL=/path/to/multi-view-evaluation/build/ETH3DMultiViewEvaluation
```

## Dataset layout

One directory per scene, as unpacked from the official `<scene>_dslr_undistorted` and
`<scene>_dslr_scan_eval` archives:

```text
<root>/courtyard/dslr_scan_eval/scan_alignment.mlp        GT scan poses (MeshLab project)
<root>/courtyard/dslr_scan_eval/scan*.ply                 laser scans referenced by the .mlp
<root>/courtyard/dslr_calibration_undistorted/cameras.txt COLMAP text cameras
<root>/courtyard/dslr_calibration_undistorted/images.txt  COLMAP text image poses
```

Cameras are parsed with pycolmap; non-pinhole camera models are recorded as explicit
limitations in scene metadata, never silently approximated as pinhole. The GT fingerprint
jointly hashes `scan_alignment.mlp` and every referenced scan PLY.

## Predictions

One point cloud per scene, **already in the ETH3D ground-truth (COLMAP) frame in
meters** — the official tool applies no alignment:

```text
preds/courtyard.ply
preds/delivery_area.ply
```

## Run

```bash
e3r benchmark run preds/ --dataset eth3d --split training \
  --protocol eth3d_training_official --root /data/eth3d \
  --method mymethod --out runs/eth3d_mymethod
```

## What the official tool owns

The official scoring is not a plain distance metric: completeness and accuracy are
voxel-normalized over two shifted voxel grids, and prediction points are classified
accurate / inaccurate / **unobserved** via beam-based free-space modeling from the scan
positions (unobserved points are excluded). eval3r never reimplements this. Metrics are
the official accuracy / completeness / F1 at the official tolerance set (1, 2, 5, 10,
20, 50 cm; the 2 cm F1 is the conventional headline number). Voxel and beam parameters
stay at the official defaults and are recorded per scene, along with the tool path,
source commit, and exact command.
