# Example: ScanNet

ScanNet's GT meshes are **BundleFusion reconstructions — not independent measurements**.
eval3r records the GT as `reconstructed` / `reconstruction_derived`, and the geometry
protocols are `native`: ScanNet has no official reconstruction benchmark, so these
follow the community single-/double-layer 5 cm F-score convention and must never be
presented as official ScanNet numbers.

## Dataset layout

Standard SensReader-style export, one directory per scene:

```text
<root>/scans/scene0000_00/scene0000_00_vh_clean_2.ply   GT mesh
<root>/scans/scene0000_00/scene0000_00.txt              scene metadata
<root>/scans/scene0000_00/pose/ intrinsic/ depth/ ...   exported trajectory data
<root>/val.txt (or <root>/splits/val.txt)               scene-id list
```

Exported poses are camera-to-world, OpenCV-style; depth is 16-bit millimeters
(`depth_unit` 0.001, invalid value 0).

## Predictions

One reconstructed mesh per scene, named by scene ID:

```text
preds/scene0000_00.ply
preds/scene0001_00.ply
```

## Run

```bash
e3r benchmark run preds/ --dataset scannet --split val \
  --protocol scannet_single_layer_geometry_5cm --root /data/scannet \
  --method mymethod --out runs/scannet_mymethod
```

## Visibility culling is protocol-defined

- `scannet_single_layer_geometry_5cm` / `scannet_double_layer_geometry_5cm` (val split):
  **no visibility culling**; `culled_fraction` is recorded as 0.
- `scannet_test_single_layer_geometry_5cm` (test split): applies the community
  NeuralRecon/TransformerFusion-style culling — the *prediction's* depth is rendered from
  the GT camera trajectory (pyrender, headless EGL) and TSDF-integrated (open3d) to
  recover the observed region; prediction geometry outside it is trimmed before scoring.

Culling never runs silently: it happens only when the protocol's masking requests it, and
the renderer, TSDF backend, their versions, the voxel size, the trajectory fingerprint,
and the per-scene culled fraction are all recorded in result metadata. This is a
visibility mask over the caller's prediction, not reconstruction — eval3r produces no new
scene geometry.

Both mesh-layer conventions exist as separate protocols because they are different
benchmarks; eval3r never silently chooses one.
