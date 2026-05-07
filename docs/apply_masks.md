# Applying Occlusion Masks

This page describes the occlusion-mask file format, transform convention, and how masks are applied during evaluation.

## Mask artifact format

For each scene, provide a directory with:

- `occlusion_mask.npy`: a 3D array `(D_x, D_y, D_z)` where
  - `0` means **visible**
  - `1` means **occluded**
- `T_mask_scene.txt`: a whitespace-delimited `4x4` matrix.

### Transform convention

`T_mask_scene` maps homogeneous **scene/world** coordinates to continuous mask voxel coordinates:

- input: `[x, y, z, 1]^T`
- output: `[i, j, k, 1]^T`

So:

```text
[i, j, k, 1]^T = T_mask_scene @ [x, y, z, 1]^T
```

`[i, j, k]` are continuous voxel coordinates used with trilinear interpolation.
Integer values index voxel centers.

## Sampling and visibility rule

During filtering:

1. Points are transformed to voxel space with `T_mask_scene`.
2. Mask values are sampled trilinearly from `occlusion_mask.npy`.
3. Out-of-bounds samples are treated as occluded (`1.0`).
4. A point is kept iff sampled mask value `< 0.5`.

If all points are marked occluded, eval3r keeps all original points as a fallback to avoid unfairly collapsing the sample.

## CLI usage

### Metric CLI

Use per-scene masks with:

- `--mask-dir`
- `--mask-name` (default: `occlusion_mask.npy`)
- `--t-mask-scene-name` (default: `T_mask_scene.txt`)
- `--mask-mode` in `{pred, gt, both}` (default: `pred`)

Example:

```bash
e3r metric all outputs/scannet/scene0001_00 \
  --gt /data/scannet/scene0001_00/gt_mesh.ply \
  --mask-dir /data/scannet_masks \
  --mask-mode both
```

### Benchmark CLI

Use the same options:

```bash
e3r benchmark run scannet outputs/scannet \
  --root /data/scannet \
  --mask-dir /data/scannet_masks \
  --t-mask-scene-name T_mask_scene.txt \
  --mask-mode pred
```

## Mask mode semantics

- `pred`: apply mask only to predicted points.
- `gt`: apply mask only to ground-truth points.
- `both`: apply mask to both predicted and ground-truth points.

Default remains `pred` for backward compatibility.
