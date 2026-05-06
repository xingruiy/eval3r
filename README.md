# Eval3r: 3D reconstruction evaluation, made explicit
## Overview

`eval3r` focuses on a small, explicit core:

- A stable on-disk **prediction format** (manifest + geometry + trajectory + cameras).
- A `PredictionWriter` / `PredictionReader` API for research code.
- Reliable geometry metrics — Chamfer (4 explicit variants), accuracy, completeness, F-score.
- Depth metrics — AbsRel, SqRel, RMSE, RMSE log, and delta accuracy (δ < 1.25).
- An `e3r` CLI for `metric`, `validate`, `inspect`, `render`, and `preset`.
- Optional headless rendering via `pyrender`.

The goals are reproducibility and explicit assumptions: no silent alignment, no silent unit conversion, no hidden default for pose conventions.

## Install

Base install:

```bash
pip install eval3r
```

Optional extras:

```bash
pip install eval3r[render]   # pyrender + pillow + imageio
pip install eval3r[dev]      # pytest + ruff + mypy + pre-commit
```

## Quick start — saving a prediction

```python
import numpy as np
import eval3r as e3r

with e3r.PredictionWriter(
    "outputs/scannet/scene0799_00",
    scene_id="scene0799_00",
    dataset="scannet",
    method="my_method",
    unit="m",
    coordinate_system="opengl",
    pose_convention="T_wc",
) as pred:
    pred.save_point_cloud(points, colors=colors)
    pred.save_mesh(vertices, faces)
    pred.save_poses(poses, timestamps=timestamps)
    pred.save_metadata({"checkpoint": "ckpt.pth"})
```

The writer warns if `unit`, `coordinate_system`, or `pose_convention` is left
unspecified — the manifest will record `"unspecified"` so downstream evaluation
can flag the ambiguity instead of guessing.

## Quick start — CLI

```bash
e3r validate outputs/scannet/scene0799_00
e3r inspect  outputs/scannet/scene0799_00
e3r metric all outputs/scannet/scene0799_00 \
    --gt /data/scannet/scene0799_00/gt_mesh.ply \
    --align none --samples 200000 --seed 42 \
    --thresholds 0.05 --chamfer-variant l1_mean_bidirectional
e3r render mesh outputs/.../geometry/pred_mesh.ply --out render.png --headless
e3r metric depth pred_depth.png --gt /data/scannet/scene0799_00/depth/0.png
```

## Supported datasets

`eval3r` ships dataset adapters that describe standard filesystem layouts.
Adapters own per-dataset assets (mesh, point cloud, depth, color, poses,
intrinsics) and are discoverable through the registry:

| dataset | GT format | description |
|---|---|---|
| `scannet` | mesh | ScanNet v2 — per-scene PLY meshes, RGB-D frames, camera poses |
| `replica` | mesh | Replica — high-quality indoor scene reconstructions |
| `dtu` | point cloud | DTU MVS — structured-light point clouds, evaluation subset of 19 scans |
| `tanks_temples` | point cloud | Tanks & Temples — laser-scan point clouds, training / intermediate / advanced subsets |
| `eth3d` | mesh / point cloud | ETH3D — high-res (dslr) and low-res (rig) tracks, COLMAP calibration |
| `tum_rgbd` | — | TUM RGB-D — handheld SLAM sequences (depth, color, poses, no geometry GT) |

## Benchmarking a method against a dataset

```bash
# List all registered adapters and inspect a specific one.
e3r datasets list
e3r datasets show scannet
e3r datasets show dtu

# Check that a dataset root matches the expected layout.
e3r datasets validate scannet --root /data/scannet \
    --split /data/scannet/splits/scannetv2_test.txt
e3r datasets validate dtu --root /data/dtu

# Run a method's predictions across a full split (generic command).
e3r benchmark run scannet outputs/scannet \
    --root /data/scannet \
    --split /data/scannet/splits/scannetv2_test.txt \
    --thresholds 0.05 --workers 8 \
    --out results.json --csv results.csv

# Run against datasets with non-default layout via adapter opts.
e3r benchmark run eth3d outputs/eth3d \
    --root /data/eth3d --track dslr

e3r benchmark run tanks_temples outputs/tnt \
    --root /data/tnt --subset training

# Pass adapter-specific overrides with -o key=value.
e3r benchmark run tum_rgbd outputs/tum \
    --root /data/tum -o intrinsics_fx=535.4 -o intrinsics_cx=320.1
```

eval3r does not ship dataset splits — pass a path to a text file with one
scene id per line. Omit `--split` to auto-discover scenes. DTU defaults to
the standard 19-scan evaluation subset; Tanks & Temples accepts `--subset`
(`training`, `intermediate`, or `advanced`); ETH3D accepts `--track` (`dslr`
or `rig`).

The benchmark reports two summaries: `summary` (mean / median / std over
**successful** scenes only) and `summary_all` (over **all** scenes with
missing or failed scenes filled in by the configured defaults — distance
metrics → `--missing-distance-default`, default `1.0` m; F-score / precision
/ recall → `--missing-fscore-default`, default `0.0`). The defaults are
recorded in the result `config` so leaderboard numbers are reproducible.

The locator looks for `eval3r_prediction.json` first, then falls back to
`<scene>/mesh.ply`, `<scene>/<scene>_mesh.ply`, `<scene>/<scene>.ply`, etc.
Custom layouts are supported via `--geometry-pattern "<scene>/out/final.ply"`
(repeatable, prepended to the defaults). Adapter overrides use `-o key=value`
(e.g. `-o depth_scale=5000`, `-o mesh_filename=scan.ply`).

## Chamfer variants

`eval3r.metrics.chamfer_distance` accepts an explicit `variant`:

| variant                    | formula                                        |
|----------------------------|------------------------------------------------|
| `l1_mean_bidirectional`    | `0.5 * (mean‖p−q‖ + mean‖q−p‖)`                |
| `l1_sum_bidirectional`     | `mean‖p−q‖ + mean‖q−p‖`                        |
| `l2_squared`               | `mean‖p−q‖² + mean‖q−p‖²`                      |
| `l2_unsquared`             | `mean‖p−q‖ + mean‖q−p‖`                        |

Be explicit about which one a paper or another codebase reports.

## Depth metrics

```bash
e3r metric depth pred_depth.png --gt gt_depth.png
e3r metric depth pred_depth.npy --gt gt_depth.npy --mask valid.npy
e3r metric depth pred_depth.png --gt gt_depth.png --json
```

| metric            | description                                |
|-------------------|--------------------------------------------|
| `abs_rel`         | mean(\|pred − gt\| / gt)                    |
| `sq_rel`          | mean((pred − gt)² / gt)                     |
| `rmse`            | sqrt(mean((pred − gt)²))                    |
| `rmse_log`        | sqrt(mean((log pred − log gt)²))            |
| `delta1`          | fraction of pixels where max(ratio, 1/ratio) < 1.25 |
| `delta2`          | same for threshold 1.25²                   |
| `delta3`          | same for threshold 1.25³                   |

Pixels where either depth is zero, negative, NaN, or Inf are excluded automatically.
An optional `--mask` can further restrict valid pixels.

## Alignment

Many reconstruction methods only predict geometry up to an unknown scale, rotation, and translation. To properly evaluate these methods, `eval3r` requires an explicit `--align` argument to align the prediction to the ground truth before computing metrics. The options are:

```bash
--align none      # default — never silently align
--align scale     # isotropic scale only
--align se3       # Umeyama R, t (or ICP without correspondences)
--align sim3      # Umeyama scale, R, t
--align icp       # point-to-point ICP from identity
```

## Feedback

eval3r is a research-oriented project. Bug reports, feature requests, and
general feedback are welcome — please open an issue on GitHub.

## License

eval3r is released under the MIT License. See `pyproject.toml` for the
canonical metadata.

eval3r does **not** redistribute any third-party datasets, splits, or
ground-truth meshes. Datasets (ScanNet, Replica, DTU, ETH3D, Tanks & Temples,
TUM RGB-D) remain under their original licenses; you must obtain them from
their respective sources and abide by those terms. Adapter code in
`eval3r/datasets/` only describes filesystem layouts — no dataset content
ships in the package.
