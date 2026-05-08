# Eval3r: 3D reconstruction evaluation, made explicit

[![PyPI version](https://img.shields.io/pypi/v/eval3r.svg)](https://pypi.org/project/eval3r/)
[![Python versions](https://img.shields.io/pypi/pyversions/eval3r.svg)](https://pypi.org/project/eval3r/)
[![Documentation Status](https://readthedocs.org/projects/eval3r/badge/?version=latest)](https://eval3r.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/xingruiy/eval3r/actions/workflows/ci.yml/badge.svg)](https://github.com/xingruiy/eval3r/actions/workflows/ci.yml)

> [!NOTE]
> Eval3r is a research-oriented project and is under active development, we strive to make it more useful for the research community. Please don't hesitate to open an issue or submit a pull request with feedback, suggestions, or improvements.

## Overview

Eval3r focuses on a small, explicit core:

- A handy CLI for 3D reconstruction evaluation and visualization.
- A stable on-disk **prediction format** (manifest + geometry + trajectory + cameras)

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

## Quick start
Suppose you have two mesh/point cloud files:

```text
pred.ply   # your reconstruction
gt.ply     # ground truth
````

You want to know: **how good is the reconstruction?**

### Run geometry metrics

```bash
e3r metric all pred.ply --gt gt.ply
```

This reports Chamfer distance, accuracy, completeness, precision, recall, and F-score.
Use this when both point clouds are already in the same coordinate frame.

### Visualize before trusting numbers

```bash
e3r metric all pred.ply --gt gt.ply --debug-plot
```

Check whether the two point clouds overlap.
If they are shifted, rotated, or scaled differently, raw metrics may look bad even when the shape is reasonable.


### Align before evaluation

#### Rigid alignment:

```bash
e3r metric all pred.ply --gt gt.ply --align icp --debug-plot
```

Use this when the prediction has correct scale but may be rotated or translated.

#### Similarity alignment:

```bash
e3r metric all pred.ply --gt gt.ply --align sim3 --debug-plot
```

Use this when the prediction may have unknown scale, such as monocular reconstruction.

#### Trajectory-based alignment:

```bash
e3r metric all pred.ply --gt gt.ply \
  --align traj_sim3 \
  --pred-traj pred_trajectory.txt \
  --gt-traj gt_trajectory.txt \
  --debug-plot
```

Use this when the method also predicts camera poses.

### Benchmark a dataset split

For a full dataset benchmark, organize predictions by scene:

```text
preds_root/
  scene0000_00/
    mesh.ply
  scene0001_00/
    mesh.ply
````

Then run:

```bash
e3r benchmark run scannet preds_root \
  --root /path/to/scannet \
  --split split.txt \
  --align sim3 \
  --debug-plot \
  --out results.json \
  --csv results.csv
```

This evaluates all scenes in the split and reports dataset-level summary statistics.

Useful options:

```bash
--align sim3              # alignment mode
--samples 200000          # points sampled per scene
--thresholds 0.05         # F-score threshold
--workers 8               # parallel workers
--debug-plot              # save per-scene visualization
--out results.json        # full benchmark result
--csv results.csv         # per-scene table
```

For custom prediction filenames:

```bash
e3r benchmark run scannet preds_root \
  --root /path/to/scannet \
  --split split.txt \
  --pred-filename "{scene_id}/mesh.ply" \
  --align sim3 \
  --csv results.csv
```

For trajectory-based alignment:

```bash
e3r benchmark run scannet preds_root \
  --root /path/to/scannet \
  --split split.txt \
  --align traj_sim3 \
  --pred-pose-dir pred_poses \
  --pred-pose-file "{scene_id}.txt" \
  --pred-pose-convention T_wc \
  --csv results.csv
```

Use `e3r metric` for one pair of geometries.
Use `e3r benchmark run` for dataset-level evaluation.

### Saving a prediction

If you want to save your method's output in Eval3r's format for downstream evaluation, use `PredictionWriter`:

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

### Document

For monocular or feedforward methods, `sim3` is often a good default.
For SLAM or pose-producing methods, prefer `traj_se3` or `traj_sim3`.

For more update-to-date examples, see the [docs](https://eval3r.readthedocs.io/en/latest/).

## License

Eval3r is released under the MIT License. See `pyproject.toml` for the canonical metadata.

## Disclaimer

Eval3r does **not** redistribute any third-party datasets. Datasets (ScanNet, Replica, DTU, ETH3D, Tanks & Temples, TUM RGB-D) remain under their original licenses; you must obtain them from their respective sources and abide by those terms. Adapter code in `eval3r/datasets/` only describes filesystem layouts — no dataset contents.