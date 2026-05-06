# Eval3r

`eval3r` is a small toolkit for saving, evaluating, and visualizing 3D reconstruction predictions. It keeps reconstruction evaluation explicit: prediction format, pose conventions, coordinate systems, units, alignment, and metric variants are recorded instead of guessed.

## Install

```bash
pip install eval3r
```

Optional extras:

```bash
pip install eval3r[render]   # pyrender + pillow + imageio
pip install eval3r[dev]      # pytest + ruff + mypy + pre-commit
```

## Quick Start

Save a prediction:

```python
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

Run the CLI:

```bash
e3r validate outputs/scannet/scene0799_00
e3r inspect outputs/scannet/scene0799_00
e3r metric all outputs/scannet/scene0799_00 \
    --gt /data/scannet/scene0799_00/gt_mesh.ply \
    --align none --samples 200000 --seed 42 \
    --thresholds 0.05 --chamfer-variant l1_mean_bidirectional
e3r render mesh outputs/.../geometry/pred_mesh.ply --out render.png --headless
e3r metric depth pred_depth.png --gt /data/scannet/scene0799_00/depth/0.png
```

## Core Features

- Stable on-disk prediction format: manifest, geometry, trajectory, and cameras.
- `PredictionWriter` and `PredictionReader` APIs for research code.
- Geometry metrics: Chamfer, accuracy, completeness, precision, recall, and F-score.
- Depth metrics: AbsRel, SqRel, RMSE, RMSE log, and delta accuracy.
- Explicit alignment modes for geometry and trajectory-based evaluation.
- Dataset adapters for common reconstruction benchmarks.

## Guides

- [Metric Computation](metric.md)
- [Trajectory Pose File Formats](trajectory.md)
