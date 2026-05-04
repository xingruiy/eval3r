# eval3r

Handy toolkit for saving, evaluating, and visualizing 3D reconstruction predictions.

`eval3r` v0.1 focuses on a small, explicit core:

- A stable on-disk **prediction format** (manifest + geometry + trajectory + cameras).
- A `PredictionWriter` / `PredictionReader` API for research code.
- Reliable geometry metrics — Chamfer (4 explicit variants), accuracy, completeness, F-score.
- An `e3r` CLI for `metric`, `validate`, `inspect`, `render`, and `preset`.
- Optional headless rendering via `pyrender`.

The goals are reproducibility and explicit assumptions: no silent alignment,
no silent unit conversion, no hidden default for pose conventions.

## Install

Base install (NumPy / SciPy / Pydantic / Typer / Rich / trimesh):

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e '.[render]'   # pyrender + pillow + imageio
pip install -e '.[dev]'      # pytest + ruff + mypy + pre-commit
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
```

## Chamfer variants

`eval3r.metrics.chamfer_distance` accepts an explicit `variant`:

| variant                    | formula                                        |
|----------------------------|------------------------------------------------|
| `l1_mean_bidirectional`    | `0.5 * (mean‖p−q‖ + mean‖q−p‖)`                |
| `l1_sum_bidirectional`     | `mean‖p−q‖ + mean‖q−p‖`                        |
| `l2_squared`               | `mean‖p−q‖² + mean‖q−p‖²`                      |
| `l2_unsquared`             | `mean‖p−q‖ + mean‖q−p‖`                        |

Be explicit about which one a paper or another codebase reports.

## Alignment

```bash
--align none      # default — never silently align
--align scale     # isotropic scale only
--align se3       # Umeyama R, t (or ICP without correspondences)
--align sim3      # Umeyama scale, R, t
--align icp       # point-to-point ICP from identity
```

