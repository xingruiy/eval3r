# eval3r

[![PyPI version](https://img.shields.io/pypi/v/eval3r.svg)](https://pypi.org/project/eval3r/)
[![Python versions](https://img.shields.io/pypi/pyversions/eval3r.svg)](https://pypi.org/project/eval3r/)
[![Documentation Status](https://readthedocs.org/projects/eval3r/badge/?version=latest)](https://eval3r.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/xingruiy/eval3r/actions/workflows/ci.yml/badge.svg)](https://github.com/xingruiy/eval3r/actions/workflows/ci.yml)

`eval3r` is a Python toolkit for **saving, validating, benchmarking, and visualizing 3D reconstruction results** with explicit assumptions and reproducible workflows.

It is designed for research and engineering settings where evaluation details matter:

- no hidden alignment behavior,
- no implicit pose convention conversions,
- no silent unit guessing,
- and a stable, inspectable prediction format.

## Why eval3r exists

Many reconstruction pipelines fail at the "last mile": prediction artifacts are hard to compare, evaluation settings are underspecified, and reproducing numbers from papers can be difficult.

`eval3r` addresses this with:

- a practical CLI (`e3r`) for everyday evaluation tasks,
- strongly typed I/O for predictions and trajectories,
- dataset adapters and presets,
- geometry + trajectory metrics,
- optional rendering and debug visualization,
- and end-to-end benchmark utilities.

## Installation

### Base package

```bash
pip install eval3r
```

### Optional extras

```bash
pip install eval3r[render]   # rendering and plotting extras
pip install eval3r[traj]     # trajectory evaluation helpers
pip install eval3r[torch]    # torch-dependent workflows
pip install eval3r[dev]      # tests, lint, type-check, pre-commit
```

## Command-line overview

After installation, `eval3r` exposes the `e3r` command.

```bash
e3r --help
```

Primary commands:

- `e3r metric ...` — compute geometry metrics for a prediction/GT pair
- `e3r benchmark ...` — evaluate many scenes and aggregate split-level metrics
- `e3r render ...` — render geometry and comparison outputs
- `e3r mask ...` — generate/inspect occlusion masks
- `e3r validate <prediction_dir>` — validate a prediction directory against manifest rules
- `e3r inspect <prediction_dir>` — inspect a prediction directory summary
- `e3r dataset ...` and `e3r preset ...` — inspect adapters and presets

## Quickstart: single-scene geometry evaluation

Assume:

```text
pred.ply   # prediction
gt.ply     # ground truth
```

### 1) Compute metrics

```bash
e3r metric all pred.ply --gt gt.ply
```

Typical outputs include Chamfer distance, accuracy/completeness, and thresholded precision/recall/F-score.

### 2) Always inspect overlap

```bash
e3r metric all pred.ply --gt gt.ply --debug-plot
```

If overlap is poor due to coordinate mismatch, metrics are not meaningful yet.

### 3) Select alignment intentionally

Use rigid alignment when scale is trustworthy:

```bash
e3r metric all pred.ply --gt gt.ply --align icp --debug-plot
```

Use similarity alignment when scale may drift (common in monocular systems):

```bash
e3r metric all pred.ply --gt gt.ply --align sim3 --debug-plot
```

Use trajectory-driven alignment when poses are available:

```bash
e3r metric all pred.ply --gt gt.ply \
  --align traj_sim3 \
  --pred-traj pred_trajectory.txt \
  --gt-traj gt_trajectory.txt \
  --debug-plot
```

## Quickstart: benchmark a dataset split

Expected prediction layout example:

```text
preds_root/
  scene0000_00/
    mesh.ply
  scene0001_00/
    mesh.ply
```

Run benchmark:

```bash
e3r benchmark preds_root \
  --dataset scannet \
  --root /path/to/scannet \
  --split split.txt \
  --align sim3 \
  --workers 8 \
  --out results.json \
  --csv results.csv
```

Useful options:

- `--samples 200000` points sampled per scene
- `--thresholds 0.05` threshold(s) for F-score-like metrics
- `--debug-plot` to save visual diagnostics
- `--pred-filename` to map custom prediction filenames

Trajectory-based benchmark alignment example:

```bash
e3r benchmark run scannet preds_root \
  --root /path/to/scannet \
  --split split.txt \
  --align traj_sim3 \
  --pred-pose-dir pred_poses \
  --pred-pose-file "{scene_id}.txt" \
  --pred-pose-convention T_wc
```

## Prediction format and Python API

`eval3r` includes a manifest-driven prediction format so files are portable and self-described.

Minimal writer example:

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

When key metadata is missing, the writer records explicit placeholders instead of guessing, enabling downstream validation and auditability.

## Supported workflows

- Scene-level metric inspection during model development.
- Dataset-level benchmark reporting for ablations and papers.
- Reproducible artifact exchange between teams using manifest-backed prediction directories.
- Alignment experiments (`icp`, `sim3`, trajectory-driven variants) with debug plots.

## Documentation

- Docs site: https://eval3r.readthedocs.io/
- Repository: https://github.com/xingruiy/eval3r

## Development

```bash
pip install -e .[dev]
pytest
```

## License and dataset disclaimer

`eval3r` is released under the MIT License.

This project does **not** redistribute third-party datasets such as ScanNet, Replica, DTU, ETH3D, Tanks & Temples, or TUM RGB-D. You must obtain and use those datasets under their original licenses.
