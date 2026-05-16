# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e .[dev]          # install with dev extras
pip install -e .[dev,render]   # also install pyrender extras

pytest                         # full test suite
pytest tests/metrics/          # single directory
pytest tests/io/test_trajectory.py::test_round_trip  # single test

ruff check eval3r/             # lint (line-length=100, target py3.10)
mypy eval3r/                   # type-check

hatch build                    # build wheel
```

The CLI entry point is `e3r` → `eval3r.cli.main:app`.

## Architecture

eval3r evaluates 3D reconstruction predictions. The core abstraction is the **Pipeline**, which composes four stages: sample → align → filter → measure. Everything revolves around:

1. **Manifest** (`eval3r/manifest/`) — A prediction lives in a directory with an `eval3r_prediction.json` manifest (Pydantic model in `manifest.py`). `PredictionWriter` writes it; `PredictionReader` loads it. Each artifact carries a SHA-256 hash and metadata (units, pose convention, coordinate system).

2. **Pipeline** (`eval3r/pipeline.py`) — `Pipeline(sampler, aligner, filters, metrics)` runs the full eval loop. `EvalConfig` sets `n_samples` (default 200k) and random `seed`. Returns `PipelineResult` with per-metric values, sample counts, and alignment info.

3. **Data flow**:
   ```
   mesh/point cloud → sample_points() → align (ICP/trajectory/none)
     → filter (occlusion/bbox/polygon) → KNN distances → metrics
   ```
   KNN distances (`d_pg`, `d_gp`) are computed once via `scipy.spatial.cKDTree` and shared across all metrics.

4. **Geometry I/O** (`eval3r/io/`) — `MeshData(vertices, faces)` and `PointCloudData(points, colors)` are the internal types. All pose sequences are `(T, 4, 4)` float64 homogeneous matrices. Trajectories use TUM format (8 cols: `timestamp tx ty tz qx qy qz qw`) or KITTI (12 cols: row-major 3×4).

5. **Benchmarks** (`eval3r/benchmark/`) — `BaseBenchmark` runs multi-scene evaluation. Dataset adapters (ScanNet, DTU, ETH3D, Replica, …) know ground-truth layout. `BenchmarkConfig` fields control alignment mode, sampling, and filtering per dataset.

6. **Alignment modes** — `none`, `icp_se3`, `icp_sim3`, `traj_se3`, `traj_sim3`. `umeyama()` in `alignment/base.py` is the closed-form SVD solver. `TrajectoryAligner` matches frames by timestamp when trajectory lengths differ.

## Key conventions

- **Pose convention** is stored as metadata (`T_wc`, `T_cw`, `unspecified`) and must be set explicitly. Same applies to `units` (`m`, `cm`, `mm`, `unspecified`) and `coordinate_system` (`colmap`, `opengl`, `opencv`, `unspecified`).
- **Quaternion order** in TUM files: `qx qy qz qw` (scipy convention). ETH3D COLMAP files use `qw qx qy qz` and the reader swaps them.
- **Optional heavy imports** (`pyrender`, etc.) are lazy-loaded via `eval3r.utils.optional.optional_import()` with a helpful error pointing to the relevant extra.
- **Line length:** 100 characters (ruff).
