# 007 — DTU adapter and official-like evaluation backend

## Goal

DTU benchmark evaluation runs locally through `e3r benchmark run` with official-like
fidelity: laser-scan GT point clouds, ObsMask/Plane culling, millimeter normalization, and
a regression-tested evaluation backend.

## Scope

- `datasets/dtu.py` implementing the `DatasetAdapter` interface from `.agent/datasets.md`:
  scan ID resolution, `stlXXX_total.ply` GT loading, `ObsMaskXXX_10.mat` and `PlaneXXX.mat`
  resolution, native mm → internal meters (source unit recorded), prediction filename
  resolution honoring the light-condition suffix (`<method>XXX_l3.ply`; do not ignore it).
- Missing Plane files recorded explicitly in scene metadata; protocol failure policy decides
  what happens (never silently skip while claiming official-like fidelity).
- `backends/dtu_eval.py` (`official_eval` registry kind): validated Python port of the
  official MATLAB point-cloud evaluation (accuracy/completeness with ObsMask/Plane and the
  0.2 mm downsampling), matching MATLAB behavior within documented tolerance; records
  whether the evaluator is the validated port or the MATLAB script.
- Capabilities: dense_geometry, independent_gt, official_local_eval=true,
  official_local_eval_method=validated_official_port.
- GroundTruthSpec: pointcloud / laser_scan / independent / dense_surface; GT fingerprinting
  (point cloud + ObsMask/Plane where practical).
- First `e3r benchmark run` integration: scene iteration, manifest resolution, per-scene
  runner invocation, aggregate results (this generalizes to later adapters).
- Tiny synthetic DTU-layout fixture (`tests/fixtures/dtu_tiny/`) with a handful of points
  and miniature ObsMask/Plane `.mat` files.
- Regression target: documented comparison against known official-like outputs, executed
  when full data is available locally (documented, not in normal CI).

## Out of Scope

- Any other dataset adapter.
- Camera/pose loading for DTU (not needed for point-cloud geometry evaluation).

## Relevant Files

- `.agent/datasets.md` — "DTU" adapter requirements and rules, adapter interface
- `.agent/protocols.md` — `dtu_official_like_pointcloud` template
- `.agent/backends.md` — "DTU evaluation backend"
- `.agent/plan.md` — "Dataset adapter for DTU" milestone
- `CLAUDE.md` — DTU cautions

## Plan

1. Implement adapter (discovery, GT/mask/plane resolution, unit handling, prediction
   resolution with light-suffix parsing).
2. Implement dtu_eval backend; validate the port against the official MATLAB code's
   published behavior on the fixture (record method + tolerance in Findings).
3. Wire `e3r benchmark run --dataset dtu --split test --protocol
   dtu_official_like_pointcloud` through the task-006 runner.
4. Fixture + tests: scene discovery, mm→m normalization, ObsMask/Plane application,
   missing-Plane behavior, light-suffix resolution, capability declaration, benchmark
   integration end-to-end on the fixture.

## Findings

(record during implementation — especially port-vs-MATLAB validation evidence)

## Decisions

(record during implementation)

## Verification

```bash
pytest tests/unit/test_dtu*.py tests/integration/test_dtu_benchmark*.py
e3r benchmark run preds/ --dataset dtu --split test --protocol dtu_official_like_pointcloud  # on fixture
```

Acceptance per `.agent/plan.md` milestone; results.json records evaluator method and Plane
availability per scene.

## Status

todo
