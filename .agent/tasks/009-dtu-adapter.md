# 009 — DTU adapter

## Goal

DTU scenes can be discovered and normalized locally through the generic benchmark-run
plumbing, with laser-scan GT point clouds, ObsMask/Plane metadata, millimeter handling, and
honest capability/provenance reporting.

## Scope

- `datasets/dtu.py` implementing the `DatasetAdapter` interface from `.agent/datasets.md`:
  scan ID resolution, `stlXXX_total.ply` GT loading, `ObsMaskXXX_10.mat` and `PlaneXXX.mat`
  resolution, native mm -> internal meters (source unit recorded), prediction filename
  resolution honoring the light-condition suffix (`<method>XXX_l3.ply`; do not ignore it).
- Missing Plane files recorded explicitly in scene metadata; protocol failure policy decides
  what happens (never silently skip while claiming official-like fidelity).
- Capabilities: dense_geometry, independent_gt, official_local_eval initially false until
  task 010 validates the official-like evaluator; local evaluation status can still be
  `supported` for eval3r-native point-cloud runs.
- GroundTruthSpec: pointcloud / laser_scan / independent / dense_surface; GT fingerprinting
  for point cloud + ObsMask/Plane where practical.
- Tiny synthetic DTU-layout fixture (`tests/fixtures/dtu_tiny/`) with a handful of points
  and miniature ObsMask/Plane `.mat` files.
- Integration with `e3r benchmark run --dataset dtu` using the generic point-cloud geometry
  runner, but not yet claiming official-like fidelity.

## Out of Scope

- Validated Python port or MATLAB wrapper for official DTU evaluation (task 010).
- Any other dataset adapter.
- Camera/pose loading for DTU (not needed for point-cloud geometry evaluation).

## Relevant Files

- `.agent/datasets.md` — "DTU" adapter requirements and rules, adapter interface
- `.agent/protocols.md` — DTU protocol template, with fidelity finalized in task 010
- `.agent/plan.md` — "Dataset adapter for DTU" milestone
- `CLAUDE.md` — DTU cautions

## Plan

1. Implement adapter discovery, GT/mask/plane resolution, unit handling, and prediction
   resolution with light-suffix parsing.
2. Add fixture data and tests for scan discovery, mm->m normalization, metadata recording,
   missing-Plane behavior, capability declaration, and GT fingerprinting.
3. Wire DTU through benchmark-run plumbing for an eval3r-native point-cloud smoke test.

## Findings

(record during implementation)

## Decisions

(record during implementation)

## Verification

```bash
pytest tests/unit/test_dtu_adapter*.py tests/integration/test_dtu_adapter_benchmark*.py
e3r benchmark run tests/fixtures/dtu_tiny/preds --dataset dtu --split test --protocol single_geometry
```

Acceptance: DTU layout and metadata resolve correctly, and results clearly avoid claiming
official-like fidelity until task 010 is complete.

## Status

todo

