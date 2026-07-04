# 012 — Pose metrics

## Goal

Trajectory evaluation (ATE, RPE, rotation/translation error) delegates to evo with explicit
association and alignment policies; `e3r metric pose` runs end-to-end.

## Scope

- `backends/trajectory_evo.py` (`trajectory` registry kind): `evaluate_ate` /
  `evaluate_rpe` wrapping evo; TUM format first (source formats beyond TUM arrive with
  their dataset adapters).
- `metrics/pose.py`: normalize evo output into `MetricResult`s; record number of
  associated and dropped poses, association parameters (`associate_max_diff`), alignment
  mode, Sim3 scale when used, backend name + version.
- Alignment modes `trajectory_se3` / `trajectory_sim3` via evo; Sim3 disallowed for
  metric-scale protocols unless explicitly allowed by the protocol.
- Diagnostic metric `alignment_scale_error = |log(s)|`.
- `single_pose` built-in protocol through the task-006 runner; alignment transforms saved
  to `alignment_transforms.json`.
- CLI: `e3r metric pose pred_tum.txt --gt gt_tum.txt --backend evo --align sim3`.
- Optional-dependency behavior: without evo, the documented `pip install 'eval3r[pose]'`
  error; tests use `pytest.importorskip("evo")`.

## Out of Scope

- Dataset pose adapters (KITTI-360, Waymo, CO3D — backlog).
- Interpolation policies beyond evo defaults (record policy; extend later).

## Relevant Files

- `.agent/metrics.md` — "Pose metrics"
- `.agent/backends.md` — "Trajectory backend"
- `.agent/protocols.md` — `single_pose` template
- `.agent/plan.md` — "Pose metrics" milestone

## Plan

1. Implement evo wrapper with output normalization and metadata capture.
2. Implement pose metric layer + diagnostics.
3. Wire runner + CLI.
4. Tests: synthetic TUM trajectories with known transforms (identity → zero ATE; known
   Sim3 scale recovered; dropped-pose accounting), missing-evo error message, wrapper
   output normalization.

## Findings

(record during implementation)

## Decisions

(record during implementation; e.g. evo API vs CLI invocation)

## Verification

```bash
pytest tests/unit/test_pose*.py
e3r metric pose pred_tum.txt --gt gt_tum.txt --backend evo --align sim3  # on fixture
```

Acceptance per `.agent/plan.md` milestone.

## Status

todo
