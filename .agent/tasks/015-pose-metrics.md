# 015 — Pose metrics

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
- Alignment modes `trajectory_se3` / `trajectory_sim3` via evo.
- Diagnostic metric `alignment_scale_error = |log(s)|`.
- `single_pose` built-in protocol through the task-007 runner; alignment transforms saved
  to `alignment_transforms.json`.
- CLI: `e3r metric pose pred_tum.txt --gt gt_tum.txt --backend evo --align sim3`.
- evo is always installed (no extra); tests exercise it directly without `pytest.importorskip`.

## Out of Scope

- Dataset pose adapters (KITTI-360, CO3D — backlog).
- Interpolation policies beyond evo defaults (record policy; extend later).

## Relevant Files

- `.agent/metrics.md` — "Pose metrics"
- `.agent/backends.md` — "Trajectory backend"
- `.agent/protocols.md` — `single_pose` template
- `.agent/plan.md` — "Pose metrics" milestone
- `eval3r/backends/trajectory_evo.py`, `eval3r/metrics/pose.py`,
  `eval3r/pipeline/pose_runner.py`, `eval3r/cli/metric.py`, `eval3r/api.py`

## Plan

1. Implement evo wrapper with output normalization and metadata capture.
2. Implement pose metric layer + diagnostics.
3. Wire runner + CLI.
4. Tests: synthetic TUM trajectories with known transforms (identity → zero ATE; known
   Sim3 scale recovered; dropped-pose accounting), missing-evo error message, wrapper
   output normalization.

## Findings

- **The old `single_pose.yaml` left RPE deltas to evo's defaults.** evo defaults RPE to
  delta = 1 frame — a hidden choice, and RPE at delta = 1 frame is not comparable to
  delta = 1 second. Pinned `delta: 1, delta_unit: frames, all_pairs: false` explicitly in
  both `rpe_*` MetricSpecs (and `offset: 0.0` in the association parameters) →
  protocol_version 0.2.0, new hash pinned in test_protocols.py. An `rpe_*` spec without
  delta parameters, or an `ate`/`rpe_*` spec without an explicit statistic, now fails
  loudly.
- **evo's API surface (1.36.4) verified before writing the wrapper**:
  `sync.associate_trajectories(traj_ref, traj_est, max_diff, offset_2)`,
  `PoseTrajectory3D.align(traj_ref, correct_scale) -> (R, t, s)` (Umeyama),
  `metrics.APE/RPE(...).get_all_statistics()` → rmse/mean/median/std/min/max/sse.
  Exceptions to wrap: `FileInterfaceException` (loading), `SyncException` (association),
  `MetricsException` (RPE with too few poses). Sim3 alignment on pred = 2·gt + offset
  recovers scale 0.5 to 1e-9 (unit + integration tests assert this analytically).
- **RPE rotation is alignment-invariant, RPE translation is not scale-invariant** —
  a constant per-frame extra rotation θ gives rpe_rotation = θ exactly regardless of
  alignment (test at θ = 10°); rigid offsets leave rpe_translation at 0, but an
  uncorrected 2× scale does not. This is why the alignment mode + estimated scale go
  into *every* pose metric's metadata, not only ATE's.
- **Acceptance-run gotcha**: a stale non-editable `eval3r` copy exists in the
  conda env's site-packages; `python -m eval3r.cli.main` resolves the *local tree* only
  when run from the repo directory (cwd precedes site-packages). Acceptance runs must be
  driven from the repo root (the PATH `e3r` is a different package entirely — see memory).
- `test_smoke.py::test_stub_command_fails_loudly_with_reason` pointed at `metric pose`
  as the canonical stub; retargeted to `e3r diff` (task 016). The unused `_not_yet`
  helper in `cli/metric.py` was removed (the `diff` stub in `cli/main.py` keeps its own).

## Decisions

Superseded design note (2026-07-08): protocol-level Sim3/adaptation gates were removed.
Users may choose pose convention, scale declaration, and alignment/adaptation method; eval3r
records the resulting adaptation instead of refusing it through the protocol.

- **evo Python API in-process, not the evo CLI.** evo is a required base pip dependency
  (a delegation backend like scipy — the official-toolbox rule does not apply to an
  eval3r_native protocol), the API returns exact float statistics without stdout parsing,
  and the version is recorded via `evo.__version__` (leading "v" stripped) in
  `backend_versions`. No fallback evaluator exists.
- `backends/trajectory_evo.py`: `EvoTrajectoryBackend` (`trajectory` kind, name `evo`)
  with the `.agent/backends.md` interface: `evaluate_ate(pred, gt, align, association)`
  and `evaluate_rpe(..., pose_relation, delta, delta_unit, all_pairs)`. Result dicts are
  self-describing: full stats, n_pred/n_gt poses, n_associated + dropped counts per side,
  association policy (`nearest_timestamp`, explicit `associate_max_diff` — the backend
  refuses to default it, explicit `offset`), and the estimated alignment (mode, R, t, s).
  ATE = APE translation_part (metres); RPE pose relations restricted to
  `translation_part` (m) / `rotation_angle_deg` (deg). TUM format only for now.
- `metrics/pose.py`: metric names `ate`, `rpe_translation`, `rpe_rotation`,
  `alignment_scale_error`; alignment modes `none` / `trajectory_se3` / `trajectory_sim3`
  (granularity `per_scene` only — one estimate per trajectory pair);
  `alignment_scale_error = |ln s|` (fails on non-positive/non-finite scale);
  `pose_metric_result` copies alignment mode/scale, association policy, and pose counts
  into every MetricResult's metadata; `n_points_pred`/`n_points_gt` carry the raw pose
  counts.
- `pipeline/pose_runner.py` mirrors the task-007/014 runners (stages resolve → load →
  align → metric → aggregate, same failure policies / SceneFailure / RunResult assembly).
  CLI shorthands `se3`/`sim3` normalize to the first-class `trajectory_se3` /
  `trajectory_sim3` before hashing. Treats alignment selection
  as prediction/run adaptation and records it instead of protocol-gating it. The alignment record
  (mode, rotation, translation, scale, `|ln s|`, n_poses_used) goes to
  `alignment_transforms.json`; association counts go to RunResult.metadata.
- API `evaluate_pose(...)` (+ lazy re-export), CLI `e3r metric pose` with `--align`,
  `--associate-max-diff`, `--backend`, `--protocol`, `--out`, `--method`; overrides
  recorded and re-hashed. CLI echoes protocol+hash, alignment mode/solver, association
  policy, backend, failure policy, then the metrics table (value/unit/statistic), the
  association counts line, and the alignment record line.
- New error type `InvalidTrajectoryError` (load/association failures); alignment refusals
  use the existing `AlignmentError`.

## Verification

```bash
ruff check .   # All checks passed!
mypy eval3r    # Success: no issues found in 93 source files
env -u FORCE_COLOR EVAL3R_ETH3D_TOOL=~/xingrui_ws/tools/multi-view-evaluation/build/ETH3DMultiViewEvaluation \
  pytest -q    # 370 passed, 3 skipped (skips = TnT real-data tests only)
mkdocs build   # OK
# analytic checks (tests/unit/test_pose_metrics.py, 20 tests): identity → ATE 0;
#   pred = 2·gt + offset → Sim3 scale 0.5 and ATE 0 (1e-9); SE3 leaves scale 1 and
#   residual ATE; unaligned constant offset (3,0,4) → ATE exactly 5; per-frame 10°
#   rotation → rpe_rotation exactly 10 deg; association drop accounting (17/20 with
#   3 GT frames dropped); no-overlap / malformed-TUM / missing-max-diff failures name
#   paths + tolerance; statistic and RPE deltas never defaulted.
# end-to-end (tests/integration/test_single_pose_metric.py, 10 tests): API + CLI,
#   run-directory completeness, results.json association counts + per-metric
#   alignment metadata, alignment_transforms.json record (scale 0.5, |ln s| = ln 2),
#   --align none override changes hash, pinned-alignment protocol refuses --align sim3,
#   CLI exit 2 on bad --align, CLI failure prints reason.
# acceptance CLI run (from the repo root; PATH e3r is foreign and site-packages has a
# stale copy):
#   python -m eval3r.cli.main metric pose pred_tum.txt --gt gt_tum.txt \
#     --backend evo --align sim3 --out <dir>
#   → ate 4.9e-10 m, rpe_translation 7.3e-10 m, rpe_rotation 0 deg,
#     alignment_scale_error 0.693147 (= ln 2, scale 0.5 recovered); config panel echoes
#     protocol v0.2.0 + hash, association policy, overrides; run directory complete.
```

## Status

done
