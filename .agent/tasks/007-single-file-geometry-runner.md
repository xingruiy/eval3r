# 007 — Single-file geometry runner

## Goal

The staged evaluation pipeline runs end-to-end for one prediction file and one ground-truth
file, writes a complete run directory through task 006, and powers `e3r metric geometry`
plus the `evaluate_geometry` Python API.

## Scope

- `pipeline/runner.py` + `pipeline/stages/*`: stage functions configured entirely by the
  protocol; shared by geometry now, depth/pose later (tasks 014/015) so failure handling
  and metadata capture do not drift.
- Single-file geometry path: resolve, load, normalize units, align, mask/cull, sample,
  metric, aggregate, report-to-json/csv.
- Failure policies `abort`, `skip_and_flag`, `score_worst`; structured errors say what
  failed, which scene, which file, which protocol required it, and how to fix it when
  obvious.
- Alignment stage for point sets: `none`, `se3`, `sim3` (Umeyama). ICP is deferred until a
  backend exists. Sim3 is disallowed for metric-scale protocols unless explicitly allowed.
  Transforms are passed to the task-006 writer.
- CLI `e3r metric geometry pred.ply --gt gt.ply --threshold 0.05 --sample 200000 --input
  pointcloud --gt-input pointcloud` and mesh variant, mapping flags onto the
  `single_geometry` protocol as recorded overrides.
- Python API: `evaluate_geometry(...)` per `.agent/plan.md`.
- Determinism: same protocol hash + seeds -> identical results; seed `derive` policy
  (per-scene seed from protocol base seed + scene_id).

## Out of Scope

- Dataset adapters and `e3r benchmark run` (task 008+).
- Dataset-specific visibility, masks, or official evaluator behavior.
- Markdown/LaTeX/HTML reports and diffing (task 016).

## Relevant Files

- `.agent/plan.md` — "Architecture", "CLI", "Python API"
- `.agent/reproducibility.md` — result directory and environment metadata expectations
- `.agent/schema.md` — `RunResult`, `SceneFailure`, `AlignmentSpec`
- `.agent/metrics.md` — pipeline order for point-cloud metrics, failure behavior

## Plan

1. Implement stages as small pure functions over a run context; runner sequences them and
   applies the failure policy per scene.
2. Wire `single_geometry` protocol through the runner for one pred/gt file pair.
3. Implement point-set SE3/Sim3 alignment and transform metadata.
4. CLI command + Python API wrapper.
5. Tests: failure-policy behavior per stage, deterministic repeat runs, override recording,
   alignment transform output, integration test for pointcloud and mesh inputs.

## Findings

(record during implementation)

## Decisions

(record during implementation; e.g. stage context shape, run directory naming)

## Verification

```bash
pytest tests/unit/test_runner*.py tests/integration/test_single_geometry*.py
e3r metric geometry tests/fixtures/geom/pred.ply --gt tests/fixtures/geom/gt.ply --threshold 0.05
```

Acceptance per `.agent/plan.md`: the `e3r metric geometry` command produces a complete run
directory whose `results.json` carries every required field.

## Status

todo

