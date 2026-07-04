# 006 — Pipeline runner and result writer

## Goal

The staged evaluation pipeline (`resolve → load → normalize → align → mask/cull → sample →
metric → aggregate → report`) runs end-to-end for single-file geometry evaluation, writes a
complete run directory, and powers `e3r metric geometry` plus the `evaluate_geometry`
Python API.

## Scope

- `pipeline/runner.py` + `pipeline/stages/*`: stage functions configured entirely by the
  protocol; shared by geometry now, depth/pose later (tasks 011/012) so failure handling
  and metadata capture do not drift.
- Failure policies `abort`, `skip_and_flag`, `score_worst`; `SceneFailure` records with
  stage, reason, traceback; structured errors say what failed, which scene, which file,
  which protocol required it, and how to fix it when obvious.
- Run directory writer per `.agent/reproducibility.md`: `results.json` (full `RunResult`),
  `results.csv`, `per_scene.csv` (stable core columns), `failures.json`, `protocol.yaml`
  copy, `config.yaml` (incl. any CLI overrides), `environment.json`,
  `backend_versions.json`, `alignment_transforms.json` when alignment runs, `logs.txt`.
- Environment capture: python version, platform, eval3r version, backend versions, command
  line, cwd, git commit + dirty state when available, timestamp. No secrets/env vars.
- Alignment stage: `none`, `se3`, `sim3` (umeyama) on points; ICP deferred until a backend
  exists (T&T official backend does its own ICP). Sim3 disallowed for metric-scale
  protocols unless explicitly allowed. Transforms saved.
- CLI `e3r metric geometry pred.ply --gt gt.ply --threshold 0.05 --sample 200000 --input
  pointcloud --gt-input pointcloud` and mesh variant, mapping flags onto the
  `single_geometry` protocol as recorded overrides.
- Python API: `evaluate_geometry(...)` per `.agent/plan.md`.
- Determinism: same protocol hash + seeds → identical results; seed `derive` policy
  (per-scene seed from protocol base seed + scene_id).

## Out of Scope

- Dataset adapters and `e3r benchmark run` (tasks 007–008 wire the same runner to scenes).
- Markdown/LaTeX/HTML reports and diffing (task 013) — json/csv only here.

## Relevant Files

- `.agent/plan.md` — "Architecture", "Result directory", "CLI", "Python API"
- `.agent/reproducibility.md` — run directory, required result fields, environment metadata
- `.agent/schema.md` — `RunResult`, `SceneFailure`, `AlignmentSpec`
- `.agent/metrics.md` — pipeline order for point-cloud metrics, failure behavior

## Plan

1. Implement stages as small pure functions over a run context; runner sequences them and
   applies the failure policy per scene.
2. Implement result writer + environment capture.
3. Wire `single_geometry` protocol through the runner for one pred/gt file pair.
4. CLI command + Python API wrapper.
5. Tests: failure-policy behavior per stage, run directory completeness (every required
   field present in results.json), determinism (repeat run → identical metrics), override
   recording, partial-coverage flags in results.json/csv.

## Findings

(record during implementation)

## Decisions

(record during implementation; e.g. run directory naming, config.yaml contents)

## Verification

```bash
pytest tests/unit/test_runner*.py tests/unit/test_result_writer*.py tests/integration/test_single_geometry*.py
e3r metric geometry tests/fixtures/geom/pred.ply --gt tests/fixtures/geom/gt.ply --threshold 0.05
```

Acceptance per `.agent/plan.md`: the `e3r metric geometry` command produces a complete run
directory whose `results.json` carries every required field.

## Status

todo
