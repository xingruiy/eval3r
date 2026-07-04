# 006 — Result writer and environment metadata

## Goal

A complete, schema-valid run directory can be written from already-computed scene metrics,
with reproducibility metadata captured consistently before the runner and dataset adapters
arrive.

## Scope

- Run directory writer per `.agent/reproducibility.md`: `results.json` (full `RunResult`),
  `results.csv`, `per_scene.csv` (stable core columns), `failures.json`, `protocol.yaml`
  copy, `config.yaml` (incl. any CLI overrides), `environment.json`,
  `backend_versions.json`, `alignment_transforms.json` when alignment runs, `logs.txt`.
- Environment capture: python version, platform, eval3r version, backend versions, command
  line, cwd, git commit + dirty state when available, timestamp. No secrets/env vars.
- `SceneFailure` serialization with stage, reason, traceback, scene id, and any structured
  hints supplied by later runner stages.
- Stable CSV column policy for aggregate and per-scene outputs; partial-coverage fields are
  present even before report rendering exists.
- Helpers for protocol/config copying, backend-version metadata, and optional alignment
  transform output.
- Tests use synthetic `MetricResult` / `RunResult` objects; no file loading or metric
  execution belongs in this slice.

## Out of Scope

- Pipeline stage execution, alignment, CLI commands, and Python evaluation APIs (task 007).
- Dataset adapters and `e3r benchmark run` (tasks 008+).
- Markdown/LaTeX/HTML reports and diffing (task 016) — json/csv only here.

## Relevant Files

- `.agent/plan.md` — "Architecture", "Result directory", "CLI", "Python API"
- `.agent/reproducibility.md` — run directory, required result fields, environment metadata
- `.agent/schema.md` — `RunResult`, `SceneFailure`, `MetricResult`, `ReportingSpec`

## Plan

1. Implement run-directory writer and stable JSON/CSV serialization helpers.
2. Implement environment and backend-version capture with secret-free metadata.
3. Implement failure and optional alignment-transform file writers.
4. Tests: run directory completeness, required result fields in `results.json`, stable CSV
   columns, protocol/config copy behavior, dirty-git metadata shape, partial-coverage fields.

## Findings

(record during implementation)

## Decisions

(record during implementation; e.g. run directory naming, config.yaml contents)

## Verification

```bash
pytest tests/unit/test_result_writer*.py tests/unit/test_environment*.py
```

Acceptance per `.agent/plan.md`: a complete run directory can be produced from fixture
results, and `results.json` carries every required field.

## Status

todo
