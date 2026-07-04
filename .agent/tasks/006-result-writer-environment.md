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

- `reports/run_directory.py::write_run_directory` orchestrates the run directory. Always writes
  `results.json` (full `RunResult`), `results.csv`, `per_scene.csv`, `failures.json`,
  `environment.json`, `backend_versions.json`, `logs.txt`; conditionally writes `protocol.yaml`
  (from an `EvalProtocol` or raw YAML), `manifest.yaml`, `config.yaml`, `alignment_transforms.json`.
  `default_run_dir_name` = `<timestamp>_<dataset>_<method>` (slugged).
- `reports/json.py`: `write_run_result_json` / `read_run_result_json` (round-trips through the
  validated model), `write_failures_json` (coverage-focused: policy + expected/evaluated/failed
  counts + per-scene reason/stage/policy_action), `dump_json`.
- `reports/csv.py`: `write_results_csv` (aggregate `metric,value`) and `write_per_scene_csv`
  with the stable `PER_SCENE_COLUMNS` (scene_id, status, failure_reason, accuracy…fscore,
  point counts, valid/culled fraction, alignment mode/scale, backends, runtime). `per_scene_rows`
  builds one row per scene from `per_scene_metrics` + `failed_scenes`, so partial coverage is
  always visible (failed scenes appear with `status=failed`).
- `core/environment.py::capture_environment` records only minimal, non-identifying facts:
  eval3r version, python version/implementation, platform, os, cpu architecture, and the
  resolved command. Per user direction it deliberately **omits** working directory, git
  commit/dirty state, timestamp, and timezone (and python executable path) — these are
  identifying/unnecessary. `.agent/reproducibility.md` "Environment metadata" updated to match,
  with an explicit must-NOT-contain list. A test asserts none of these leak.
- Schema reconciliation (docs+model together): `RunResult.backend_versions` was
  `dict[str, str]` but backend metadata is nested (`{kind: {name, library, version,
  approximate}}`) per `.agent/backends.md` / `.agent/reproducibility.md`. Widened to
  `dict[str, Any]` (accepts a nested metadata dict or a plain version string); existing flat
  fixtures still validate.
- Tests (`test_result_writer.py` 8, `test_environment.py` 4): core-file presence, required
  `results.json` fields + round-trip, failures/partial-coverage, stable per-scene columns with
  both ok and failed scenes, aggregate CSV, conditional files present/absent, protocol copy,
  environment shape + no-secrets + git in/out of repo.

## Decisions

- Environment capture lives in `core/environment.py` (a small, justified extension to the
  `.agent/plan.md` layout — it is fundamental reproducibility metadata used by the runner to
  fill `RunResult.environment`, not report rendering).
- `results.json` is the **full** `RunResult` (self-describing), not the friendlier nested view
  sketched in `.agent/reproducibility.md`; the nested example remains a documentation aid.
- Run-directory writing takes an already-computed `RunResult`; no metric execution or file
  loading happens here (that is task 007). `environment`/`backend_versions` default to the
  values already on the result so a directory is self-describing without extra inputs.
- The orchestrator lives in a new `reports/run_directory.py` module (layout has `reports/{json,
  csv,...}`; a focused orchestrator module fits the "small modules" style).

## Verification

```bash
pytest tests/unit/test_result_writer*.py tests/unit/test_environment*.py
```

Acceptance per `.agent/plan.md`: a complete run directory can be produced from fixture
results, and `results.json` carries every required field.

Outcomes (feature/repo-foundation):

```text
pytest            -> 124 passed (adds 12 writer + environment tests)
ruff check .      -> All checks passed!
mypy eval3r       -> Success: no issues found in 85 source files
end-to-end        -> write_run_directory produces results/csv/failures/env/backends/logs
```

## Status

done
