# Failure policy and partial coverage

Scenes fail — a missing prediction file, a required mask that isn't there, an empty
point cloud after culling. eval3r's position: **failures are part of the result**, and an
aggregate over a subset of scenes must never look like an aggregate over all of them.

## Policies

Every protocol pins one:

```yaml
failure_policy:
  policy: abort | skip_and_flag | score_worst
  worst_values: {fscore: 0.0}   # required per metric for score_worst
```

- `abort` — the first scene failure stops the run with the full reason. Right for
  single-file evaluation and official protocols.
- `skip_and_flag` — failed scenes are skipped; the run completes, but every aggregate is
  labeled **partial** and the failed scenes are listed with reasons.
- `score_worst` — failed scenes contribute protocol-pinned worst values to aggregates
  (so a method is not rewarded for crashing on hard scenes).

## Failure accounting

Every result records: expected scene count, evaluated scene count, the failed scenes with
their pipeline stage (`resolve`/`load`/`normalize`/`align`/`mask`/`sample`/`metric`/...)
and verbatim reasons (plus traceback when available), the policy in effect, and whether
aggregates are partial. `failures.json` in the run directory holds the structured list.

## Partial coverage is always visible

If any expected scene was not evaluated, a partial-coverage banner appears in **every**
output format — the CLI summary, `results.json`, `results.csv`, `per_scene.csv`, and the
Markdown/LaTeX/HTML reports:

> ⚠ PARTIAL COVERAGE: 310 of 312 scenes evaluated, 2 failed (failure policy:
> skip_and_flag). Aggregate metrics do not cover every expected scene — see the failure
> table for per-scene reasons.

`e3r diff` shows each run's coverage with the same banner behavior and warns when two
runs' scene coverage differs — an average over 310 scenes is not comparable to an average
over 312.
