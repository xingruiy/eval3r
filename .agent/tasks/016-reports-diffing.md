# 016 — Reports and diffing

## Goal

Markdown, LaTeX, and HTML reports with visible partial-coverage banners; `e3r diff` with
strict protocol-hash checking; error-colored point-cloud debug outputs.

## Scope

- `reports/markdown.py`, `reports/latex.py`, `reports/html.py` (json/csv exist from task
  006): per-scene table + aggregate table; every format shows scene coverage
  (expected/evaluated/failed), failure reasons, failure policy, and a partial-coverage
  banner when aggregates are partial.
- Report headers surface protocol name+hash, fidelity, GT provenance/independence, local
  evaluation status, alignment mode (incl. depth scale mode + granularity), confidence
  policy, sampling counts.
- `e3r diff runs/a runs/b` + `diff_runs()` API: strict mode refuses differing protocol
  hashes; `--loose` allowed but output labeled non-strict; warn on the comparability
  triggers listed in `.agent/reproducibility.md` (GT provenance, coverage, failure policy,
  alignment, confidence, sampling, backend officialness).
- Debug outputs when protocol requests them: `error_colored.ply` (per-point distance
  colormap via point-cloud backend), distance histogram PNG.
- CLI summary output mirrors coverage/banner behavior.

## Out of Scope

- Plot styling beyond a basic histogram; interactive HTML.
- New metrics or protocol fields.

## Relevant Files

- `.agent/reproducibility.md` — "Report comparability", "Scene coverage and failures", per-scene CSV
- `.agent/plan.md` — "Reporting and diffing" milestone, `e3r diff` behavior
- `.agent/schema.md` — `ReportingSpec`

## Plan

1. Implement report renderers off `RunResult` (no recomputation).
2. Implement diff loader + strict/loose comparison + warning triggers.
3. Implement colored-PLY and histogram debug writers.
4. Tests: partial coverage visible in every format (string assertions), strict diff
   refusal on hash mismatch, loose diff labeling, warning triggers, colored PLY point
   count matches input.

## Findings

(record during implementation)

## Decisions

(record during implementation)

## Verification

```bash
pytest tests/unit/test_reports*.py tests/unit/test_diff*.py
e3r diff runs/method_a runs/method_b  # fixture runs with matching and mismatching hashes
```

Acceptance per `.agent/plan.md` milestone.

## Status

todo
