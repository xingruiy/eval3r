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
- `.agent/schema.md` — `ReportingSpec`, new "Run diff" section
- `eval3r/reports/table.py`, `markdown.py`, `latex.py`, `html.py`, `plots.py`, `diff.py`,
  `run_directory.py`; `eval3r/cli/diff.py`, `eval3r/cli/main.py`; `eval3r/api.py`

## Plan

1. Implement report renderers off `RunResult` (no recomputation).
2. Implement diff loader + strict/loose comparison + warning triggers.
3. Implement colored-PLY and histogram debug writers.
4. Tests: partial coverage visible in every format (string assertions), strict diff
   refusal on hash mismatch, loose diff labeling, warning triggers, colored PLY point
   count matches input.

## Findings

- **`ReportingSpec` was already hash-excluded** (`NON_SEMANTIC_TOPLEVEL_KEYS` in
  `core/hashing.py` contains `"reporting"`), so enabling markdown/latex/html formats or
  debug outputs never changes a protocol hash — no protocol version bumps in this slice
  and no new protocol fields.
- **Per-point distances were computed and discarded** inside
  `evaluate_geometry_metrics`; debug outputs need them. Rather than recomputing NN
  queries, `DirectionalDistances` now also carries the *cleaned* pred/gt point arrays
  (row-aligned with the distances), `evaluate_geometry_metrics` accepts a precomputed
  `distances=` argument, and the runner computes distances once up front only when the
  reporting spec requests debug outputs — identical values flow to metrics and debug.
- **Only the eval3r-native geometry path can emit debug outputs.** Official-toolbox
  paths (DTU port, TnT toolbox, ETH3D binary) own their distance computation internally;
  their `SceneOutcome.debug` stays `None` and `debug/debug_outputs.json` simply carries
  no record for them. Documented in `.agent/reproducibility.md`.
- **LaTeX escaping ate a test's expectations**: `skip_and_flag` renders as
  `skip\_and\_flag`, so cross-format string assertions must match each format's own
  spelling; also the `%`-comment title line originally carried the raw (unescaped) title.
- **Python 3.10 forbids backslashes inside f-string expressions** — the LaTeX coverage
  paragraph needed its `\textbf{partial}` fragment hoisted out of the f-string.
- The last CLI stub (`e3r diff`) is gone, so
  `test_smoke.py::test_stub_command_fails_loudly_with_reason` became a real check: diff
  of a missing run directory exits 1 naming the missing `results.json`. `cli/vis.py`
  is documented as intentionally unregistered (debug artifacts live in
  `reports/plots.py`; no interactive viewer).

## Decisions

- **matplotlib added as a required base dependency** (pyproject + `.agent/plan.md` +
  `.agent/backends.md` dependency lists) — used only for histogram PNGs (headless via
  the Agg `Figure`/`FigureCanvasAgg`, no pyplot global state, no display needed) and the
  `turbo` error colormap. It never participates in metric computation.
- `reports/table.py` is the format-agnostic layer: `build_report_data(RunResult) ->
  ReportData` (title, header entries, `CoverageInfo`, banner text, aggregate table,
  per-scene table with dynamic metric columns keyed by scene[/frame], failure list).
  Renderers (`render_markdown` → `results.md`, `render_latex` → `results.tex` fragment
  with escaping, `render_html` → self-contained `report.html` with inline CSS and a red
  banner div) never recompute metrics — they render the `RunResult` only. Banner fires
  when `evaluated < expected or failed > 0` and names the failure policy.
- `write_run_directory` gained `formats=` (default: the protocol's
  `reporting.formats`); json/csv stay unconditional. Debug outputs are written by the
  API layer (`api._write_debug_outputs`) after the run directory, via the run's
  pointcloud backend: `debug/<scene>_error.ply` (turbo colormap over pred→gt distance,
  vmax = max distance, recorded), `debug/<scene>_histogram.png` (both directions, 64
  bins), and `debug/debug_outputs.json` recording colormap, vmax, bins, and point
  counts. Wired for both the single-file geometry runner and the benchmark native
  geometry path (`SceneOutcome.debug` → `*RunOutput.debug_scenes`).
- `reports/diff.py`: `diff_runs(run_a, run_b, *, loose=False) -> RunDiff` (pydantic
  models `RunIdentity` / `MetricDelta` / `DiffWarning` / `RunDiff`, documented in
  `.agent/schema.md` "Run diff"). Strict mode raises the new
  `RunComparisonError` on protocol-hash mismatch, naming both runs/hashes and pointing
  at `--loose`; loose results carry `strict=False`. Warnings implement all
  `.agent/reproducibility.md` comparability triggers; "backend officialness" compares
  fidelity + the `official_eval` backend-versions entry. Per-scene deltas cover common
  scene labels; one-sided scenes are listed explicitly.
- `e3r diff` (`cli/diff.py`, registered top-level in `main.py`): echoes both run
  identities (method, protocol+hash, coverage) with per-run PARTIAL COVERAGE banners,
  a NON-STRICT panel under `--loose`, the comparability-warning table (field column
  no-wrap so tests/greps can match trigger names), aggregate and per-scene delta
  tables, and refusals as a red panel + exit 1 — never a bare exit code.
- Public API: `eval3r.diff_runs` lazy re-export (also via `eval3r.api`);
  `eval3r.reports` re-exports the renderers, debug writers, and diff types.

## Verification

```bash
ruff check .   # All checks passed!
mypy eval3r    # Success: no issues found in 94 source files
env -u FORCE_COLOR EVAL3R_ETH3D_TOOL=~/xingrui_ws/tools/multi-view-evaluation/build/ETH3DMultiViewEvaluation \
  pytest -q    # 411 passed, 3 skipped (skips = TnT real-data tests only)
mkdocs build   # OK
# unit (tests/unit/test_reports.py, 17 tests): partial-coverage banner + coverage counts
#   + failure reasons asserted in markdown/latex/html; no banner on full coverage;
#   header surfaces hash/fidelity/GT provenance/local-eval/alignment/confidence/
#   sampling/failure policy; latex escaping (raw specials absent); html escaped +
#   self-contained; run_directory writes/omits formats; colored PLY point count ==
#   input (50), histogram PNG magic bytes, debug manifest params, no-op when not
#   requested.
# unit (tests/unit/test_diff.py, 16 tests): strict refusal names both hashes +
#   --loose; loose labeled non-strict + warns on hash; matching runs -> no warnings,
#   exact deltas (accuracy +0.02, fscore -0.06); None values -> None delta; all nine
#   comparability triggers parametrized (provenance, local-eval status, coverage,
#   failure policy, alignment mode, confidence, sampling counts, fidelity,
#   official_eval backend entry); missing run dir names results.json; direct
#   results.json path accepted.
# integration (tests/integration/test_reports_diff.py, 5 tests): API run with all
#   formats + debug enabled -> results.md/.tex/report.html + debug/{error.ply,
#   histogram.png,debug_outputs.json}, PLY count == manifest n_points_pred; API
#   strict/loose across a threshold-changed hash; CLI acceptance
#   `e3r diff runs/method_a runs/method_b` exit 0 with warning-free comparison; CLI
#   refusal exit 1 + loose NON-STRICT; fabricated partial run -> CLI shows PARTIAL
#   COVERAGE banner + scene_coverage warning.
# acceptance run (from the repo root; PATH e3r is foreign and site-packages has a
# stale copy):
#   python -m eval3r.cli.main diff runs/method_a runs/method_b
#   -> both run panels (method, protocol v0.1.0 + hash, 1/1 coverage), "no
#      comparability warnings", aggregate + per-scene delta tables (all deltas 0).
```

## Status

done
