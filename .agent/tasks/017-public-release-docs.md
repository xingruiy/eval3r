# 017 — Public release docs and packaging

## Goal

Release-ready documentation, examples, packaging checks, and PyPI preparation for the
implemented core library without freezing a third-party plugin API prematurely.

## Scope

- README per the structure in `.agent/plan.md`, reflecting the implemented feature set
  rather than aspirational adapters.
- Docs completion pass: the `docs/` mkdocs pages listed in `.agent/plan.md`
  ("Documentation plan") including install, quickstart, prediction_format, fidelity,
  confidence, failure_policy, dataset examples.
- Examples for built-in workflows only: single-file geometry, benchmark run on supported
  adapters, protocol inspection, report/diff usage.
- Public API documentation for the implemented stable functions: `evaluate_geometry`,
  `run_benchmark`, `load_protocol`, `diff_runs`.
- Explicit stability note: internal registries exist, but third-party dataset/backend
  plugin entry points are not yet a public compatibility contract.
- PyPI packaging preparation: `python -m build`, `twine check`, wheel install smoke test,
  versioned docs build; version per the semver policy in `.agent/plan.md`.

## Out of Scope

- New adapters or metrics.
- Third-party plugin APIs, entry-point groups, external adapter packages, or a plugin
  compatibility policy. These are deferred until at least 3 built-in adapters are
  implemented and there is 1 concrete external-adapter use case.
- 1.0.0 itself; this task prepares a pre-1.0 public release.

## Relevant Files

- `.agent/plan.md` — public release milestone, "Versioning policy", "Documentation plan",
  README structure
- `README.md`, `mkdocs.yml`, `docs/` (13 pages + 4 examples), `pyproject.toml`,
  `eval3r/__init__.py`

## Plan

1. Complete README and mkdocs user documentation for implemented workflows.
2. Add examples that exercise built-in functionality without external plugin packages.
3. Document public APIs and current stability boundaries.
4. Packaging checks: `python -m build`, `twine check`, wheel install smoke test, docs build.
5. Prepare release checklist and test-PyPI dry run notes; do not publish unless explicitly
   requested.

## Findings

- **README.md was still the planning-doc stub** ("Final Planning Documents" pointing at
  `.agent/`); it is also the PyPI landing page (`readme = "README.md"`), so it was fully
  rewritten per the plan's README structure. Relative `docs/...` links were made absolute
  GitHub URLs so they survive PyPI rendering.
- **`docs/` had only `index.md`**; the full "Documentation plan" page set now exists:
  index, install, quickstart, prediction_format, protocols, fidelity, datasets, metrics,
  backends, confidence, failure_policy, schema, reproducibility, and
  examples/{dtu,scannet,tanks_temples,eth3d}. Every page was written against the current
  code and `.agent/*.md`, and CLI snippets were checked against the real `--help` output
  (e.g. `dataset inspect --root --split`, `benchmark run` flag set, per-adapter
  prediction-naming errors read from `resolve_prediction` implementations).
- **The example result table in the README uses real validated numbers** (DTU scan 24:
  accuracy 0.343 / completeness 0.248 / overall 0.295 mm from the task-010 validation run,
  labeled as such) instead of invented values.
- `.agent/plan.md` still described `docs/` as "future" user documentation; both mentions
  updated to reflect that the pages exist, with the rule that planning docs and user docs
  must be fixed together when they disagree.
- The built wheel packages all 10 builtin protocol YAMLs
  (`eval3r.protocols` package-data), verified by inspecting the wheel contents.
- pip >= 25.1 is needed for `pip install --group dev` (PEP 735); the install docs give the
  plain-pip fallback.

## Decisions

- **Version 0.3.0 → 0.5.0** per the plan's semver policy: 0.4.x was the Tanks and
  Temples wrapper + ETH3D adapter, 0.5.x is depth + pose metrics — all shipped; reports
  and diffing (task 016) land within 0.5.x. 0.6.x remains reserved for the deferred
  plugin API. Test fixtures that carry a literal `eval3r_version` string were left as-is
  (they are arbitrary fixture values, not assertions on the package version;
  `capture_environment` imports `__version__` so it tracks automatically).
- **Stability wording** (README "Development status" + docs/index "Stability"): the
  supported public surface is the CLI commands, the top-level API functions
  (`evaluate_geometry`, `evaluate_depth`, `evaluate_pose`, `run_benchmark`,
  `load_protocol`, `diff_runs`), the protocol YAML schema + hashing, and the versioned
  `results.json` schema. Internal dataset/backend registries are explicitly **not** a
  compatibility contract yet.
- **Public API documentation lives in quickstart.md** ("Python API" section, one
  subsection per function with a runnable snippet) rather than a new off-plan `api.md`
  page — the plan's documentation list is the source of truth for the page set.
- `build` and `twine` added to the dev dependency group (dev tooling, not runtime).
- Docs describe only implemented workflows: the four real adapters + custom, the 10
  built-in protocols, and the deferred-adapter backlog is mentioned as planned, not
  offered.

## Release checklist (for the actual publish, when requested)

```text
1. git checkout main && merge the release branch; working tree clean
2. ruff check . && mypy eval3r && pytest (with EVAL3R_ETH3D_TOOL / EVAL3R_TNT_* set
   where available) && mkdocs build --strict
3. confirm version: pyproject.toml [project].version == eval3r.__version__
4. rm -rf dist && python -m build && twine check dist/*
5. fresh-venv wheel smoke test:
     python -m venv /tmp/rel && /tmp/rel/bin/pip install dist/eval3r-*.whl
     /tmp/rel/bin/e3r --help && /tmp/rel/bin/e3r protocol list
6. test-PyPI dry run:  twine upload --repository testpypi dist/*
     then: pip install --index-url https://test.pypi.org/simple/ \
             --extra-index-url https://pypi.org/simple eval3r
     (extra-index-url needed: heavy deps like open3d are not on test PyPI)
7. real upload:        twine upload dist/*     (credentials never committed; use
     ~/.pypirc or TWINE_* env vars locally)
8. git tag v<version> && push tag; verify pip install eval3r + e3r --help from PyPI
```

Nothing was uploaded in this task (test-PyPI or otherwise) — publishing requires the
user's explicit request and credentials.

## Verification

```bash
ruff check .            # All checks passed!
mypy eval3r             # Success: no issues found in 94 source files
env -u FORCE_COLOR EVAL3R_ETH3D_TOOL=~/xingrui_ws/tools/multi-view-evaluation/build/ETH3DMultiViewEvaluation \
  pytest -q             # 411 passed, 3 skipped (skips = TnT real-data tests only)
mkdocs build --strict   # OK (all 17 pages in nav, no warnings)

rm -rf dist && python -m build   # Successfully built eval3r-0.5.0.tar.gz and .whl
twine check dist/*               # PASSED (both artifacts)
# wheel contents: all 10 protocols/builtin/*.yaml present

# fresh-venv smoke test (python -m venv; wheel + full dependency set installed cleanly):
#   e3r --help                       -> works from the wheel entry point
#   eval3r.__version__               -> 0.5.0
#   load_protocol('single_geometry') -> loads from packaged YAML
#   e3r protocol list                -> Built-in protocols (10)
#   evaluate_geometry on synthetic clouds (500 pts, +0.001/axis offset)
#     -> accuracy 0.001732 (= 0.001*sqrt(3), analytically correct),
#        full run directory written (results.json, per_scene.csv, protocol.yaml, ...)
```

Acceptance met: the built artifact installs cleanly in a fresh venv, `e3r --help` works
from the wheel, docs build strictly, and the examples cover only implemented workflows
with the plugin-API non-promise stated explicitly.

## Status

done
