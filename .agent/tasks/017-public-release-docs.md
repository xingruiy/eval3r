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
- `.agent/datasets.md` — adapter interface (documented as internal until plugin work is resumed)
- `.agent/backends.md` — registry (documented as internal until plugin work is resumed)

## Plan

1. Complete README and mkdocs user documentation for implemented workflows.
2. Add examples that exercise built-in functionality without external plugin packages.
3. Document public APIs and current stability boundaries.
4. Packaging checks: `python -m build`, `twine check`, wheel install smoke test, docs build.
5. Prepare release checklist and test-PyPI dry run notes; do not publish unless explicitly
   requested.

## Findings

(record during implementation)

## Decisions

(record during implementation; e.g. public API stability wording, release version)

## Verification

```bash
pytest
python -m build && twine check dist/*
pip install dist/eval3r-*.whl && e3r --help
mkdocs build
```

Acceptance: the built artifact installs cleanly, `e3r --help` works from the wheel, docs
build, and examples cover implemented workflows without promising plugin compatibility.

## Status

todo
