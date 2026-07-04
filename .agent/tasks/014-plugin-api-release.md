# 014 — Plugin API and public release

## Goal

Stable dataset/backend/protocol plugin APIs with worked examples, and eval3r published to
PyPI with versioned docs.

## Scope

- Stabilize and document the `DatasetAdapter` plugin interface (entry-point group, e.g.
  `eval3r.datasets`) so third-party packages can register adapters; same for backends
  (`eval3r.backends`) and protocol search paths (user protocol directories).
- `datasets/custom.py`: generic directory-layout adapter for unlisted datasets, driven by a
  manifest (clearly `eval3r_native`).
- Examples: one custom dataset adapter package and one custom protocol YAML, in docs +
  `examples/`.
- Public API freeze for 1.0-track: `evaluate_geometry`, `run_benchmark`, `load_protocol`,
  `diff_runs`; deprecation policy note.
- Docs completion pass: the `docs/` mkdocs pages listed in `.agent/plan.md`
  ("Documentation plan") including install, quickstart, prediction_format, fidelity,
  confidence, failure_policy, dataset examples.
- README per the structure in `.agent/plan.md`.
- PyPI packaging: build, twine check, versioned docs deploy; version per the semver policy
  in `.agent/plan.md`.

## Out of Scope

- New adapters or metrics.
- 1.0.0 itself (this task targets the 0.6.x plugin milestone; 1.0 is a later decision).

## Relevant Files

- `.agent/plan.md` — "Plugin API and public release" milestone, "Versioning policy",
  "Documentation plan", README structure
- `.agent/datasets.md` — adapter interface (the contract being frozen)
- `.agent/backends.md` — registry (the contract being frozen)

## Plan

1. Entry-point-based discovery for datasets/backends; user protocol path support.
2. Custom adapter + examples.
3. Docs pass + README.
4. Packaging: `python -m build`, `twine check`, test-PyPI dry run, then release.

## Findings

(record during implementation)

## Decisions

(record during implementation; e.g. entry-point group names, protocol search order)

## Verification

```bash
pytest
python -m build && twine check dist/*
pip install dist/eval3r-*.whl && e3r --help
mkdocs build
```

Acceptance per `.agent/plan.md`: `pip install eval3r` (from the built artifact) and
`e3r --help` work; example plugin discoverable.

## Status

todo
