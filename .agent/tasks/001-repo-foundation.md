# 001 — Repository foundation

## Goal

A pip-installable `eval3r` package skeleton with working CLI entry point, lint, type check,
test runner, docs build, and CI — so every later slice lands on green tooling.

## Scope

- `pyproject.toml`: package metadata, base dependencies only (`numpy`, `scipy`, `pandas`,
  `pyyaml`, `typer`, `rich`, `pydantic`), optional extras exactly as listed in
  `.agent/backends.md` (`mesh`, `open3d`, `pose`, `colmap`, `depth`, `torch`, `faiss`,
  `waymo`, `all` — note `all` intentionally excludes torch/faiss/waymo).
- Package skeleton per the repository layout in `.agent/plan.md` (empty modules with
  docstrings are fine; do not stub logic that later tasks own).
- `e3r` console entry point via typer; `e3r --help` lists planned command groups
  (`metric`, `benchmark`, `dataset`, `protocol`, `diff`). `benchmark` is a subcommand
  group (`run`, `validate`) — see the CLI section of `.agent/plan.md`.
- Tooling config: `ruff`, `pytest`, `mypy` (project choice: mypy; record if changed),
  `mkdocs` with a minimal `docs/index.md`.
- GitHub Actions CI: ruff + pytest + mypy + mkdocs build on push/PR.
- `.gitignore`, `.env.example` (names/comments only).

## Out of Scope

- Any schema models, metrics, protocols, adapters, or backends (tasks 002+).
- Publishing to PyPI (task 014).

## Relevant Files

- `.agent/plan.md` — "Repository layout", "Dependency policy", "Milestones → Repository foundation", "CLI"
- `.agent/backends.md` — "Base dependencies", "Optional extras"
- `CLAUDE.md` — git workflow, data/credential rules

## Plan

1. Write `pyproject.toml` with base deps, extras, `e3r` entry point, ruff/mypy/pytest config.
2. Create the package directory tree from `.agent/plan.md` layout.
3. Implement `cli/main.py` with typer app and empty command groups.
4. Add mkdocs config + `docs/index.md` stub; add CI workflow.
5. Add a smoke test: `import eval3r` and CLI `--help` invocation via typer test runner.

## Findings

(record during implementation)

## Decisions

(record during implementation; e.g. mypy vs pyright, minimum Python version)

## Verification

```bash
pip install -e .
e3r --help
ruff check .
pytest
mypy eval3r
mkdocs build
```

All commands must pass. Acceptance per `.agent/plan.md`: package imports, `e3r --help` works,
pytest runs, docs build locally.

## Status

todo
