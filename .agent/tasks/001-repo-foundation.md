# 001 — Repository foundation

## Goal

A pip-installable `eval3r` package skeleton with working CLI entry point, lint, type check,
test runner, docs build, and CI — so every later slice lands on green tooling.

## Scope

- `pyproject.toml`: package metadata, single required dependency list exactly as in
  `.agent/backends.md` (`numpy`, `scipy`, `pandas`, `pyyaml`, `typer`, `rich`, `pydantic`,
  `trimesh`, `plyfile`, `open3d`, `evo`, `pycolmap`, `imageio`, `opencv-python`). No
  `[project.optional-dependencies]` / extras — all dependencies are required. The project
  does not depend on torch, faiss, or waymo tooling.
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
- Publishing to PyPI (task 017).

## Relevant Files

- `.agent/plan.md` — "Repository layout", "Dependency policy", "Milestones → Repository foundation", "CLI"
- `.agent/backends.md` — "Dependencies"
- `CLAUDE.md` — git workflow, data/credential rules

## Plan

1. Write `pyproject.toml` with the single required dependency list, `e3r` entry point, ruff/mypy/pytest config.
2. Create the package directory tree from `.agent/plan.md` layout.
3. Implement `cli/main.py` with typer app and empty command groups.
4. Add mkdocs config + `docs/index.md` stub; add CI workflow.
5. Add a smoke test: `import eval3r` and CLI `--help` invocation via typer test runner.

## Findings

- Implemented on branch `feature/repo-foundation` (off `longhaul`).
- Package tree generated per `.agent/plan.md` "Repository layout": 82 docstring-only
  modules across `core/`, `datasets/`, `protocols/`, `metrics/`, `backends/`,
  `pipeline/stages/`, `reports/`, `cli/`. Each module docstring names the task slice that
  owns its real logic.
- `eval3r/cli/main.py` builds the full typer command tree in one module. Command groups:
  `metric` (`geometry`/`depth`/`pose`), `benchmark` (`run`/`validate`), `dataset`
  (`inspect`), `protocol` (`show`), plus top-level `diff`. Every leaf command is a stub that
  raises `NotImplementedError` with an explicit reason naming the owning task (no silent
  stubs), satisfying CLAUDE.md's error/CLI verbosity rules.
- `pycolmap` is declared in `dependencies` but not installed locally and not imported by the
  skeleton (`grep` confirms zero references), so local verification passes without it.
  Local `pip install -e . --no-deps` used since the rest of the dep set was already present;
  CI installs the full set.
- `mkdocs.yml` uses the plain (built-in) theme with `nav: Home only` to keep
  `mkdocs build --strict` self-contained; other `docs/*.md` deferred to later tasks.

## Decisions

- **Type checker:** mypy (project default per plan). Config: `ignore_missing_imports = true`
  (untyped scientific deps), `python_version = "3.10"`.
- **Minimum Python:** 3.10 (local interpreter 3.10.12); CI matrix 3.10 + 3.11.
- **Version:** `0.3.0` — fresh rewrite line; `longhaul` history is post-`dev` (which ended at
  0.2.x).
- **Build backend:** setuptools + wheel.
- **Dev tooling** (`ruff`, `mypy`, `pytest`, `mkdocs`) declared under `[dependency-groups] dev`,
  keeping runtime `dependencies` extras-free per CLAUDE.md.
- `py.typed` marker and serialization tests deferred to task 002 as planned.

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

Outcomes (feature/repo-foundation):

```text
e3r --help        -> exit 0, lists metric/benchmark/dataset/protocol/diff
ruff check .      -> All checks passed!
pytest            -> 3 passed
mypy eval3r       -> Success: no issues found in 83 source files
mkdocs build      -> built (mkdocs build --strict, exit 0)
import eval3r     -> 0.3.0
```

Note: local install used `pip install -e . --no-deps` (runtime deps already present; pycolmap
declare-only and unimported). CI validates the full dependency set.

## Status

done
