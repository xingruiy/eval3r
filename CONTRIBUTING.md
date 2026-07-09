# Contributing to eval3r

Thanks for contributing to `eval3r`. This project is a research-oriented 3D
reconstruction evaluation library, so correctness, explicitness, and
reproducibility matter more than compact output or minimal dependencies.

This repository is agent friendly. Coding agents are welcome to contribute, and
the `.agent/` files plus `CLAUDE.md` are maintained as durable guidance for
agent-assisted development.

## Development setup

Use Python 3.10 or newer.

```bash
git clone https://github.com/xingruiy/eval3r
cd eval3r
pip install -e .
pip install -r docs/requirements.txt
```

Development tools are declared in `pyproject.toml` under the `dev` dependency
group. Install them with the workflow you use for dependency groups, or install
the listed tools directly in your environment.

## Project rules

Before changing behavior, read the relevant source-of-truth files:

- `.agent/plan.md` for project scope and roadmap
- `.agent/schema.md` for schemas and result structure
- `.agent/protocols.md` for protocol behavior
- `.agent/datasets.md` for dataset adapters
- `.agent/metrics.md` for metric definitions
- `.agent/backends.md` for backend delegation
- `.agent/reproducibility.md` for result metadata and hashing

If code and docs disagree, update both in the same change. Do not silently
change protocol behavior, schema fields, metric semantics, dataset conventions,
or result metadata.

## What belongs in eval3r

Good contributions include:

- protocol definitions
- dataset adapters
- prediction manifests
- metric definitions
- benchmark orchestration
- result schemas
- reproducibility records
- reports

Delegate common geometry, camera, trajectory, and IO work to established
libraries where practical. Do not add learned reconstruction methods, default
mesh repair, large proprietary dataset downloads, or fake stand-ins for official
evaluators.

## Tests and verification

Every behavior-changing change needs tests. Use tiny fixtures for datasets and
normal CI paths. Tests for official-evaluation wrappers must invoke the real
official tool when it is available, and skip clearly when it is absent; they must
not substitute a fake evaluator that fabricates official-looking numbers.

Before opening a pull request, run:

```bash
ruff check .
pytest
mypy eval3r
mkdocs build
```

If a command is not configured or cannot run in your environment, say so in the
pull request and include the exact error.

## Pull request checklist

- Tests cover new or changed behavior.
- Documentation is updated for schema, protocol, metric, dataset, CLI, backend,
  or result-format changes.
- Protocol changes update the YAML, protocol version, hash expectations, and
  regression fixtures as needed.
- Result-affecting behavior is recorded in metadata.
- Errors and CLI output state what failed, which inputs were involved, and how
  to fix it when obvious.
- Official-evaluation paths use the real official code or toolbox.

## License

By contributing, you agree that your contributions are licensed under the MIT
License in `LICENSE`.
