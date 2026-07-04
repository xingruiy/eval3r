# 008 — Benchmark-run adapter plumbing

## Goal

`e3r benchmark run` can execute the task-007 runner across scenes from a generic
`DatasetAdapter`, using a manifest to resolve predictions and aggregating per-scene outputs
without any dataset-specific official behavior.

## Scope

- Dataset adapter registry and base interface from `.agent/datasets.md`.
- Generic benchmark orchestration: scene iteration, protocol/dataset capability validation,
  manifest loading, prediction resolution, per-scene runner invocation, aggregate result
  assembly, partial-coverage accounting.
- CLI: `e3r benchmark run <pred_root> --dataset <name> --split <split> --protocol <name>`.
- Capability gating: unsupported, server-only, missing-public-GT, external-asset, and
  external-renderer statuses fail before computation with clear messages.
- A tiny in-repo `custom`/fixture adapter for integration tests only, using point-cloud
  files and the `single_geometry` protocol family.
- No official evaluator, dataset-specific culling, or benchmark-specific score semantics.

## Out of Scope

- DTU, ScanNet, Tanks and Temples, ETH3D, or any real dataset adapter (tasks 009+).
- Plugin entry points for third-party adapters (task 017).
- New metrics, protocols, or report formats.

## Relevant Files

- `.agent/datasets.md` — adapter interface, capability declaration, local evaluation status
- `.agent/protocols.md` — protocol validation checklist
- `.agent/reproducibility.md` — scene coverage and failure accounting
- `.agent/plan.md` — "Evaluation runner", "CLI", dataset support strategy

## Plan

1. Implement adapter registration and lookup for built-in adapters.
2. Implement manifest-driven prediction resolution and scene loop orchestration.
3. Add capability/local-evaluation preflight checks.
4. Wire benchmark CLI to the task-007 runner and task-006 writer.
5. Tests: adapter lookup, manifest resolution, capability refusal paths, partial coverage,
   end-to-end benchmark run on the tiny fixture adapter.

## Findings

- The benchmark loop reuses the task-007 `evaluate_geometry_scene` verbatim per scene, adding
  only a `resolve` stage (prediction + GT path resolution) in front. Same `SceneOutcome` /
  `SceneFailure` types, so failure accounting does not drift between single-file and benchmark.
- Preflight is a hard gate *before* any prediction is loaded: `server_only_eval` capability or a
  non-`supported` `local_evaluation().status` raises `BenchmarkError` with the concrete status
  and reason — no local numbers are ever produced for a server-only/asset-gated split.
- `score_worst` is realized by injecting per-scene `MetricResult`s (value from
  `failure_policy.worst_values`, `metadata.scored_worst=True`) so worst-scored scenes still
  flow through aggregation while remaining listed in `failed_scenes`.
- Manifest is loaded from `--manifest`, else `<pred_root>/manifest.yaml`, else inferred as
  `<scene>.ply`; inferred manifests are validated, written to the run dir, and flagged
  (`RunResult.metadata.manifest_inferred`, config, and the manifest's own metadata).

## Decisions

- Dataset registry maps a name to a factory `(root: Path | None) -> DatasetAdapter` (real
  adapters bind to a dataset root). Built-in `custom` adapter registered lazily in
  `datasets.default_registry()`. Unknown names raise `UnknownDatasetError` listing available.
- `custom` adapter layout: `<root>/splits/<split>.txt` + `<root>/gt/<scene>.ply`. GT is
  user-supplied, so provenance/independence/density are `unknown` and fidelity stays
  eval3r-native — it never claims independent GT.
- New errors `DatasetError`, `UnknownDatasetError`, `BenchmarkError` in `core/errors.py`.
- Aggregation is `per_scene_then_mean` for every metric this slice (pooled `global` F-score
  deferred until the geometry metric exposes per-scene precision/recall counts).
- Benchmark orchestration lives in `pipeline/benchmark.py` (reuses `pipeline/stages/aggregate.py`);
  the API adds `run_benchmark(...)`; CLI adds `e3r benchmark run` / `validate` and
  `e3r dataset list` / `inspect`. The tests register a `fixture` adapter (a `custom` adapter at
  the committed fixture root) so the doc-style `--dataset fixture --split tiny` command works.

## Verification

```bash
ruff check .   # All checks passed!
mypy eval3r    # Success: no issues found in 88 source files
pytest -q      # 169 passed
mkdocs build   # OK
# in-process CLI (PATH `e3r` is a different installed package in this env):
python -c "import sys; from eval3r.datasets import default_registry, CustomAdapter; from pathlib import Path; \
  default_registry().register('fixture', lambda r: CustomAdapter(Path('tests/fixtures/benchmark/dataset_root'))); \
  sys.argv=['e3r','benchmark','run','tests/fixtures/benchmark/preds','--dataset','fixture','--split','pair',\
  '--protocol','single_geometry','--out','/tmp/bench_run']; from eval3r.cli.main import app; app()"
```

Acceptance met: benchmark orchestration runs on a non-official fixture adapter, writes a
complete run directory (incl. `manifest.yaml`), reports partial coverage under `skip_and_flag`,
and refuses server-only / non-supported splits before loading predictions. Covered by
`tests/unit/test_dataset_registry.py`, `tests/unit/test_benchmark_runner.py`, and
`tests/integration/test_benchmark_fixture.py`.

## Status

done

