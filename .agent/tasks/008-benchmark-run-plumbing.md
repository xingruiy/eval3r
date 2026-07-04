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

(record during implementation)

## Decisions

(record during implementation; e.g. adapter registry location and preflight error format)

## Verification

```bash
pytest tests/unit/test_dataset_registry*.py tests/unit/test_benchmark_runner*.py tests/integration/test_benchmark_fixture*.py
e3r benchmark run tests/fixtures/benchmark_preds --dataset fixture --split tiny --protocol single_geometry
```

Acceptance: benchmark orchestration works for a non-official fixture adapter and refuses
unsupported dataset/protocol combinations before loading predictions.

## Status

todo

