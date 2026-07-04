# 010 — DTU official-like evaluation backend

## Goal

DTU benchmark evaluation runs locally through `e3r benchmark run` with official-like
fidelity by using the task-009 adapter plus a regression-tested official-like evaluator
for ObsMask/Plane culling and DTU's point-cloud scoring behavior.

## Scope

- `backends/dtu_eval.py` (`official_eval` registry kind): validated Python port of the
  official MATLAB point-cloud evaluation (accuracy/completeness with ObsMask/Plane and the
  0.2 mm downsampling), matching MATLAB behavior within documented tolerance; records
  whether the evaluator is the validated port or the MATLAB script.
- Update DTU capabilities for the official-like protocol:
  `official_local_eval=true`, `official_local_eval_method=validated_official_port`.
- The `dtu_official_like_pointcloud` protocol runs through `e3r benchmark run` and records
  evaluator method, Plane availability, mask/culling metadata, GT fingerprint, and protocol
  fidelity.
- Regression target: documented comparison against known official-like outputs, executed
  when full data is available locally (documented, not in normal CI).

## Out of Scope

- Any other dataset adapter.
- Generic DTU file layout and metadata parsing already covered by task 009.
- Camera/pose loading for DTU.

## Relevant Files

- `.agent/datasets.md` — "DTU" adapter requirements and rules, adapter interface
- `.agent/protocols.md` — `dtu_official_like_pointcloud` template
- `.agent/backends.md` — "DTU evaluation backend"
- `.agent/plan.md` — "Dataset adapter for DTU" milestone
- `CLAUDE.md` — DTU cautions

## Plan

1. Implement dtu_eval backend; validate the port against the official MATLAB code's
   published behavior on the fixture (record method + tolerance in Findings).
2. Wire `e3r benchmark run --dataset dtu --split test --protocol
   dtu_official_like_pointcloud` through the task-008 benchmark plumbing and task-007
   runner.
3. Tests: ObsMask/Plane application, missing-Plane official-like behavior, capability
   declaration, evaluator metadata, benchmark integration end-to-end on the fixture.

## Findings

(record during implementation — especially port-vs-MATLAB validation evidence)

## Decisions

(record during implementation)

## Verification

```bash
pytest tests/unit/test_dtu_eval*.py tests/integration/test_dtu_official_like_benchmark*.py
e3r benchmark run preds/ --dataset dtu --split test --protocol dtu_official_like_pointcloud  # on fixture
```

Acceptance per `.agent/plan.md` milestone; results.json records evaluator method and Plane
availability per scene.

## Status

todo
