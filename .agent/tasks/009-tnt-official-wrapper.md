# 009 — Tanks and Temples official wrapper

## Goal

Tanks and Temples training-split evaluation runs locally by wrapping the official
evaluation code (fidelity `official`), while intermediate/advanced splits are cleanly
refused as server-only.

## Scope

- `datasets/tanks_temples.py`: resolve public training scenes, GT point cloud, crop-volume
  JSON, `.log` trajectory, alignment transform; per-scene thresholds come from official
  metadata/backend output, never a global constant.
- `backends/tnt_official.py` (`official_eval` registry kind): invoke the official
  evaluation script (vendored or user-supplied checkout — record which and its
  version/commit); normalize its output into `MetricResult` / results.json; record backend
  version, command, per-scene threshold, alignment/ICP details the backend performed.
- Capabilities: training → official_local_eval=true, method=official_script_wrapper;
  intermediate/advanced → server_only_eval=true, method=server_only.
- CLI behavior: running a server-only protocol
  (`tanks_temples_intermediate_server_only`) fails clearly before any computation, telling
  the user to submit to the official server.
- GT fingerprinting: GT point cloud, crop file, alignment file, backend version.
- Tiny fixture with fabricated official-script outputs for output-parsing tests
  (`tests/fixtures/tanks_temples_tiny/`); a dry-run/wrapper test that skips when the
  official code is absent.

## Out of Scope

- Reimplementing the official evaluation (explicitly forbidden without regression-tested
  reason).
- Full-dataset validation in CI (documented manual step when data is available).

## Relevant Files

- `.agent/datasets.md` — "Tanks and Temples" requirements and rules
- `.agent/protocols.md` — `tanks_temples_training_official` + server-only stub
- `.agent/backends.md` — "Tanks and Temples backend"
- `.agent/plan.md` — milestone
- `CLAUDE.md` — Tanks and Temples cautions

## Plan

1. Implement adapter resolution of the five per-scene artifacts.
2. Implement wrapper backend: subprocess or import invocation, output parsing,
   version/command recording.
3. Wire capability gating so the CLI refuses server-only splits with a clear message.
4. Tests: artifact resolution, output parsing from fixture text, server-only refusal,
   per-scene threshold recording, capability declarations.

## Findings

(record during implementation — official code source, version pinning approach)

## Decisions

(record during implementation; e.g. vendored vs external checkout of official code)

## Verification

```bash
pytest tests/unit/test_tnt*.py
e3r benchmark run preds/ --dataset tanks_temples --split training --protocol tanks_temples_training_official  # with official code available
e3r benchmark run preds/ --dataset tanks_temples --split intermediate --protocol tanks_temples_intermediate_server_only  # must refuse
```

Acceptance per `.agent/plan.md` milestone; refusal path verified in tests.

## Status

todo
