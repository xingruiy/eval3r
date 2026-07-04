# 002 — Core schema models

## Goal

All Pydantic models from `.agent/schema.md` implemented in `eval3r/core/`, with
serialization tests and sample JSON/YAML fixtures, so every later slice validates against
one authoritative schema.

## Scope

- Enumerations: `PredictionModality`, `GroundTruthModality`, `GTProvenance`,
  `GTIndependence`, `GTDensity`, `SourcePoseFormat`, `NormalizedConvention`, `ScaleType`,
  `Fidelity`, `LocalEvaluationStatus`, `OfficialLocalEvalMethod`.
- Models: `DatasetCapabilities`, `DatasetVariant`, `GroundTruthSpec`, `LocalEvaluationSpec`,
  `Reconstruction`, `SceneData`, `UsesGTSpec`, `ConfidenceManifestSpec`,
  `ScenePredictionEntry`, `PredictionManifest`, `AlignmentSpec` (including the depth scale
  modes `scale_median` / `scale_least_squares` / `scale_affine` and `per_sequence`
  granularity), `SamplingSideSpec`, `SamplingSpec`, `CullingSpec`, `MaskingSpec`,
  `ConfidenceSpec`, `MetricSpec`, `AggregationSpec`, `FailurePolicySpec`, `ReportingSpec`,
  `EvalProtocol`, `MetricResult`, `SceneFailure`, `RunResult`.
- `RunResult` must include the full field set in `.agent/schema.md` (protocol_version,
  method_version, metric_definitions, aggregation, failure_policy, manifest_path,
  protocol_path, …) so `results.json` is self-describing.
- Round-trip serialization tests (model → JSON → model) for every model.
- Sample fixtures: one valid protocol YAML, one valid manifest YAML, one results.json,
  under `tests/fixtures/schema/`.
- Validation-error tests: bad enum value, missing required field, wrong shape.

## Out of Scope

- Protocol loading/hashing logic (task 003).
- Result writing to disk (task 006).

## Relevant Files

- `.agent/schema.md` — entire file (source of truth for every field)
- `.agent/reproducibility.md` — "Required result fields" (RunResult must cover them)
- `CLAUDE.md` — schema rules (any field change updates docs + models + tests + fixtures together)

## Plan

1. Implement enums and small specs in `core/types.py` / `core/schema.py`.
2. Implement manifest, protocol, and result models (`core/manifest.py`, `core/protocol.py`,
   `core/result.py`).
3. Write round-trip and validation tests; add fixtures.
4. Cross-check field-for-field against `.agent/schema.md`; fix doc or code if they diverge
   (same change, per CLAUDE.md).

## Findings

(record during implementation)

## Decisions

(record during implementation; e.g. pydantic v2 config choices, Path serialization form)

## Verification

```bash
pytest tests/unit/test_schema*.py
ruff check . && mypy eval3r
```

Every model round-trips; every required RunResult field from `.agent/reproducibility.md`
exists on the model.

## Status

todo
