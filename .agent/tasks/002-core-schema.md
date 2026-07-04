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

- All models from `.agent/schema.md` implemented and transcribed **field-for-field** (no
  schema changes, so `.agent/schema.md` remains authoritative and in sync — no doc edit needed).
- Module layout: enums → `core/types.py` (as `Literal` type aliases, not `enum.Enum`); shared
  spec models → `core/schema.py`; manifest models → `core/manifest.py`; `EvalProtocol` →
  `core/protocol.py`; `MetricResult`/`SceneFailure`/`RunResult` → `core/result.py`.
- `core/__init__.py` re-exports every public model (single import surface) with an `__all__`.
  Import graph is acyclic: `types → schema → {manifest, protocol}`, `result → schema+manifest`.
- Tests (`tests/unit/test_schema.py`, 34 cases + smoke): parametrized JSON round-trip for one
  instance of **every** exported model (a coverage test asserts no exported model lacks an
  instance); fixture validation; required-`RunResult`-fields check against
  `.agent/reproducibility.md`; and validation-error cases (bad enum, missing field, wrong
  shape, extra field).
- Fixtures under `tests/fixtures/schema/`: hand-written minimal `protocol.yaml` and
  `manifest.yaml` (exercise documented defaults) + `results.json` generated from a constructed
  `RunResult` (guaranteed valid and round-tripping).
- Verified `RunResult` covers every field in `.agent/reproducibility.md` "Required result
  fields" via `test_run_result_has_all_required_fields`.

## Decisions

- **Pydantic v2** (2.12). Shared base `E3RModel(BaseModel)` with
  `model_config = ConfigDict(extra="forbid")` so unexpected keys are explicit validation
  errors, not silent typos; each model has a `metadata`/`parameters` dict for open-ended data,
  so forbidding stray top-level keys costs no flexibility.
- Enums as `Literal` type aliases (`TypeAlias`) so YAML/JSON string values map directly with no
  conversion layer, matching `.agent/schema.md` exactly.
- Nested-model defaults use `Field(default_factory=...)` instead of the doc's `= Model()`
  literal (behaviourally identical in pydantic; avoids a shared-instance foot-gun). Scalar and
  mutable-collection defaults (`[]`, `{}`) mirror the doc directly (safe under pydantic).
- `Path` fields serialize to strings in JSON mode and re-parse to `Path`, so round-trip
  equality holds.
- Added a small helper alias `IntrinsicsSource` in `types.py` for the repeated
  `Literal["predicted","gt","dataset_default","unknown"]` used by `Reconstruction` and
  `PredictionManifest` (same literal set the doc inlines; not a new schema field).

## Verification

```bash
pytest tests/unit/test_schema*.py
ruff check . && mypy eval3r
```

Every model round-trips; every required RunResult field from `.agent/reproducibility.md`
exists on the model.

Outcomes (feature/repo-foundation):

```text
pytest            -> 36 passed (3 smoke + 33 schema)
ruff check .      -> All checks passed!
mypy eval3r       -> Success: no issues found in 83 source files
```

## Status

done
