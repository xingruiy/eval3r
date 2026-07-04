# 003 — Protocol loader and canonical hashing

## Goal

Protocol YAML files load into validated `EvalProtocol` models, get a stable canonical
sha256 hash, and are discoverable through a protocol registry; `e3r protocol show` works.

## Scope

- `protocols/loader.py`: YAML → `EvalProtocol` with clear validation errors (file, field,
  which protocol required it).
- `core/hashing.py`: canonical hashing exactly per `.agent/schema.md` / `.agent/reproducibility.md`:
  parse → validate → serialize to JSON with sorted keys → strip non-semantic content →
  UTF-8 → sha256 → `sha256:` prefix.
- The non-semantic exclusion list (notes, formatting, report-only preferences that do not
  affect metrics) must be **explicit in code and covered by tests**.
- `protocols/registry.py`: name → protocol lookup over `protocols/builtin/`.
- Built-in YAML files, matching the templates in `.agent/protocols.md` exactly:
  `single_geometry`, `single_depth`, `single_pose`, `dtu_official_like_pointcloud`,
  `scannet_single_layer_geometry_5cm`, `scannet_double_layer_geometry_5cm`,
  `tanks_temples_training_official`, `tanks_temples_intermediate_server_only`,
  `eth3d_training_official_like`. (ScanNet protocols are `eval3r_native`; `single_depth`
  uses `mode: scale_median`; backend_preferences keys are registry kinds.)
- CLI: `e3r protocol show <name>` prints protocol, version, hash.
- Expected-hash regression test for each built-in protocol.

## Out of Scope

- Executing any protocol (task 007).
- Adapter-side validation of dataset/protocol compatibility (task 008+).

## Relevant Files

- `.agent/protocols.md` — protocol templates and validation checklist
- `.agent/schema.md` — "Canonical hashing"
- `.agent/reproducibility.md` — "Protocol hashing" (which fields hash, which may be excluded)

## Plan

1. Implement loader with pydantic validation and helpful error context.
2. Implement canonical hashing + explicit exclusion list.
3. Write built-in YAMLs from `.agent/protocols.md`; complete the double-layer ScanNet file
   by copying the single-layer fields with the documented variant changes.
4. Registry + `e3r protocol show`.
5. Tests: hash stable across YAML reformatting/comments; hash changes when a threshold,
   sampling count, masking method, or aggregation changes; expected-hash table.

## Findings

- Loader (`protocols/loader.py`): `load_protocol_file` / `load_protocol_text` /
  `load_protocol_data`; Pydantic `ValidationError` is reformatted to one `loc: msg` line per
  failing field and wrapped in `ProtocolValidationError(source, detail, name=...)` so the
  message names the file and the failing fields. Verified live: an unknown name and an
  incomplete YAML both fail with explicit, actionable messages.
- Hashing (`core/hashing.py`): validate → `model_dump(mode="json")` → strip non-semantic keys →
  `json.dumps(sort_keys=True, separators=(",",":"))` → sha256 → `sha256:` prefix.
  `canonical_protocol_payload()` is exposed so the exclusion policy is inspectable and tested.
- Registry (`protocols/registry.py`): discovers `builtin/*.yaml` by stem; `load_protocol`
  accepts a built-in name or a `.yaml`/`.yml` path. `eval3r.load_protocol` re-exported at
  top level (plan's public API). CLI `e3r protocol show <name>` + `e3r protocol list`
  implemented in `cli/protocol.py` (rich tables, verbose policy echo) and wired into `main.py`.
- Two doc-internal inconsistencies reconciled (both docs updated in the same change):
  1. `single_depth` uses `valid_region.method: valid_depth`, which was absent from
     `CullingSpec.method` in `.agent/schema.md`. Added `valid_depth` to the enum in both
     `.agent/schema.md` and `core/schema.py`, with a note; extended `test_schema` coverage
     implicitly via protocol load.
  2. DTU template had `version: 2014` (YAML int) but `DatasetVariant.version` is `str | None`;
     quoted it as `"2014"` in the built-in YAML. (The loader's clear error surfaced this.)
- The `tanks_temples_intermediate_server_only` doc snippet omits the eight required
  execution-policy fields (`alignment`, `confidence`, `masking`, `sampling`, `metrics`,
  `aggregation`, `failure_policy`, `reporting`), so it cannot validate against `EvalProtocol`
  as written. Filled them with explicit "no local evaluation" values (`metrics: []`, culling
  `none`, etc.) and set the GT block `local_evaluation_status: server_only` for consistency.
- 9 built-in protocols load, validate, and hash. Expected-hash regression table pinned in
  `tests/unit/test_protocols.py`.

## Decisions

- **Non-semantic hash exclusion list (explicit + tested)**:
  - recursive keys (any nesting depth): `notes`, `reason` — free-form human prose.
  - top-level keys: `reporting` — report-only output preferences that never change a metric.
  Everything else is hashed, including `name`, `protocol_version`, `schema_version`, `fidelity`,
  `dataset`, `ground_truth`, `local_evaluation`, `alignment`, `confidence`, `masking`,
  `sampling`, `metrics`, `aggregation`, `failure_policy`, `backend_preferences`. Tests assert
  stability under notes/reason/reporting/key-order changes and sensitivity to threshold,
  sampling count, masking method, and aggregation changes.
- `load_protocol` resolves a value that looks like a path (`.yaml`/`.yml` suffix or existing
  file) as a file, otherwise as a built-in name — so both `e3r protocol show <name>` and
  `e3r protocol show ./my.yaml` work.
- Registry uses `functools.lru_cache` for the builtin index (directory scanned once).

## Verification

```bash
pytest tests/unit/test_protocol*.py tests/unit/test_hash*.py
e3r protocol show scannet_single_layer_geometry_5cm
```

Acceptance per `.agent/plan.md`: `e3r protocol show scannet_single_layer_geometry_5cm`.

Outcomes (feature/repo-foundation):

```text
pytest            -> 69 passed (smoke + schema + hashing + protocols)
ruff check .      -> All checks passed!
mypy eval3r       -> Success: no issues found in 83 source files
e3r protocol show scannet_single_layer_geometry_5cm -> rich output, exit 0
e3r protocol show <unknown> -> explicit "not a known built-in" message, exit 1
```

## Status

done
