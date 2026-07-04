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

- Executing any protocol (task 006).
- Adapter-side validation of dataset/protocol compatibility (tasks 007+).

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

(record during implementation)

## Decisions

(record during implementation; e.g. exact exclusion list contents)

## Verification

```bash
pytest tests/unit/test_protocol*.py tests/unit/test_hash*.py
e3r protocol show scannet_single_layer_geometry_5cm
```

Acceptance per `.agent/plan.md`: `e3r protocol show scannet_single_layer_geometry_5cm`.

## Status

todo
