# Task 022 — Provenance-driven prediction adaptation + two-level protocol identity

## Goal

Separate the *scientific comparability contract* (the protocol, hashed) from *prediction
adaptation* (how a prediction enters the protocol's frame given its declared provenance — not
hashed). Make prediction provenance load-bearing — especially `scale` (`metric`/`relative`/
`unknown`), which is currently recorded but never consumed — so a relative-scale prediction is
either auto-resolved within the protocol's allowed envelope or refused loudly, never silently
scored as garbage. Add a compact `--as cw@opencv@sim3` override grammar as the ergonomic front
door, replacing the scattered per-modality flags, and record exactly what adaptation was
applied and why.

Numbers for the existing metric path with metric predictions must be **byte-for-byte
unchanged** (regression guarantee): adaptation is a no-op on that path.

## Scope

- New `eval3r/core/adaptation.py`:
  - `AdaptationOverride` — parsed compact override (`direction`, `axes`, `world_frame`,
    `scale`, `alignment_mode`, `unit`), all optional.
  - `parse_adaptation_tokens(s)` — **order-independent** grammar; each token maps to exactly
    one axis via an alias vocabulary; unknown/duplicate-axis tokens raise with the full valid
    vocabulary.
  - `AdaptationRecord` — the non-hashed fingerprint written into results: `pose_convention`,
    `world_frame`, `unit`, `alignment` (effective mode), `reason`, `within_envelope`,
    `source`.
  - `resolve_adaptation(protocol, provenance, override)` — the brain: resolves convention /
    world-frame / unit from provenance (override wins on conflict); resolves the effective
    alignment mode from the protocol default + provenance-implied scale resolution, gated by
    the protocol's allowed envelope (auto-adapt within; refuse loudly when the envelope forbids
    what provenance requires).
- Schema: extend `AlignmentSpec` with the allowed envelope (`allowed_modes`,
  `scale_resolution`); add `RunResult.adaptation: AdaptationRecord`. Make
  `PredictionManifest.scale` / `Reconstruction.scale` drive the resolver.
- Hashing: envelope fields stay in the hash; runners **stop recomputing the hash on
  adaptation** (adaptation no longer mutates the hashed protocol). Re-pin the 12 builtin hashes
  and bump their `protocol_version`s; set explicit envelopes on the builtins.
- Pipeline: refactor the three `apply_*_overrides` into a shared `resolve_adaptation` call whose
  `AdaptationRecord` the stages consume; add adaptation plumbing to the benchmark path (no
  override channel today).
- Surface: `--as / --adapt` option on `metric geometry|depth|pose` and `benchmark run`;
  `adapt=` param on `api.evaluate_*` and `api.run_benchmark`; keep the existing
  `--pred-world-frame` / `--pred-pose-convention` / `--gt-pose-convention` / `--align` flags as
  aliases that populate the same override.
- Result / diff: write the fingerprint into `RunResult.adaptation` and run-directory
  `config.yaml`; `e3r diff` **warns** (not rejects) when two runs share a protocol hash but
  differ in adaptation.
- Docs: `.agent/schema.md`, `.agent/protocols.md`, `.agent/reproducibility.md`,
  `.agent/datasets.md` updated in the same change.

## Out of Scope

- Refusing-by-default on provenance mismatch — user chose **auto-adapt + record** within the
  envelope (refusal is only the guardrail when the envelope forbids the required adaptation).
- Changing metric formulas, sampling, masking, or culling behavior.
- Verified handedness mappings for the formats task 021 left as raising (CO3D,
  Tanks and Temples `.log`).
- Removing the legacy per-modality flags (kept as aliases).

## Relevant Files

- New: `eval3r/core/adaptation.py`, `tests/unit/test_adaptation.py`.
- Schema/hash: `eval3r/core/schema.py` (`AlignmentSpec` envelope), `eval3r/core/types.py`,
  `eval3r/core/result.py` (`RunResult.adaptation`), `eval3r/core/hashing.py`,
  `tests/unit/test_protocols.py` (re-pin `EXPECTED_HASHES`), `eval3r/protocols/builtin/*.yaml`.
- Pipeline: `eval3r/pipeline/runner.py`, `pose_runner.py`, `depth_runner.py`,
  `benchmark.py`, `stages/align.py` (Sim3-on-metric gate consults the envelope),
  `stages/normalize.py` (reuse `normalize_world_frame` / `normalize_to_meters` as-is).
- Surface: `eval3r/api.py`, `eval3r/cli/metric.py`, `eval3r/cli/benchmark.py`,
  `eval3r/reports/run_directory.py`, diff module (`diff_runs`).
- Reuse (do not reinvent): `PoseConventionTransform` (`core/pose_convention.py`),
  `convention_for` (`datasets/conventions.py`).
- Docs: `.agent/schema.md`, `.agent/protocols.md`, `.agent/reproducibility.md`,
  `.agent/datasets.md`, `docs/quickstart.md`.

## Plan

Two-level identity. **Protocol (hashed)** declares the science + the *allowed alignment
envelope*: which alignment modes are scientifically defensible for this benchmark and whether
scale resolution is permitted (`scale_resolution: forbidden|allowed|required_if_relative`).
**AdaptationRecord (not hashed)** captures what was actually done to *this* prediction.

```
provenance (manifest/adapter)  ─┐
compact override (--as ...)     ─┼─▶  resolve_adaptation()  ─▶  AdaptationRecord
protocol.alignment envelope     ─┘                              (effective, non-hashed)
```

Resolver logic for effective alignment: start at `protocol.alignment.mode`; if provenance
implies scale resolution (`scale` in `{relative, unknown}` and GT metric) or the override names
a mode, compute the desired mode and gate it against `allowed_modes` / `scale_resolution`:
desired ∈ envelope → apply, `within_envelope=True`, record the reason (auto-adapt); desired ∉
envelope → raise a rich error naming prediction, declared scale, protocol, envelope, and fix.
Compose `(direction, axes)` into the existing `SourcePoseFormat` (pose) or `WorldAxes`
(geometry) via `convention_for`. Envelope defaults: metric official / official-like
(`dtu_*`, `eth3d_*`, `tanks_temples_*`, ScanNet metric) → envelope reflecting the official
method, `scale_resolution: forbidden`; `single_*` / eval3r-native → permissive,
`scale_resolution: allowed`.

Grammar vocabulary (order-independent, aliased): `cw|c2w`→cam_to_world, `wc|w2c`→world_to_cam,
`opencv|cv`, `opengl|gl`, `metric|relative|unknown`, `none|se3|rigid|sim3|scale_median|scale_ls`,
`m|cm|mm|um`. A bare `opencv|opengl` token is camera-axes for pose, world-frame for geometry.

## Findings

Recorded during Phase-1 exploration (source of the two defects this task fixes):

- **`PredictionManifest.scale` is dead metadata** — recorded at `manifest.py:60`,
  `schema.py:112`, but no code consumes it. Scale handling is delegated entirely to
  `protocol.alignment.mode`, so a relative prediction under a metric SE3 protocol produces
  meaningless numbers with no signal. This task makes it load-bearing.
- **`alignment` is inside the canonical hash** (`hashing.py:27-31` excludes only
  `reporting`/`notes`/`reason`). Every alignment/convention tweak recomputes
  `compute_protocol_hash` on a mutated protocol copy (`runner.py:339`, `pose_runner.py:337`),
  minting a new protocol identity; `allow_override=True` only grants permission and still
  changes the hash.
- **The right pattern already exists in one place**: `pred_world_frame` is recorded in the
  `overrides` dict *without* touching the hash (`runner.py:337-339`) — the seed this task
  generalizes into `AdaptationRecord`.
- The three `apply_*_overrides` functions are the uniform injection point
  (`runner.py:87`, `pose_runner.py:88`, `depth_runner.py:148`). The benchmark path has **no**
  override plumbing (`benchmark.py` uses the protocol as-is at `:623`) — the real gap for the
  chosen scope.
- No `@`-delimited override grammar exists anywhere today; convention is specified via full
  `SourcePoseFormat` literals mapped in `datasets/conventions.py:30-39`.

## Decisions

Confirmed with the user (senior CV researcher) before planning:

1. **Auto-adapt + record** is the default when provenance implies an adaptation — applied
   automatically within the protocol's allowed envelope; a loud refusal only when the envelope
   forbids the adaptation provenance requires (the guardrail against silently rescaling a
   metric benchmark).
2. **Two-level identity** — protocol hash covers the science + allowed envelope; the concrete
   per-prediction choice is a separate non-hashed fingerprint; diff warns on differing
   fingerprints. The 12 pinned hashes are reworked once (with `protocol_version` bumps).
3. **Order-independent tokens** grammar.
4. Scope covers the single-file metric commands **and** the benchmark path, plus `api.*`.
5. Legacy per-modality flags are kept as aliases (no silent removal, per schema/back-compat
   rules).

## Verification

- Implemented `tests/unit/test_adaptation.py`: order-independence, alias parsing,
  unknown/duplicate token errors with vocabulary, relative-scale auto-adaptation under
  `single_geometry`, refusal under `dtu_official_like_pointcloud`, and pose convention
  composition via `convention_for`.
- Re-pinned all 12 built-in protocol hashes after adding explicit envelopes and bumping
  protocol versions. `tests/unit/test_protocols.py::test_builtin_hash_regression` passes.
- Updated integration expectations for `--align none` on depth/pose: metric values still expose
  scale/offset error, while `protocol_hash` remains the base protocol hash and
  `RunResult.adaptation` records the effective alignment.
- Added diff coverage for adaptation warnings: same protocol hash plus different
  `AdaptationRecord` emits a warning; unchanged records do not.
- Verification commands run:
  - `ruff check .` → pass.
  - `mypy eval3r` → pass.
  - `mkdocs build` → pass.
  - `PYTHONPATH=. pytest -k 'not camera_pycolmap and not load_scene_records_cameras_and_non_pinhole_limitation'` → 530 passed, 6 skipped, 9 deselected.
  - Focused changed tests:
    `PYTHONPATH=. pytest tests/unit/test_adaptation.py tests/unit/test_diff.py tests/integration/test_single_depth_metric.py::test_align_none_override_exposes_scale_error tests/integration/test_single_pose_metric.py::test_align_none_override_exposes_offset_and_changes_hash tests/integration/test_single_pose_metric.py::test_pinned_alignment_protocol_refuses_override tests/unit/test_pose_convention_runner.py::test_convert_stage_fails_for_unmapped_format -q` → 27 passed.
- Environment limitation: bare `pytest` failed collection in this shell because installed ROS
  launch-testing plugins did not import the local package; `PYTHONPATH=. pytest` collected.
  The full `PYTHONPATH=. pytest` run then failed only pycolmap-dependent tests because
  `pycolmap` is not installed in this environment. Those tests are unrelated to task 022 and
  were excluded in the broad verification command above.

Drive the CLI **in-process** (the PATH `e3r` is a different package).

## Status

done
