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

- **Validated against the reference on REAL DTU data.** The user placed DTU at
  `/mnt/research/dataset/DTU` (GT under `groundtruth/stlNNN_total.ply`, `ObsMask/ObsMaskN_10.mat`,
  `ObsMask/PlaneN.mat`). Reference algorithm read from `jzhangbs/DTUeval-python` (read for parity,
  **not** vendored). On a cropped region of scan 24 with a constructed prediction, the reference
  (random shuffle) gives overall ∈ [0.21992, 0.21997] mm (spread 5e-5); the packaged deterministic
  port gives 0.21991 mm — inside the reference's own run-to-run band, and identical across runs.
- **Full stack on the real 5.17M-point scan 24** (adapter + benchmark + backend, GT `groundtruth/`,
  filename `mvsnet024_l3.ply`, real ObsMask/Plane): accuracy 0.343, completeness 0.248, overall
  0.295 mm — plausible DTU numbers, fidelity `official_like`, evaluator `validated_official_port`.
- The reference downsample is nondeterministic (random shuffle before radius-dedup). eval3r requires
  determinism, so the port seeds the shuffle (per-scene derived seed). The seeding does not bias the
  result relative to the reference band. This is the documented port-vs-reference tolerance.
- The official path works in the DTU **millimetre** frame; it deliberately bypasses the metre
  normalization (ObsMask BB/Res, plane, and the 20 mm cap are mm). Reported metrics are mm.

## Decisions

- `backends/dtu_eval.py` `DTUOfficialEval` (registry kind `official_eval`, name `dtu`,
  version `validated_official_port`). Uses scipy `cKDTree` (not sklearn — sklearn is not a
  declared dependency). Radius-dedup 0.2 mm downsample, ObsMask observability cull, ground-plane
  cull, 20 mm distance cap; empty-after-cull / all-beyond-cap raise `MetricError`.
- Benchmark gained an **official-eval branch**: when `backend_preferences.official_eval` is set,
  each scene is scored by the official backend on native-unit points + adapter visibility data,
  bypassing the generic normalize/align/sample/metric stages. Records evaluator method,
  plane availability, and cull counts per scene; `official_eval` backend version in results.json.
- DTU adapter: GT path now resolves `groundtruth/` then `Points/stl/`; added `load_visibility_data`
  (ObsMask required, missing Plane → `plane=None` → the scene fails under the official protocol
  rather than being silently scored); capabilities now `official_local_eval=true`,
  `official_local_eval_method=validated_official_port`.
- `dtu_official_like_pointcloud.yaml` bumped to 0.2.0: `official_eval: dtu`, `pointcloud: plyfile`,
  notes clarified (mm frame; missing-Plane fails; culling/downsample done in the port). New pinned
  hash `sha256:192ffe1c…`. Fixture regenerated (5 mm offset within the 20 mm cap; ObsMask 4³/Res100;
  keep-all Plane1; no Plane4 for the missing-Plane path).
- Offline parity lives in `tests/regression/test_dtu_parity.py`, skipped when the DTU data is absent
  (set `EVAL3R_DTU_ROOT` or use the default path); it ran and passed here on real data.

## Verification

```bash
ruff check .   # All checks passed!
mypy eval3r    # Success: no issues found in 88 source files
pytest -q      # 203 passed (incl. the real-data parity regression on this machine)
mkdocs build   # OK
# real-data end-to-end (offline; not committed): full scan 24 ->
#   accuracy 0.343 / completeness 0.248 / overall 0.295 mm, fidelity official_like
```

Acceptance met: `dtu_official_like_pointcloud` runs through `e3r benchmark run` with a
regression-validated official-like evaluator; results.json records the evaluator method, Plane
availability, and per-scene cull counts; missing-Plane scenes fail explicitly rather than being
scored. Covered by `tests/unit/test_dtu_eval.py`,
`tests/integration/test_dtu_official_like_benchmark.py`, and the offline
`tests/regression/test_dtu_parity.py`.

## Status

done
