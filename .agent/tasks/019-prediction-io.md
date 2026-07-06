# 019 — Prediction writer/reader (official eval3r-native prediction layout)

## Goal

A prediction writer/reader module that defines the **official eval3r-native on-disk
prediction layout**: any reconstruction method can use `PredictionWriter` to export its
predictions plus aux files (poses/trajectories, cameras, confidence) into a
self-contained directory with a validated `manifest.yaml`, and `read_prediction_dir`
resolves and verifies that directory. `e3r benchmark run` consumes the layout unchanged
(`load_or_infer_manifest` already honors `<pred_root>/manifest.yaml`).

## Scope

- Canonical layout (documented as *the* eval3r-native layout in
  `docs/prediction_format.md`):

  ```text
  <pred_root>/
    manifest.yaml                  # PredictionManifest (task-002 schema, unchanged)
    <scene_id>/
      mesh.ply | pointcloud.ply | pointmap.npy | depth/ | depth.png
      trajectory_tum.txt           # aux: per-scene poses (TUM) — feeds task-018 alignment
      cameras.json                 # aux: camera file
      confidence.npy | confidence/
  ```

  Canonical filenames map 1:1 onto `ScenePredictionEntry` fields; manifest paths stay
  relative to `pred_root` so the directory is relocatable.

- `eval3r/predictions/` package:
  - `layout.py` — canonical filename per entry field; modality↔entry consistency rules
    (e.g. a `mesh` manifest must have `mesh` entries).
  - `writer.py` — `PredictionWriter(root, method=…, dataset=…, variant=…, split=…,
    prediction_modality=…, scale=…, coordinate_frame=…, source_pose_format=…,
    uses_gt=…, …)`; `add_scene(scene_id, mesh=…, pointcloud=…, trajectory=…, …)`
    accepting source file paths (copied into the layout) and numpy arrays where eval3r
    owns a writer (pointcloud/pointmap via the pointcloud backend, trajectory → TUM
    text, confidence → `.npy`); meshes are file-copy only (eval3r builds no geometry).
    `finalize()` / context-manager exit validates and writes `manifest.yaml` with
    per-file sha256 fingerprints + `eval3r_version` in `metadata` dicts (no schema
    change — `metadata` is the designated open-ended field). Refuses duplicate scenes,
    entries inconsistent with the declared modality, and missing source files.
  - `reader.py` — `read_prediction_dir(root, verify=False)` → resolved manifest with
    absolute per-scene paths; explicit errors naming scene, field, and missing file;
    `verify=True` re-hashes files against recorded fingerprints.
- CLI `e3r prediction` sub-app (`cli/prediction.py`): `validate <root>` (schema +
  existence + fingerprints; per-scene outcome table; lists every failure; non-zero
  exit) and `show <root>` (manifest summary + scene/file table).
- Public API re-exports: `PredictionWriter`, `read_prediction_dir`.
- Integration test: write a prediction dir with the writer → `run_benchmark` on the
  existing tiny fixture dataset end-to-end.
- Docs: rewrite `docs/prediction_format.md` around the official layout with writer and
  reader examples; short layout note in `.agent/schema.md`; `.agent/plan.md`
  prediction-manifest section cross-referenced.

## Out of Scope

- Any change to the `PredictionManifest` schema fields (layout metadata lives in the
  existing `metadata` dicts).
- Format conversion at write time (no mesh→pointcloud sampling; the protocol owns
  sampling at evaluation time).
- Dataset-specific prediction name resolution (e.g. DTU `<method>XXX_l3.ply` stays in
  the DTU adapter).
- Upload/packaging tooling for benchmark servers.

## Relevant Files

- `.agent/schema.md` — "Prediction manifest"
- `.agent/plan.md` — "Prediction manifest" section
- `eval3r/core/manifest.py`, `eval3r/pipeline/benchmark.py` (`load_or_infer_manifest`),
  `eval3r/predictions/` (new), `eval3r/cli/prediction.py` (new), `eval3r/api.py`,
  `eval3r/__init__.py`, `docs/prediction_format.md`

## Plan

1. Layout rules + writer (path copies first, then array inputs, fingerprinting).
2. Reader with verification.
3. CLI validate/show.
4. Benchmark integration test.
5. Docs, verification.

## Findings

- **`load_or_infer_manifest` consumed the writer's output with zero benchmark changes**:
  a `run_benchmark` on a `PredictionWriter` directory records `manifest_inferred: false`,
  preserves the method name and `metadata.layout`, and the run dir's `manifest.yaml`
  echoes the written one verbatim (integration test asserts all three).
- No production writers existed for TUM trajectories, pointmaps, or confidence — only
  loaders (`trajectory_evo` reads TUM; `depth_common` reads `.npy`) and per-test TUM
  helpers. Task 019 added `write_tum_trajectory` ((N, 8) rows, strictly increasing
  timestamps enforced) plus `.npy`/PLY array paths in the writer; the TUM output
  round-trips through `EvoTrajectoryBackend.load_trajectory` (asserted in tests).
- `PredictionManifest` has no top-level `split` field — the split lives in
  `DatasetVariant.split`; the writer's `split=` kwarg maps there.
- A failed `add_scene` initially left a partial scene directory that the
  stale-output guard then refused on retry (caught by the malformed-array test).
  `add_scene` is now atomic per scene: since a pre-existing scene dir is refused
  up front, everything under it belongs to the current call and is removed on any
  placement failure, so the scene can be re-added after fixing the input.
- Pointmaps legitimately contain NaN (invalid pixels), so the pointmap array path
  allows non-finite values while the pointcloud array path rejects them.
- Reconfirmed the stale `eval3r` 0.3.0 in site-packages shadows the repo when a
  script's `sys.path[0]` is not the repo root (first smoke run mixed old CLI with
  new modules); in-process smoke/tests must run from the repo root or with
  `PYTHONPATH` set.

## Decisions

- One dataclass, `PredictionCheck`, backs both entry points: `check_prediction_dir`
  never raises on per-scene problems (the CLI renders its `issues` as a table) and
  `read_prediction_dir` is the strict form that aggregates **every** problem —
  each naming scene, field, and absolute path — into one `PredictionLayoutError`
  (new `Eval3rError` subclass), never just the first.
- Layout provenance lives entirely in the schema's `metadata` dicts (no field
  changes): top-level `layout: "eval3r-native-v1"` + `eval3r_version`; per-scene
  `fingerprints: {field: "sha256:<hex>"}` via the existing `file_fingerprint`.
  Directory fields (`depth_dir`/`confidence_dir`) get a joint hash over
  `relpath:sha256` lines of the sorted file list, so any added/removed/edited
  frame changes it.
- `verify=True` never passes vacuously: a resolved file with no recorded
  fingerprint is itself reported (hand-written manifests validate with
  `--no-verify`).
- Copy-only for meshes, depth, and camera files (eval3r builds no geometry);
  array inputs only where eval3r owns a plain writer (pointcloud/pointmap/
  confidence/trajectory). Copied files keep their source suffix on the canonical
  stem (`mesh.obj` stays `.obj`); array outputs get canonical extensions.
- Modality↔entry consistency is enforced at `add_scene` (declared modality's field
  required; aux fields always allowed); `colmap_reconstruction` is refused — no
  canonical single-file layout exists for a COLMAP model.
- Safety refusals: existing `manifest.yaml` is never overwritten; scene ids must
  be plain directory names; per-scene confidence files require declared
  `confidence.present`; the context manager finalizes only on clean exit, so a
  directory that errored mid-export never carries a manifest.
- `e3r prediction validate` defaults to fingerprint verification (`--no-verify`
  opts out); it prints the manifest panel, a per-scene ok/fail table, and every
  failure verbatim (exit 1 on any).
- Public re-export: `read_prediction_dir` as the usual lazy wrapper;
  `PredictionWriter` via PEP 562 module `__getattr__` (a wrapper function cannot
  stand in for a class users construct and isinstance-check).

## Verification

```bash
ruff check .   # All checks passed!
mypy eval3r    # Success: no issues found in 101 source files
EVAL3R_ETH3D_TOOL=~/xingrui_ws/tools/multi-view-evaluation/build/ETH3DMultiViewEvaluation \
  pytest -q    # 463 passed, 3 skipped (the 3 = TnT real-toolbox tests only)
mkdocs build   # OK (docs/prediction_format.md rewritten, already in nav)
# in-process smoke (PYTHONPATH=repo; PATH `e3r` is a different package):
#   writer (file copy + pointcloud/trajectory arrays) -> read_prediction_dir(verify=True)
#   -> `e3r prediction validate` exit 0 ("2/2 scenes valid, fingerprints verified"),
#   `show` table; tampering one PLY -> verify raises naming scene/field/path with
#   recorded+actual sha256, validate exit 1; fresh writer dir -> run_benchmark on the
#   fixture adapter: fscore 1.0, manifest_inferred False.
```

New tests: `tests/unit/test_prediction_writer.py` (7),
`tests/unit/test_prediction_reader.py` (6),
`tests/integration/test_prediction_cli.py` (5, incl. writer→`run_benchmark`
end-to-end proving the layout is consumed unchanged as a declared manifest).

## Status

done
