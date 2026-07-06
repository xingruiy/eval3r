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

(to be filled during implementation)

## Decisions

(to be filled during implementation)

## Verification

(to be filled during implementation)

## Status

todo
