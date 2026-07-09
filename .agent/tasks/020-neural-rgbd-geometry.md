# 020 — Neural-RGBD geometry adapter (culled/source mesh; depth & pose deferred)

## Goal

Give `eval3r` a real Neural-RGBD (`neural_rgbd`) dataset adapter and geometry protocols so
NRGBD mesh predictions can be scored through the normal
`run_benchmark(dataset="neural_rgbd", ...)` path. Neural-RGBD is a standard indoor
neural-surface / SLAM reconstruction benchmark; the community reports geometry metrics
(accuracy / completeness / chamfer / precision / recall / F-score) against the released
**culled** GT mesh. This slice implements **geometry only** — depth and pose are deferred.

## Scope

- `NeuralRGBDAdapter` in `eval3r/datasets/neural_rgbd.py` (replaces the stub), mesh-only,
  modeled on `ScanNetAdapter`.
- Protocol-driven **culled vs source** GT-mesh variant selection (never silent).
- Two auto-discovered built-in protocols:
  `eval3r/protocols/builtin/neural_rgbd_geometry_culled.yaml` and
  `neural_rgbd_geometry_source.yaml` — both `native`, `alignment: none`,
  surface-area sampling `n_points: 200000`, F-score threshold `0.05` (matches the reference
  driver `../3d-eval/scripts/run_da3_nrgbd_eval.py`).
- Registry wiring (`registry.py` + `datasets/__init__.py`).
- A minimal, general, opt-in `bind_protocol(protocol)` hook on the geometry benchmark loop so
  an adapter whose GT geometry depends on the protocol (NRGBD culled vs source mesh) can
  select the right file before `load_scene` is called. The dataset-specific decision stays in
  the adapter; the pipeline only offers the hook.
- Tiny fixtures + unit and integration tests; docs updates.

## Out of Scope

- **Depth** evaluation (`neural_rgbd_depth.yaml`) and **pose** evaluation — deferred.
- `load_trajectory`, `focal.txt` / `poses.txt` parsing, and resolving the OpenGL-vs-OpenCV
  pose convention. Recorded as a note only; irrelevant to mesh-to-mesh geometry scoring.
- Any eval-time TSDF / visibility culling of predictions — the culled GT mesh is a released
  asset, not an eval-time step.
- Plugin/entry-point registration; large NRGBD data download inside the package.

## Relevant Files

- `.agent/datasets.md` — "Neural-RGBD" adapter requirements (OpenGL convention; separate
  culled vs uncropped mesh variants; synthetic vs real provenance).
- `.agent/protocols.md` — built-in protocol set.
- `eval3r/datasets/neural_rgbd.py` (rewritten), `eval3r/datasets/scannet.py` (template),
  `eval3r/datasets/registry.py`, `eval3r/datasets/__init__.py`.
- `eval3r/pipeline/benchmark.py` — geometry loop; `bind_protocol` hook.
- `eval3r/protocols/builtin/neural_rgbd_geometry_{culled,source}.yaml` (new),
  `scannet_single_layer_geometry_5cm.yaml` (template).
- `tests/fixtures/neural_rgbd_tiny/` (new), `tests/unit/test_neural_rgbd_adapter.py` (new),
  `tests/integration/test_neural_rgbd_benchmark.py` (new),
  `tests/unit/test_dataset_registry.py`.
- Docs: `.agent/datasets.md`, `.agent/protocols.md`, `.agent/plan.md`, `docs/datasets.md`.

## Plan

1. Task file + README index (this file; mark `in_progress`).
2. Adapter: mesh-only, `mesh_variant` default `culled`, `bind_protocol`, `_mesh_variant`,
   honest capabilities (`independent_gt=True`, synthetic-exact GT), provenance classifier.
3. Pipeline: opt-in `bind_protocol` hook before the scene loop.
4. Registry + `__init__` wiring.
5. Two protocol YAMLs (culled/source), auto-discovered.
6. Fixtures (tiny boxes: source, slightly-smaller culled, near-perfect pred) + tests.
7. Docs.
8. Verify: ruff / mypy / pytest / mkdocs, then the real-data check against
   `../3d-eval/datasets/nrgbd_meshes/official` scoring each scene's own `neural_rgbd.ply`
   vs `gt_mesh_culled.ply`, cross-checked against the reference single-file `metric geometry`
   numbers.

## Findings

Confirmed from the real dataset at `../3d-eval/datasets` and the reference driver
`../3d-eval/scripts/run_da3_nrgbd_eval.py`:

- GT meshes at `nrgbd_meshes/official/<scene>/`: `gt_mesh.ply` (uncropped/source),
  `gt_mesh_culled.ply` (culled), `neural_rgbd.ply` (the NRGBD method's own reconstruction —
  a real prediction for verification). 10 scenes: breakfast_room, complete_kitchen,
  green_room, grey_white_room, icl_living_room, kitchen, morning_apartment, staircase,
  thin_geometry, whiteroom. (The `datasets/nrgbd/<scene>/` trajectory tree has 9 — no
  icl_living_room — but geometry needs only the mesh tree.)
- Reference geometry eval is **mesh-to-mesh, `alignment.mode: none`** (its
  `alignment_transforms.json` records the identity), **threshold 0.05**, **sample 200000**,
  default GT source `official_culled`.
- Geometry is convention-free: no camera is used, so the OpenGL-vs-OpenCV pose-convention
  question (reference declares `cam_to_world_opencv`; eval3r docs note native OpenGL) does not
  affect the score. It matters only for the deferred depth/pose work; recorded as a note.
- These NRGBD scenes are synthetic with exact artist-mesh GT → `provenance: synthetic_exact`,
  `independence: independent`, `independent_gt=True` (unlike ScanNet's reconstruction-derived
  mesh). `depth_unit` for the depth tree is `0.001` (mm→m), noted for the future depth task.
- Architectural note: the geometry loop reads `scene.gt_mesh` from `load_scene(scene_id)`,
  which takes no protocol; NRGBD's mesh path depends on the protocol variant, hence the
  opt-in `bind_protocol` hook.

## Decisions

- **Geometry is mesh-to-mesh, `alignment: none`.** Prediction and GT share the dataset world
  frame; the reference driver records the identity alignment. The pose convention (OpenGL vs
  OpenCV) therefore does not affect the score — recorded as a note, deferred to depth/pose.
- **Culled/source is a protocol variant** (`neural_rgbd_geometry_culled` /
  `neural_rgbd_geometry_source`), mapped by `_mesh_variant` from `dataset.variant`; an
  unknown/absent variant raises naming both protocols. Two separate YAMLs enforce "do not mix
  culled and uncropped under one protocol".
- **`bind_protocol` opt-in hook** added to `run_benchmark_geometry` (one guarded block): the
  interface's `load_scene(scene_id)` gets no protocol, but NRGBD's GT mesh path depends on the
  variant. The hook lets the adapter pick the mesh before scenes load; the dataset-specific
  decision stays in the adapter, and adapters without protocol-dependent GT ignore it.
- **GT is `synthetic_exact` / `independent`, `independent_gt=True`.** The released scenes are
  synthetic with exact artist meshes. A `_KNOWN_REAL_SCENES` classifier (empty today) keeps
  the synthetic/real distinction for any future real-scene bundle.
- Protocols are `native` community 5cm-F-score conventions (no official Neural-RGBD
  server/toolbox), so no official-toolbox wrapper and no fake stand-in.
- `culled_fraction` kept in the culled protocol and recorded as 0 (culling is pre-baked into
  the released mesh, not an eval-time step).

## Verification

```bash
ruff check .                    # All checks passed!
mypy eval3r                     # Success: no issues found in 101 source files
pytest -q                       # 473 passed, 6 skipped (official-toolbox env skips).
                                # 8 pre-existing failures are ModuleNotFoundError: pycolmap,
                                # unrelated to this task (no camera/pycolmap/eth3d changes).
pytest -q tests/unit/test_neural_rgbd_adapter.py \
          tests/integration/test_neural_rgbd_benchmark.py \
          tests/unit/test_dataset_registry.py \
          tests/unit/test_protocols.py            # 59 passed
mkdocs build                    # Documentation built in 0.15 seconds
```

New tests: `tests/unit/test_neural_rgbd_adapter.py` (15), `tests/integration/
test_neural_rgbd_benchmark.py` (2), plus `test_dataset_registry.py` and `test_protocols.py`
(two new builtins + pinned hashes) updated. Fixtures: `tests/fixtures/neural_rgbd_tiny/`
(tiny box meshes: source, smaller culled, near-perfect pred).

Real-data check (`../3d-eval/datasets/nrgbd_meshes/official`, 10 scenes) via the adapter
`run_benchmark(dataset="neural_rgbd", split="all", protocol="neural_rgbd_geometry_culled")`,
scoring each scene's own `neural_rgbd.ply` vs `gt_mesh_culled.ply`: mean fscore 0.916,
precision 0.969, recall 0.877 — the method's own reconstruction scores strongly, as expected.
Cross-checked against the reference single-file path (`eval3r.cli.main metric geometry
--threshold 0.05 --sample 200000`): green_room adapter 0.9885 vs reference 0.9888; kitchen
adapter 0.6891 vs reference 0.6875 — agreement within sampling-seed noise, confirming the
adapter+protocol path reproduces the reference mesh-to-mesh numbers.

Known limitation: depth and pose evaluation are deferred (`neural_rgbd_depth.yaml`,
focal.txt/poses.txt loading, OpenGL-convention resolution) — see the backlog in
`.agent/tasks/README.md`.

## Status

done
