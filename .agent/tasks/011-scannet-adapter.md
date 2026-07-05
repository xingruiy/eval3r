# 011 — ScanNet adapter

## Goal

ScanNet validation-split geometry evaluation runs through `e3r benchmark run` under the
`eval3r_native` single-layer and double-layer 5cm protocols, with explicit visibility
culling and honest reconstruction-derived GT labeling.

## Scope

- `datasets/scannet.py`: official train/val/test scene ID resolution; exported SensReader
  layout (color/depth/pose/intrinsic directories); camera-to-world OpenCV-style pose
  normalization (recording source format); `depth_unit = 0.001` for 16-bit mm depth;
  GT mesh variant resolution; GT mesh hashing.
- Single-layer vs double-layer handled strictly via protocol/variant — never silently chosen.
- Protocol-defined visibility culling (`gt_visibility`, tolerance from protocol); culled
  fraction recorded per scene; never silently enabled or disabled.
- GroundTruthSpec: mesh / reconstructed / reconstruction_derived / dense_surface —
  never labeled independent.
- Mesh prediction path: surface sampling via mesh backend (200k, derived seeds) per protocol.
- `e3r benchmark run --dataset scannet --split val --protocol
  scannet_single_layer_geometry_5cm` end-to-end; `per_scene.csv` and `results.json` output.
- Tiny fixture (`tests/fixtures/scannet_tiny/`): 1–2 miniature scenes with tiny meshes,
  a few poses/intrinsics files in exported layout.

## Out of Scope

- Depth-metric evaluation on ScanNet sequences (possible later; task 014 adds depth metrics).
- Semantic/instance benchmarks (out of project scope entirely).

## Relevant Files

- `.agent/datasets.md` — "ScanNet" requirements and rules
- `.agent/protocols.md` — both ScanNet protocol templates (note: fidelity `eval3r_native`,
  community TransformerFusion-style convention; name the regression reference in Findings)
- `.agent/plan.md` — "Built-in adapter implications → ScanNet", milestone
- `CLAUDE.md` — ScanNet cautions

## Plan

1. Implement adapter: layout parsing, pose/intrinsics loading + normalization, mesh
   variant + fingerprint resolution, capability declaration.
2. Implement visibility-culling input resolution used by the mask stage.
3. Wire benchmark run; verify per-scene metadata (culled_fraction, mesh hash) lands in
   results.
4. Fixture + tests: scene discovery, pose normalization against a hand-computed case,
   depth_unit, variant selection, culling on/off strictly by protocol, GT provenance
   labeling, end-to-end fixture benchmark.

## Findings

- **Real data** at `/mnt/dataset/ScanNet` (standard NeuralRecon-style export). Per scene:
  `scans/<scene>/{color,depth,pose,intrinsic_depth.txt}` + `<scene>_vh_clean_2.ply` (GT mesh) +
  `<scene>.txt` (depthWidth/Height). Poses are 4×4 **cam-to-world OpenCV**; depth is 16-bit mm.
  Split lists at `<root>/{val,test,train}.txt`. Test scenes (incl. their GT meshes) are present
  locally, so test culling is validatable here.
- **Reference culling (found online).** Atlas `eval_mesh` (magicleap/Atlas) is the base metric:
  threshold 0.05, down_sample 0.02, both-direction nearest distances, precision/recall/fscore —
  and does **no** culling itself. NeuralRecon (zju3dv/NeuralRecon) adds the visibility culling by
  **rendering prediction depth from GT poses (pyrender) + TSDF-fusing (open3d) into a trimmed
  mesh**, then running `eval_mesh`. TransformerFusion ships precomputed occlusion masks — those
  are exactly `groundtruth/scannet_test_occlusion_masks/` in the data, but **test-split only**.
- **User decisions (this task):** val = **no** culling; culling applies to the **test** split
  only; implement the pyrender + TSDF-fusion culling; **change the ground rules** to permit
  render+TSDF as an evaluation-time visibility-culling mechanism.
- **De-risked on real data:** on scene0707_00 (GT-as-prediction + a hallucinated box 10 m away),
  the render+TSDF+KDTree-trim keeps 98% of real geometry and culls 100% of the far box
  (~3 s/scene). Full stack on real scenes: val fscore≈1.0 culled_fraction 0.0; test fscore≈0.99
  culled_fraction≈0.025. Convention: pyrender camera pose = `cam2world @ diag(1,-1,-1,1)`
  (OpenCV→OpenGL); open3d extrinsic = `inv(cam2world)`.

## Decisions

- Ground rules amended (CLAUDE.md + `.agent/backends.md` + `.agent/plan.md`): TSDF/render are
  forbidden **as reconstruction**, but allowed **solely** for evaluation-time visibility culling,
  always recorded in metadata, never silent. Added `pyrender` dependency (needs EGL).
- New `visibility` backend kind + `backends/visibility_render.py` `RenderTsdfVisibilityCull`
  (name `render_tsdf`): pyrender EGL depth render → open3d ScalableTSDFVolume → scipy cKDTree trim
  (vectorised, deterministic). Records renderer/TSDF versions, voxel/sdf_trunc, tolerance, pose
  counts, observed voxels, trajectory fingerprint, culled_fraction.
- Implemented the real `Open3dPointCloudBackend` (was a task-004 stub) because the ScanNet
  protocol declares `pointcloud: open3d`; registered it + the visibility backend in
  `default_registry`.
- `datasets/scannet.py` `ScanNetAdapter`: split/layout parsing, cam2world-OpenCV poses,
  `depth_unit=0.001`, GT mesh `_vh_clean_2.ply` + fingerprint, `load_trajectory` (drops non-finite
  poses), capabilities (independent_gt False, official_local_eval False, requires_external_renderer
  True). Single/double layer taken from the protocol variant (`metadata.layer_convention`), never
  guessed.
- Diagnostics split: `metrics/diagnostics.py` `partition_specs` / `build_diagnostic_metrics`;
  `culled_fraction` is injected by the runner/benchmark (0.0 on the no-cull path), never sent to
  the point-set metric layer. Fixed the generic runner to handle diagnostic metrics too.
- Benchmark gained a `gt_visibility` branch (`_evaluate_scene_visibility_culled`) mirroring the
  DTU official branch: resolve mesh pred + mesh GT + trajectory, cull, sample, score, inject
  culled_fraction + cull metadata; records the `visibility` backend version.
- Protocols: `scannet_single/double_layer_geometry_5cm` (val) bumped to 0.2.0 with culling removed
  (masking all `none`); new `scannet_test_single_layer_geometry_5cm` (test, gt_visibility, tolerance
  0.05, render params). Pinned hashes updated in `test_protocols.py`.
- Reference named in fixtures/protocol notes: Atlas `eval_mesh` (metric) +
  NeuralRecon/TransformerFusion (culling). Fidelity stays `eval3r_native`.

## Verification

```bash
ruff check .            # All checks passed!
mypy eval3r             # Success: no issues found in 89 source files
pytest -q               # 226 passed
mkdocs build            # OK
# real-data (offline, not committed): val scene0568_00 -> fscore 1.0, culled_fraction 0.0;
#   test scene0707_00 -> fscore 0.995, culled_fraction 0.025, renderer pyrender EGL recorded.
```

Fixture `tests/fixtures/scannet_tiny/` (2-m slab GT + prediction with a far unobserved box):
val keeps the far box (precision<0.95, culled_fraction 0), test culls it (culled_fraction 0.5,
precision 1.0). Covered by `tests/unit/test_scannet_adapter.py`,
`tests/unit/test_visibility_render.py` (skips cleanly without a headless GL context), and
`tests/integration/test_scannet_benchmark.py`. Report labels GT reconstruction-derived,
fidelity eval3r_native.

## Status

done
