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

(record during implementation — name the community reference implementation used for any
regression numbers)

## Decisions

(record during implementation)

## Verification

```bash
pytest tests/unit/test_scannet*.py tests/integration/test_scannet_benchmark*.py
e3r benchmark run preds/ --dataset scannet --split val --protocol scannet_single_layer_geometry_5cm  # on fixture
```

Acceptance per `.agent/plan.md` milestone; report clearly labels GT as
reconstruction-derived and fidelity as eval3r_native.

## Status

todo
