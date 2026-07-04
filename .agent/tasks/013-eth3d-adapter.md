# 013 — ETH3D adapter

## Goal

ETH3D training-split point-cloud evaluation runs locally under
`eth3d_training_official_like`, with COLMAP text camera parsing via pycolmap, honest
non-pinhole handling, and the official-like claim validated against the official
multi-view-evaluation tool.

## Scope

- `datasets/eth3d.py`: COLMAP text camera parsing (world_to_cam_colmap → internal
  camera-to-world OpenCV meters, source format recorded); training GT scan resolution;
  occlusion-aware GT preparation respected; test split marked server_only.
- `backends/camera_pycolmap.py` (`camera` registry kind): pycolmap-backed camera loading;
  minimal internal pinhole path allowed only for scenes whose camera model actually is
  simple pinhole — unsupported models fail with a clear warning, never silently
  approximated (record limitation in scene metadata).
- Protocol validation (per the note in `.agent/protocols.md`): regression-compare eval3r's
  accuracy/completeness/F1 against the official ETH3D multi-view-evaluation tool on a
  fixture; the official tool reports tolerances 1/2/5/10/20/50 cm — extend the protocol's
  metric list to the official tolerance set if needed for comparability, bumping protocol
  version + expected hash.
- Capabilities: dense_geometry, independent_gt, training official_local_eval per outcome of
  the validation (validated_official_port) — record honestly.
- Tiny fixture (`tests/fixtures/eth3d_tiny/`): miniature COLMAP text files + tiny GT cloud.

## Out of Scope

- Depth or pose ETH3D protocols.
- Server-only test-split evaluation.

## Relevant Files

- `.agent/datasets.md` — "ETH3D" requirements and rules
- `.agent/protocols.md` — `eth3d_training_official_like` (including the tolerance note)
- `.agent/backends.md` — "Camera backend"
- `CLAUDE.md` — ETH3D cautions

## Plan

1. Implement COLMAP text parsing through pycolmap; convention-normalization test against a
   hand-computed pose.
2. Implement adapter GT/scene resolution and capability declaration.
3. Run the official multi-view-evaluation tool on the fixture; align eval3r behavior and
   protocol metric list; update protocol version/hash/docs together if changed.
4. Tests: camera parsing (pinhole + one nontrivial model), non-pinhole warning path,
   server-only test split refusal, end-to-end fixture benchmark.

## Findings

(record during implementation — validation deltas vs official tool)

## Decisions

(record during implementation — final tolerance list, fidelity label justification)

## Verification

```bash
pytest tests/unit/test_eth3d*.py
e3r benchmark run preds/ --dataset eth3d --split training --protocol eth3d_training_official_like  # on fixture
```

Acceptance per `.agent/plan.md` milestone. pycolmap is always installed, so no
`pytest.importorskip` guard is needed.

## Status

todo
