# 013 — ETH3D adapter

## Goal

ETH3D training-split point-cloud evaluation runs locally under
`eth3d_training_official`, with COLMAP text camera parsing via pycolmap, honest
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
- `.agent/protocols.md` — `eth3d_training_official` (including the tolerance note)
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

- **The official scoring cannot be ported, only wrapped.** Reading the official
  `ETH3D/multi-view-evaluation` source (completeness.cc / accuracy.cc): completeness and
  accuracy are **voxel-normalized** — per-point results are averaged within voxel cells
  (default 0.01 m) over **two shifted voxel grids**, then averaged over cells — and
  accuracy classifies each reconstruction point as accurate / inaccurate / **unobserved**
  using beam-based free-space modeling from the laser-scan positions (beam start radius
  0.001125 m, divergence half-angle 0.011°); unobserved points and unobserved-only cells
  are excluded per tolerance. Plain pred↔gt distance fractions cannot reproduce this, so
  a "validated_official_port" via eval3r's generic metric path was rejected; the honest
  implementation is a subprocess wrapper (`official_script_wrapper`), like Tanks and
  Temples. The plan's regression-comparison of eval3r-native metrics vs the tool is
  therefore moot: the wrapper *is* the official computation.
- **Official-tool build break (mechanical, patched).** At commit `0daa4f4` the build
  fails against PCL 1.12 headers because a stale `add_definitions(-std=c++11)` in
  CMakeLists.txt wins the compile line, although upstream's own last commit ("Use the
  C++17 standard", `CMAKE_CXX_STANDARD 17`) intends C++17. Changed that one flag to
  `-std=c++17` in the local checkout (`~/xingrui_ws/tools/multi-view-evaluation`,
  build at `build/ETH3DMultiViewEvaluation`) — build-flags only, no scoring source
  touched, matching upstream intent; recorded here and in `.agent/backends.md`.
- **Analytic fixture reproduced exactly by the real binary.** `tests/fixtures/eth3d_tiny`
  (single scan, identity MLP; 2 exact prediction points, 2 points 3 cm in front of their
  scan points along the scan rays, 1 point 1 m behind a scan point = unobserved) has
  hand-derived scores 0.5/0.5/1/1/1/1 at tolerances 0.01–0.5 for all of
  accuracy/completeness/F1 — the real binary outputs exactly these values, confirming
  the beam/voxel/unobserved semantics above.
- **No real ETH3D dataset is present locally** (nothing under /mnt); validation is
  fixture-based against the real binary (which the task scoped). The official per-scene
  downloads are .7z and no 7z extractor is installed, so no real scene was pulled.
- pycolmap 4.0.4 (conda env `dl`) reads COLMAP text models; `image.cam_from_world` is a
  **method** in this version (property in older ones) — the backend handles both.
  Convention check pinned against a hand-computed pose (q=(1/√2,0,1/√2,0), t=(1,2,3) →
  C=(3,−2,−1)).

## Decisions

- `backends/eth3d_official.py` `Eth3dOfficialEval` (registry kind `official_eval`, name
  `eth3d_official`, `input_mode="scan_mlp"`, method `official_script_wrapper`): resolves
  the binary from `tool_path` / `EVAL3R_ETH3D_TOOL` / `ETH3D_MULTI_VIEW_EVALUATION`
  (user-supplied external build; explicit error naming the repo when absent), runs it
  once per scene with the full tolerance list, parses the four summary lines
  (misaligned/missing output → MetricError), records command, tool path, tool source
  commit, and the official-default voxel/beam parameters.
- `backends/camera_pycolmap.py` `PycolmapCameraBackend` (kind `camera`): pycolmap-parsed
  COLMAP models, `world_to_cam_colmap` → cam-to-world OpenCV recorded on the returned
  `ColmapCameraSet`; `pinhole_intrinsics` raises explicitly for any non-
  PINHOLE/SIMPLE_PINHOLE model (never a silent approximation); non-pinhole presence adds
  a limitation note the adapter copies into scene metadata.
- `datasets/eth3d.py` `Eth3dAdapter`: high-res DSLR multi-view benchmark only (13
  training / 12 test scenes pinned; low-res multi-camera is a different benchmark, out of
  scope). `official_artifacts` parses `scan_alignment.mlp` (same MLMesh/filename fields
  the official tool reads) and validates every referenced scan PLY; `gt_fingerprint`
  jointly hashes .mlp + scans. GT: laser_scan / independent / dense_surface, metres.
  `local_evaluation("test")` → server_only (refused in preflight).
- Protocol is `eth3d_training_official` with fidelity **`official`** (the real official tool
  produces the numbers, exactly like `tanks_temples_training_official`), version 0.2.0.
  Metric list extended to the full official tolerance set as 18 explicit entries
  (`accuracy_1cm` … `fscore_50cm`, thresholds in metres) since aggregation keys by metric
  name. Alignment stays `none` (the tool applies none; predictions must be in the GT
  COLMAP frame). masking.pred_culling documents the tool's own observability exclusion
  (`visibility_mask`/`gt`); eval3r applies no additional masking. New pinned hash
  `sha256:67a7f8a0…` in test_protocols.py; docs updated in the same change.
- Benchmark loop: official dispatch generalized from a boolean to `input_mode`
  (`point_arrays` DTU / `artifacts` TnT / `scan_mlp` ETH3D) with a new
  `_evaluate_scene_eth3d_official` branch that maps each metric spec to the official
  output column by name-prefix + threshold.
- No fake evaluator anywhere: wrapper/benchmark tests drive the real binary and skip
  cleanly (with the env-var hint) when `EVAL3R_ETH3D_TOOL` is unset. `.env.example`
  documents the variable.

## Verification

```bash
ruff check .   # All checks passed!
mypy eval3r    # Success: no issues found in 90 source files
EVAL3R_ETH3D_TOOL=~/xingrui_ws/tools/multi-view-evaluation/build/ETH3DMultiViewEvaluation \
  pytest -q    # 286 passed, 3 skipped (the 3 = TnT real-data tests; all ETH3D tests ran)
pytest tests/unit/test_eth3d_official.py tests/integration/test_eth3d_benchmark.py -q -rs
               # without the env var: 10 passed, 3 skipped with explicit reasons
mkdocs build   # OK
# real official binary, direct: fixture → Completenesses/Accuracies/F1 = 0.5 0.5 1 1 1 1
# real official binary, via CLI (in-process python -m eval3r.cli.main):
#   benchmark run tests/fixtures/eth3d_tiny/preds --dataset eth3d --split training \
#     --protocol eth3d_training_official --root tests/fixtures/eth3d_tiny/dataset_root
#   → all 18 metrics match the analytic values; results.json records
#     evaluator_method=official_script_wrapper, tool_path, tool_commit=0daa4f4…,
#     voxel_size=0.01, beam params, fidelity=official, gt provenance=laser_scan
```

Acceptance per `.agent/plan.md` milestone met (protocol name updated there in the same
change). pycolmap is always installed; no `pytest.importorskip` used.

## Status

done
