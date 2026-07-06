# 018 — Alignment module (SE3/Sim3 with mandatory visualization)

## Goal

A first-class alignment module that estimates SE3/Sim3 transforms between two geometries
that live in **different coordinate frames and scales**, and that **always gives the user
a way to visualize the alignment** — most real-world alignments are quirky, and a residual
number alone does not show a flipped, mirrored, or locally-stuck registration.

Exactly two estimation modes (per explicit user decision — **no FPFH / feature-based
global registration, ever**):

1. **closest-point ICP** on the geometries themselves (Open3D point-to-point ICP,
   centroid + RMS-radius-scale init, `with_scaling` for Sim3);
2. **trajectory-first**: align the predicted camera trajectory onto the GT trajectory
   (evo timestamp association + Umeyama SE3/Sim3, already owned by the task-015 backend),
   then propagate that transform to the prediction geometry.

## Scope

- `backends/registration_open3d.py` (currently a stub): `registration` registry kind,
  name `open3d`, per the `.agent/backends.md` interface:
  `icp(source, target, init_transform, max_correspondence_distance, max_iterations,
  with_scaling)` point-to-point ICP; records init transform, max correspondence
  distance, iteration cap, convergence criteria, fitness, inlier RMSE, correspondence
  count. Explicit failures on empty inputs. Registered in `core/registry.py`.
- Alignment semantics (documented in `.agent/schema.md`; no new schema literals needed):

  | Protocol spec | Meaning |
  |---|---|
  | `mode: se3\|sim3`, `solver: icp`, `estimate_on: pointcloud` | closest-point ICP; `with_scaling = (mode == sim3)` |
  | `mode: se3\|sim3`, `solver: umeyama`, `estimate_on: trajectory` | evo-associated trajectory Umeyama, 4×4 applied to the geometry |
  | `mode: icp` | refused — the mode must state the transform class (`se3`/`sim3` + `solver: icp`) so scale handling stays explicit |

- `pipeline/stages/align.py`: dispatch the two new solver paths; `AlignmentResult` gains
  optional fitness / association metadata. ICP `max_correspondence_distance` is required
  (protocol `parameters`), never defaulted silently; trajectory mode requires explicit
  `associate_max_diff`. Sim3-on-metric-scale guard applies to every scale-changing path.
  Existing corresponded-Umeyama path unchanged.
- `pipeline/runner.py` / `pipeline/benchmark.py`: pass the needed backends + optional
  pred/GT trajectory paths into the align stage (benchmark: `ScenePredictionEntry
  .trajectory` / `SceneData.gt_trajectory`; a missing trajectory when trajectory
  alignment is requested is an explicit scene failure at stage `align`). Record
  `registration` / `trajectory` backend versions when used.
- Visualization `reports/alignment_vis.py`: `alignment_before.ply` /
  `alignment_after.ply` overlays (pred + gt merged, two fixed colors), orthographic
  projection PNG (before/after × XY/XZ/YZ, subsampled, residual + scale annotation,
  matplotlib Agg), `alignment_vis.json` manifest recording colors, subsample seed/count,
  and the transform shown.
- Standalone entry points that **always** write the visualization artifacts:
  - CLI `e3r align` (new `cli/align.py`, registered in `cli/main.py`) with `--mode`,
    `--solver icp|trajectory`, `--pred-trajectory/--gt-trajectory/--associate-max-diff`,
    `--max-corr-dist D|auto` (auto = 5% of GT bbox diagonal, echoed + recorded),
    `--input/--gt-input mesh|pointcloud`, `--out DIR`; writes `alignment.json`,
    `pred_aligned.ply`, overlays + PNG. Rich echo of resolved config; verbose failures.
  - Python API `align_geometries(...)` in `api.py`, re-exported from `eval3r/__init__.py`.
- Pipeline runs: when a run directory is written and a non-`none` alignment actually
  ran, the overlay artifacts land in `debug/` automatically (debug output, not
  evaluation behavior → no protocol/schema gate, no hash change).
- Tests: known-Sim3 ICP recovery, se3 case, determinism, empty-input and
  missing-parameter failures, trajectory-mode analytic propagation + missing-file
  errors, mode-`icp` refusal, metric-scale Sim3 guard on both paths, visualization
  artifacts + manifest, CLI in-process end-to-end and refusal exits.
- Docs: `.agent/backends.md` registration implementation note, `.agent/schema.md`
  semantics table, `docs/alignment.md` + mkdocs nav.

## Out of Scope

- FPFH / feature-descriptor / RANSAC global registration (explicitly excluded by user).
- Trajectory alignment for pose metrics (task 015 owns it, via evo).
- Depth scale alignment modes (task 014).
- Interactive 3D viewer (headless artifacts only; the PLYs open in MeshLab/CloudCompare).
- Colored-ICP, multi-scale ICP schedules, point-to-plane estimation.
- Changing any built-in protocol's alignment behavior.

## Relevant Files

- `.agent/backends.md` — "Registration backend"
- `.agent/schema.md` — "Alignment schema"
- `.agent/plan.md` — "Alignment policy"
- `eval3r/backends/registration_open3d.py`, `eval3r/backends/trajectory_evo.py`,
  `eval3r/pipeline/stages/align.py`, `eval3r/pipeline/runner.py`,
  `eval3r/pipeline/benchmark.py`, `eval3r/reports/alignment_vis.py` (new),
  `eval3r/cli/align.py` (new), `eval3r/api.py`, `eval3r/__init__.py`

## Plan

1. Registration backend (point-to-point ICP with full parameter capture) + registry.
2. Align-stage dispatch: ICP path (centroid/RMS init) and trajectory path (evo).
3. Visualization module.
4. `e3r align` CLI + `align_geometries` API (always visualize).
5. Runner/benchmark hookup incl. debug artifacts.
6. Tests, docs, verification.

## Findings

- **Open3D point-to-point ICP recovers analytic Sim3/SE3 exactly** (scale to 1e-6,
  points to 1e-9) from the centroid + RMS-radius init on anisotropic clouds — but a
  *near-uniform cube* rotated 20° converged to a locally-stuck optimum (scale 0.63
  instead of 0.70) with fitness 1.0 and a small residual. This is precisely the
  quirk class the mandatory visualization exists for, and it shaped the test
  fixtures: recovery tests use anisotropic (4 x 2 x 0.5) clouds so the optimum is
  well-determined; the symmetric-cloud failure mode is documented in the fixture
  docstring.
- Open3D ICP has no randomness, but its multithreaded reductions are only
  reproducible to float summation order (~2e-15 matrix differences between
  identical runs); the determinism test asserts allclose(atol=1e-12), not
  bit-equality, and the backend docstring says so.
- evo's `align` (used by `align_trajectories`) returns (rotation, translation,
  scale) in the x_gt ≈ s·R·x_pred + t convention; propagating that 4x4 to the
  prediction geometry reproduces an analytically constructed SE3 to 1e-9 (1.5e-7
  through a f4 PLY round-trip).
- `capture_alignment_vis` transforms the *same* subsample for before/after, so
  the two overlays are point-for-point comparable and the capture cost is one
  seeded subsample per side (default cap 100k points/side; benchmark memory stays
  bounded).

## Decisions

- Two solvers only; quirky alignments are handled by mandatory visualization, not by
  feature-based global registration (user decision, recorded in agent memory).
- Mode `icp` is refused rather than aliased so the transform class (rigid vs similarity)
  is always explicit in the protocol.
- ICP coarse init estimates **no rotation** (centroid translation + RMS-radius scale
  for Sim3 only): rotation without correspondences would need features, which are
  excluded; ICP owns the rotation, and an init too far off is visible in the
  before-overlay and in the explicit zero-correspondence error.
- `max_correspondence_distance` (ICP) and `associate_max_diff` (trajectory) are
  never defaulted anywhere; `e3r align --max-corr-dist auto` resolves to 5% of the
  GT bbox diagonal and records the rule + resolved value in the alignment record.
- `align_geometries` passes `metric_scale=False` (Sim3 allowed): the standalone
  tool exists exactly to bridge frames/scales and applies no metrics; protocol
  runs keep the Sim3-on-metric-scale guard on every scale-changing path.
- Pipeline debug overlays (`debug/<scene>_alignment_{before,after}.ply` +
  projections PNG + `alignment_vis.json`) are written whenever a non-`none`
  alignment ran and a run directory is written — debug output, not evaluation
  behavior, so no protocol/schema gate and no hash change.
- `registration` / `trajectory` backend versions are recorded in
  `backend_versions` (runner + benchmark) only when the protocol's alignment
  actually uses them.
- Benchmark trajectory sources: prediction trajectory from the manifest entry
  (`ScenePredictionEntry.trajectory`, resolved relative to pred_root), GT
  trajectory from `SceneData.gt_trajectory`; a missing one is an explicit scene
  failure at stage `align`.
- CI got headless-GL system libs (libegl1/libgl1/mesa) plus an explicit
  "diagnose headless OpenGL" step that builds a pyrender `OffscreenRenderer`, so
  an EGL regression surfaces as a named step failure instead of opaque
  visibility-test errors.

## Verification

```bash
ruff check .   # All checks passed!
mypy eval3r    # Success: no issues found in 96 source files
EVAL3R_ETH3D_TOOL=~/xingrui_ws/tools/multi-view-evaluation/build/ETH3DMultiViewEvaluation \
  pytest -q    # 444 passed, 3 skipped (the 3 = TnT real-toolbox tests only)
mkdocs build   # OK (docs/alignment.md in nav)
# analytic smoke, API: gt = 0.7*R(20°)@pred + t → align_geometries(mode=sim3,
#   solver=icp, max_corr_dist="auto") recovers scale 0.6994, fitness 1.0, and writes
#   all 6 artifacts; trajectory solver recovers R/t to 1e-9 and propagates to the
#   cloud (1.5e-7 through f4 PLY).
# CLI in-process: e3r align --mode sim3 --max-corr-dist auto → exit 0, config panel
#   echoes the auto-resolved distance; missing --max-corr-dist → exit 1 with the
#   full reason; --mode icp → exit 2.
# projection PNG inspected: before-rows show the misaligned red/blue overlays,
#   after-rows interleave; annotation carries mode/solver/scale/residual/fitness.
```

New tests: `tests/unit/test_registration_open3d.py` (12),
`tests/unit/test_align_stage.py` (14), `tests/unit/test_alignment_vis.py` (3),
`tests/integration/test_align_cli.py` (6, incl. the automatic `debug/` overlays +
recorded registration backend version in a protocol-driven ICP evaluation run).

## Status

done
