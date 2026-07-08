# 023 — Trajectory alignment debug visualization

## Goal

Write trajectory-alignment debug artifacts whenever trajectory alignment is actually
estimated, for both pose evaluation and geometry evaluation with
`estimate_on: trajectory`.

## Scope

- Capture associated prediction positions before alignment, after alignment, and GT
  positions from the same evo association/alignment pass used for metrics.
- Write trajectory debug artifacts under `debug/scenes/<scene_id>/`.
- Keep run-level manifests for discoverability and compatibility.
- Move newly written geometry alignment and distance debug artifacts to per-scene
  debug directories while retaining top-level manifest files.
- Add focused tests for writers and runner/API/CLI integration.

## Out of Scope

- No protocol fields or protocol hash changes.
- No scoring changes.
- No replacement or reimplementation of evo trajectory metrics.
- No changes to official-toolbox evaluation paths.

## Relevant Files

- `eval3r/backends/trajectory_evo.py`
- `eval3r/pipeline/pose_runner.py`
- `eval3r/pipeline/runner.py`
- `eval3r/pipeline/stages/align.py`
- `eval3r/reports/alignment_vis.py`
- `eval3r/reports/plots.py`
- `eval3r/api.py`
- `eval3r/cli/metric.py`
- `eval3r/cli/benchmark.py`
- `tests/unit/test_alignment_vis.py`
- `tests/unit/test_align_stage.py`
- `tests/unit/test_pose_metrics.py`
- `tests/integration/test_single_pose_metric.py`
- `tests/integration/test_align_cli.py`

## Plan

1. Extend evo backend output with trajectory visualization data from the existing
   association/alignment pass.
2. Add trajectory visualization capture/writer functions and run-level indexes.
3. Thread captures through pose and geometry trajectory-first runners.
4. Update debug artifact layout to `debug/scenes/<scene_id>/...` with top-level
   manifests.
5. Add/update tests and documentation, then run targeted tests and `mkdocs build`.

## Findings

- The evo backend already performed the single association/alignment pass needed for
  metrics. The implementation now captures associated prediction positions before
  alignment, the aligned prediction positions, and associated GT positions from that
  same pass.
- Pose API/CLI paths previously wrote only core run-directory files when called with
  `return_run=True` / direct CLI writing; trajectory debug artifacts now use the same
  report writer path as geometry and benchmark runs.
- Existing geometry debug artifacts were written directly under `debug/`; they now
  land under `debug/scenes/<scene_id>/` with top-level manifests preserved.
- Full-suite verification is blocked by the environment missing `pycolmap`; the
  failures are in pre-existing camera/ETH3D tests, not in trajectory-debug paths.

## Decisions

- No protocol fields were added and no protocol hash semantics changed.
- `TrajectoryAlignmentVisData` carries arrays separately from normal result metadata,
  so `results.json` and `alignment_transforms.json` stay compact and JSON-schema stable.
- Top-level manifests remain the stable entry points:
  `debug/debug_outputs.json`, `debug/alignment_vis.json`,
  `debug/trajectory_alignment_vis.json`, and `debug/debug_index.json`.
- Heavy per-scene artifacts use the large-run layout
  `debug/scenes/<scene_id>/...`.
- `mode: none` pose runs still record identity alignment metadata as before but do not
  write trajectory visualization artifacts.

## Verification

- `python -m compileall eval3r` — OK.
- `PYTHONPATH=. pytest tests/unit/test_alignment_vis.py tests/unit/test_reports.py tests/unit/test_align_stage.py tests/unit/test_pose_metrics.py tests/integration/test_single_pose_metric.py tests/integration/test_align_cli.py tests/integration/test_reports_diff.py tests/integration/test_single_depth_metric.py -q` — OK, 93 passed, 1 Matplotlib Axes3D warning.
- `ruff check .` — OK.
- `mkdocs build` — OK.
- `PYTHONPATH=. pytest -q` — 533 passed, 6 skipped, 8 failed because `pycolmap`
  is not installed in this environment (`tests/unit/test_camera_pycolmap.py` and
  `tests/unit/test_eth3d_adapter.py`).

## Status

done
