# Task 021 — Pose / coordinate convention transforms as a first-class feature

## Goal

Make pose/coordinate convention transformation a first-class, validated step. A method
may emit poses in OpenGL axes, world-to-camera (`Tcw`) instead of camera-to-world
(`Twc`), or geometry in an OpenGL world frame; eval3r must transform such predictions to
its internal convention (camera-to-world, OpenCV axes, meters) **before** metrics, prove
each transform's truthfulness, and pick the declared conventions up automatically from
the prediction writer/reader, CLI, and API.

## Scope

- A transformer class converting poses **and** geometry between all axis/direction/world
  modes, validating every result (`eval3r/core/pose_convention.py`).
- A single `SourcePoseFormat -> PoseConvention` mapping table
  (`eval3r/datasets/conventions.py`), raising for unverified formats.
- New prediction-side `world_frame` field; reuse of the existing `source_pose_format` /
  `normalized_convention` fields to drive transforms.
- Pipeline insertion: a "convert" stage in the pose runner; `normalize_world_frame` in
  the geometry normalize stage (single-file + benchmark + visibility-culled paths);
  culling-pose normalization in the ScanNet adapter.
- CLI flags (`--pred-pose-convention` / `--gt-pose-convention` on `metric pose`,
  `--pred-world-frame` on `metric geometry`) and API params with manifest auto-pickup.
- `PredictionWriter` writes `world_frame` (and `normalized_convention`); reader needs no
  change (full manifest validation already covers the new field).

## Out of Scope

- Expanding `NormalizedConvention` or adding a protocol-required-target field (the target
  is the fixed internal frame → no protocol-hash churn).
- Verified base-convention mappings for CO3D / Tanks and Temples `.log` (raise for now).
- Any change to SE3/Sim3 alignment semantics — convention is a separate upstream step.

## Relevant Files

- `eval3r/core/pose_convention.py` — `PoseConvention`, `PoseConventionTransform`,
  matrix/TUM/world math, `read_tum_rows`, validators (filled from a stub).
- `eval3r/datasets/conventions.py` — `convention_for`, `normalized_convention_target`,
  the mapping table (filled from a stub).
- `eval3r/core/errors.py` — new `PoseConventionError`.
- `eval3r/core/types.py` — new `WorldAxes` alias.
- `eval3r/core/schema.py`, `eval3r/core/manifest.py` — `world_frame` field.
- `eval3r/core/result.py` — `"convert"` added to `SceneFailure.stage`.
- `eval3r/pipeline/pose_runner.py` — "convert" stage + `_normalize_trajectory_convention`.
- `eval3r/pipeline/stages/normalize.py` — `normalize_world_frame`.
- `eval3r/pipeline/runner.py`, `eval3r/pipeline/benchmark.py` — geometry insertion.
- `eval3r/datasets/scannet.py` — culling-pose normalization.
- `eval3r/cli/metric.py`, `eval3r/api.py`, `eval3r/predictions/writer.py` — flags,
  echo, params, writer kwarg.
- `.agent/schema.md`, `docs/quickstart.md` — docs.
- `tests/unit/test_pose_convention.py`, `tests/unit/test_pose_convention_runner.py`,
  additions to `tests/unit/test_prediction_writer.py`.

## Plan

Route every pose conversion through the canonical internal representation (c2w, OpenCV)
so each hop is one validated primitive. The OpenCV↔OpenGL flip is the single involution
`F = diag(1, -1, -1, 1)` (proper 180° about X); direction change is matrix inverse; the
same `F` serves camera-axis (right-multiply on c2w) and world-frame (global left-multiply
on points). Target is always the fixed internal frame, so no protocol field / no hash
churn. Validation always on: finiteness, homogeneous row, orthonormality + `det=+1`,
axis-only center preservation, direction inverse-consistency, and round-trip.

## Findings

- **ATE-translation is blind to a camera-axis (OpenCV↔OpenGL) flip on a c2w trajectory**:
  right-multiplying a c2w pose by `F` rotates the camera axes but leaves the translation
  column (camera centre) unchanged, so positions — and hence ATE — are identical. The
  axis mismatch only shows up in orientation-dependent metrics (RPE-rotation) or in any
  geometry built from the poses. A **direction** mismatch (`Tcw` vs `Twc`) does move
  centres and is caught by ATE. Tests use RPE-rotation for the axis case and ATE for the
  direction case accordingly — this is a real property, not a test artifact.
- Adding `"convert"` to `SceneFailure.stage` (a serialized Literal in `core/result.py`)
  was required so an unmapped-format failure validates; `.agent/schema.md` updated to
  match. Not protocol-hash-affecting (results are not hashed).
- ScanNet culling-pose normalization is a validated no-op today (its export is already
  `cam_to_world_opencv`); it exercises the `(N,4,4)` matrix path in production and makes
  the `CameraTrajectory` (c2w OpenCV) contract correct for any future OpenGL/w2c adapter.
- `world_frame` placed on the prediction side only (`Reconstruction`,
  `PredictionManifest`), not on `GroundTruthSpec`/`EvalProtocol` — confirmed builtin
  protocol hashes are unchanged (`test_hashing.py` still passes with its pinned hashes).

## Decisions

1. Geometry is in scope: a declared world-frame mismatch applies a deterministic global
   axis transform to vertices before geometry metrics (not left to alignment).
2. Canonicalize to the fixed internal frame; declare both sides. Reuse existing
   serialized fields; add a hash-neutral prediction-only `world_frame`.
3. Raise for unverified source formats (`unknown`, `co3d_frame_annotations`,
   `tanks_temples_log`) rather than guess a handedness.

## Verification

Commands run (this environment):

```text
ruff check eval3r                                   -> clean
pytest tests/unit/test_pose_convention.py -q        -> 42 passed
pytest tests/unit/test_pose_convention_runner.py -q -> 5 passed
pytest tests/unit/test_prediction_writer.py -q      -> passed (incl. 2 new)
pytest tests/unit/test_hashing.py test_schema.py \
       test_prediction_reader.py test_pose_metrics.py \
       test_scannet_adapter.py -q                   -> 88 passed (hashes unchanged)
pytest tests/ -q                                    -> 8 pre-existing failures, all
    pycolmap ModuleNotFoundError (pycolmap absent in this env; unrelated files), rest pass
```

End-to-end (through the in-process API, not the PATH-shadowed `e3r`): a GT TUM + a
same-trajectory pred in OpenGL c2w with `pred_pose_format="cam_to_world_opengl",
align="none"` drives RPE-rotation from ~112° to ~0; a `world_to_cam_opencv` pred drives
ATE from ~5.6 m to ~0. `result.metadata["convention"]` records
`pred_transformed`/`target`. A geometry pred flipped into the OpenGL world frame scores
chamfer ~0 with `pred_world_frame="opengl"` and large without.

### Not verified / limitations

- No real-dataset OpenGL/w2c adapter exists yet to exercise a non-no-op culling-pose
  conversion end to end; validated on synthetic `(N,4,4)` stacks and the ScanNet no-op.
- CO3D / Tanks and Temples `.log` handedness remains unverified and intentionally raises.

## Status

done
