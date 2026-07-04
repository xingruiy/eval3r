# 009 — DTU adapter

## Goal

DTU scenes can be discovered and normalized locally through the generic benchmark-run
plumbing, with laser-scan GT point clouds, ObsMask/Plane metadata, millimeter handling, and
honest capability/provenance reporting.

## Scope

- `datasets/dtu.py` implementing the `DatasetAdapter` interface from `.agent/datasets.md`:
  scan ID resolution, `stlXXX_total.ply` GT loading, `ObsMaskXXX_10.mat` and `PlaneXXX.mat`
  resolution, native mm -> internal meters (source unit recorded), prediction filename
  resolution honoring the light-condition suffix (`<method>XXX_l3.ply`; do not ignore it).
- Missing Plane files recorded explicitly in scene metadata; protocol failure policy decides
  what happens (never silently skip while claiming official-like fidelity).
- Capabilities: dense_geometry, independent_gt, official_local_eval initially false until
  task 010 validates the official-like evaluator; local evaluation status can still be
  `supported` for eval3r-native point-cloud runs.
- GroundTruthSpec: pointcloud / laser_scan / independent / dense_surface; GT fingerprinting
  for point cloud + ObsMask/Plane where practical.
- Tiny synthetic DTU-layout fixture (`tests/fixtures/dtu_tiny/`) with a handful of points
  and miniature ObsMask/Plane `.mat` files.
- Integration with `e3r benchmark run --dataset dtu` using the generic point-cloud geometry
  runner, but not yet claiming official-like fidelity.

## Out of Scope

- Validated Python port or MATLAB wrapper for official DTU evaluation (task 010).
- Any other dataset adapter.
- Camera/pose loading for DTU (not needed for point-cloud geometry evaluation).

## Relevant Files

- `.agent/datasets.md` — "DTU" adapter requirements and rules, adapter interface
- `.agent/protocols.md` — DTU protocol template, with fidelity finalized in task 010
- `.agent/plan.md` — "Dataset adapter for DTU" milestone
- `CLAUDE.md` — DTU cautions

## Plan

1. Implement adapter discovery, GT/mask/plane resolution, unit handling, and prediction
   resolution with light-suffix parsing.
2. Add fixture data and tests for scan discovery, mm->m normalization, metadata recording,
   missing-Plane behavior, capability declaration, and GT fingerprinting.
3. Wire DTU through benchmark-run plumbing for an eval3r-native point-cloud smoke test.

## Findings

- DTU GT is millimetres, so unit normalization had to become real. Added a `normalize`
  pipeline stage (`unit_to_meters` + `LoadedGeometry.transformed(scale_matrix)`) and threaded
  `pred_unit` / `gt_unit` through `evaluate_geometry_scene`; the benchmark loop reads them from
  `Reconstruction.unit` / `GroundTruthSpec.unit`. Verified end-to-end: a 50 mm prediction offset
  reports `accuracy = 0.05 m`. Single-file / custom paths pass `None`/`"m"` → no-op.
- The benchmark always builds a manifest, but an *inferred* one is just a naive `<scene>.ply`
  fallback and was clobbering DTU's `<method>NNN_l3.ply` resolution. Fix: pass the manifest to
  `resolve_prediction` only when it was *declared* (`resolve_manifest = None if inferred`); the
  inferred manifest is still written to the run dir. Adapters then use their own inference.
- DTU prediction filenames encode the light condition (`mvsnet001_l3.ply`); a regex parses
  method/scan/light and the resolver never drops the `_l3` suffix. Missing Plane files are
  recorded (`plane_available`, `plane_path`), never silently ignored — required so task 010
  cannot claim official-like fidelity for a scan without its Plane file.

## Decisions

- `datasets/dtu.py` `DTUAdapter` over `<root>/Points/stl/stl<NNN>_total.ply`,
  `<root>/ObsMask/ObsMask<N>_10.mat`, `<root>/ObsMask/Plane<N>.mat`, `<root>/splits/<split>.txt`.
  Registered as `dtu`. Native unit `mm`; GT is `pointcloud/laser_scan/independent/dense_surface`.
- `official_local_eval = False` / method `none` for now (honest: task 010 wires the ObsMask/Plane
  evaluator). `local_evaluation.status = supported` because the public laser-scan GT makes
  eval3r-native point-cloud runs locally evaluable — labelled eval3r-native, not official.
- `gt_fingerprint` is a joint hash of GT + ObsMask + Plane (missing files omitted), so a scan
  with a Plane fingerprints differently from one without.
- Fixture root named `dataset_root/` (not `data/`) to stay clear of the `.gitignore data/` rule;
  `.mat` masks written with `scipy.io.savemat`.

## Verification

```bash
ruff check .   # All checks passed!
mypy eval3r    # Success: no issues found in 88 source files
pytest -q      # 190 passed
mkdocs build   # OK
# in-process CLI (PATH `e3r` is a different installed package in this env):
python -c "import sys; sys.argv=['e3r','benchmark','run','tests/fixtures/dtu_tiny/preds',\
  '--dataset','dtu','--split','one','--protocol','single_geometry',\
  '--root','tests/fixtures/dtu_tiny/dataset_root','--out','/tmp/dtu_run']; \
  from eval3r.cli.main import app; app()"   # accuracy 0.05 m (50 mm normalized)
```

Acceptance met: DTU scans, GT, ObsMask/Plane, mm units, and light-suffix prediction filenames
resolve correctly; missing Plane files are recorded; results are eval3r-native and the adapter
does not claim official-like fidelity (deferred to task 010). Covered by
`tests/unit/test_dtu_adapter.py`, `tests/unit/test_normalize.py`, and
`tests/integration/test_dtu_adapter_benchmark.py`.

## Status

done

