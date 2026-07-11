# Task 028: Known-Good Benchmark Validation

## Goal

Run comprehensive benchmark validation against known-good evaluations across every
implemented dataset/protocol path in the current repo, and establish a durable external
reproducibility corpus of datasets, predictions, official/reference outputs, inventories,
and run records that can be reused and extended in future checks.

The validation must prove eval3r agrees with the relevant real evaluator or named
reference result. It must not invent, mock, stub, or vendor fake official evaluators.

## Scope

- DTU `dtu_native_pointcloud` against real DTU data, ObsMask/Plane files, and known
  reference outputs.
- Tanks and Temples `tanks_temples_training_official` against the real official
  toolbox on public training scenes.
- ETH3D `eth3d_training_official` against the real official
  `ETH3DMultiViewEvaluation` binary.
- ScanNet native geometry protocols:
  `scannet_single_layer_geometry_5cm`,
  `scannet_double_layer_geometry_5cm`, and
  `scannet_test_single_layer_geometry_5cm` when GT trajectory and renderer support
  are available.
- Neural-RGBD geometry protocols:
  `neural_rgbd_geometry_culled` and `neural_rgbd_geometry_source`.
- Local sanity checks for `single_geometry`, `single_depth`, and `single_pose` using
  analytic or tiny fixtures already present in the repo.
- Produce a durable missing-assets report in this task file's Findings section.
- Define an external, append-only data/reference organization so new datasets, methods,
  scene subsets, and reference outputs can be added over time without committing large or
  licensed artifacts to this repo.

## Out of Scope

- Adding new dataset adapters or new benchmark protocols.
- Implementing learned reconstruction methods.
- Downloading or committing proprietary, licensed, or large raw benchmark data into this
  repo.
- Local evaluation of server-only splits.
- Replacing official code with a lookalike evaluator.
- Treating paper tables as comparable when the exact scene set, prediction artifact,
  protocol, and units cannot be matched.

## Relevant Files

- `.agent/plan.md`
- `.agent/schema.md`
- `.agent/protocols.md`
- `.agent/datasets.md`
- `.agent/metrics.md`
- `.agent/backends.md`
- `.agent/reproducibility.md`
- `eval3r/protocols/builtin/*.yaml`
- `eval3r/pipeline/benchmark.py`
- `eval3r/backends/dtu_eval.py`
- `eval3r/backends/tnt_official.py`
- `eval3r/backends/eth3d_official.py`
- `eval3r/datasets/dtu.py`
- `eval3r/datasets/tanks_temples.py`
- `eval3r/datasets/eth3d.py`
- `eval3r/datasets/scannet.py`
- `eval3r/datasets/neural_rgbd.py`
- `tests/unit/test_dtu_eval.py`
- `tests/unit/test_tnt_official.py`
- `tests/unit/test_eth3d_official.py`
- `tests/unit/test_benchmark_runner.py`
- `docs/examples/dtu.md`
- `docs/examples/tanks_temples.md`
- `docs/examples/eth3d.md`
- `docs/examples/scannet.md`

## Plan

1. Preflight the current environment.
   - Record whether the real benchmark roots are available for DTU, Tanks and
     Temples, ETH3D, ScanNet, and Neural-RGBD.
   - Record whether official tool env vars are configured:
     `EVAL3R_TNT_TOOLBOX`, `EVAL3R_TNT_PYTHON`, `EVAL3R_ETH3D_TOOL`.
   - Record whether any user-supplied prediction roots and manifests exist for the
     known methods being checked.
   - Record missing data/tooling explicitly in Findings before attempting full runs.

2. Run repo-local sanity checks.
   - Run the normal unit tests for schemas, protocol hashing, dataset adapters,
     official-wrapper parsing, and benchmark plumbing.
   - Run the real official-wrapper tests only when their real external tools are
     configured; otherwise record the exact skip reason as a missing-tool finding.
   - Use tiny fixtures only for adapter/math/control-flow checks, never as evidence
     of official benchmark fidelity.

3. Validate DTU known-good behavior.
   - Required data: DTU root with `Points/stl` or `groundtruth`, `ObsMask`, `Plane`,
     split file, and prediction PLYs such as `<method>024_l3.ply`.
   - Reference source: official MATLAB/DTUeval-python output for the exact scan and
     prediction. The currently documented parity point is scan 24:
     accuracy `0.343`, completeness `0.248`, overall `0.295` mm.
   - Run:
     ```bash
     e3r benchmark run <DTU_PREDS> --dataset dtu --split test \
       --protocol dtu_native_pointcloud --root <DTU_ROOT> \
       --method <METHOD> --out runs/known_good/dtu_<METHOD>
     ```
   - Compare per-scene and aggregate `accuracy`, `completeness`, and `overall` in
     millimetres. Record tolerance and reason, including any reference
     run-to-run randomness.

4. Validate Tanks and Temples official behavior.
   - Required data/tooling: official `isl-org/TanksAndTemples` toolbox evaluation
     directory, pinned toolbox Python environment, public training GT scene
     directories, and prediction PLYs.
   - Run:
     ```bash
     e3r benchmark run <TNT_PREDS> --dataset tanks_temples --split training \
       --protocol tanks_temples_training_official --root <TNT_ROOT> \
       --method <METHOD> --out runs/known_good/tnt_<METHOD>
     ```
   - Compare eval3r parsed precision/recall/F-score and per-scene `distance_tau`
     against the official toolbox output from the same command/prediction.
   - Record toolbox path, toolbox commit, Python executable, command, and any
     mechanical compatibility patch. Stop and escalate if a fix could change scores.

5. Validate ETH3D official behavior.
   - Required data/tooling: built real `ETH3DMultiViewEvaluation` binary, ETH3D
     training scene roots, and prediction PLYs already in ETH3D GT/COLMAP frame.
   - Run:
     ```bash
     e3r benchmark run <ETH3D_PREDS> --dataset eth3d --split training \
       --protocol eth3d_training_official --root <ETH3D_ROOT> \
       --method <METHOD> --out runs/known_good/eth3d_<METHOD>
     ```
   - Compare accuracy/completeness/F1 at 1, 2, 5, 10, 20, and 50 cm against the
     official binary output. The 2 cm F1 is the headline comparison but all
     tolerance columns must be checked.
   - Record binary path, source commit when discoverable, official voxel/beam
     defaults, and exact command.

6. Validate ScanNet native community protocols.
   - Required data: exported ScanNet scenes, split files, GT meshes, and method
     prediction meshes for a named published/reproducible method.
   - Prefer validation against the same reference driver named in the protocol notes
     such as Atlas/NeuralRecon-style `eval_mesh`, using the exact same scene subset,
     layer convention, threshold, sampling count, seed policy, and visibility-culling
     status.
   - Run validation split protocols first:
     ```bash
     e3r benchmark run <SCANNET_PREDS> --dataset scannet --split val \
       --protocol scannet_single_layer_geometry_5cm --root <SCANNET_ROOT> \
       --method <METHOD> --out runs/known_good/scannet_single_<METHOD>
     e3r benchmark run <SCANNET_PREDS> --dataset scannet --split val \
       --protocol scannet_double_layer_geometry_5cm --root <SCANNET_ROOT> \
       --method <METHOD> --out runs/known_good/scannet_double_<METHOD>
     ```
   - Run `scannet_test_single_layer_geometry_5cm` only when finite poses, depth
     intrinsics, pyrender/EGL, Open3D TSDF support, and the reference culling path are
     all available.
   - Record that ScanNet outputs are native community numbers, not official ScanNet
     benchmark numbers.

7. Validate Neural-RGBD native community protocols.
   - Required data: official mesh root with each scene's `gt_mesh.ply`,
     `gt_mesh_culled.ply`, and known prediction meshes, ideally the shipped
     `neural_rgbd.ply`.
   - Reference source: Neural-RGBD reference driver or paper/release table for the
     exact scene set and culled/source variant.
   - Run:
     ```bash
     e3r benchmark run <NRGBD_PREDS> --dataset neural_rgbd --split all \
       --protocol neural_rgbd_geometry_culled --root <NRGBD_ROOT> \
       --method <METHOD> --out runs/known_good/nrgbd_culled_<METHOD>
     e3r benchmark run <NRGBD_PREDS> --dataset neural_rgbd --split all \
       --protocol neural_rgbd_geometry_source --root <NRGBD_ROOT> \
       --method <METHOD> --out runs/known_good/nrgbd_source_<METHOD>
     ```
   - Compare accuracy, completeness, chamfer, precision, recall, and 5 cm F-score.
     Record sampling randomness/tolerance policy.

8. Inspect every produced run directory.
   - Confirm `results.json`, `protocol.yaml`, `config.yaml`, `environment.json`,
     `backend_versions.json`, and report files exist when requested.
   - Confirm scene coverage and failures are visible.
   - Confirm official-toolbox metadata includes tool paths, commands, versions/commits,
     thresholds/defaults, and any compatibility patches.
   - Confirm result metadata records protocol hash, GT fingerprint, masking/culling,
     sampling, confidence, alignment/adaptation, backend versions, command, Python
     version, platform, and git commit when available.

9. Record final comparison table.
   - For each dataset/protocol/method/scene-set, record eval3r result, reference
     result, absolute delta, relative delta where meaningful, tolerance, pass/fail, and
     source of truth.
   - For blocked entries, record the exact missing data, missing code, missing env var,
     or unavailable reference result.

## Reproducibility Corpus Plan

Keep all benchmark data, official tool checkouts, method predictions, reference outputs,
and produced benchmark runs outside the repo. Treat that external tree as a persistent
reproducibility corpus: it is not a source checkout, but it should be organized,
fingerprinted, and append-only enough that future eval3r versions can be checked against
the same artifacts.

A recommended local layout is:

```text
/data/eval3r_known_good/
  README.md
  inventory.yaml
  datasets/
    dtu/
    tanks_temples_training/
    eth3d_training/
    scannetv2_exported/
    neural_rgbd_meshes_official/
  predictions/
    dtu/<method>/<version_or_commit>/
    tanks_temples/<method>/<version_or_commit>/
    eth3d/<method>/<version_or_commit>/
    scannet/<method>/<version_or_commit>/
    neural_rgbd/<method>/<version_or_commit>/
  references/
    dtu/<method>/<version_or_commit>/
    tanks_temples/<method>/<version_or_commit>/
    eth3d/<method>/<version_or_commit>/
    scannet/<method>/<version_or_commit>/
    neural_rgbd/<method>/<version_or_commit>/
  tools/
    TanksAndTemples/
    multi-view-evaluation/
  runs/
    <eval3r_commit>/<dataset>/<protocol>/<method>/<version_or_commit>/
```

Set these environment variables for reproducible commands:

```bash
export EVAL3R_KNOWN_GOOD_ROOT=/data/eval3r_known_good
export EVAL3R_DTU_ROOT=$EVAL3R_KNOWN_GOOD_ROOT/datasets/dtu
export EVAL3R_TNT_ROOT=$EVAL3R_KNOWN_GOOD_ROOT/datasets/tanks_temples_training
export EVAL3R_ETH3D_ROOT=$EVAL3R_KNOWN_GOOD_ROOT/datasets/eth3d_training
export EVAL3R_SCANNET_ROOT=$EVAL3R_KNOWN_GOOD_ROOT/datasets/scannetv2_exported
export EVAL3R_NRGBD_ROOT=$EVAL3R_KNOWN_GOOD_ROOT/datasets/neural_rgbd_meshes_official
export EVAL3R_TNT_TOOLBOX=$EVAL3R_KNOWN_GOOD_ROOT/tools/TanksAndTemples/python_toolbox/evaluation
export EVAL3R_TNT_PYTHON=/path/to/tnt_toolbox_python
export EVAL3R_ETH3D_TOOL=$EVAL3R_KNOWN_GOOD_ROOT/tools/multi-view-evaluation/build/ETH3DMultiViewEvaluation
```

Maintain the corpus inventory at:

```text
$EVAL3R_KNOWN_GOOD_ROOT/inventory.yaml
```

The inventory is the source of truth for what the external corpus contains. It should
list every dataset, scene subset, prediction root, reference source, expected metric
keys, comparison tolerance, and artifact fingerprint policy. Minimal shape:

```yaml
schema_version: 1
corpus_root: ${EVAL3R_KNOWN_GOOD_ROOT}
policy:
  storage: external_only
  append_only_artifacts: true
  no_large_files_in_eval3r_repo: true
datasets:
  dtu:
    root: ${EVAL3R_DTU_ROOT}
    protocol: dtu_native_pointcloud
    split: test
    scenes: [scan24]
    dataset_fingerprint:
      policy: "file-size + mtime + sha256 for small metadata, full sha256 for selected files"
    methods:
      mvsnet:
        version: paper_or_commit_id
        predictions: ${EVAL3R_KNOWN_GOOD_ROOT}/predictions/dtu/mvsnet/paper_or_commit_id
        reference: ${EVAL3R_KNOWN_GOOD_ROOT}/references/dtu/mvsnet/paper_or_commit_id/scan24_reference.yaml
        source: "official MATLAB or DTUeval-python output for the same prediction"
        tolerance:
          accuracy_mm: 0.001
          completeness_mm: 0.001
          overall_mm: 0.001
        status: prepared
```

### Corpus Maintenance Rules

- Do not commit datasets, predictions, reference logs, generated run directories, or
  official tool checkouts to this repo.
- Keep this task file and future docs as pointers and policy only; the external
  `inventory.yaml` and artifact fingerprints are the reproducibility corpus index.
- Add new methods or dataset/protocol pairs by appending new inventory entries; do not
  overwrite old prediction/reference directories when a method version, eval3r version,
  protocol version, or official-tool version changes.
- Store raw official/reference logs next to normalized YAML summaries. The raw log is the
  evidence; the YAML is for automated comparison.
- Record license/access notes for each dataset and prediction artifact so future agents
  know what cannot be redistributed.
- Prefer stable identifiers: dataset release, scene list, method paper name, method git
  commit or release tag, prediction export date, official-tool commit, eval3r commit, and
  protocol hash.
- When adding a new dataset or protocol later, first update the inventory schema entry and
  task notes with required files, official/reference source, metric keys, units, and
  comparison tolerance before running eval3r.
- Treat missing or changed fingerprints as a reproducibility event. Do not silently reuse
  old reference numbers against changed data or predictions.

### DTU Data

Required dataset files:

- `Points/stl/stl<NNN>_total.ply` or `groundtruth/stl<NNN>_total.ply`
- `ObsMask/ObsMask<N>_10.mat`
- `ObsMask/Plane<N>.mat`
- `splits/test.txt`

Required predictions:

- Point clouds in DTU millimetre model frame.
- Conventional names such as `<method>024_l3.ply`, preserving the light-condition
  suffix.
- Prefer at least scan 24 first because docs record a known parity point:
  `accuracy=0.343`, `completeness=0.248`, `overall=0.295` mm.

Required reference output:

- Official MATLAB output or a real DTUeval-python/reference output on the exact same
  prediction file.
- Store raw logs plus a normalized YAML:

```yaml
scene: scan24
method: mvsnet
protocol: dtu_native_pointcloud
source: "DTU official MATLAB script, commit/path/date"
metrics:
  accuracy: 0.343
  completeness: 0.248
  overall: 0.295
unit: mm
```

### Tanks and Temples Data

Required official code:

- Real `isl-org/TanksAndTemples` checkout.
- `python_toolbox/evaluation/run.py` present.
- Dedicated Python environment compatible with the toolbox's pinned `open3d==0.9`.

Required dataset files per public training scene:

- `<Scene>/<Scene>.ply`
- `<Scene>/<Scene>.json`
- `<Scene>/<Scene>_trans.txt`
- `<Scene>/<Scene>_COLMAP_SfM.log`
- `<Scene>/<Scene>_mapping_reference.txt`

Required predictions:

- One point cloud per scene, named `<Scene>.ply`, or an explicit eval3r manifest.
- Start with `Barn` if preparing a small smoke subset, then expand to all training
  scenes: `Barn`, `Caterpillar`, `Church`, `Courthouse`, `Ignatius`, `Meetingroom`,
  `Truck`.

Required reference output:

- Raw official toolbox stdout/stderr for each scene.
- Normalized YAML per scene with `precision`, `recall`, `fscore`, and `distance_tau`.
- Source must be the same official toolbox invocation or a preserved official log for
  the identical prediction.

### ETH3D Data

Required official code:

- Real `ETH3D/multi-view-evaluation` checkout.
- Built `ETH3DMultiViewEvaluation` binary.

Required dataset files per training scene:

- `<scene>/dslr_scan_eval/scan_alignment.mlp`
- All scan PLYs referenced by `scan_alignment.mlp`
- `<scene>/dslr_calibration_undistorted/cameras.txt`
- `<scene>/dslr_calibration_undistorted/images.txt`
- `<scene>/dslr_calibration_undistorted/points3D.txt`

Required predictions:

- One point cloud per scene, named `<scene>.ply`, already in ETH3D GT/COLMAP frame in
  metres.
- Start with `courtyard` for a small real-data smoke subset, then expand to all public
  training scenes present under the root.

Required reference output:

- Raw official binary stdout/stderr for each scene.
- Normalized YAML with accuracy/completeness/F1 at 1, 2, 5, 10, 20, and 50 cm.
- Record binary path, source commit if discoverable, and command line.

### ScanNet Data

Required dataset files:

- Exported ScanNet v2 scene directories under `scans/<scene>/`.
- `<scene>_vh_clean_2.ply`
- `pose/*.txt`, `intrinsic_depth.txt`, depth metadata, and depth frames for the
  test visibility-culling protocol.
- Split file at `<root>/val.txt`, `<root>/test.txt`, or under `<root>/splits/`.

Required predictions:

- Meshes in ScanNet world frame in metres.
- Eval3r-native manifest preferred, especially when scene IDs or filenames do not
  match `<scene>.ply`.

Required reference output:

- Named reference driver output for the exact same predictions, protocol variant,
  scene subset, sampling count, threshold, and culling status.
- Acceptable references are logs from the community driver named in the protocol notes
  or a paper/release table only when the exact artifacts and scene subset are known.
- Record explicitly that these are native community comparisons, not official ScanNet
  benchmark results.

### Neural-RGBD Data

Required dataset files:

- Official mesh root with one directory per scene.
- `gt_mesh.ply`
- `gt_mesh_culled.ply`
- Optional `gt_trajectory_tum.txt` if later trajectory-first checks are added.

Required predictions:

- Mesh predictions in the dataset world frame in metres.
- Prefer the shipped `neural_rgbd.ply` as the first method if available, because it is
  a known release artifact.

Required reference output:

- Reference driver output or paper/release table for the same scene set and mesh
  variant.
- Separate reference files for culled and source protocols because completeness/recall
  are not comparable across variants.

### Prediction Manifest Preparation

For each method/dataset pair, create or validate an eval3r-native `manifest.yaml` unless
the adapter's conventional filename resolver is intentionally being tested. Use:

```bash
e3r prediction validate <PRED_ROOT>
e3r prediction show <PRED_ROOT>
```

Each manifest must record:

- `method` and `version`
- dataset, variant, and split
- prediction modality
- coordinate frame and source pose format
- unit and scale type
- `uses_gt` provenance
- confidence policy if the method self-filtered
- per-scene relative paths
- fingerprints when available

### Preflight Acceptance Criteria

Before running full comparisons, the preparation step is complete only if:

- Every planned dataset root exists and contains the required files for the selected
  scene subset.
- Every official tool path resolves and reports its source version or commit when
  possible.
- Every prediction root validates or has a documented adapter-specific naming reason.
- Every reference output has a source, command/log, metric units, scene set, and
  tolerance.
- Any missing dataset/tool/reference is recorded in Findings with the exact path or env
  var that was checked.

## Findings

### Preflight (2026-07-10, supersedes the stale initial inspection)

The initial "no real data anywhere" inspection only looked inside the repo/shell env; a
machine-wide preflight found most assets present:

| Benchmark | GT data | Official tool | Known-method predictions |
|---|---|---|---|
| DTU | present: `/mnt/research/dataset/DTU` — `groundtruth/stl<NNN>_total.ply` (full official STLs), `ObsMask/ObsMask<N>_10.mat` + `Plane<N>.mat`. GT lives in `groundtruth/`, not `Points/stl/`; **no split file**. Scan dirs on disk are the standard 15-scan eval set. | dtu_eval port validated (task 010); **MATLAB absent** on this machine — official MATLAB path unavailable, DTUeval-python reference checkout to be used | missing — download approved (MVSNet-style `<method>NNN_l3.ply` releases) |
| Tanks and Temples training | present: `/mnt/dataset/tnt` — all 7 training scenes with full official file set (`<S>.ply`, `.json`, `_trans.txt`, `_COLMAP_SfM.log`, `_mapping_reference.txt`) | toolbox checkout `~/xingrui_ws/tools/TanksAndTemples` @ `2a0d1b25` (isl-org origin); **no open3d==0.9-capable Python on machine** — open3d 0.9.0.0 has cp35–cp37 Linux wheels only (verified on PyPI), uv's oldest CPython is 3.8.20, so uv cannot provide it; dedicated conda py3.7 env planned | missing — download approved |
| ETH3D | **missing** — no ETH3D data on any volume | built binary `~/xingrui_ws/tools/multi-view-evaluation/build/ETH3DMultiViewEvaluation` @ `0daa4f4d` (ETH3D origin) | missing |
| ScanNet | present: `/mnt/dataset/ScanNet` — 1613 scans incl. test split, splits at root + `splits/`, TransformerFusion `mask_test_scans/`, `groundtruth/scannet_test_occlusion_masks/` | native protocol (no official local tool by design) | present: `~/xingrui_ws/codes/prediction/finerecon-4cm` — **official FineRecon release meshes**, 100 test scenes (user-confirmed provenance) |
| Neural-RGBD | present: `~/xingrui_ws/codes/3d-eval/datasets/nrgbd_meshes/official/` — per-scene `gt_mesh.ply`, `gt_mesh_culled.ply`, `gt_trajectory_tum.txt`, splits; plus raw sequences in `nrgbd/` (7.6G, for the deferred depth/pose slice) | native protocol | present: shipped `neural_rgbd.ply` per scene in the mesh root |

- No `EVAL3R_*` env vars set in the shell; corpus README documents the canonical exports.
- No MATLAB (`which matlab` empty): DTU official MATLAB path is recorded unavailable.
- Reproducibility corpus created at `/mnt/research/eval3r-data` (user-chosen root) with
  `datasets/ predictions/ references/ tools/ runs/ scripts/ analysis/`, `README.md`, and
  `inventory.yaml`. Existing dataset roots and tool checkouts are symlinked, not copied;
  Neural-RGBD data intentionally left in place (user decision) and symlinked.

Execution outcomes per phase are appended below.

## Decisions

- Corpus root is `/mnt/research/eval3r-data` (user decision); datasets stay at their
  home locations and are symlinked — Neural-RGBD explicitly NOT moved (user decision).
- Statistical repetition policy (user requirement): each eval3r benchmark is repeated —
  2 same-seed repeats (determinism check, must be near-bit-identical) + 5 seed-varied
  repeats (explicit recorded seed overrides) reporting per-metric mean ± std with
  error-bar plots; program-invocable references (T&T toolbox, ETH3D binary,
  DTUeval-python) are invoked ≥3× to measure their own spread. Pass criterion:
  |eval3r_mean − reference| within the inventory tolerance set before comparison.
- Every validated row must keep alignment/trajectory debug visualizations (tasks 018/023
  artifacts) or, where no alignment runs, per-scene pred-vs-GT overlay renders.
- T&T toolbox Python: uv cannot supply it (open3d 0.9.0.0 = cp37 max; uv min CPython
  3.8.20) — use a dedicated conda py3.7 env recorded in the inventory.
- DTU reference: MATLAB absent, so the source of truth is a real external
  DTUeval-python checkout run on the identical prediction files (recorded commit).
- Treat official-toolbox output from the same invocation as the source of truth for
  Tanks and Temples and ETH3D.
- Treat DTU MATLAB/reference-port output on the same scan and prediction as the source
  of truth for DTU.
- Treat ScanNet and Neural-RGBD as native community validations, not official
  benchmark validations.
- Do not average only successful scenes without recording failed or skipped scenes.
- Do not mark this task done unless every available benchmark path is either verified
  against a real source of truth or explicitly recorded as blocked by missing assets.

## Verification

Planned baseline commands:

```bash
pytest
pytest tests/unit/test_dtu_eval.py tests/unit/test_tnt_official.py tests/unit/test_eth3d_official.py
pytest tests/unit/test_benchmark_runner.py tests/unit/test_protocols.py tests/unit/test_hashing.py
```

Planned preflight commands:

```bash
env | rg '^(EVAL3R_|TANKS|ETH3D|DTU|SCANNET|NRGBD|NEURAL|TNT)'
find . -maxdepth 4 -type d \( -name '*dtu*' -o -name '*scannet*' -o -name '*eth3d*' -o -name '*tanks*' -o -name '*neural*' -o -name '*nrgbd*' \) -print | sort
```

Execution outcomes will be appended here with exact commands, real outputs/summaries,
result directories, and pass/fail deltas.

### Phase 1 — repo-local sanity (2026-07-10)

```bash
python -m pytest -q
# 541 passed, 6 skipped in 31.02s
```

All 6 skips are the intended real-official-tool guards, reasons verbatim:
- ETH3D (×3): "real ETH3D multi-view-evaluation binary not configured; set EVAL3R_ETH3D_TOOL…"
- T&T (×3): "real Tanks and Temples toolbox not configured; set EVAL3R_TNT_TOOLBOX, EVAL3R_TNT_PYTHON (pinned open3d==0.9) and EVAL3R_TNT_DATA…"

The ETH3D binary exists on this machine, so the ETH3D skips were re-run with the tool
configured:

```bash
EVAL3R_ETH3D_TOOL=~/xingrui_ws/tools/multi-view-evaluation/build/ETH3DMultiViewEvaluation \
  python -m pytest -q tests/unit/test_eth3d_official.py tests/integration/test_eth3d_benchmark.py
# 13 passed in 2.68s   (wrapper exercised against the real official binary)
```

T&T official tests remain skipped until the dedicated open3d==0.9 py3.7 env exists (phase 4).

### Phase 2 — ScanNet × FineRecon (2026-07-10)

Result-affecting repo changes made for this phase (both user-approved):

1. **Run-config sampling seed override.** The statistical repetition policy needs
   explicit, recorded seed variation, but `run_benchmark_geometry` hardcoded
   `DEFAULT_BASE_SEED`. Added `base_seed` run-config parameter threaded through
   `run_benchmark_geometry` → per-scene evaluators → `sample_geometry`, exposed as
   `e3r benchmark run --seed N` and `run_benchmark(base_seed=N)`, recorded in
   `config.yaml` and `results.json` `metadata.base_seed`. Seed remains protocol-pinned
   when a protocol sets an explicit integer. Test:
   `test_base_seed_is_run_config_recorded_and_threads_to_sampling`. Docs:
   `.agent/metrics.md`, `.agent/reproducibility.md`, `docs/quickstart.md`.
2. **Pose orthonormality gate recalibrated for real data.** First real ScanNet run
   failed all scenes at resolve: SensReader text-export poses deviate ~1.0–1.3e-6
   from orthonormal, just over the 1e-6 gate in
   `eval3r/core/pose_convention.py::assert_orthonormal`. Corrupt rotations deviate
   ≥1e-2, so the gate was widened to `ORTHO_TOL = 1e-4` (RᵀR and det checks; poses
   are used as-is downstream, the gate only controls acceptance). User approved
   "widen tolerance" over re-orthonormalization or a CLI flag. Test:
   `test_convert_accepts_text_precision_rotation_noise`.

FineRecon release registered in the corpus: 100 test-split meshes fingerprinted
(`predictions/scannet/finerecon/official-release-4cm.sha256`, archive sha256 alongside),
inventory entry with provenance and comparability note (paper uses TransformerFusion
occlusion masks; eval3r protocol does render+TSDF culling → paper numbers are
indicative-only).

3-scene smoke (scene0707–0709): 3/3 evaluated after the pose-gate fix; F-score 0.706,
culled_fraction 0.585, run metadata complete (base_seed, protocol hash, visibility
backend versions). Full campaign launched detached (PID recorded in log):
7 × 100-scene runs (2 same-seed + 5 seed-varied) →
`$CORPUS/runs/aaa6f1e/scannet/scannet_test_single_layer_geometry_5cm/finerecon/rep-*/`,
log `$CORPUS/runs/scannet_finerecon_campaign.log`.

**Campaign complete (2026-07-11): 7 × 100/100 scenes, 0 failures.** Aggregate over 6
seeds (same-seed pair bit-identical, determinism OK):
accuracy 0.074556 ± 0.000116, completeness 0.050620 ± 0.000023,
chamfer 0.062588 ± 0.000062, precision 0.69068, recall 0.74427,
fscore 0.714470 ± 0.000073, culled_fraction 0.430752 (seed-independent, as expected).
Vs FineRecon paper Table 3 "Ours/4cm" (indicative-only reference):
- fscore 0.7145 vs 0.756 → Δ −0.0415, **PASS** within the pre-declared ±0.05.
- chamfer 0.0626 vs 0.0515 → Δ +0.0111, **outside** the pre-declared ±0.01.
  Attribution: eval3r completeness (5.06cm) ≈ paper completeness (5.11cm, 1cm-variant
  row), while accuracy is 7.46 vs 5.25cm — the whole gap is in the pred→GT direction,
  the expected signature of the paper's TransformerFusion occlusion-mask *trimming*
  (removes prediction surface in unobserved regions) vs eval3r's render+TSDF
  visibility culling. Recorded as a protocol-convention difference, not an eval3r
  defect; the protocol was not tweaked to chase the table. Possible follow-up task:
  a TransformerFusion-mask protocol variant for exact paper comparability
  (out of scope for 028 — no new protocols).
Artifacts: `.../finerecon/analysis/` (aggregate.md/csv, errorbars.png, determinism.md).
ScanNet numbers are native community numbers, not official ScanNet benchmark results.

### Phase 4 — Tanks and Temples (2026-07-10, in progress)

- Toolbox env created: conda `tnt-toolbox`, Python 3.7.16, `open3d==0.9.0.0`,
  `matplotlib 3.5.3`, `numpy 1.21.6` — satisfies the official toolbox pins exactly.
  uv could not provide this (open3d 0.9 wheels stop at cp37; uv CPython floor is 3.8.20).
- Real-toolbox tests previously skipped now pass:
  `EVAL3R_TNT_TOOLBOX=… EVAL3R_TNT_PYTHON=… EVAL3R_TNT_DATA=/mnt/dataset/tnt
  pytest tests/unit/test_tnt_official.py tests/integration/test_tnt_benchmark.py`
  → **11 passed in 336.70s**, including a genuine official `run.py` Barn evaluation
  (distance_tau 0.01 read from official output; command/toolbox/interpreter provenance
  asserted).
- Known-method predictions: the dataset ships `<Scene>_COLMAP.ply` for all 7 training
  scenes (the benchmark's own COLMAP reconstruction, with training-set results published
  in the T&T paper) — chosen as the known method; no download needed.
- **Campaign complete (2026-07-11): PASS_EXACT.** Direct official toolbox ×3 per scene
  → deterministic (bit-identical). eval3r wrapped runs ×3 → bit-identical, and
  per-scene precision/recall/fscore equal the direct toolbox output on **all 7 scenes**
  (Δ = 0.0000): Barn F=0.5003 (τ=0.010), Caterpillar F=0.5570 (τ=0.005), Church
  F=0.5310 (τ=0.025), Courthouse F=0.4563 (τ=0.025), Ignatius F=0.8055 (τ=0.003),
  Meetingroom F=0.3431 (τ=0.010), Truck F=0.6745 (τ=0.005); mean F=0.5525. Per-scene
  taus read from official output (not a global constant — verified). Evidence:
  `references/tanks_temples/colmap/dataset-shipped/` (raw toolbox outputs ×21 +
  reference.yaml), runs under `runs/aaa6f1e/tanks_temples/.../colmap/rep-{1,2,3}`.
  Paper-table cross-check intentionally not asserted (provenance of the shipped
  reconstruction vs the paper's pipeline configuration is not verifiable; the
  same-invocation toolbox output is the recorded source of truth).

### Phase 6 — Neural-RGBD (2026-07-10)

- Both protocols smoke-passed on the 9-scene `eval3r_current` split (icl_living_room
  excluded), shipped `neural_rgbd.ply` as the known method:
  culled → acc 0.02087 / comp 0.08223 / chamfer 0.05155 / F 0.90762;
  source → completeness 0.895 / recall 0.351 (variants correctly not comparable).
  Debug error clouds + histograms emitted per scene.
- Named reference: real **GO-Surf** driver (`tools/mesh_metrics.py::compute_metrics`,
  commit 71bb125), 3 repeats × 9 scenes on identical files → mean
  Acc 0.01699 / Comp 0.07958 / C-L1 0.04828 / F 0.91580.
- Delta vs eval3r explained and **proven sampling-density-driven**: GO-Surf's own
  distance/threshold functions re-run at eval3r's pinned 200k samples/side give
  Acc 0.020879 / Comp 0.08224 / C-L1 0.051559 / F 0.907674 — agrees with eval3r to
  ~1e-5 absolute (F within 6e-5). Evidence: `references/neural_rgbd/.../reference.yaml`
  (+ `diag_gosurf_200k.log`). Metric formulas validated against the community driver.
- Repetition campaign (2 same-seed + 5 seed-varied × both protocols) launched detached.

### Phase 5 — ETH3D (2026-07-10)

- Data acquired into the corpus: `multi_view_training_dslr_scan_eval.7z` (all 13
  training scenes' GT scans + `scan_alignment.mlp`) and
  `courtyard_dslr_undistorted.7z` (calibration + images); archives fingerprinted
  (`datasets/eth3d_training_downloads/archives.sha256`), extracted to
  `datasets/eth3d_training/`.
- Known-method prediction: ETH3D's own shipped COLMAP SfM sparse points
  (`dslr_calibration_undistorted/points3D.txt`, 33,487 pts) converted to
  `courtyard.ply` — a real benchmark-shipped reconstruction artifact (same approach
  as T&T's shipped COLMAP ply).
- Reference: direct `ETH3DMultiViewEvaluation` runs ×3 on the identical file —
  **bit-identical across repeats** (deterministic binary).
- **eval3r `eth3d_training_official` parity: EXACT on all 18 values**
  (accuracy/completeness/F1 at 1/2/5/10/20/50 cm), e.g. F1@2cm 0.0293307,
  F1@50cm 0.785295. Evidence: `references/eth3d/colmap-sfm/dataset-shipped/courtyard/`
  (raw logs + reference.yaml), run dir `runs/smoke/eth3d_colmap_sfm_courtyard`.
- Misconfigured-tool path verified: without `EVAL3R_ETH3D_TOOL` the run fails with
  the explicit remediation message (no silent fallback).
- Repetition (official path has no sampling): 3 eval3r runs — `results.json` metrics
  dicts compare `==` across all three (`rep-{1,2,3}` under
  `runs/aaa6f1e/eth3d/eth3d_training_official/colmap-sfm/`). Wrapper deterministic.
- Blocked (recorded): leaderboard cross-check vs a published method — ETH3D does not
  distribute submitted reconstructions; other 12 scenes' dense-calibration archives
  can be added later the same append-only way.

### Phase 6 addendum — Neural-RGBD repetition campaign (2026-07-10)

- Campaign complete (14 runs). Culled protocol aggregate over 6 seeds:
  accuracy 0.02088 ± 1.6e-5, completeness 0.08222 ± 4.9e-4, chamfer 0.05155 ± 2.5e-4,
  fscore 0.90764 ± 1.6e-4 → **all PASS** vs the as-is GO-Surf reference within the
  pre-declared tolerances; same-seed pairs bit-identical (determinism OK).
- Source protocol aggregate recorded (internal stats; no external reference exists
  for the uncropped variant): fscore 0.48838 ± 2.3e-4, determinism OK.
- Artifacts: `runs/aaa6f1e/neural_rgbd/<protocol>/neural_rgbd/analysis/`
  (aggregate.md/csv, errorbars.png, determinism.md).

### Phase 3 — DTU (2026-07-10, in progress)

- Adapter already accepts `groundtruth/` GT layout (task 010); added
  `<root>/Points/stl -> groundtruth` symlink for the *external* DTUeval-python reference
  tool, and `<root>/splits/test.txt` with the canonical 22-scan MVSNet evaluation set
  (verified against CasMVSNet and PatchmatchNet `lists/dtu/test.txt`; the previous
  15-scan NeuS-style set is kept as `splits/neus15.txt`).
- Known-method predictions: R-MVSNet release on Google Drive is not fetchable by gdown
  (permission/quota). Instead using DTU's own reference reconstructions
  (`Points_MVS.zip`: camp/furu/tola `<method>NNN_l3.ply`) fetched selectively via HTTP
  range requests for the 22 eval scans — published means (R-MVSNet Tab.1, verified from
  the PDF): Camp 0.835/0.554/0.695, Furu 0.613/0.941/0.777, Tola 0.342/1.19/0.766 mm.
- Reference evaluator: real `jzhangbs/DTUeval-python` checkout at corpus
  `tools/DTUeval-python` (commit 45b4fec); MATLAB absent on this machine (recorded).
- **Campaign complete (2026-07-11): PASS.** All 22 scans × {camp ×7 reps, furu ×1,
  tola ×1} evaluated 22/22 with 0 failures; real DTUeval-python reference run on every
  camp scan (+3 repeats on scan1 and scan24; reference repeat spread ≤ 7.5e-5 mm).
  - Per-scene exact-file parity: worst |eval3r − DTUeval-python| over 22 scans × 3
    metrics = **9.6e-5 mm** (scan13 accuracy) — same order as the reference's own
    run-to-run spread.
  - Camp aggregate over 6 seeds: accuracy 0.836108 ± 1.0e-5, completeness
    0.566368 ± 4.0e-6, overall 0.701238 ± 6.8e-6 vs reference means
    0.836102 / 0.566369 / 0.701236 → deltas ≤ 6.2e-6 mm, **PASS** at the
    pre-declared 0.001 tolerance. Same-seed pair bit-identical.
  - Published-MATLAB cross-check (secondary anchor): accuracy matches ≤1e-3 for all
    three methods (camp 0.8361 vs 0.835, furu 0.6125 vs 0.613, tola 0.3426 vs 0.342);
    completeness consistently +0.008–0.015 (camp 0.5664 vs 0.554, furu 0.9493 vs
    0.941, tola 1.2049 vs 1.19) — the known MATLAB↔DTUeval-python difference, not an
    eval3r artifact, since the exact-file reference agrees to 1e-4.
  - Evidence: `references/dtu/camp/points_mvs_2014/` (22 raw logs + reference.yaml),
    runs + `analysis/` under `runs/aaa6f1e/dtu/dtu_native_pointcloud/{camp,furu,tola}`.
- **scan1 parity (camp prediction, identical 637MB PLY, first fetched file):**
  eval3r `dtu_native_pointcloud` acc/comp/overall = 0.504416 / 0.210567 / 0.357491 mm
  vs DTUeval-python over 3 repeats (unseeded shuffle): mean 0.504423 / 0.210579 /
  0.357501, spread ~1e-5. eval3r sits **inside the reference's own run-to-run
  spread** (matches repeats 2–3 to 1e-6 on accuracy) → PASS at the 0.001mm tolerance.
  Run dir: `$CORPUS/runs/smoke/dtu_camp_scan1`; raw logs + normalized
  `scan1_reference.yaml` under `$CORPUS/references/dtu/camp/points_mvs_2014/`.

## Final comparison table (phase 7; DTU rows pending campaign completion)

Repetition policy: 2 same-seed + 5 seed-varied eval3r runs where sampling applies
(mean ± std over 6 distinct seeds); official-tool paths (no sampling) use ×3 identical
runs. Program-invocable references run ≥3×. All analysis artifacts (aggregate.md/csv,
errorbars.png, determinism.md) live next to the reps under
`$CORPUS/runs/aaa6f1e/<dataset>/<protocol>/<method>/analysis/`.

| dataset/protocol | method | headline metric: eval3r (±std) | reference | Δ | tol | verdict | source of truth |
|---|---|---|---|---|---|---|---|
| dtu_native_pointcloud | camp | overall 0.701238 ± 0.0000068 mm (6 seeds) | 0.701236 (DTUeval-python, 22 scans, identical files) | +2.3e-6 | 0.001 | **PASS** (worst per-scene Δ 9.6e-5) | real DTUeval-python @45b4fec on identical files; MATLAB absent (recorded) |
| dtu_native_pointcloud | furu / tola | overall 0.7809 / 0.7738 mm (×1) | 0.777 / 0.766 (published MATLAB means) | +0.004 / +0.008 | indicative | **PASS** (accuracy ≤1e-3 for all methods; completeness offset = known MATLAB↔port difference) | published R-MVSNet Tab.1 (secondary anchor) |
| tanks_temples_training_official | colmap (shipped) | fscore 0.552529 (×3 bit-identical) | 0.552529 (direct toolbox ×3, deterministic) | −4e-7 | 1e-4 | **PASS_EXACT** (all 7 scenes Δ=0.0000) | official toolbox @2a0d1b25, same invocation, pinned open3d 0.9 env |
| eth3d_training_official | colmap-sfm (shipped points3D) | fscore_2cm 0.0293307 (×3 bit-identical) | 0.0293307 (direct binary ×3, deterministic) | 0 | 1e-6 | **PASS_EXACT** (all 18 values) | official ETH3DMultiViewEvaluation @0daa4f4d, same input |
| scannet_test_single_layer_geometry_5cm | finerecon (official 4cm release) | fscore 0.71447 ± 0.00007 | 0.756 (paper Table 3, 4cm) | −0.0415 | 0.05 | **PASS** (indicative) | paper table; protocol differs (mask-trim vs render-cull), documented |
| scannet_test_single_layer_geometry_5cm | finerecon | chamfer 0.06259 ± 0.00006 m | 0.0515 (paper Table 3, 4cm) | +0.0111 | 0.01 | **OUTSIDE TOL** — attributed to trimming-convention difference (acc-direction only; completeness matches to 0.5mm) | same |
| neural_rgbd_geometry_culled | neural_rgbd (shipped) | fscore 0.90764 ± 0.00016 | 0.91580 (real GO-Surf driver ×3) | −0.0082 | 0.02 | **PASS**; matched-200k diagnostic agrees to ~1e-5 | GO-Surf @71bb125 on identical files |
| neural_rgbd_geometry_source | neural_rgbd | fscore 0.48838 ± 0.00023 | none (no external ref for uncropped GT) | – | – | recorded (internal stats only) | – |
| scannet val single/double layer | – | – | – | – | – | **BLOCKED**: no val-split predictions of a known method | – |
| eth3d leaderboard cross-check | – | – | – | – | – | **BLOCKED**: ETH3D does not distribute submitted reconstructions | – |
| dtu MATLAB official script | – | – | – | – | – | **BLOCKED**: MATLAB not installed on this machine | – |

Determinism: every same-seed / same-config pair compared `==` on aggregate metrics —
no drift observed anywhere. Run-dir inspection: all run dirs checked by
`scripts/check_run_metadata.py` → 0 problems (required files, reproducibility fields,
base_seed in config + metadata, coverage/failure visibility).

### Close-out verification (2026-07-11)

```bash
ruff check .                                   # all checks passed
python -m pytest -q                            # 543 passed, 6 skipped (the 6 are the
                                               # ETH3D/T&T real-tool env-var guards;
                                               # both tools were exercised for real in
                                               # phases 1/4 with the env configured)
python $CORPUS/scripts/check_run_metadata.py   # 41 run dirs checked, 0 problems
```

Every implemented dataset/protocol path is either verified against a real source of
truth or explicitly recorded blocked (table above). The reproducibility corpus at
`/mnt/research/eval3r-data` (inventory.yaml, predictions, references with raw logs,
runs with analysis/error-bar plots, scripts) is the durable artifact index; nothing
large or licensed entered this repo.

## Status

done
