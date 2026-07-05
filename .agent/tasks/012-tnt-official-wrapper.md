# 012 — Tanks and Temples official wrapper

## Goal

Tanks and Temples training-split evaluation runs locally by wrapping the official
evaluation code (fidelity `official`), while intermediate/advanced splits are cleanly
refused as server-only.

## Scope

- `datasets/tanks_temples.py`: resolve public training scenes, GT point cloud, crop-volume
  JSON, `.log` trajectory, alignment transform; per-scene thresholds come from official
  metadata/backend output, never a global constant.
- `backends/tnt_official.py` (`official_eval` registry kind): invoke the official
  evaluation script (vendored or user-supplied checkout — record which and its
  version/commit); normalize its output into `MetricResult` / results.json; record backend
  version, command, per-scene threshold, alignment/ICP details the backend performed.
- Capabilities: training → official_local_eval=true, method=official_script_wrapper;
  intermediate/advanced → server_only_eval=true, method=server_only.
- CLI behavior: running a server-only protocol
  (`tanks_temples_intermediate_server_only`) fails clearly before any computation, telling
  the user to submit to the official server.
- GT fingerprinting: GT point cloud, crop file, alignment file, backend version.
- Tiny fixture with fabricated official-script outputs for output-parsing tests
  (`tests/fixtures/tanks_temples_tiny/`); a dry-run/wrapper test that skips when the
  official code is absent.

## Out of Scope

- Reimplementing the official evaluation (explicitly forbidden without regression-tested
  reason).
- Full-dataset validation in CI (documented manual step when data is available).

## Relevant Files

- `.agent/datasets.md` — "Tanks and Temples" requirements and rules
- `.agent/protocols.md` — `tanks_temples_training_official` + server-only stub
- `.agent/backends.md` — "Tanks and Temples backend"
- `.agent/plan.md` — milestone
- `CLAUDE.md` — Tanks and Temples cautions

## Plan

1. Implement adapter resolution of the five per-scene artifacts.
2. Implement wrapper backend: subprocess or import invocation, output parsing,
   version/command recording.
3. Wire capability gating so the CLI refuses server-only splits with a clear message.
4. Tests: artifact resolution, output parsing from fixture text, server-only refusal,
   per-scene threshold recording, capability declarations.

## Findings

- **Real data** at `/mnt/dataset/tnt` (7 training scenes: Barn, Caterpillar, Church,
  Courthouse, Ignatius, Meetingroom, Truck). Per scene `<Scene>/`: `<Scene>.ply` (laser-scan
  GT, metres), `<Scene>.json` (crop volume), `<Scene>_trans.txt` (4x4 alignment),
  `<Scene>_COLMAP_SfM.log` (.log trajectory), `<Scene>_mapping_reference.txt`, plus
  `<Scene>_COLMAP.ply` (a reconstruction). No local official-toolbox checkout exists.
- **Official code (found online, source-verified):** `isl-org/TanksAndTemples`
  `python_toolbox/evaluation`. `run.py` CLI is
  `--dataset-dir <scene_dir> --traj-path <log> --ply-path <pred> [--out-dir]`; it reads
  `<scene>.ply/.json/_trans.txt/_COLMAP_SfM.log/_mapping_reference.txt` from `dataset-dir`
  by naming convention, does trajectory alignment + 3-stage ICP + crop, and prints exactly
  `distance tau : %.3f` / `precision : %.4f` / `recall : %.4f` / `f-score : %.4f`. Per-scene
  `dTau` comes from `config.py` `scenes_tau_dict` (Barn 0.01, Caterpillar 0.005, Church 0.025,
  Courthouse 0.025, Ignatius 0.003, Meetingroom 0.01, Truck 0.005). The toolbox pins
  `open3d==0.9`.
- **Real-toolbox validation (this task):** cloned the toolbox (commit `2a0d1b25`) and drove it
  through the real `TntOfficialEval` wrapper on real Barn data (`Barn_COLMAP.ply` as the
  prediction). The wrapper resolved the checkout, recorded the commit, built the correct
  command, launched `run.py` (which loaded the real GT/crop/trans/log/mapping and reached
  `trajectory_alignment`), and the run failed at `o3d.registration.RANSACConvergenceCriteria()`
  — the open3d 0.9→0.19 API break (`open3d.registration` moved to `open3d.pipelines.registration`),
  **not** a wrapper bug. The wrapper surfaced it verbatim (traceback + stderr + command) as a
  `MetricError`. A full real score needs the toolbox's pinned `open3d==0.9` env; that is the
  documented manual step. Patching the official code is out of scope (do-not-reimplement rule).

## Decisions

- **External user-supplied checkout, not vendored.** The official toolbox is treated like the
  DTU MATLAB path: an external tool, located from an explicit `toolbox_dir` or the
  `EVAL3R_TNT_TOOLBOX` / `TANKSANDTEMPLES_TOOLBOX` env var. Vendoring was rejected (its
  `open3d==0.9` pin conflicts with the project's open3d, and CLAUDE.md forbids reimplementing
  official evaluation). Absent toolbox → explicit `BackendError` naming the env vars + repo URL.
- `backends/tnt_official.py` `TntOfficialEval` (`official_eval` kind, name `tnt_official`,
  method `official_script_wrapper`, `input_mode="artifacts"`): subprocess-invokes the official
  `run.py`, records command + toolbox dir + git commit, and parses the printed summary via a
  pure `parse_official_output`. Per-scene `dTau` is read from the official output, never
  hardcoded in eval3r.
- `datasets/tanks_temples.py` `TanksAndTemplesAdapter`: resolves the five per-scene artifacts
  (`official_artifacts`), GT `laser_scan`/`independent`/`dense_surface` in metres, joint
  `gt_fingerprint` over GT+crop+trans, official training/intermediate/advanced scene lists.
  Per-split gate: `local_evaluation('intermediate'|'advanced')` → `server_only` (preflight
  refuses before any computation); training → supported.
- Benchmark gained a file-based official branch (`_evaluate_scene_tnt_official`), dispatched by
  the evaluator's `input_mode == "artifacts"` (keeps the DTU point-array branch intact).
- Protocol YAMLs (`tanks_temples_training_official`, `tanks_temples_intermediate_server_only`)
  were already present and unchanged → pinned hashes unchanged.
- Testing without the heavy real toolbox: a fake-toolbox fixture (`fake_toolbox/run.py`) mirrors
  the official CLI + output format exactly, so the real command-build + subprocess + parse path
  is exercised end-to-end in CI; parse and absent-toolbox paths tested directly.

## Verification

```bash
ruff check .            # All checks passed!
mypy eval3r             # Success: no issues found in 89 source files
pytest -q               # 249 passed
mkdocs build            # OK
# real toolbox (offline, not committed): wrapper drove isl-org/TanksAndTemples @2a0d1b25 on
#   real Barn; correct command + recorded commit; failed inside toolbox on open3d 0.9 vs 0.19
#   (needs pinned open3d==0.9 env) — surfaced verbatim, not a wrapper defect.
```

Covered by `tests/unit/test_tnt_adapter.py` (13), `tests/unit/test_tnt_official.py` (8, incl.
fake-toolbox subprocess run, parse, absent-toolbox refusal), `tests/integration/test_tnt_benchmark.py`
(training run via fake toolbox with per-scene dTau recorded; intermediate split refused
server-only). Fixture: `tests/fixtures/tanks_temples_tiny/`.

## Status

done
