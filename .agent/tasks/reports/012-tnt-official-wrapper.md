# Task 012 report — Tanks and Temples official wrapper

**Status:** done · **Commits:** `7d8d081` (initial), `<rework>` (real-toolbox rework)

## What was built and why

Local, official-fidelity evaluation of the Tanks and Temples **training** split by wrapping
the official `isl-org/TanksAndTemples` `python_toolbox/evaluation` toolbox, plus clean
server-only refusal of the intermediate/advanced splits (GT withheld). eval3r resolves the
five per-scene artifacts and delegates precision/recall/F-score, ICP, cropping, and the
per-scene distance threshold `dTau` to the official code — it never reimplements them.

## What changed

- `eval3r/datasets/tanks_temples.py` — `TanksAndTemplesAdapter`: resolves `<Scene>.ply` (GT,
  metres), `.json` (crop), `_trans.txt` (alignment), `_COLMAP_SfM.log` (trajectory),
  `_mapping_reference.txt`; joint `gt_fingerprint`; per-split `local_evaluation` gate
  (training supported; intermediate/advanced server_only).
- `eval3r/backends/tnt_official.py` — `TntOfficialEval` (`official_eval`, `input_mode="artifacts"`):
  subprocess-runs the official `run.py`, parses its summary, records command + toolbox dir +
  git commit + **interpreter used**. `dTau` read from official output. Interpreter configurable
  via `python_executable` / `EVAL3R_TNT_PYTHON`.
- `eval3r/pipeline/benchmark.py` — file-based official branch `_evaluate_scene_tnt_official`,
  dispatched by `input_mode == "artifacts"`; records `python_executable` in metadata.
- Registries wired (`core/registry.py`, `datasets/registry.py`, `datasets/__init__.py`).
- Tests: `tests/unit/test_tnt_adapter.py`, `tests/unit/test_tnt_official.py`,
  `tests/integration/test_tnt_benchmark.py`; fixture `tests/fixtures/tanks_temples_tiny/`.

## How it was verified

- `ruff check .`, `mypy eval3r`, `pytest -q`, `mkdocs build` — all green (see task file for the
  final counts of the rework run).
- **Real official toolbox, real data:** created pinned env
  `conda create -n tnt_toolbox python=3.7 && pip install open3d==0.9.0.0 matplotlib numpy`, then
  drove the **unmodified** toolbox (commit `2a0d1b25`) through the real `TntOfficialEval`
  wrapper on real Barn (`/mnt/dataset/tnt/Barn`, prediction `Barn_COLMAP.ply`):
  **precision 0.4569 / recall 0.5529 / f-score 0.5003 at dTau 0.01**, ~150s. The real-toolbox
  tests were run with `EVAL3R_TNT_TOOLBOX` / `EVAL3R_TNT_PYTHON` / `EVAL3R_TNT_DATA` set and
  passed; they skip cleanly when unset.

## Official-toolbox compat / result-affecting note

The toolbox pins `open3d==0.9`. Under newer open3d it fails at
`o3d.registration.RANSACConvergenceCriteria()`. This is **not** a pure namespace move:
open3d 0.9's `RANSACConvergenceCriteria(max_iteration, max_validation)` and the `checkers`-less
`registration_ransac_based_on_correspondence` signature differ semantically from newer open3d,
so porting them would change the trajectory alignment used for scoring — a **result-affecting**
change. Per the "Official code / toolbox rule" the toolbox is run **byte-for-byte unmodified**
under its pinned interpreter instead. No official code was patched; nothing needed escalation
because the pinned-env path avoids the porting decision entirely.

## Not covered / limitations / follow-ups

- CI does not run the real toolbox (heavy, needs the checkout + `open3d==0.9` env + real GT);
  those tests skip there. Real-toolbox validation is a documented local step.
- Only Barn was scored end-to-end (representative); the other six training scenes use the same
  path and per-scene `dTau` from official output.
- The external toolbox checkout and the `tnt_toolbox` conda env are local, not committed.
