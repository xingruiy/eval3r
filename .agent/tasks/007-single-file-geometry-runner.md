# 007 — Single-file geometry runner

## Goal

The staged evaluation pipeline runs end-to-end for one prediction file and one ground-truth
file, writes a complete run directory through task 006, and powers `e3r metric geometry`
plus the `evaluate_geometry` Python API.

## Scope

- `pipeline/runner.py` + `pipeline/stages/*`: stage functions configured entirely by the
  protocol; shared by geometry now, depth/pose later (tasks 014/015) so failure handling
  and metadata capture do not drift.
- Single-file geometry path: resolve, load, normalize units, align, mask/cull, sample,
  metric, aggregate, report-to-json/csv.
- Failure policies `abort`, `skip_and_flag`, `score_worst`; structured errors say what
  failed, which scene, which file, which protocol required it, and how to fix it when
  obvious.
- Alignment stage for point sets: `none`, `se3`, `sim3` (Umeyama). ICP is deferred until a
  backend exists. Transforms are passed to the task-006 writer.
- CLI `e3r metric geometry pred.ply --gt gt.ply --threshold 0.05 --sample 200000 --input
  pointcloud --gt-input pointcloud` and mesh variant, mapping flags onto the
  `single_geometry` protocol as recorded overrides.
- Python API: `evaluate_geometry(...)` per `.agent/plan.md`.
- Determinism: same protocol hash + seeds -> identical results; seed `derive` policy
  (per-scene seed from protocol base seed + scene_id).

## Out of Scope

- Dataset adapters and `e3r benchmark run` (task 008+).
- Dataset-specific visibility, masks, or official evaluator behavior.
- Markdown/LaTeX/HTML reports and diffing (task 016).

## Relevant Files

- `.agent/plan.md` — "Architecture", "CLI", "Python API"
- `.agent/reproducibility.md` — result directory and environment metadata expectations
- `.agent/schema.md` — `RunResult`, `SceneFailure`, `AlignmentSpec`
- `.agent/metrics.md` — pipeline order for point-cloud metrics, failure behavior

## Plan

1. Implement stages as small pure functions over a run context; runner sequences them and
   applies the failure policy per scene.
2. Wire `single_geometry` protocol through the runner for one pred/gt file pair.
3. Implement point-set SE3/Sim3 alignment and transform metadata.
4. CLI command + Python API wrapper.
5. Tests: failure-policy behavior per stage, deterministic repeat runs, override recording,
   alignment transform output, integration test for pointcloud and mesh inputs.

## Findings

- Stages are pure functions over a `LoadedGeometry` (points *or* a mesh handle). The runner
  tracks a `stage` variable and captures any `Eval3rError` into a `SceneFailure` tagged with
  the failing stage, so failure accounting is exact without per-stage try/except sprawl.
- Umeyama alignment needs 1:1 correspondence (equal, matched point counts). That is the only
  correspondence a single-file comparison can assume without a registration/ICP backend, so
  unequal counts fail explicitly; correspondence-free ICP is deferred to task 015+.
- `single_geometry`'s built-in threshold is already 0.05, so `--threshold 0.05` (and default
  pointcloud sampling) leave the canonical protocol hash unchanged — a useful determinism
  check. Only a *behavior-changing* override (different threshold, sample count, mesh modality)
  moves the hash.
- The shell `e3r` on PATH in the `dl` conda env is a *different, older* `eval3r` package
  (it has `--thresholds`, `--align icp_se3`, `--chamfer-variant`). `import eval3r` still
  resolves to this source tree, and the in-process CLI (`typer.testing.CliRunner`, or
  `python -c "from eval3r.cli.main import app; app()"`) exercises our real command.

## Decisions

Superseded design note (2026-07-08): Sim3 is no longer disallowed by metric-scale
protocols. Prediction adaptation is user/run configuration, not a protocol permission gate;
selected scale-resolving alignment runs and is recorded.

- New errors `AlignmentError`, `CullingError`, `SceneEvaluationError` (the last carries
  scene_id + stage for the `abort` policy) live in `core/errors.py`.
- Single-file masking supports only `method: none`; dataset masks (obs/visibility/official)
  raise `CullingError` pointing at `e3r benchmark run` (task 011+) rather than silently
  evaluating unmasked geometry.
- Seeds: explicit ints pass through; `derive`/`None` derives a stable 32-bit seed from a base
  seed (default 0) and `scene_id:role`, so repeated runs match and pred/gt differ.
- Timestamps: `RunResult.timestamp` is UTC with a `Z` suffix and the default run-dir name uses
  a UTC `now`, so neither leaks the local timezone (privacy). `environment.json` stays minimal.
- CLI flags (`--threshold`, `--sample`, `--input`, `--gt-input`) map onto a deep-copied
  protocol as recorded overrides; the canonical hash is recomputed from the overridden copy.
- `typer.Argument`/`typer.Option` added to ruff `flake8-bugbear.extend-immutable-calls`
  (the framework's intended default-argument pattern) — fixes B008 on `Path`-typed params.

## Verification

```bash
ruff check .                 # All checks passed!
mypy eval3r                  # Success: no issues found in 86 source files
pytest -q                    # 144 passed
mkdocs build                 # OK
# CLI (driven in-process; PATH `e3r` is a different installed package in this env):
python -c "import sys; sys.argv=['e3r','metric','geometry','tests/fixtures/geom/pred.ply',\
  '--gt','tests/fixtures/geom/gt.ply','--threshold','0.05','--out','/tmp/e3r_run']; \
  from eval3r.cli.main import app; app()"
```

Acceptance per `.agent/plan.md`: the `e3r metric geometry` command produces a complete run
directory (`results.json`, `results.csv`, `per_scene.csv`, `failures.json`, `environment.json`,
`backend_versions.json`, `protocol.yaml`, `config.yaml`, `alignment_transforms.json`, `logs.txt`)
whose `results.json` carries every field `.agent/reproducibility.md` requires. Verified by
`tests/integration/test_single_geometry.py`.

## Status

done
