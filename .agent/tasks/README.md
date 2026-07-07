# eval3r Task Index

Canonical index of implementation task slices. Statuses: `todo`, `in_progress`, `done`, `blocked`.

Resume rule: start from the first non-`done` task below. Before resuming any task, re-read the
relevant `.agent/*.md` docs and the actual code — task notes can be stale.

## Ordering and status

| # | Task file | Slice | Status |
|---|-----------|-------|--------|
| 001 | [001-repo-foundation.md](001-repo-foundation.md) | Package skeleton, pyproject, CI, lint, test, docs build, `e3r --help` | done |
| 002 | [002-core-schema.md](002-core-schema.md) | Pydantic models from `.agent/schema.md`, serialization tests, fixtures | done |
| 003 | [003-protocol-loader-hashing.md](003-protocol-loader-hashing.md) | Protocol YAML loader, canonical hashing, registry, built-in YAMLs | done |
| 004 | [004-backend-registry.md](004-backend-registry.md) | Backend registry (selection + version recording), scipy/trimesh/plyfile backends | done |
| 005 | [005-geometry-metrics.md](005-geometry-metrics.md) | Point-set geometry metrics with synthetic analytic tests | done |
| 006 | [006-result-writer-environment.md](006-result-writer-environment.md) | Run directory writer, JSON/CSV output, environment metadata | done |
| 007 | [007-single-file-geometry-runner.md](007-single-file-geometry-runner.md) | Staged single-file geometry runner, `e3r metric geometry`, Python API | done |
| 008 | [008-benchmark-run-plumbing.md](008-benchmark-run-plumbing.md) | Generic dataset adapter registry, manifest resolution, `e3r benchmark run` | done |
| 009 | [009-dtu-adapter.md](009-dtu-adapter.md) | DTU adapter, ObsMask/Plane metadata, mm normalization, fixture | done |
| 010 | [010-dtu-official-evaluator.md](010-dtu-official-evaluator.md) | DTU official-like evaluator backend and protocol integration | done |
| 011 | [011-scannet-adapter.md](011-scannet-adapter.md) | ScanNet adapter, mesh variants, culling, benchmark integration | done |
| 012 | [012-tnt-official-wrapper.md](012-tnt-official-wrapper.md) | Tanks and Temples adapter + official script wrapper | done |
| 013 | [013-eth3d-adapter.md](013-eth3d-adapter.md) | ETH3D adapter, pycolmap cameras, official-tolerance validation | done |
| 014 | [014-depth-metrics.md](014-depth-metrics.md) | Depth IO, masks, scale alignment modes, `e3r metric depth` | done |
| 015 | [015-pose-metrics.md](015-pose-metrics.md) | evo backend, ATE/RPE, `e3r metric pose` | done |
| 016 | [016-reports-diffing.md](016-reports-diffing.md) | Markdown/LaTeX/HTML reports, partial-coverage banners, `e3r diff` | done |
| 017 | [017-public-release-docs.md](017-public-release-docs.md) | Release docs, examples, packaging checks, PyPI prep | done |
| 018 | [018-alignment-module.md](018-alignment-module.md) | SE3/Sim3 alignment (closest-point ICP + trajectory-first; no FPFH), mandatory visualization, `e3r align` | done |
| 019 | [019-prediction-io.md](019-prediction-io.md) | Official eval3r-native prediction layout: `PredictionWriter` / `read_prediction_dir`, `e3r prediction` | done |
| 020 | [020-neural-rgbd-geometry.md](020-neural-rgbd-geometry.md) | Neural-RGBD geometry adapter + culled/source mesh protocols (depth/pose deferred) | done |
| 021 | [021-pose-convention-transforms.md](021-pose-convention-transforms.md) | First-class pose/coordinate convention transforms (axis/direction/world-frame) with validation, wired through pose+geometry runners, CLI, API, writer | done |
| 022 | [022-prediction-adaptation.md](022-prediction-adaptation.md) | Provenance-driven prediction adaptation + two-level protocol identity: allowed alignment envelope (hashed) vs non-hashed `AdaptationRecord`; `scale` made load-bearing; compact `--as cw@opencv@sim3` grammar; metric + benchmark paths | done |

Tasks 001–021 are done; task 022 is the next slice (first non-`done`). Remaining work beyond it
is the intentionally deferred backlog below (each item
needs its conventions pinned first — create a numbered task file when one is picked up),
Neural-RGBD depth/pose (geometry shipped in task 020), and the actual PyPI publish, which
happens only on explicit user request (see the release checklist in task 017).

Tasks must be executed in order unless a task's Scope says otherwise; each depends on the
deliverables of the previous ones (schemas before adapters, metrics before runners, result
writing before execution, single-file evaluation before dataset benchmarks, generic
benchmark plumbing before real dataset adapters).

## Backlog (no task files yet — intentionally deferred)

Per CLAUDE.md, complex dataset adapters wait until the schema and protocol behavior they
need is documented and proven by the core slices above:

```text
7-Scenes adapter          (depth/pose; pinned Kinect intrinsics; 65535 invalid depth)
Neural-RGBD depth/pose    (geometry done in task 020; depth/pose still deferred:
                           OpenGL convention, focal.txt/poses.txt, neural_rgbd_depth.yaml)
Replica adapter           (requires named + fingerprinted rendered trajectory bundle)
Hypersim adapter          (ray-distance depth; meters_per_asset_unit; paid mesh assets)
CO3D adapter              (frame_annotations.jgz; COLMAP-derived GT; object-centric)
BlendedMVS adapter        (MVSNet cam.txt; verify parser against primary implementation)
KITTI-360 adapter         (pose + sparse LiDAR; fisheye cameras out of minimal scope)
Plugin API                (defer until at least 3 built-in adapters are implemented and
                           there is 1 concrete external-adapter use case)
```

## Conventions for task files

Every task file has: Goal, Scope, Out of Scope, Relevant Files, Plan, Findings, Decisions,
Verification, Status. Mark the active task `in_progress` before editing code; record durable
findings and decisions (not transcripts); record verification commands and outcomes; mark
`done` only after verification passes and update this index in the same change. The task file's
Findings/Decisions/Verification are the task's report — there are no separate report files.
Official-eval paths must be verified against the **real** official toolbox (no fake stand-in);
see the "Official code / toolbox rule" in `CLAUDE.md`.
