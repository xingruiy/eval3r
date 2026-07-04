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
| 009 | [009-dtu-adapter.md](009-dtu-adapter.md) | DTU adapter, ObsMask/Plane metadata, mm normalization, fixture | todo |
| 010 | [010-dtu-official-evaluator.md](010-dtu-official-evaluator.md) | DTU official-like evaluator backend and protocol integration | todo |
| 011 | [011-scannet-adapter.md](011-scannet-adapter.md) | ScanNet adapter, mesh variants, culling, benchmark integration | todo |
| 012 | [012-tnt-official-wrapper.md](012-tnt-official-wrapper.md) | Tanks and Temples adapter + official script wrapper | todo |
| 013 | [013-eth3d-adapter.md](013-eth3d-adapter.md) | ETH3D adapter, pycolmap cameras, official-tolerance validation | todo |
| 014 | [014-depth-metrics.md](014-depth-metrics.md) | Depth IO, masks, scale alignment modes, `e3r metric depth` | todo |
| 015 | [015-pose-metrics.md](015-pose-metrics.md) | evo backend, ATE/RPE, `e3r metric pose` | todo |
| 016 | [016-reports-diffing.md](016-reports-diffing.md) | Markdown/LaTeX/HTML reports, partial-coverage banners, `e3r diff` | todo |
| 017 | [017-public-release-docs.md](017-public-release-docs.md) | Release docs, examples, packaging checks, PyPI prep | todo |

**Next task: 009**

Tasks must be executed in order unless a task's Scope says otherwise; each depends on the
deliverables of the previous ones (schemas before adapters, metrics before runners, result
writing before execution, single-file evaluation before dataset benchmarks, generic
benchmark plumbing before real dataset adapters).

## Backlog (no task files yet — intentionally deferred)

Per CLAUDE.md, complex dataset adapters wait until the schema and protocol behavior they
need is documented and proven by the core slices above:

```text
7-Scenes adapter          (depth/pose; pinned Kinect intrinsics; 65535 invalid depth)
Neural-RGBD adapter       (OpenGL convention; culled vs source mesh variants)
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
`done` only after verification passes and update this index in the same change.
