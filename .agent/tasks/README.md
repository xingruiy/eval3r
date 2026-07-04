# eval3r Task Index

Canonical index of implementation task slices. Statuses: `todo`, `in_progress`, `done`, `blocked`.

Resume rule: start from the first non-`done` task below. Before resuming any task, re-read the
relevant `.agent/*.md` docs and the actual code — task notes can be stale.

## Ordering and status

| # | Task file | Slice | Status |
|---|-----------|-------|--------|
| 001 | [001-repo-foundation.md](001-repo-foundation.md) | Package skeleton, pyproject, CI, lint, test, docs build, `e3r --help` | todo |
| 002 | [002-core-schema.md](002-core-schema.md) | Pydantic models from `.agent/schema.md`, serialization tests, fixtures | todo |
| 003 | [003-protocol-loader-hashing.md](003-protocol-loader-hashing.md) | Protocol YAML loader, canonical hashing, registry, built-in YAMLs | todo |
| 004 | [004-backend-registry.md](004-backend-registry.md) | Backend registry, optional-dep errors, scipy/trimesh/plyfile backends | todo |
| 005 | [005-geometry-metrics.md](005-geometry-metrics.md) | Point-set geometry metrics with synthetic analytic tests | todo |
| 006 | [006-runner-result-writer.md](006-runner-result-writer.md) | Staged pipeline runner, run directory writer, `e3r metric geometry` | todo |
| 007 | [007-dtu-adapter.md](007-dtu-adapter.md) | DTU adapter, ObsMask/Plane, dtu_eval backend, protocol, fixture | todo |
| 008 | [008-scannet-adapter.md](008-scannet-adapter.md) | ScanNet adapter, mesh variants, culling, `e3r benchmark run` integration | todo |
| 009 | [009-tnt-official-wrapper.md](009-tnt-official-wrapper.md) | Tanks and Temples adapter + official script wrapper | todo |
| 010 | [010-eth3d-adapter.md](010-eth3d-adapter.md) | ETH3D adapter, pycolmap cameras, official-tolerance validation | todo |
| 011 | [011-depth-metrics.md](011-depth-metrics.md) | Depth IO, masks, scale alignment modes, `e3r metric depth` | todo |
| 012 | [012-pose-metrics.md](012-pose-metrics.md) | evo backend, ATE/RPE, `e3r metric pose` | todo |
| 013 | [013-reports-diffing.md](013-reports-diffing.md) | Markdown/LaTeX/HTML reports, partial-coverage banners, `e3r diff` | todo |
| 014 | [014-plugin-api-release.md](014-plugin-api-release.md) | Plugin API, examples, PyPI release | todo |

**Next task: 001**

Tasks must be executed in order unless a task's Scope says otherwise; each depends on the
deliverables of the previous ones (schemas before adapters, metrics before runners,
single-file evaluation before dataset benchmarks).

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
Waymo adapter             (optional heavy backend; pose + sparse LiDAR)
```

## Conventions for task files

Every task file has: Goal, Scope, Out of Scope, Relevant Files, Plan, Findings, Decisions,
Verification, Status. Mark the active task `in_progress` before editing code; record durable
findings and decisions (not transcripts); record verification commands and outcomes; mark
`done` only after verification passes and update this index in the same change.
