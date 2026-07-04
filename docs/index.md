# eval3r

`eval3r` is a plain 3D reconstruction evaluation library. It evaluates meshes, point
clouds, depth predictions, and trajectories under **explicit, dataset-aware protocols**.

It focuses on protocol definitions, dataset adapters, prediction manifests, metric
definitions, benchmark orchestration, result schemas, reproducibility records, and reports.
Commodity geometry, camera, trajectory, and IO work is delegated to existing libraries.

`eval3r` is research-oriented: correctness, explicitness, and inspectability outrank
packaging minimalism. Every dependency is required and always installed.

!!! note "Design source of truth"
    While this user documentation is being written incrementally alongside the
    implementation, the authoritative design documents live in `.agent/` in the
    repository (`plan.md`, `schema.md`, `protocols.md`, `datasets.md`, `metrics.md`,
    `backends.md`, `reproducibility.md`).

## Status

This project is under active construction. The command-line interface (`e3r`) currently
exposes its command groups — `metric`, `benchmark`, `dataset`, `protocol`, and `diff` — while
the individual commands are implemented slice by slice (see `.agent/tasks/`).

```bash
e3r --help
```
