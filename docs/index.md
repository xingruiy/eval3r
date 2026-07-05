# eval3r

`eval3r` is a plain 3D reconstruction evaluation library. It evaluates meshes, point
clouds, depth predictions, and trajectories under **explicit, dataset-aware protocols**.

It focuses on protocol definitions, dataset adapters, prediction manifests, metric
definitions, benchmark orchestration, result schemas, reproducibility records, and reports.
Commodity geometry, camera, trajectory, and IO work is delegated to existing libraries.

`eval3r` is research-oriented: correctness, explicitness, and inspectability outrank
packaging minimalism. Every dependency is required and always installed.

## Status

This project is under active construction (see `.agent/tasks/`). The command-line
interface (`e3r`) implements:

- `e3r metric geometry|depth|pose` — single-file evaluation under explicit protocols
- `e3r benchmark run|validate` — dataset-split evaluation (DTU, ScanNet, Tanks and
  Temples, ETH3D)
- `e3r dataset list|inspect` — adapter capabilities and scene discovery
- `e3r protocol list|show` — protocol inspection with canonical hashes
- `e3r diff` — compare two run directories; refuses mismatched protocol hashes unless
  `--loose` is passed, and then labels the output NON-STRICT

Run directories can include Markdown / LaTeX / HTML reports (protocol
`reporting.formats`) and debug outputs (error-colored point clouds, distance
histograms) when the protocol requests them. Every report format shows scene
coverage, failure reasons, and a partial-coverage banner when aggregates are partial.

```bash
e3r --help
```
