# eval3r

`eval3r` is a plain 3D reconstruction evaluation library. It evaluates meshes, point
clouds, depth predictions, and camera trajectories under **explicit, dataset-aware
protocols**.

It focuses on protocol definitions, dataset adapters, prediction manifests, metric
definitions, benchmark orchestration, result schemas, reproducibility records, and
reports. Commodity geometry, camera, trajectory, and IO work is delegated to existing
libraries; official benchmark behavior is wrapped, never faked.

`eval3r` is research-oriented: correctness, explicitness, and inspectability outrank
packaging minimalism. Every dependency is required and always installed. Silent, terse,
or lossy output is treated as a defect.

## What eval3r is not

There is no TSDF integration, RGB-D fusion, volumetric fusion, online mapping, SLAM
tracking, or learned reconstruction inside eval3r. Depth sequences are evaluated per
frame; they are never turned into scene reconstructions. (One narrow exception exists:
protocol-requested visibility culling for the community ScanNet convention, which trims
the caller's prediction to the observed region and records everything it did.)

## Implemented surface

The command-line interface (`e3r`):

- `e3r metric geometry|depth|pose` — single-file evaluation under explicit protocols
- `e3r benchmark run|validate` — dataset-split evaluation (DTU, ScanNet, Tanks and
  Temples, ETH3D)
- `e3r dataset list|inspect` — adapter capabilities and scene discovery
- `e3r protocol list|show` — protocol inspection with canonical hashes
- `e3r diff` — compare two run directories; refuses mismatched protocol hashes unless
  `--loose` is passed, and then labels the output NON-STRICT

The Python API: [`evaluate_geometry`, `evaluate_depth`, `evaluate_pose`, `run_benchmark`,
`load_protocol`, `diff_runs`](quickstart.md#python-api).

Run directories can include Markdown / LaTeX / HTML reports (protocol
`reporting.formats`) and debug outputs (error-colored point clouds, distance
histograms) when the protocol requests them. Every report format shows scene
coverage, failure reasons, and a partial-coverage banner when aggregates are partial.

## Stability

`eval3r` is pre-1.0. The supported public surface is:

- the CLI commands above,
- the Python API functions above,
- the protocol YAML schema and its canonical hashing,
- the `results.json` result schema (versioned via `schema_version`).

Internal registries for dataset adapters and backends exist, but **third-party plugin
entry points are not yet a public compatibility contract**. Registering your own adapter
or backend against the internal interfaces may break between minor versions; a plugin API
will be stabilized once more built-in adapters have proven the interface.

## Documentation

- [Installation](install.md) — package install, external official tools
- [Quickstart](quickstart.md) — CLI and Python API
- [Prediction format](prediction_format.md) — manifests, conventions, file layouts
- [Protocols](protocols.md) — what a protocol pins, built-ins, hashing
- [Fidelity](fidelity.md) — official / official_like / eval3r_native / server_only
- [Datasets](datasets.md) — adapters, capabilities, ground-truth honesty
- [Metrics](metrics.md) — geometry, depth, and pose metric definitions
- [Backends](backends.md) — delegation policy and version recording
- [Confidence](confidence.md) — confidence filtering policies
- [Failure policy](failure_policy.md) — failure accounting and partial coverage
- [Result schema](schema.md) — run directories, `results.json`, run diffs
- [Reproducibility](reproducibility.md) — what is recorded and why
- Examples: [DTU](examples/dtu.md), [ScanNet](examples/scannet.md),
  [Tanks and Temples](examples/tanks_temples.md), [ETH3D](examples/eth3d.md)
