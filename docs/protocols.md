# Protocols

Protocols define how evaluation is performed. They are the main unit of comparability in
eval3r: **a metric number is not meaningful without the protocol that produced it**, and
two numbers are only comparable when their protocol hashes match.

A protocol is not just a list of metrics. Every protocol explicitly pins:

```text
dataset variant
ground-truth provenance, independence, density
local evaluation status
prediction modality
alignment (mode, what it is estimated on, solver, granularity)
masking and culling
confidence policy
sampling (method, count, seed policy — per side)
metric definitions (statistics, thresholds, reductions, aggregation order)
aggregation
failure policy
reporting outputs
backend preferences
```

Metrics never silently choose alignment, scale handling, thresholds, masking, culling,
sampling counts or seeds, confidence thresholds, or aggregation order. If a protocol
requires something (a mask file, a depth unit) and it is missing, the scene fails under
the protocol's failure policy — there is no silent fallback.

## Built-in protocols

```bash
e3r protocol list
e3r protocol show <name>
```

| Protocol | Fidelity | Purpose |
|---|---|---|
| `single_geometry` | eval3r_native | two-file point-cloud/mesh comparison (debugging, quick checks) |
| `single_depth` | eval3r_native | one depth file pair or two frame directories |
| `single_pose` | eval3r_native | one TUM trajectory pair (evo ATE/RPE, Sim3 by default) |
| `dtu_official_like_pointcloud` | official_like | DTU test split with ObsMask/Plane culling, mm-native GT |
| `scannet_single_layer_geometry_5cm` | eval3r_native | community single-layer 5 cm F-score convention, val split |
| `scannet_double_layer_geometry_5cm` | eval3r_native | double-layer variant of the above |
| `scannet_test_single_layer_geometry_5cm` | eval3r_native | test split with protocol-defined visibility culling |
| `tanks_temples_training_official` | official | wraps the official toolbox on public-GT training scenes |
| `tanks_temples_intermediate_server_only` | server_only | stub; local evaluation is refused |
| `eth3d_training_official` | official | wraps the official multi-view-evaluation binary (1–50 cm tolerances) |

ScanNet protocols are `eval3r_native` because ScanNet has no official
geometry-reconstruction benchmark: they follow the community single-/double-layer 5 cm
convention, and results must never be presented as official ScanNet benchmark numbers.

## Canonical protocol hashing

Every protocol has a canonical hash: the YAML is parsed and validated into the
`EvalProtocol` model, serialized to canonical JSON with sorted keys, and sha256-hashed
(`sha256:...`). The hash is stable across YAML formatting and comments, and changes
whenever any behavior-affecting field changes — dataset variant, GT spec, alignment,
masking, confidence, sampling, metrics, aggregation, failure policy, or backend
preferences that affect semantics.

Report-only preferences (the `reporting` block: output formats, debug outputs) are
explicitly excluded — turning on an HTML report never changes a protocol hash.

Every result carries its protocol hash, and `e3r diff` refuses strict comparisons across
differing hashes.

## Overrides

Some protocols allow CLI/API overrides for quick experiments (`--threshold`, `--sample`,
`--align`, ...). Overrides are always recorded in `config.yaml` and `results.json`, and
they change the effective protocol hash — an overridden run is never silently comparable
to a canonical one. Official protocols generally disallow overrides.

## Custom protocols

Any command that takes `--protocol` accepts a path to your own protocol YAML. The file
must validate against the full protocol schema (every section above is required — the
explicitness is the point). Start from a built-in:

```bash
e3r protocol show single_geometry   # prints the resolved fields
```

and see the protocol model in [Result schema](schema.md) /
[`.agent/protocols.md`](https://github.com/xingruiy/eval3r/blob/main/.agent/protocols.md)
in the repository for full annotated examples. Changing protocol behavior should bump the
protocol's `protocol_version`; the hash changes automatically.

## Naming

Built-in names are descriptive:
`<dataset>_<variant>_<metric_family>_<important_threshold_or_convention>` — e.g.
`scannet_single_layer_geometry_5cm`, never `scannet_geometry`. Names must not hide
important protocol choices.
