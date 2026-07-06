# Result schema

All external inputs and outputs are validated Pydantic models. `results.json` is
self-describing: it carries the metric definitions, policies, and provenance actually
used, so a result file can be interpreted without the original protocol file.

## Run directory

Each run writes a directory:

```text
runs/2026-07-04_153000_dtu_mymethod/
  results.json               the full RunResult (always)
  results.csv                aggregate metrics (always)
  per_scene.csv              per-scene metrics + status/coverage columns (always)
  failures.json              structured scene failures (when any)
  protocol.yaml              copy of the resolved protocol (reporting-controlled)
  manifest.yaml              copy or inferred prediction manifest (benchmark runs)
  config.yaml                resolved configuration incl. recorded overrides
  environment.json           minimal, non-identifying environment facts
  backend_versions.json      every backend used, with versions and parameters
  alignment_transforms.json  per-scene/per-granularity alignment records (when alignment ran)
  logs.txt
  results.md / results.tex / report.html    when reporting.formats requests them
  debug/                     when the protocol requests debug outputs
    debug_outputs.json       parameters that shaped them (colormap, vmax, bins, counts)
    <scene>_error.ply        prediction points colored by pred→gt distance
    <scene>_histogram.png    distance histograms, both directions
```

## `results.json` (RunResult)

Key fields (see `e3r protocol show` and the repository's `.agent/schema.md` for the full
model):

```text
schema_version, eval3r_version, timestamp, command
method, method_version
dataset {dataset, variant, split, version, fingerprint}
protocol, protocol_version, protocol_hash, fidelity
ground_truth {modality, provenance, independence, density, fingerprint, unit, ...}
local_evaluation {status, reason, public_gt_available, official_server_required}
n_scenes_expected, n_scenes_evaluated, failed_scenes[], failure_policy
metrics {name: value}                    aggregates
metric_definitions[]                     the MetricSpecs actually used
per_scene_metrics[]                      every per-scene MetricResult
alignment, masking, sampling, confidence_policy, aggregation
uses_gt {pose, intrinsics, scale}        from the manifest
backend_versions {kind: {name, version, ...}}
environment, git_commit (optional), manifest_path, protocol_path
```

Each per-scene `MetricResult` carries its own context: value, unit, threshold, statistic,
scene/frame IDs, protocol hash, backend, `n_points_pred` / `n_points_gt` /
`n_pixels_valid`, `valid_fraction`, `culled_fraction`, and metric-specific metadata
(e.g. depth scale-alignment mode and estimated scale, pose association counts).

## Schema versioning

`schema_version` is required in `results.json`. Result fields are never renamed or
removed without a compatibility note; new optional fields get defaults; new required
fields bump the schema version. Protocol versions are independent — changing a protocol
threshold bumps the protocol version and hash, not the result schema.

## Run diff

`diff_runs(run_a, run_b, *, loose=False)` compares two written run directories and
returns a `RunDiff`; strict mode (default) raises `RunComparisonError` when protocol
hashes differ.

```python
class RunDiff(BaseModel):
    strict: bool                       # False marks an explicit --loose comparison
    run_a: RunIdentity                 # method, dataset, protocol + hash, fidelity, coverage
    run_b: RunIdentity
    warnings: list[DiffWarning]        # field, reason, value_a, value_b
    metrics: list[MetricDelta]         # name, value_a, value_b, delta = B - A
    per_scene: dict[str, list[MetricDelta]]
    scenes_only_in_a: list[str]
    scenes_only_in_b: list[str]
```

Warnings fire on the comparability triggers listed in
[Reproducibility](reproducibility.md#report-comparability): protocol hash (loose mode
only), GT provenance/independence, local evaluation status, scene coverage, failure
policy, alignment mode, confidence policy, sampling counts, and backend officialness.
