# Quickstart

Every command below echoes its resolved configuration before running — protocol name and
hash, dataset and variant, the alignment / masking / sampling / confidence / failure
policies actually in effect, and the input and output paths verbatim — and writes a
complete [run directory](schema.md#run-directory).

## Single-file geometry

```bash
e3r metric geometry pred.ply \
  --gt gt.ply \
  --threshold 0.05 \
  --input pointcloud \
  --gt-input pointcloud
```

Meshes are sampled before distance metrics (never raw vertices unless a protocol says
so):

```bash
e3r metric geometry pred_mesh.ply \
  --gt gt_mesh.ply \
  --input mesh --gt-input mesh \
  --sample 200000 --threshold 0.05
```

This uses the `single_geometry` protocol by default; `--threshold` / `--sample` are
recorded as protocol overrides (they change the protocol hash, so overridden runs are
never silently comparable to un-overridden ones). Prediction adaptation is separate:
`--as opengl@sim3` records how the prediction entered the protocol's frame without
minting a new protocol identity.

## Single-file depth

```bash
e3r metric depth pred_depth.png \
  --gt gt_depth.png \
  --depth-unit 0.001 --gt-depth-unit 0.001 \
  --align scale_median --align-granularity per_frame
```

`--depth-unit` is metres per stored unit and is **required** for integer depth files
(e.g. `0.001` for millimetre PNGs). Passing two directories instead of two files
evaluates a depth *sequence*, matched frame-by-frame by filename stem and aggregated per
frame — never fused into scene geometry. Scale alignment
(`none | scale_median | scale_least_squares | scale_affine`) and its granularity are
always explicit and recorded. Alignment mode can also be supplied through the common
adaptation grammar, for example `--as relative@scale_median`; the run writes an
`adaptation` record to `results.json` and `config.yaml`.

## Single-file pose

```bash
e3r metric pose pred_tum.txt \
  --gt gt_tum.txt \
  --align sim3
```

Trajectories are TUM format (`timestamp x y z qx qy qz qw`); metrics are delegated to
[evo](https://github.com/MichaelGrupp/evo). ATE / RPE statistics, the association policy
and tolerance, the alignment mode, and the estimated Sim3 scale are all recorded.

If a prediction uses a different pose convention than the protocol's internal one
(camera-to-world, OpenCV axes), declare it and eval3r transforms — and validates — the
trajectory *before* association and alignment:

```bash
e3r metric pose pred_tum.txt --gt gt_tum.txt --align none \
  --pred-pose-convention cam_to_world_opengl   # or world_to_cam_colmap, etc.
```

The compact adaptation grammar is shared by geometry, depth, pose, and benchmark
runs. Tokens are order-independent, so `--as wc@opengl@trajectory_sim3` and
`--adapt trajectory_sim3@opengl@wc` are equivalent.

For geometry, a prediction built in an OpenGL world frame is rotated into the internal
frame before metrics with `--pred-world-frame opengl`. The transform is a proper
rotation (never a reflection), validated on every call, and recorded in the run; the
`SourcePoseFormat` / `world_frame` fields written by `PredictionWriter` are picked up
automatically. Unverified conventions (`unknown`, `co3d_frame_annotations`,
`tanks_temples_log`) raise rather than guess a handedness.

## Dataset benchmarks

```bash
e3r dataset list
e3r dataset inspect dtu --root /data/DTU --split test

e3r benchmark validate preds/ --dataset dtu --split test \
  --protocol dtu_official_like_pointcloud --root /data/DTU

e3r benchmark run preds/ --dataset dtu --split test \
  --protocol dtu_official_like_pointcloud --root /data/DTU
```

`benchmark validate` checks preflight and per-scene prediction resolution without
evaluating. Runs are refused before any computation when the split is not locally
evaluable (e.g. Tanks and Temples intermediate, ETH3D test — those are server-only).
See the worked examples: [DTU](examples/dtu.md), [ScanNet](examples/scannet.md),
[Tanks and Temples](examples/tanks_temples.md), [ETH3D](examples/eth3d.md).
Official and official-like protocols declare restrictive adaptation envelopes. A
manifest with `scale: relative` under such a protocol is refused with an actionable
message instead of being silently rescaled.

## Protocols

```bash
e3r protocol list
e3r protocol show scannet_single_layer_geometry_5cm
```

`protocol show` prints the resolved fields and the canonical protocol hash that every
result carries. See [Protocols](protocols.md).

## Diffing runs

```bash
e3r diff runs/method_a runs/method_b
```

Strict by default: runs with different protocol hashes are refused with the full reason.
`--loose` allows the comparison but labels the output NON-STRICT, and comparability
warnings (differing GT provenance, coverage, failure policy, alignment, confidence,
adaptation, sampling, backend officialness) are always printed. See
[Reproducibility](reproducibility.md#report-comparability).

## Python API

The stable public functions live at the package top level.

### `evaluate_geometry`

```python
from eval3r import evaluate_geometry

result = evaluate_geometry(
    "pred.ply", gt="gt.ply",
    input_type="mesh", gt_type="mesh",   # or "pointcloud" (default)
    threshold=0.05, sample=200_000,       # recorded protocol overrides
    adapt="opengl@sim3",                  # optional non-hashed prediction adaptation
    protocol="single_geometry",           # built-in name or a protocol YAML path
    out_dir="runs/my_run",                # optional: write a full run directory
)
print(result.metrics)          # aggregate metrics
print(result.protocol_hash)    # carried by every result
```

Returns a `RunResult` (the same Pydantic model serialized to `results.json`).

### `evaluate_depth`

```python
from eval3r import evaluate_depth

result = evaluate_depth(
    "pred_depth.png", gt="gt_depth.png",
    depth_unit=0.001, gt_depth_unit=0.001,   # metres per stored unit (required for ints)
    align="scale_median", align_granularity="per_frame",  # align is adaptation
    protocol="single_depth",
)
```

`pred`/`gt` may also be two directories of frames (a depth sequence).

### `evaluate_pose`

```python
from eval3r import evaluate_pose

result = evaluate_pose(
    "pred_tum.txt", gt="gt_tum.txt",
    align="sim3",                 # none | se3 | sim3 (trajectory_* accepted)
    adapt="wc@opengl@trajectory_sim3",
    associate_max_diff=0.01,      # timestamp association tolerance in seconds
    protocol="single_pose",
)
```

### `run_benchmark`

```python
from eval3r import run_benchmark

result = run_benchmark(
    "preds/",
    dataset="dtu", split="test",
    protocol="dtu_official_like_pointcloud",
    root="/data/DTU",              # dataset root the adapter needs (GT files)
    manifest=None,                 # optional manifest YAML; else resolved or inferred
    adapt=None,                    # optional --as/--adapt grammar
    out_dir="runs/dtu_mymethod",
)
```

### `load_protocol`

```python
from eval3r import load_protocol

protocol = load_protocol("dtu_official_like_pointcloud")  # or a YAML path
print(protocol.name, protocol.fidelity)
```

Returns the validated `EvalProtocol` model.

### `diff_runs`

```python
from eval3r import diff_runs

diff = diff_runs("runs/method_a", "runs/method_b")   # raises RunComparisonError on
                                                     # mismatched protocol hashes
diff = diff_runs("runs/method_a", "runs/method_b", loose=True)  # labeled non-strict
for w in diff.warnings:
    print(w.field, w.reason)
for m in diff.metrics:
    print(m.name, m.value_a, m.value_b, m.delta)     # delta = B - A
```

Returns a `RunDiff` (see [Result schema](schema.md#run-diff)).
