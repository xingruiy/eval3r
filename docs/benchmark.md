# Benchmark CLI

`e3r benchmark` runs a method's predictions against a full dataset split and
produces aggregate metrics (Chamfer, accuracy, completeness, F-score) plus a
per-scene CSV/JSON breakdown. It is the multi-scene counterpart of
[`e3r metric`](metric.md).

```text
e3r benchmark <preds_root> [--dataset <name>] --root <gt_root> [...options]
```

The first positional is **always** the predictions root — one subdirectory per
scene (or any structure the prediction locator can resolve, e.g. a manifest).

## Two modes

`e3r benchmark` decides what ground truth to load based on whether
`--dataset` is given:

| Mode               | Trigger                          | GT layout        | Thresholds & sampling |
| ------------------ | -------------------------------- | ---------------- | --------------------- |
| Registered dataset | `--dataset <name>`               | adapter class    | dataset preset        |
| Manual             | omit `--dataset`, pass `--gt-path` | path template  | required on the CLI   |

## Mode 1: registered dataset

Use this when the data sits in one of the layouts an adapter already
understands (`scannet`, `replica`, `dtu`, `eth3d`, `tum_rgbd`,
`tanks_temples`). The adapter handles file discovery; the matching
[preset](#presets) supplies sensible defaults.

```bash
e3r benchmark outputs/scannet \
    --dataset scannet \
    --root /data/scannet \
    --split /data/scannet/splits/scannetv2_test.txt \
    --thresholds 0.05 --workers 8 \
    --out results.json --csv results.csv
```

To reshape an adapter's expected layout (different filenames or subdirs), pass
`-o key=value` overrides. Each adapter exposes its own knobs — see
`e3r dataset show <name>` for the list.

```bash
e3r benchmark outputs/eth3d --dataset eth3d \
    --root /data/eth3d -o track=dslr

e3r benchmark outputs/tum --dataset tum_rgbd \
    --root /data/tum -o intrinsics_fx=535.4 -o intrinsics_cx=320.1
```

## Mode 2: manual layout (no adapter)

Use this when your data does not match any registered adapter. Omit
`--dataset` and point at the ground truth via a `{scene_id}` template:

```bash
e3r benchmark outputs/mine \
    --root /data/mydataset \
    --gt-path '{scene_id}/gt.ply' \
    --scenes-file splits/val.txt \
    --thresholds 0.05 --align none --unit m
```

The GT file may be a mesh (any trimesh-loadable format with faces) or a point cloud — eval3r probes the first scene to decide which.

### Required flags in manual mode

- `--root`: directory the `--gt-path` template is resolved against.
- `--gt-path`: path template with `{scene_id}` substitution.
- A scene source — exactly one of:
  - `--scenes 's1,s2,s3'` (inline comma-separated list)
  - `--scenes-file <path>` (one scene id per line)
  - `--split <path>` (same format as `--scenes-file`)
- `--thresholds`: explicit since no preset is consulted in manual mode
  (e.g. `--thresholds 0.05` for metres, `--thresholds 1.0 2.0 5.0` for mm).

### Caveats

- `GenericAdapter` only exposes ground-truth geometry. Trajectory-based
  alignment modes (`traj_*`) need pose data, which the generic adapter does
  not provide. Pass poses externally with `--pred-pose-dir` if you need
  trajectory alignment, or write a real adapter.
- Adapter `-o` overrides are rejected in manual mode: the generic adapter is
  configured solely through `--gt-path` and the scene source.

## Presets

Presets live in `eval3r/presets/` and ship one entry per registered dataset.
Each is a small dict bundling evaluation policy (thresholds, sampling,
alignment, Chamfer variant, unit), **not** GT layout. Inspect them with:

```bash
e3r preset list
e3r preset show scannet
```

Any preset value can be overridden on the CLI (`--thresholds`,
`--align`, `--samples`, `--seed`, `--chamfer-variant`).

## Output

The benchmark reports two summaries plus coverage:

- `summary` — mean / median / std / n over **successful** scenes only.
- `summary_all` — mean / median / std / n over **all** scenes; missing or
  failed scenes are filled with the configured defaults
  (`--missing-distance-default`, `--missing-fscore-default`).
- `coverage` — counts of `n_total`, `n_evaluated`, `n_missing_pred`,
  `n_missing_gt`, `n_failed`.

Use `--out result.json` for the full machine-readable payload (per-scene
results, paths, errors) and `--csv result.csv` for a flat per-scene table.

## Combining with other features

- **Occlusion masks** — pass `--mask-dir` plus optional path patterns. See
  [Occlusion Masks](masks.md) for the file-format and
  transform conventions.
- **Trajectory alignment** — set `--align traj_sim3` (or similar) and provide
  poses either via the prediction manifest or `--pred-pose-dir`. See
  [Trajectory Formats](trajectory.md) for accepted file formats.
- **Cropping to GT bbox** — `--crop` plus `--crop-margin <metres>`.

## Quick reference

| Flag                          | Purpose                                                   |
| ----------------------------- | --------------------------------------------------------- |
| `--dataset NAME`              | Use a registered adapter. Omit for manual mode.           |
| `--root PATH`                 | Ground-truth filesystem root. Always required.            |
| `--split PATH`                | Scene-id list. Optional for adapters that auto-discover.  |
| `--gt-path TEMPLATE`          | Manual-mode GT template (e.g. `{scene_id}/gt.ply`).       |
| `--scenes "id1,id2"`          | Manual-mode inline scene list.                            |
| `--scenes-file PATH`          | Manual-mode scene file (one id per line).                 |
| `-o key=value`                | Adapter-specific override (registered mode only).         |
| `--thresholds X [Y ...]`      | F-score / precision / recall thresholds.                  |
| `--align {none,sim3,...}`     | Alignment mode applied before metrics.                    |
| `--samples N`, `--seed N`     | Override preset sampling defaults.                        |
| `--mask-dir PATH`             | Root for mask path patterns. See `masks.md`.              |
| `--out result.json`           | Write full per-scene + summary JSON.                      |
| `--csv result.csv`            | Write per-scene CSV.                                      |
| `--json`                      | Emit the JSON payload to stdout (no rich table).          |
| `--workers N`                 | Process workers; defaults to `min(8, ncpu)`.              |
| `--fail-on-missing`           | Hard-error if any scene's prediction can't be located.    |
