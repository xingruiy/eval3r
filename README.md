# eval3r

`eval3r` is a plain 3D reconstruction evaluation library. It evaluates meshes, point
clouds, depth predictions, and camera trajectories against dataset references under
**explicit, dataset-aware protocols**, and records enough metadata for every number to be
understood later.

The central idea:

> `eval3r` loads predictions and references, normalizes dataset conventions, applies an
> explicit protocol, computes metrics, and records what actually happened — including
> what the "ground truth" really is, what was aligned, masked, culled, and sampled, which
> scenes failed and why, and which backend versions produced the numbers.

It is a research tool: correctness, explicitness, and inspectability outrank packaging
minimalism. Silent defaults are treated as defects.

## Installation

```bash
pip install eval3r
```

Until the first PyPI release, install from source:

```bash
git clone https://github.com/xingruiy/eval3r
pip install ./eval3r
```

Requires Python >= 3.10. All dependencies are required and always installed — there is no
optional-extra system. A few **official evaluation tools** are external and user-supplied
(they are not pip packages); see the
[install docs](https://github.com/xingruiy/eval3r/blob/main/docs/install.md) for the Tanks
and Temples toolbox, the ETH3D multi-view-evaluation binary, and the optional DTU MATLAB
path.

## Quick example

Compare a predicted point cloud against a reference:

```bash
e3r metric geometry pred.ply --gt gt.ply --threshold 0.05 --input pointcloud --gt-input pointcloud
```

This runs the `single_geometry` protocol, prints the resolved configuration (protocol name
and hash, alignment / masking / sampling / confidence / failure policies in effect), and
writes a complete run directory (`results.json`, `per_scene.csv`, `protocol.yaml`,
`environment.json`, `backend_versions.json`, ...).

The same thing from Python:

```python
from eval3r import evaluate_geometry

result = evaluate_geometry("pred.ply", gt="gt.ply", threshold=0.05, out_dir="runs/quick")
print(result.metrics)
```

## Supported prediction types

```text
mesh
pointcloud
pointmap                 (explicit 3D points in a declared frame)
single_depth
depth_sequence           (per-frame depth metrics; never fused into scene geometry)
camera_trajectory        (TUM format)
```

`eval3r` never converts depth sequences into scene reconstructions. There is no TSDF
integration, RGB-D fusion, or SLAM tracking as a reconstruction method (a narrowly scoped
visibility-culling exception exists for the community ScanNet evaluation convention and is
always recorded in result metadata).

## Supported dataset adapters

| Dataset | Split | Fidelity | Ground truth | Local evaluation |
|---|---|---|---|---|
| DTU | test | `native` | laser scan (independent) | validated port of the official MATLAB evaluation (ObsMask + Plane); optional MATLAB path |
| Tanks and Temples | training | `official` | laser scan (independent) | wraps the real official toolbox (user-supplied checkout, pinned `open3d==0.9` interpreter) |
| ETH3D high-res DSLR | training | `official` | laser scan (independent) | wraps the real official `multi-view-evaluation` binary (user-supplied build) |
| ScanNet | val / test | `native` | BundleFusion mesh (reconstruction-derived) | community single-/double-layer 5 cm F-score convention; never official numbers |
| custom | — | `native` | user-declared | single-file and simple-layout evaluation |

Tanks and Temples intermediate/advanced and ETH3D test are **server-only**: eval3r refuses
to produce local official-looking numbers for withheld-GT splits.

## Protocol-driven evaluation

A metric number is meaningless without the protocol that produced it. Every protocol pins,
explicitly: dataset variant, ground-truth provenance/independence/density, local
evaluability, alignment, masking and culling, confidence policy, sampling (methods, counts,
seeds), metric definitions (statistics, thresholds, reductions, aggregation order), failure
policy, and backend preferences. Metrics never silently choose any of these.

Each protocol has a canonical **hash** over its behavior-affecting fields. Results carry
the hash; `e3r diff` refuses to compare runs whose hashes differ unless you explicitly pass
`--loose` (and then labels the output NON-STRICT).

```bash
e3r protocol list
e3r protocol show dtu_native_pointcloud
```

Protocols also carry one of three **fidelity** labels: `official` (wraps the real
official tool), `native` (evaluated locally by eval3r, including validated local ports and
eval3r-defined protocols), or `server` (withheld-GT server evaluation only). eval3r never
substitutes a lookalike evaluator for an official one.

## Backend delegation

Commodity work is delegated to mature libraries and recorded in result metadata: trimesh /
Open3D (meshes), plyfile / Open3D (point clouds), scipy (nearest neighbor), evo
(trajectories), pycolmap (COLMAP cameras), imageio / OpenCV (depth IO), and the real
official evaluation tools for DTU / Tanks and Temples / ETH3D. Backend names and versions
are written into every result.

## Example benchmark run

```bash
e3r benchmark run preds/ \
  --dataset dtu \
  --split test \
  --protocol dtu_native_pointcloud \
  --root /data/DTU
```

## Example result table

Per-scene and aggregate tables land in `per_scene.csv`, `results.csv`, and (when the
protocol requests them) `results.md` / `results.tex` / `report.html`:

| scene | accuracy (mm) | completeness (mm) | overall (mm) |
|---|---|---|---|
| scan24 | 0.343 | 0.248 | 0.295 |
| scan37 | failed (load) | — | — |

(The scan24 row is a real number from validating eval3r's DTU port against the official
MATLAB evaluation; the scan37 row illustrates how a failed scene appears.)

Partial coverage is never hidden: if any expected scene failed, every report format shows a
partial-coverage banner, the failure policy, and the per-scene failure reasons.

## Development status

`eval3r` is pre-1.0 (current version 0.5.x). The implemented, tested surface is:

- `e3r metric geometry|depth|pose` — single-file evaluation
- `e3r benchmark run|validate` — dataset-split evaluation (DTU, ScanNet, Tanks and Temples, ETH3D)
- `e3r dataset list|inspect`, `e3r protocol list|show`
- `e3r diff` — strict result comparison with comparability warnings
- Python API: `evaluate_geometry`, `evaluate_depth`, `evaluate_pose`, `run_benchmark`,
  `load_protocol`, `diff_runs`

**Stability:** the CLI commands, the Python API functions above, the protocol YAML schema,
and the `results.json` schema are the supported public surface. Internal registries for
dataset adapters and backends exist, but third-party plugin entry points are **not yet a
public compatibility contract** — they will be stabilized after more built-in adapters
prove the interface.

Documentation lives in
[`docs/`](https://github.com/xingruiy/eval3r/blob/main/docs/index.md) (built with mkdocs).

## Contributing and license

Contributions are welcome. Please read
[`CONTRIBUTING.md`](https://github.com/xingruiy/eval3r/blob/main/CONTRIBUTING.md)
before opening a pull request.

`eval3r` is distributed under the
[`MIT License`](https://github.com/xingruiy/eval3r/blob/main/LICENSE).

## Citation

If eval3r is useful in your research, please cite the software:

```bibtex
@software{eval3r,
  author  = {Yang, Xingrui},
  title   = {eval3r: dataset-aware 3D reconstruction evaluation under explicit protocols},
  url     = {https://github.com/xingruiy/eval3r},
  version = {0.5.0},
  year    = {2026},
}
```

Please also cite the datasets and official evaluation tools you evaluate against (DTU,
Tanks and Temples, ETH3D, ScanNet); eval3r wraps or follows their protocols, it does not
replace them.
