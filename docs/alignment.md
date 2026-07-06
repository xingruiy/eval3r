# Alignment

Predictions often live in a different coordinate frame — and sometimes a different
scale — than the ground truth. eval3r's alignment module estimates an SE3 (rigid)
or Sim3 (rigid + uniform scale) transform mapping the prediction onto the GT, and
**always gives you a way to look at the result**: a residual number alone does not
show a flipped, mirrored, or locally-stuck registration.

## Two estimation paths — nothing else

| Path | When to use it |
|---|---|
| **Closest-point ICP** (`--solver icp`) | The geometries overlap and a coarse centroid init is good enough. Open3D point-to-point ICP; Sim3 uses scaled point-to-point estimation with an RMS-radius scale init. |
| **Trajectory-first** (`--solver trajectory`) | You have the predicted and GT camera trajectories (TUM format). They are timestamp-associated (evo) and Umeyama-aligned; the resulting transform is propagated to the geometry. |

There is deliberately **no feature-based global registration** (FPFH etc.). If ICP
from the centroid init cannot find your alignment, the overlays will show it —
supply a better prealignment or use the trajectory path.

## `e3r align`

```bash
# ICP, with scale correction; 'auto' = 5% of the GT bbox diagonal (echoed + recorded)
e3r align pred.ply --gt gt_scan.ply --mode sim3 --max-corr-dist auto --out runs/align1

# explicit correspondence distance in metres
e3r align pred.ply --gt gt_scan.ply --mode se3 --max-corr-dist 0.05 --out runs/align2

# trajectory-first: align the camera trajectories, propagate to the geometry
e3r align pred.ply --gt gt_scan.ply --solver trajectory \
  --pred-trajectory pred_traj.txt --gt-trajectory gt_traj.txt \
  --associate-max-diff 0.01 --out runs/align3
```

Required, never defaulted: `--max-corr-dist` for ICP, `--associate-max-diff` for
the trajectory solver. `--mode icp` does not exist — the mode states the transform
class (`se3` or `sim3`) so scale handling is always explicit.

Every run writes:

```text
alignment.json               transform, solver, every shaping parameter, backend versions
pred_aligned.ply             the transformed prediction
alignment_before.ply         pred (red) + gt (blue) overlay before alignment
alignment_after.ply          the same overlay after alignment
alignment_projections.png    before/after x XY/XZ/YZ orthographic views, annotated
alignment_vis.json           colors, subsample seed/counts, the transform shown
```

Open the PLYs in MeshLab or CloudCompare and check the after-overlay: the red and
blue clouds should interleave everywhere, not just where the residual was measured.

The same entry point exists in Python:

```python
from eval3r import align_geometries

out = align_geometries(
    "pred.ply", "gt_scan.ply",
    out_dir="runs/align1", mode="sim3", solver="icp", max_corr_dist="auto",
)
print(out.alignment["scale"], out.alignment["residual_rmse"])
```

## Alignment inside evaluation runs

Protocols request geometry alignment through their `alignment` spec
(see the semantics table in `.agent/schema.md`):

```yaml
alignment:
  mode: sim3                 # or se3
  estimate_on: pointcloud    # or trajectory
  solver: icp                # or umeyama (trajectory-first when estimate_on: trajectory)
  parameters:
    max_correspondence_distance: 0.05   # ICP: required, metres
    # associate_max_diff: 0.01          # trajectory: required, seconds
```

For trajectory-first alignment in a benchmark, the prediction trajectory comes
from the manifest entry (`trajectory:`) and the GT trajectory from the dataset
adapter; a missing one is an explicit per-scene failure at stage `align`.

Whenever a non-`none` alignment actually runs, the run directory automatically
receives the same overlays under `debug/` (`<scene>_alignment_before.ply`, …,
`debug/alignment_vis.json`), and the applied transform is recorded in
`alignment_transforms.json` and in the `registration` / `trajectory` backend
versions.

Two guards protect metric comparability:

- **Sim3 on metric-scale protocols is refused** unless the protocol explicitly
  allows it (`alignment.parameters.allow_sim3: true` or `allow_override: true`) —
  rescaling the prediction changes what the metrics mean.
- **ICP never runs silently**: a protocol must select `solver: icp`, and its
  correspondence distance must be pinned in the protocol.
