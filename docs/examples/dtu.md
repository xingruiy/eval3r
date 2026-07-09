# Example: DTU

DTU's reference is an **independent laser-scanned point cloud** — one of the strongest
ground truths eval3r supports. Native units are millimeters; eval3r normalizes to meters
internally and records the source unit.

## Dataset layout

```text
<root>/Points/stl/stl024_total.ply    laser-scan GT point cloud (mm)
       (or <root>/groundtruth/stl024_total.ply — real releases vary)
<root>/ObsMask/ObsMask24_10.mat       observability mask (object volume)
<root>/ObsMask/Plane24.mat            background-plane cull (may be missing per scan)
<root>/splits/test.txt                scan-id list, one per line (e.g. "scan24")
```

Missing Plane files are recorded explicitly in scene metadata — never silently ignored,
and never skipped while claiming the native DTU protocol was followed.

## Predictions

Conventional DTU filenames, with the light-condition suffix (`l3` = all lights on):

```text
preds/mvsnet024_l3.ply
preds/mvsnet037_l3.ply
```

or explicit paths via a [manifest](../prediction_format.md). Predictions are point clouds
in the GT frame; DTU-native millimeter scale is handled by the adapter.

## Run

```bash
e3r dataset inspect dtu --root /data/DTU --split test

e3r benchmark validate preds/ --dataset dtu --split test \
  --protocol dtu_native_pointcloud --root /data/DTU

e3r benchmark run preds/ --dataset dtu --split test \
  --protocol dtu_native_pointcloud --root /data/DTU \
  --method mvsnet --out runs/dtu_mvsnet
```

## What the protocol does

`dtu_native_pointcloud` (fidelity **native**) runs a validated Python port
of the official DTU MATLAB evaluation: ObsMask culling for accuracy, Plane +
observability culling for completeness, and the official downsampling behavior. It
reports `accuracy` / `completeness` / `overall` (mean distances, in DTU's conventional
millimeters). The port was validated against the reference implementation on real DTU
data (scan 24: accuracy 0.343 / completeness 0.248 / overall 0.295 mm). The MATLAB
official script remains available as an optional external path; which evaluator ran is
recorded in result metadata.
