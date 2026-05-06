## Metric Computation

Metric computation is enabled through the `e3r metric` command. The predicted geometry is compared against the ground truth using nearest-neighbor distance computations to derive metrics such as Chamfer distance, accuracy, completeness, precision, recall, and F-score.

## Align-then-evaluate Paradigm

Modern 3D reconstruction networks often predict geometry and camera trajectories only **up to an unknown scale**. Even methods that claim to predict metric-scale geometry can still have small scale, rotation, or translation errors. Because of this, directly comparing a predicted mesh or point cloud against the ground-truth geometry may give misleadingly poor results.

`eval3r` provides explicit alignment modes before computing geometry metrics. The alignment step transforms the predicted geometry into the ground-truth coordinate system, after which metrics such as Chamfer distance, accuracy, completeness, precision, recall, and F-score are computed.

The general evaluation pipeline is:

```text
prediction geometry
        ↓
sample points
        ↓
optional alignment to ground truth
        ↓
nearest-neighbor distance computation
        ↓
Chamfer / accuracy / completeness / F-score
````

Suppose we have two point clouds or meshes:

```text
pred.ply
gt.ply
```

A direct evaluation without alignment can be run with:

```bash
e3r metric all pred.ply --gt gt.ply
```

Example output:

```text
            eval3r — geometry metrics            
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ metric          ┃ value                       ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ chamfer         │ 2.237587                    │
│ chamfer_variant │ l1_mean_bidirectional       │
│ accuracy        │ 1.767983                    │
│ completeness    │ 2.707190                    │
│ f-score @ 0.05  │ 0.0136  (P=0.0137 R=0.0134) │
│ samples         │ 200000                      │
│ seed            │ 42                          │
│ sample_method   │ area                        │
│ align_mode      │ none                        │
└─────────────────┴─────────────────────────────┘
```

In this case, `align_mode` is `none`, so the prediction and ground truth are compared in their original coordinate systems. If the prediction is shifted, rotated, or scaled relative to the ground truth, the metric values will be dominated by the coordinate mismatch rather than the actual reconstruction quality.

The two geometries can also be visualized in the same coordinate system using the `--debug-plot` option:

![](images/init.png)

Without alignment, the predicted and ground-truth shapes may appear far apart or differently scaled. This is usually not a meaningful way to evaluate feedforward reconstruction methods unless the method is expected to recover the exact global coordinate frame.

### ICP Alignment

A common way to evaluate geometric quality is to align the predicted geometry to the ground truth using **Iterative Closest Point**, or ICP.

ICP estimates a rigid transformation between the predicted and ground-truth point clouds. This usually includes rotation and translation, but not scale. It is useful when the prediction has the correct metric scale but is not perfectly registered to the ground-truth coordinate system.

Run ICP-aligned evaluation with:

```bash
e3r metric all pred.ply --gt gt.ply --align icp
```

Example output:

```text
            eval3r — geometry metrics            
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ metric          ┃ value                       ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ chamfer         │ 0.256818                    │
│ chamfer_variant │ l1_mean_bidirectional       │
│ accuracy        │ 0.266786                    │
│ completeness    │ 0.246850                    │
│ f-score @ 0.05  │ 0.1197  (P=0.1152 R=0.1245) │
│ samples         │ 200000                      │
│ seed            │ 42                          │
│ sample_method   │ area                        │
│ align_mode      │ icp                         │
│ align_scale     │ 1.000000                    │
└─────────────────┴─────────────────────────────┘
```

The much lower Chamfer distance after ICP alignment indicates that a large part of the original error came from global misalignment rather than local reconstruction quality.

![](images/icp.png)

ICP alignment is suitable when:

```text
The predicted geometry is already roughly in the correct scale.
The predicted and ground-truth geometries have enough overlap.
The initialization is reasonably close.
The prediction does not contain too many severe outliers.
```

However, ICP is not guaranteed to find the correct alignment. It is a local optimization method and can fail when the initial geometry is too far from the ground truth or when the scene has repeated structures.

### Sim(3) Alignment

For many feedforward 3D reconstruction models, the predicted geometry may be correct up to a **similarity transformation**. A similarity transformation, or Sim(3), includes:

```text
scale
rotation
translation
```

Even models that predict metric scale may produce slightly inaccurate scale estimates. In that case, rigid ICP alignment may still leave residual scale error. `eval3r` therefore also supports Sim(3) alignment.

Run Sim(3)-aligned evaluation with:

```bash
e3r metric all pred.ply --gt gt.ply --align sim3
```

Example output:

```text
            eval3r — geometry metrics            
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ metric          ┃ value                       ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ chamfer         │ 0.263353                    │
│ chamfer_variant │ l1_mean_bidirectional       │
│ accuracy        │ 0.267911                    │
│ completeness    │ 0.258795                    │
│ f-score @ 0.05  │ 0.1171  (P=0.1122 R=0.1224) │
│ samples         │ 200000                      │
│ seed            │ 42                          │
│ sample_method   │ area                        │
│ align_mode      │ sim3                        │
│ align_scale     │ 0.970263                    │
└─────────────────┴─────────────────────────────┘
```

Here, `align_scale` is `0.970263`, meaning the predicted geometry was rescaled during alignment before metric computation.

![](images/sim3.png)

Sim(3) alignment is useful when:

```text
The method predicts geometry up to scale.
The method is monocular or feedforward.
The global metric scale is unreliable.
You want to evaluate shape quality rather than absolute scale recovery.
```

However, Sim(3)-aligned results should be reported carefully. If the task requires metric reconstruction, then scale-aligned metrics may overestimate practical performance. In that case, it is often useful to report both:

```text
without alignment: evaluates metric/global-frame correctness
with Sim(3) alignment: evaluates geometric shape quality
```

### Choosing an Alignment Mode

`eval3r` supports several alignment modes:

```text
none
  No alignment. The prediction and ground truth are compared directly.

icp
  Rigid ICP alignment. Estimates rotation and translation, but keeps scale fixed.

sim3
  Similarity alignment. Estimates scale, rotation, and translation.
```

A practical recommendation:

```text
Use --align none when evaluating metric-scale reconstruction or global-frame accuracy.

Use --align icp when the method should recover scale but may have small pose-frame registration errors.

Use --align sim3 when evaluating monocular or feedforward predictions that are only defined up to scale.
```

For papers, it is best to explicitly report the alignment protocol. For example:

```text
We report Chamfer distance and F-score after Sim(3) alignment between the predicted and ground-truth point clouds.
```

or:

```text
We report metrics without alignment to evaluate metric-scale reconstruction accuracy.
```

Avoid reporting aligned metrics without saying which alignment was used.

### Debugging Alignment

Alignment can be time-consuming, especially for large meshes or dense point clouds. It is also not perfect. It requires a reasonably good reconstruction to start with and may fail when:

```text
the prediction has too many outliers
the prediction and ground truth have little overlap
the initial pose is very poor
the scene has repeated structures
the sampled points are too sparse
the geometry is incomplete
```

To inspect the alignment quality, use the debug visualization option:

```bash
e3r metric all pred.ply --gt gt.ply --align icp --debug-plot
```

or:

```bash
e3r metric all pred.ply --gt gt.ply --align sim3 --debug-plot
```

The debug plot should be checked before trusting the numbers. Good metrics after a failed alignment are not meaningful.

A recommended workflow is:

```bash
# 1. Direct comparison
e3r metric all pred.ply --gt gt.ply --debug-plot

# 2. Rigid alignment
e3r metric all pred.ply --gt gt.ply --align icp --debug-plot

# 3. Similarity alignment
e3r metric all pred.ply --gt gt.ply --align sim3 --debug-plot
```

Then compare the results. If `none`, `icp`, and `sim3` give very different numbers, the reconstruction likely has a global-frame or scale issue. If `icp` and `sim3` are similar, the prediction is probably close to the correct scale.

### Important Notes

Alignment changes the interpretation of the metrics.

```text
No alignment:
  Measures reconstruction quality plus global coordinate accuracy.

ICP alignment:
  Measures reconstruction quality after rigid registration.

Sim(3) alignment:
  Measures reconstruction quality after scale, rotation, and translation correction.
```

Therefore, `eval3r` always records the alignment mode in the metric output:

```text
align_mode
align_scale
```

This makes results easier to reproduce and avoids mixing incompatible evaluation protocols.