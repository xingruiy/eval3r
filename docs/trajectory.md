# Trajectory Pose File Formats

`eval3r` stores poses internally as `(T, 4, 4)` homogeneous matrices. The pose convention is metadata: loaders do not invert or convert poses silently.

## Auto-detected Text Formats

`load_trajectory_auto(path, convention=...)` reads whitespace-separated `.txt` files and detects the format from the number of columns in the first non-empty, non-comment row.

Blank lines and lines starting with `#` are ignored.

| Columns | Format | Timestamps | Matrix Layout |
|---------|--------|------------|---------------|
| 8 | TUM | yes | `timestamp tx ty tz qx qy qz qw` |
| 13 | timestamp + 3x4 matrix | yes | `timestamp` followed by row-major 3x4 transform |
| 17 | timestamp + 4x4 matrix | yes | `timestamp` followed by row-major 4x4 transform |

Any other column count raises `ValueError`.

Timestamps are stored in the returned `Trajectory` and are used by trajectory alignment to match corresponding frames when the prediction and ground truth have different lengths.

## TUM Format

TUM rows contain a timestamp, translation, and quaternion:

```text
timestamp tx ty tz qx qy qz qw
```

Example:

```text
0.000000 0.0 0.0 0.0 0.0 0.0 0.0 1.0
0.033333 0.1 0.0 0.0 0.0 0.0 0.0 1.0
```

The quaternion order is `qx qy qz qw`.

## Timestamped Matrix Formats

A timestamped 3x4 row contains one timestamp followed by the first three rows of a 4x4 transform in row-major order:

```text
timestamp r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
```

When loaded, `eval3r` appends the homogeneous bottom row:

```text
0 0 0 1
```

A timestamped 4x4 row contains one timestamp followed by all sixteen matrix values in row-major order:

```text
timestamp m00 m01 m02 m03 m10 m11 m12 m13 m20 m21 m22 m23 m30 m31 m32 m33
```

## Direct KITTI Loader

`load_trajectory_kitti(path, convention=...)` reads the classic KITTI format:

```text
r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
```

This format has 12 columns and no timestamps. It is available through the direct KITTI loader, not through `load_trajectory_auto`, because `load_trajectory_auto` expects timestamped trajectory files for frame matching.

`save_trajectory_kitti(path, poses)` writes this 12-column format.

## Saved Prediction Poses

`PredictionWriter.save_poses(...)` writes both trajectory representations into a prediction package:

- TUM text with timestamps.
- KITTI-style 12-column text without timestamps.

When writing poses, specify the convention explicitly:

```python
writer.save_poses(poses, timestamps=timestamps, convention="T_wc")
```

or:

```python
writer.save_poses(poses, timestamps=timestamps, convention="T_cw")
```

## Pose Convention

All trajectory loaders accept a `convention` parameter:

```text
T_wc
T_cw
unspecified
```

The convention is stored in the returned `Trajectory`. It is interpreted later by trajectory alignment and trajectory metrics when extracting camera centers.

Use:

```text
T_wc = camera-to-world transform
T_cw = world-to-camera transform
```

Trajectory-based alignment requires `T_wc` or `T_cw`. If the convention is left as `unspecified`, alignment cannot reliably compute camera centers.
