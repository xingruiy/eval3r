"""TUM and KITTI trajectory I/O.

Pose convention is stored as metadata only — never silently converted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval3r.utils.typing import PathLike, Poses


@dataclass
class Trajectory:
    poses: Poses  # (T, 4, 4)
    timestamps: np.ndarray | None  # (T,) or None
    convention: str  # "T_wc", "T_cw", or "unspecified"


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n == 0.0:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _rot_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Return (qx, qy, qz, qw)."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


def save_trajectory_tum(
    path: PathLike,
    poses: Poses,
    timestamps: np.ndarray | None = None,
) -> Path:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (T, 4, 4), got {poses.shape}")
    n = poses.shape[0]
    if timestamps is None:
        timestamps = np.arange(n, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if timestamps.shape[0] != n:
        raise ValueError(
            f"timestamps length {timestamps.shape[0]} does not match poses count {n}"
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for t, T in zip(timestamps, poses):
        tx, ty, tz = T[:3, 3]
        qx, qy, qz, qw = _rot_to_quat(T[:3, :3])
        lines.append(f"{t:.9f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}")
    out.write_text("\n".join(lines) + "\n")
    return out


def load_trajectory_tum(path: PathLike, convention: str = "unspecified") -> Trajectory:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trajectory file not found: {p}")
    rows = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append([float(x) for x in s.split()])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{p} contains no trajectory entries")
    if arr.shape[1] != 8:
        raise ValueError(f"TUM trajectory expects 8 cols, got {arr.shape[1]}")
    timestamps = arr[:, 0].copy()
    poses = np.tile(np.eye(4), (arr.shape[0], 1, 1))
    poses[:, :3, 3] = arr[:, 1:4]
    for i, q in enumerate(arr[:, 4:8]):
        poses[i, :3, :3] = _quat_to_rot(q[0], q[1], q[2], q[3])
    return Trajectory(poses=poses, timestamps=timestamps, convention=convention)


def save_trajectory_kitti(path: PathLike, poses: Poses) -> Path:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (T, 4, 4), got {poses.shape}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    flat = poses[:, :3, :].reshape(poses.shape[0], 12)
    lines = [" ".join(f"{x:.9e}" for x in row) for row in flat]
    out.write_text("\n".join(lines) + "\n")
    return out


def load_trajectory_kitti(path: PathLike, convention: str = "unspecified") -> Trajectory:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trajectory file not found: {p}")
    arr = np.loadtxt(p, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 12:
        raise ValueError(f"KITTI trajectory expects 12 cols, got {arr.shape[1]}")
    poses = np.tile(np.eye(4), (arr.shape[0], 1, 1))
    poses[:, :3, :] = arr.reshape(-1, 3, 4)
    return Trajectory(poses=poses, timestamps=None, convention=convention)


def load_trajectory_auto(path: PathLike, convention: str = "unspecified") -> Trajectory:
    """Load trajectory from a text file, auto-detecting the format.

    Supported formats (one pose per line, whitespace-separated):

      - 8 fields:  timestamp tx ty tz qx qy qz qw (TUM)
      - 13 fields: timestamp + flattened 3x4 matrix (row-major)
      - 17 fields: timestamp + flattened 4x4 matrix (row-major)

    The leading timestamp is used to match corresponding frames between
    trajectories of differing lengths during alignment.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trajectory file not found: {p}")

    lines = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    if not lines:
        raise ValueError(f"{p} contains no trajectory entries")

    first_row = [float(x) for x in lines[0].split()]
    cols = len(first_row)

    if cols == 8:
        return load_trajectory_tum(p, convention=convention)

    if cols == 13:
        return _load_trajectory_flat(p, convention, 13)

    if cols == 17:
        return _load_trajectory_flat(p, convention, 17)

    raise ValueError(
        f"Cannot determine pose format from {p}: {cols} columns per row. "
        f"Expected 8 (TUM), 13 (timestamp + 3x4), or 17 (timestamp + 4x4)."
    )


def _load_trajectory_flat(path: Path, convention: str, cols: int) -> Trajectory:
    """Load a trajectory from timestamp-prefixed flat-matrix rows.

    *cols* is 13 (3x4 → padded to 4x4) or 17 (4x4).
    """
    rows = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append([float(x) for x in s.split()])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape[0] == 0:
        raise ValueError(f"{path} contains no trajectory entries")
    timestamps = arr[:, 0].copy()
    vals = arr[:, 1:]
    if cols == 13:
        flat = vals.reshape(-1, 3, 4)
        poses = np.tile(np.eye(4), (flat.shape[0], 1, 1))
        poses[:, :3, :] = flat
    else:
        poses = vals.reshape(-1, 4, 4)
    return Trajectory(poses=poses, timestamps=timestamps, convention=convention)
