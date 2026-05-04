from __future__ import annotations

import numpy as np

from eval3r.io.trajectory import (
    load_trajectory_kitti,
    load_trajectory_tum,
    save_trajectory_kitti,
    save_trajectory_tum,
)


def _random_pose(rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    T = np.eye(4)
    T[:3, :3] = Q
    T[:3, 3] = rng.normal(size=3)
    return T


def test_tum_round_trip(tmp_path, rng) -> None:
    poses = np.stack([_random_pose(rng) for _ in range(5)])
    timestamps = np.linspace(0.0, 1.0, 5)
    out = tmp_path / "traj_tum.txt"
    save_trajectory_tum(out, poses, timestamps)
    loaded = load_trajectory_tum(out, convention="T_wc")
    assert np.allclose(loaded.timestamps, timestamps, atol=1e-6)
    assert np.allclose(loaded.poses[:, :3, 3], poses[:, :3, 3], atol=1e-6)
    # Rotations are equivalent up to numerical noise.
    for i in range(5):
        R_ref = poses[i, :3, :3]
        R_got = loaded.poses[i, :3, :3]
        assert np.allclose(R_got @ R_ref.T, np.eye(3), atol=1e-5)


def test_kitti_round_trip(tmp_path, rng) -> None:
    poses = np.stack([_random_pose(rng) for _ in range(3)])
    out = tmp_path / "traj_kitti.txt"
    save_trajectory_kitti(out, poses)
    loaded = load_trajectory_kitti(out)
    assert np.allclose(loaded.poses, poses, atol=1e-6)
