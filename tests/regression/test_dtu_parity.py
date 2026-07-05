"""Offline DTU parity regression (documented; skipped when real DTU data is absent).

Validates the packaged ``evaluate_dtu`` against an independent transcription of the
official reference algorithm (jzhangbs/DTUeval-python), run on a cropped region of a
real DTU scan. The reference randomly shuffles before its radius-dedup downsample, so
it is nondeterministic; the packaged port is deterministic (seeded) and must land
inside the reference's own run-to-run band.

This test needs the DTU dataset locally. Set ``EVAL3R_DTU_ROOT`` or place it at the
default path below; otherwise the test skips with an explicit reason. It is the
"executed when full data is available locally" regression from task 010.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData
from scipy.io import loadmat
from scipy.spatial import cKDTree

from eval3r.backends.dtu_eval import evaluate_dtu

DTU_ROOT = Path(os.environ.get("EVAL3R_DTU_ROOT", "/mnt/research/dataset/DTU"))
SCAN = 24
CROP_LO = np.array([-40.0, -40.0, 620.0])
CROP_HI = np.array([20.0, 40.0, 720.0])

pytestmark = pytest.mark.skipif(
    not (DTU_ROOT / "groundtruth" / f"stl{SCAN:03d}_total.ply").is_file(),
    reason=f"real DTU data not found at {DTU_ROOT}; offline parity regression skipped.",
)


def _read_ply(path: Path) -> np.ndarray:
    v = PlyData.read(str(path))["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float64)


def _reference(pred, stl, obs, bb, res, plane, rng, *, thresh=0.2, max_dist=20.0, patch=60.0):
    """Independent transcription of the reference DTU algorithm (random shuffle)."""
    pred = pred.copy()
    rng.shuffle(pred, axis=0)
    tree = cKDTree(pred)
    idxs = tree.query_ball_point(pred, r=thresh)
    keep = np.ones(len(pred), dtype=bool)
    for c, ix in enumerate(idxs):
        if keep[c]:
            keep[ix] = False
            keep[c] = True
    down = pred[keep]
    bb = bb.astype(np.float64)
    inb = ((down >= bb[:1] - patch) & (down < bb[1:] + patch * 2)).sum(-1) == 3
    data_in = down[inb]
    grid = np.around((data_in - bb[:1]) / res).astype(np.int64)
    gin = ((grid >= 0) & (grid < np.array(obs.shape)[None])).sum(-1) == 3
    in_obs = obs[grid[gin][:, 0], grid[gin][:, 1], grid[gin][:, 2]].astype(bool)
    data_in_obs = data_in[gin][in_obs]
    d2s, _ = cKDTree(stl).query(data_in_obs, k=1)
    acc = d2s[d2s < max_dist].mean()
    hom = np.concatenate([stl, np.ones((len(stl), 1))], -1)
    above = (plane.reshape(1, 4) * hom).sum(-1) > 0
    s2d, _ = cKDTree(data_in).query(stl[above], k=1)
    comp = s2d[s2d < max_dist].mean()
    return (acc + comp) / 2.0


def test_packaged_backend_matches_reference_band() -> None:
    stl = _read_ply(DTU_ROOT / "groundtruth" / f"stl{SCAN:03d}_total.ply")
    stl = stl[np.all((stl >= CROP_LO) & (stl < CROP_HI), axis=1)]
    om = loadmat(str(DTU_ROOT / "ObsMask" / f"ObsMask{SCAN}_10.mat"))
    obs, bb, res = om["ObsMask"], om["BB"], float(np.asarray(om["Res"]).reshape(-1)[0])
    plane_mat = loadmat(str(DTU_ROOT / "ObsMask" / f"Plane{SCAN}.mat"))
    plane = plane_mat["P"].astype(np.float64).reshape(4)

    rng0 = np.random.default_rng(0)
    idx = rng0.choice(len(stl), size=int(0.7 * len(stl)), replace=False)
    data = stl[idx] + rng0.normal(scale=0.3, size=(len(idx), 3))

    # reference band from several random-shuffle runs
    ref = np.array([
        _reference(data, stl, obs, bb, res, plane, np.random.default_rng(100 + s))
        for s in range(3)
    ])

    port = evaluate_dtu(data, stl, obs_mask=obs, bb=bb, res=res, plane=plane, seed=0)
    port2 = evaluate_dtu(data, stl, obs_mask=obs, bb=bb, res=res, plane=plane, seed=0)

    # deterministic, and within the reference's own nondeterministic band (+ small tol).
    assert port.overall == port2.overall
    assert ref.min() - 1e-3 <= port.overall <= ref.max() + 1e-3
