"""Task 010 DTU official-eval backend tests: ObsMask/Plane culling, cap, determinism.

Expectations are hand-computed on tiny synthetic inputs. Downsampling is effectively
disabled (points are spaced far apart relative to a tiny downsample radius) so each
mechanism can be checked in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval3r.backends.dtu_eval import DTUOfficialEval, evaluate_dtu
from eval3r.core.errors import MetricError

# ObsMask big enough to index grid coords up to 20; keep-all by default.
OBS = np.ones((21, 21, 21), dtype=np.uint8)
BB = np.array([[0, 0, 0], [20, 20, 20]], dtype=np.int16)
RES = 1.0
KEEP_ALL_PLANE = np.array([0.0, 0.0, 0.0, 1.0])  # [x y z 1]·P = 1 > 0 always
TINY = 1e-6  # downsample radius: no merging of points spaced >= 1 mm


def _eval(pred, gt, *, obs=OBS, bb=BB, res=RES, plane=KEEP_ALL_PLANE, max_dist=20.0):
    return evaluate_dtu(
        np.asarray(pred, float), np.asarray(gt, float),
        obs_mask=obs, bb=bb, res=res, plane=plane,
        downsample_mm=TINY, max_dist_mm=max_dist, patch_mm=60.0, seed=0,
    )


def test_basic_accuracy_completeness_and_overall() -> None:
    gt = [[0, 0, 0], [10, 0, 0]]
    pred = [[1, 0, 0], [11, 0, 0]]  # each offset 1 mm
    r = _eval(pred, gt)
    assert r.accuracy == pytest.approx(1.0)
    assert r.completeness == pytest.approx(1.0)
    assert r.overall == pytest.approx(1.0)
    assert r.n_data_in_obs == 2
    assert r.n_stl_above == 2


def test_obs_mask_culls_unobserved_points() -> None:
    obs = OBS.copy()
    obs[11, 0, 0] = 0  # mark the second pred point's voxel unobserved
    gt = [[0, 0, 0], [10, 0, 0]]
    pred = [[1, 0, 0], [11, 0, 0]]
    r = _eval(pred, gt, obs=obs)
    # only the observed pred point contributes to accuracy
    assert r.n_data_in_obs == 1
    assert r.accuracy == pytest.approx(1.0)


def test_plane_cull_removes_gt_below_plane() -> None:
    # P = [-1,0,0,5]: keep x<5. GT [10,0,0] is removed; only [0,0,0] remains.
    plane = np.array([-1.0, 0.0, 0.0, 5.0])
    gt = [[0, 0, 0], [10, 0, 0]]
    pred = [[1, 0, 0], [11, 0, 0]]
    r = _eval(pred, gt, plane=plane)
    assert r.n_stl_above == 1
    assert r.completeness == pytest.approx(1.0)  # [0,0,0] -> nearest pred [1,0,0]


def test_max_dist_cap_excludes_far_points() -> None:
    gt = [[0, 0, 0]]
    pred = [[1, 0, 0], [30, 0, 0]]  # 30 mm > 20 mm cap -> excluded from the mean
    r = _eval(pred, gt, max_dist=20.0)
    assert r.accuracy == pytest.approx(1.0)


def test_deterministic_for_fixed_seed() -> None:
    rng = np.random.default_rng(1)
    gt = rng.normal(scale=5.0, size=(200, 3)) + np.array([10.0, 10.0, 10.0])
    pred = gt + rng.normal(scale=0.2, size=gt.shape)
    obs = np.ones((40, 40, 40), dtype=np.uint8)
    bb = np.array([[0, 0, 0], [30, 30, 30]], dtype=np.int16)
    a = evaluate_dtu(pred, gt, obs_mask=obs, bb=bb, res=1.0, plane=KEEP_ALL_PLANE, seed=7)
    b = evaluate_dtu(pred, gt, obs_mask=obs, bb=bb, res=1.0, plane=KEEP_ALL_PLANE, seed=7)
    assert a.accuracy == b.accuracy
    assert a.completeness == b.completeness


def test_all_points_beyond_cap_raises() -> None:
    gt = [[0, 0, 0]]
    pred = [[100, 0, 0]]  # 100 mm, beyond 20 mm cap -> nothing survives
    with pytest.raises(MetricError):
        _eval(pred, gt, max_dist=20.0)


def test_backend_requires_visibility_keys() -> None:
    backend = DTUOfficialEval()
    with pytest.raises(MetricError) as exc:
        backend.evaluate(
            np.array([[1.0, 0, 0]]), np.array([[0.0, 0, 0]]),
            {"obs_mask": OBS, "bb": BB, "res": RES, "plane": None},  # missing plane
        )
    assert "plane" in str(exc.value)


def test_backend_info_records_validated_port() -> None:
    info = DTUOfficialEval().backend_info()
    assert info.kind == "official_eval"
    assert info.name == "dtu"
    assert info.version == "validated_official_port"
    assert info.approximate is False
