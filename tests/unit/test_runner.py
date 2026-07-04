"""Task 007 runner unit tests: overrides, Umeyama alignment, seeds, failure policy.

Expected transforms/values are constructed in the test, never copied from a run.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval3r.core.errors import AlignmentError, SceneEvaluationError
from eval3r.core.hashing import protocol_hash
from eval3r.core.schema import AlignmentSpec, FailurePolicySpec
from eval3r.pipeline.runner import apply_geometry_overrides, run_single_file_geometry
from eval3r.pipeline.stages.align import align_geometry, umeyama
from eval3r.pipeline.stages.load import LoadedGeometry
from eval3r.pipeline.stages.sample import derive_seed
from eval3r.protocols import load_protocol

FIX = "tests/fixtures/geom"


def _proto():
    return load_protocol("single_geometry")


# --- overrides -----------------------------------------------------------------


def test_threshold_override_changes_hash_and_thresholds() -> None:
    base = _proto()
    proto, overrides = apply_geometry_overrides(
        base, input_type="pointcloud", gt_type="pointcloud", threshold=0.02, sample=None
    )
    assert overrides["threshold"] == 0.02
    for m in proto.metrics:
        if m.name in ("precision", "recall", "fscore"):
            assert m.threshold == 0.02
    # 0.02 differs from the built-in 0.05, so the canonical hash must change.
    assert protocol_hash(proto) != protocol_hash(base)


def test_identity_override_preserves_builtin_hash() -> None:
    base = _proto()
    proto, overrides = apply_geometry_overrides(
        base, input_type="pointcloud", gt_type="pointcloud", threshold=0.05, sample=None
    )
    # Overriding with the values already in the built-in protocol changes nothing.
    assert protocol_hash(proto) == protocol_hash(base)
    assert overrides == {"input_type": "pointcloud", "gt_type": "pointcloud", "threshold": 0.05}


def test_mesh_override_forces_surface_sampling_with_default_count() -> None:
    proto, overrides = apply_geometry_overrides(
        _proto(), input_type="mesh", gt_type="mesh", threshold=None, sample=None
    )
    assert proto.prediction_modality == "mesh"
    assert proto.sampling.pred.method == "surface_area"
    assert proto.sampling.pred.n_points == 200_000
    assert overrides["input_type"] == "mesh"


def test_sample_override_sets_point_counts() -> None:
    proto, _ = apply_geometry_overrides(
        _proto(), input_type="pointcloud", gt_type="pointcloud", threshold=None, sample=1234
    )
    assert proto.sampling.pred.method == "random_points"
    assert proto.sampling.pred.n_points == 1234
    assert proto.sampling.gt.n_points == 1234


# --- seed derivation -----------------------------------------------------------


def test_derive_seed_is_deterministic_and_scene_specific() -> None:
    assert derive_seed("derive", "sceneA") == derive_seed("derive", "sceneA")
    assert derive_seed(None, "sceneA") == derive_seed("derive", "sceneA")
    assert derive_seed("derive", "sceneA") != derive_seed("derive", "sceneB")
    # explicit integers pass through unchanged
    assert derive_seed(42, "sceneA") == 42


# --- Umeyama alignment ---------------------------------------------------------


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_umeyama_recovers_rigid_transform() -> None:
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    rot = _random_rotation(rng)
    t = np.array([1.0, -2.0, 0.5])
    dst = src @ rot.T + t
    matrix, scale = umeyama(src, dst, with_scale=False)
    assert scale == pytest.approx(1.0)
    moved = src @ matrix[:3, :3].T + matrix[:3, 3]
    assert np.allclose(moved, dst, atol=1e-9)


def test_umeyama_recovers_similarity_transform() -> None:
    rng = np.random.default_rng(1)
    src = rng.normal(size=(40, 3))
    rot = _random_rotation(rng)
    s = 2.5
    t = np.array([0.3, 0.4, -0.5])
    dst = s * (src @ rot.T) + t
    matrix, scale = umeyama(src, dst, with_scale=True)
    assert scale == pytest.approx(s, rel=1e-6)
    moved = src @ matrix[:3, :3].T + matrix[:3, 3]
    assert np.allclose(moved, dst, atol=1e-8)


def _pc(points: np.ndarray) -> LoadedGeometry:
    from pathlib import Path

    return LoadedGeometry(kind="pointcloud", path=Path("mem"), points=np.asarray(points, float))


def test_align_none_is_identity_passthrough() -> None:
    pred = _pc(np.array([[0.0, 0, 0], [1, 0, 0]]))
    gt = _pc(np.array([[5.0, 0, 0], [6, 0, 0]]))
    out, res = align_geometry(
        pred, gt, AlignmentSpec(mode="none"), scene_id="s", metric_scale=True
    )
    assert out is pred
    assert res.mode == "none"
    assert res.matrix == np.eye(4).tolist()


def test_sim3_disallowed_on_metric_scale_without_permission() -> None:
    rng = np.random.default_rng(2)
    src = rng.normal(size=(20, 3))
    with pytest.raises(AlignmentError) as exc:
        align_geometry(
            _pc(src), _pc(2 * src), AlignmentSpec(mode="sim3"),
            scene_id="s", metric_scale=True,
        )
    assert "metric-scale" in str(exc.value)


def test_sim3_allowed_with_explicit_parameter() -> None:
    rng = np.random.default_rng(3)
    src = rng.normal(size=(20, 3))
    dst = 2.0 * src + np.array([1.0, 0, 0])
    out, res = align_geometry(
        _pc(src), _pc(dst),
        AlignmentSpec(mode="sim3", parameters={"allow_sim3": True}),
        scene_id="s", metric_scale=True,
    )
    assert res.scale == pytest.approx(2.0, rel=1e-6)
    assert res.n_correspondences == 20
    assert out.kind == "pointcloud"


def test_sim3_requires_correspondence() -> None:
    with pytest.raises(AlignmentError) as exc:
        align_geometry(
            _pc(np.zeros((5, 3))), _pc(np.zeros((6, 3))),
            AlignmentSpec(mode="sim3", parameters={"allow_sim3": True}),
            scene_id="s", metric_scale=True,
        )
    assert "1:1 correspondence" in str(exc.value)


def test_icp_mode_is_deferred_not_silent() -> None:
    with pytest.raises(AlignmentError) as exc:
        align_geometry(
            _pc(np.zeros((3, 3))), _pc(np.zeros((3, 3))),
            AlignmentSpec(mode="icp"), scene_id="s", metric_scale=False,
        )
    assert "icp" in str(exc.value).lower()


# --- failure policy ------------------------------------------------------------


def _proto_with_policy(policy: str, worst: dict | None = None):
    proto = _proto().model_copy(deep=True)
    proto.failure_policy = FailurePolicySpec(policy=policy, worst_values=worst or {})
    return proto


def test_abort_policy_raises_scene_error_on_missing_file() -> None:
    with pytest.raises(SceneEvaluationError) as exc:
        run_single_file_geometry("does_not_exist.ply", f"{FIX}/gt.ply", _proto())
    assert exc.value.stage == "load"
    assert "does_not_exist.ply" in exc.value.reason


def test_skip_and_flag_records_failure_without_metrics() -> None:
    proto = _proto_with_policy("skip_and_flag")
    run = run_single_file_geometry("missing.ply", f"{FIX}/gt.ply", proto)
    assert run.result.n_scenes_evaluated == 0
    assert len(run.result.failed_scenes) == 1
    assert run.result.failed_scenes[0].stage == "load"
    assert all(v is None for v in run.result.metrics.values())


def test_score_worst_fills_worst_values() -> None:
    worst = {"accuracy": 9.9, "fscore": 0.0}
    proto = _proto_with_policy("score_worst", worst)
    run = run_single_file_geometry("missing.ply", f"{FIX}/gt.ply", proto)
    assert run.result.n_scenes_evaluated == 0
    assert run.result.metrics["accuracy"] == 9.9
    assert run.result.metrics["fscore"] == 0.0
    # a metric with no configured worst value stays visibly None
    assert run.result.metrics["completeness"] is None
