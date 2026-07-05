"""Task 004 backend tests: registry lookup/errors, NN correctness, mesh determinism, IO.

All backends are always installed, so no ``pytest.importorskip`` is used.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from eval3r.backends import (
    PlyfilePointCloudBackend,
    ScipyNNBackend,
    TrimeshMeshBackend,
)
from eval3r.core.errors import InvalidGeometryError
from eval3r.core.registry import (
    BackendError,
    BackendRegistry,
    UnknownBackendError,
    UnknownBackendKindError,
    default_registry,
)

# --- registry ------------------------------------------------------------------


def test_default_registry_has_builtin_backends() -> None:
    reg = default_registry()
    assert reg.available("nearest_neighbor") == ["scipy"]
    assert reg.available("mesh") == ["trimesh"]
    assert reg.available("pointcloud") == ["open3d", "plyfile"]
    assert reg.available("official_eval") == ["dtu", "eth3d_official", "tnt_official"]
    # COLMAP camera parsing via pycolmap (task 013).
    assert reg.available("camera") == ["pycolmap"]
    # Depth IO (task 014): imageio default, OpenCV for PFM.
    assert reg.available("depth_io") == ["imageio", "opencv"]
    # Trajectory ATE/RPE via evo (task 015).
    assert reg.available("trajectory") == ["evo"]
    # ScanNet visibility culling (task 011): render + TSDF trim.
    assert reg.available("visibility") == ["render_tsdf"]


def test_require_returns_backend() -> None:
    reg = default_registry()
    assert reg.require("nearest_neighbor", "scipy").name == "scipy"
    assert reg.get("mesh", "trimesh").name == "trimesh"


def test_unknown_backend_name_error_names_kind_and_available() -> None:
    reg = default_registry()
    with pytest.raises(UnknownBackendError) as exc:
        reg.require("nearest_neighbor", "faiss")
    msg = str(exc.value)
    assert "nearest_neighbor" in msg
    assert "faiss" in msg
    assert "scipy" in msg  # available names are listed


def test_unknown_kind_error() -> None:
    reg = default_registry()
    with pytest.raises(UnknownBackendKindError):
        reg.require("teleporter", "scipy")


def test_register_rejects_unknown_kind_and_nameless_backend() -> None:
    reg = BackendRegistry()
    with pytest.raises(UnknownBackendKindError):
        reg.register("teleporter", ScipyNNBackend())
    with pytest.raises(BackendError):
        reg.register("nearest_neighbor", object())


def test_backend_versions_metadata_shape() -> None:
    reg = default_registry()
    meta = reg.backend_versions({"nearest_neighbor": "scipy", "mesh": "trimesh"})
    assert meta["nearest_neighbor"]["library"] == "scipy"
    assert meta["nearest_neighbor"]["approximate"] is False
    assert set(meta["mesh"]) == {"name", "library", "version", "approximate"}


# --- nearest neighbor ----------------------------------------------------------


def test_nn_distances_hand_computed() -> None:
    nn = ScipyNNBackend()
    query = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    reference = np.array([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    d = nn.nearest_distances(query, reference)
    # nearest to (0,0,0) is (1,0,0) -> 1.0; nearest to (10,0,0) is (1,0,0) -> 9.0
    np.testing.assert_allclose(d, [1.0, 9.0])


def test_nn_identical_clouds_zero_distance() -> None:
    nn = ScipyNNBackend()
    pts = np.random.default_rng(0).random((50, 3))
    d = nn.nearest_distances(pts, pts)
    np.testing.assert_allclose(d, 0.0, atol=1e-12)


def test_nn_empty_input_fails_explicitly() -> None:
    nn = ScipyNNBackend()
    with pytest.raises(InvalidGeometryError):
        nn.nearest_distances(np.zeros((0, 3)), np.ones((3, 3)))


def test_nn_wrong_shape_fails() -> None:
    nn = ScipyNNBackend()
    with pytest.raises(InvalidGeometryError):
        nn.nearest_distances(np.ones((5, 2)), np.ones((5, 3)))


def test_nn_nonfinite_fails() -> None:
    nn = ScipyNNBackend()
    bad = np.array([[0.0, 0.0, np.nan]])
    with pytest.raises(InvalidGeometryError):
        nn.nearest_distances(bad, np.ones((3, 3)))


# --- mesh ----------------------------------------------------------------------


def test_mesh_sampling_is_deterministic_for_seed() -> None:
    mesh_backend = TrimeshMeshBackend()
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    a = mesh_backend.sample_surface(box, 1000, seed=42)
    b = mesh_backend.sample_surface(box, 1000, seed=42)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (1000, 3)


def test_mesh_sampling_differs_across_seeds() -> None:
    mesh_backend = TrimeshMeshBackend()
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    a = mesh_backend.sample_surface(box, 1000, seed=1)
    b = mesh_backend.sample_surface(box, 1000, seed=2)
    assert not np.array_equal(a, b)


def test_mesh_sampling_normals_optional() -> None:
    mesh_backend = TrimeshMeshBackend()
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    pts, normals = mesh_backend.sample_surface(box, 100, seed=0, return_normals=True)
    assert pts.shape == (100, 3)
    assert normals.shape == (100, 3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)


def test_mesh_roundtrip_load_export(tmp_path) -> None:
    mesh_backend = TrimeshMeshBackend()
    box = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    out = tmp_path / "mesh.ply"
    mesh_backend.export_mesh(box, out)
    loaded = mesh_backend.load_mesh(out)
    assert len(loaded.faces) > 0
    np.testing.assert_allclose(sorted(loaded.extents), [1.0, 2.0, 3.0])


# --- point cloud ---------------------------------------------------------------


def test_pointcloud_roundtrip(tmp_path) -> None:
    pc = PlyfilePointCloudBackend()
    pts = np.random.default_rng(3).random((200, 3)).astype(np.float64)
    out = tmp_path / "cloud.ply"
    pc.save_pointcloud(pts, out)
    loaded = pc.load_pointcloud(out)
    assert loaded.shape == (200, 3)
    np.testing.assert_allclose(loaded, pts, rtol=0, atol=1e-6)


def test_pointcloud_missing_file_fails() -> None:
    pc = PlyfilePointCloudBackend()
    with pytest.raises(InvalidGeometryError):
        pc.load_pointcloud("/nonexistent/cloud.ply")


def test_pointcloud_save_wrong_shape_fails(tmp_path) -> None:
    pc = PlyfilePointCloudBackend()
    with pytest.raises(InvalidGeometryError):
        pc.save_pointcloud(np.ones((10, 2)), tmp_path / "bad.ply")
