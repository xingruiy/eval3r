# 004 — Backend registry and first backends

## Goal

A backend registry keyed by kind (`mesh`, `pointcloud`, `nearest_neighbor`, `registration`,
`trajectory`, `camera`, `depth_io`, `official_eval`) for backend selection and version
recording, plus the three first backends: scipy NN, trimesh mesh, plyfile point cloud. All
backends are always installed, so there is no missing-dependency install path.

## Scope

- `core/registry.py` (`BackendRegistry`): `get`, `available`, `require` per
  `.agent/backends.md`; `require` selects a named backend and raises an explicit error when
  the name is unknown for that kind (naming the kind and the available names). No install
  hints — all backends are always installed.
- Backend protocol interfaces (`MeshBackend`, `PointCloudBackend`, `NNBackend`, …) as
  typing.Protocol definitions.
- `backends/nn_scipy.py`: cKDTree nearest distances; empty-input failure before backend call.
- `backends/mesh_trimesh.py`: load mesh, deterministic seeded surface sampling
  (`sample_surface`), optional normals, export.
- `backends/pointcloud_plyfile.py`: load/save Nx3 float arrays, finite-coordinate validation.
- Backend version metadata collection (name, library, version, approximate flag) for
  `backend_versions.json` / result metadata.
- Registry keys must match `backend_preferences` keys used in protocol YAMLs (registry kinds).

## Out of Scope

- Open3D, evo, pycolmap, official-eval backends (added with the tasks that need them).
  The project does not include torch/FAISS NN backends.
- Any metric computation (task 005).

## Relevant Files

- `.agent/backends.md` — entire file (interfaces, rules, delegation policy)
- `.agent/schema.md` — note under `EvalProtocol` on backend_preferences keys
- `.agent/plan.md` — "Dependency policy"

## Plan

1. Registry with kind/name registration, availability probing, and `require` errors.
2. Implement the three backends behind the Protocol interfaces.
3. Version metadata helper.
4. Tests: registry lookup, unknown-backend-name error message text, mesh sampling determinism
   (same seed → same points), NN correctness on tiny hand-computed point sets,
   plyfile round-trip. No `pytest.importorskip` — all backends are installed.

## Findings

(record during implementation)

## Decisions

(record during implementation; e.g. entry-point vs import-time registration)

## Verification

```bash
pytest tests/unit/test_backend*.py
python -c "from eval3r.core.registry import BackendRegistry"
```

Requesting an unknown backend name for a kind produces the documented explicit error message.

## Status

todo
