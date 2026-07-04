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

- `core/registry.py`: `BackendRegistry` (`register`/`get`/`available`/`require`/`info`/
  `backend_versions`), `BACKEND_KINDS` tuple matching protocol `backend_preferences` keys,
  `BackendInfo` dataclass (kind/name/library/version/approximate + `as_metadata()`), and
  Protocol interfaces `Backend`/`MeshBackend`/`PointCloudBackend`/`NNBackend`.
  `UnknownBackendError` names the kind and lists available names; `UnknownBackendKindError`
  covers unknown kinds. No install hints (all backends always installed).
- Three backends behind the interfaces: `ScipyNNBackend` (cKDTree, exact, empty/shape/finite
  validation before the call), `TrimeshMeshBackend` (load + deterministic seeded
  `sample_surface`, optional normals, export; no repair), `PlyfilePointCloudBackend`
  (load/save (N,3) float, finite validation, optional colors as debug metadata).
- `backend_versions()` produces the `kind -> {name, library, version, approximate}` metadata
  structure from `.agent/backends.md` for result recording (task 006).
- Tests (`tests/unit/test_backends.py`, 20 cases): registry lookup, unknown-name/kind error
  text, `register` guards, NN hand-computed distances + identical-cloud zeros + empty/shape/
  non-finite failures, mesh sampling determinism (same seed → identical points) and
  seed-sensitivity, normals unit length, mesh + point-cloud round-trips. No `importorskip`.

## Decisions

- **Registration is lazy at first `default_registry()` call**, not import-time: the registry
  module imports the three backend classes inside `default_registry()` and registers
  singleton instances. This keeps `import eval3r.core.registry` side-effect-free and avoids an
  import cycle (backend modules import `BackendInfo` from the registry module; the registry
  imports backend classes only when the singleton is first built).
- Backends expose `name: str` + `backend_info() -> BackendInfo`; the registry's `info()` and
  `backend_versions()` read metadata through `backend_info()`.
- Determinism relies on trimesh 4.12 `sample_surface(..., seed=)`; no global RNG mutation.
- `plyfile` has no reliable `__version__`; version is read via `importlib.metadata`.

## Verification

```bash
pytest tests/unit/test_backend*.py
python -c "from eval3r.core.registry import BackendRegistry"
```

Requesting an unknown backend name for a kind produces the documented explicit error message.

Outcomes (feature/repo-foundation):

```text
pytest            -> 87 passed (adds 20 backend tests)
ruff check .      -> All checks passed!
mypy eval3r       -> Success: no issues found in 83 source files
python -c "from eval3r.core.registry import BackendRegistry" -> ok
```

## Status

done
