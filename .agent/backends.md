# eval3r Backends

Backends provide small adapter interfaces around external libraries and official evaluation tools. They keep eval3r focused on protocols, schemas, dataset adapters, metric definitions, and reproducibility records.

eval3r is a research-oriented project, so all dependencies are required and always installed. There is no optional-extra system. MATLAB is the only optional external tool (it is not a pip package), used solely for the DTU MATLAB official-script path.

## Delegation policy

The library delegates:

```text
mesh loading
mesh surface sampling
point-cloud loading
nearest-neighbor search
ICP and registration
trajectory association and ATE / RPE computation
COLMAP parsing
camera models beyond minimal pinhole
image and depth IO
official benchmark scripts where faithful reimplementation is risky
```

The library does not delegate:

```text
protocol meaning
schema validation
result structure
failure accounting
canonical hashing
metric-definition choices
```

## Dependencies

All dependencies are required and installed by a plain `pip install eval3r`:

```toml
dependencies = [
  "numpy",
  "scipy",
  "pandas",
  "pyyaml",
  "typer",
  "rich",
  "pydantic",
  "trimesh",
  "plyfile",
  "open3d",
  "evo",
  "pycolmap",
  "imageio",
  "opencv-python",
  "pyrender",
]
```

`pyrender` is used only by the `visibility` backend for offscreen depth rendering during
evaluation-time visibility culling (ScanNet single-/double-layer). It needs a headless GL
context; set `PYOPENGL_PLATFORM=egl` (the `visibility` backend sets this itself when unset).

There is no `[project.optional-dependencies]` table and no extras. Every backend below is always importable.

The project intentionally does not depend on:

```text
PyTorch
FAISS
Waymo tooling
```

There is no torch/FAISS nearest-neighbor backend and no Waymo adapter.

MATLAB is the only optional external tool. It is not a pip package; it is used only for the DTU MATLAB official-script path, and whether it was used is recorded in result metadata. A missing MATLAB must fail with an explicit message naming the DTU MATLAB path and the alternative validated Python port.

Default backend preferences should point at `trimesh`/`plyfile` for basic mesh/point-cloud IO and `scipy` for nearest-neighbor, using Open3D only where a protocol explicitly asks for it.

## Backend registry

Backends should be registered by capability:

```text
mesh
pointcloud
nearest_neighbor
registration
trajectory
camera
depth_io
official_eval
```

The registry should expose:

```python
class BackendRegistry:
    def get(self, kind: str, name: str) -> Backend: ...
    def available(self, kind: str) -> list[str]: ...
    def require(self, kind: str, name: str) -> Backend: ...
```

`require` selects a named backend and fails with an explicit message if the name is unknown for that kind (naming the kind and the available names). It does not carry install hints, because all backends are always installed.

Backend names and versions must be written into result metadata.

## Mesh backend

Delegates to:

```text
trimesh
Open3D
```

Interface:

```python
class MeshBackend(Protocol):
    name: str

    def load_mesh(self, path: Path) -> Any: ...

    def sample_surface(
        self,
        mesh: Any,
        n_points: int,
        seed: int,
        return_normals: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]: ...

    def export_mesh(self, mesh: Any, path: Path) -> None: ...
```

Rules:

```text
surface sampling must be deterministic for a given seed
normal return must be explicit
raw vertices are not a mesh metric default
mesh repair is not performed unless explicitly protocol-defined
```

Default preference:

```text
trimesh for basic mesh loading and surface sampling
Open3D as optional backend when already required by a protocol
```

## Point-cloud backend

Delegates to:

```text
Open3D
plyfile
numpy
```

Interface:

```python
class PointCloudBackend(Protocol):
    name: str

    def load_pointcloud(self, path: Path) -> np.ndarray: ...

    def save_pointcloud(
        self,
        points: np.ndarray,
        path: Path,
        colors: np.ndarray | None = None,
    ) -> None: ...
```

Rules:

```text
return Nx3 float array for coordinates
preserve colors only as optional metadata / debug output
validate finite coordinates before metric computation
record source unit if available
```

## Nearest-neighbor backend

Delegates to:

```text
scipy.spatial.cKDTree
Open3D
```

Interface:

```python
class NNBackend(Protocol):
    name: str

    def nearest_distances(
        self,
        query_points: np.ndarray,
        reference_points: np.ndarray,
    ) -> np.ndarray: ...
```

Default:

```text
scipy
```

Rules:

```text
empty inputs fail clearly before backend call
chunking is allowed for memory control if it does not change results
approximate nearest-neighbor search must be explicitly declared in backend metadata
backend version is recorded
```

## Registration backend

Delegates to:

```text
Open3D
```

Interface:

```python
class RegistrationBackend(Protocol):
    name: str

    def icp(
        self,
        source: np.ndarray,
        target: np.ndarray,
        init_transform: np.ndarray,
        max_correspondence_distance: float,
        max_iterations: int,
        criteria: dict,
    ) -> RegistrationResult: ...
```

Rules:

```text
ICP never runs silently
ICP must be enabled in the protocol or CLI override if allowed
init transform is recorded
max correspondence distance is recorded
iteration count and convergence criteria are recorded
fitness and residual RMSE are recorded
```

## Trajectory backend

Delegates to:

```text
evo
```

Interface:

```python
class TrajectoryBackend(Protocol):
    name: str

    def evaluate_ate(
        self,
        pred_path: Path,
        gt_path: Path,
        align: str,
        association: dict,
    ) -> dict: ...

    def evaluate_rpe(
        self,
        pred_path: Path,
        gt_path: Path,
        align: str,
        association: dict,
    ) -> dict: ...
```

Rules:

```text
prefer evo for ATE / RPE
record association policy
record number of associated and dropped poses
record alignment mode and Sim3 scale when used
fallback evaluator, if any, must be labeled limited
```

## Camera backend

Delegates to:

```text
pycolmap
minimal internal pinhole parser
CO3D parser when implemented
```

Interface:

```python
class CameraBackend(Protocol):
    name: str

    def load_cameras(self, path: Path) -> CameraSet: ...

    def load_trajectory(self, path: Path) -> CameraTrajectory: ...
```

Rules:

```text
minimal internal pinhole model is only for simple pinhole datasets
COLMAP text/binary parsing should use pycolmap where possible
non-pinhole cameras must not be silently approximated as pinhole
source pose format and normalized convention are recorded
```

Dataset notes:

```text
ETH3D uses COLMAP text format and may need nontrivial camera models.
CO3D camera data lives in frame_annotations.jgz.
BlendedMVS uses MVSNet cam.txt with world-to-camera extrinsics.
```

## Depth IO backend

Delegates to:

```text
imageio
OpenCV
h5py if needed by Hypersim
numpy
```

Interface:

```python
class DepthBackend(Protocol):
    name: str

    def load_depth(self, path: Path, depth_unit: float | None) -> np.ndarray: ...

    def load_mask(self, path: Path) -> np.ndarray: ...
```

Rules:

```text
depth_unit is required for integer depth where units are not self-describing
invalid values are passed to masking logic
Hypersim Euclidean ray distance must be converted only by a dataset-aware path
no depth sequence is integrated into scene geometry by a backend
```

## Official evaluation backends

Official wrappers should be used when reimplementation risk is high.

### Tanks and Temples backend

Responsibilities:

```text
invoke official evaluation script or vendored official-compatible code
resolve crop volume
resolve alignment transform
resolve .log trajectory if required
record per-scene threshold
record backend version and command
normalize output into MetricResult / results.json
```

Rules:

```text
official fidelity requires official backend behavior
training split can be local if GT is public
intermediate / advanced splits are server-only
```

Implementation (task 012): the `tnt_official` backend (`official_eval` kind,
method `official_script_wrapper`) wraps the official `isl-org/TanksAndTemples`
`python_toolbox/evaluation` toolbox. It is a **user-supplied external checkout**, not a
pip package (like the DTU MATLAB path); it is located from an explicit `toolbox_dir` or
the `EVAL3R_TNT_TOOLBOX` / `TANKSANDTEMPLES_TOOLBOX` environment variable. The backend
invokes the official `run.py` as a subprocess
(`--dataset-dir <scene_dir> --traj-path <log> --ply-path <pred> --out-dir <tmp>`), parses
its printed `precision` / `recall` / `f-score` / `distance tau` summary, and records the
command, the resolved toolbox dir, the toolbox git commit, and the interpreter used. The
per-scene threshold (`dTau`) is read from the official output, never hardcoded in eval3r.
When the toolbox is absent the backend fails explicitly (naming the env vars and the repo
URL) so a run never emits unofficial numbers.

**Pinned-interpreter, unmodified toolbox.** The toolbox pins `open3d==0.9` (its
`requirements.txt`), whose `open3d.registration` namespace **and** RANSAC convergence
semantics (`RANSACConvergenceCriteria(max_iteration, max_validation)` and a `checkers`-less
`registration_ransac_based_on_correspondence` signature) differ from newer open3d. Porting
those calls to open3d 0.19 is a *result-affecting* change (it alters the trajectory
alignment used for scoring), so per the "Official code / toolbox rule" the toolbox is run
**byte-for-byte unmodified** under its own pinned interpreter instead of being ported. The
interpreter is configured via an explicit `python_executable` or the `EVAL3R_TNT_PYTHON`
env var (default: the current interpreter), and is recorded in result metadata. A
`conda create -n tnt_toolbox python=3.7 && pip install open3d==0.9.0.0 matplotlib` env
satisfies the pin. Verified end-to-end on real Barn data (commit `2a0d1b25`, prediction =
`Barn_COLMAP.ply`): precision 0.4569 / recall 0.5529 / f-score 0.5003 at dTau 0.01 in ~150s.

**Testing.** No fake toolbox exists (a fake official evaluator is forbidden). The
end-to-end wrapper/benchmark tests drive the **real** toolbox and skip cleanly (like the
MATLAB path) when `EVAL3R_TNT_TOOLBOX` / `EVAL3R_TNT_PYTHON` / `EVAL3R_TNT_DATA` are not
set; the parse, toolbox/interpreter resolution, and absent-toolbox paths are unit-tested
directly and always run.

### DTU evaluation backend

Responsibilities:

```text
implement or wrap official-like point-cloud evaluation
use ObsMask and Plane files
match MATLAB / validated Python reimplementation behavior within tolerance
record missing Plane fallback explicitly
normalize units and output
```

Rules:

```text
official_like fidelity requires regression testing
record whether the evaluator is an official MATLAB script, a validated Python port, or another backend
skipping ObsMask / Plane downgrades protocol fidelity
```

## Backend metadata

Every run should record:

```text
backend kind
backend name
backend version
library version
important parameters
whether approximate algorithms were used
```

Example:

```json
{
  "nearest_neighbor": {
    "name": "scipy",
    "library": "scipy",
    "version": "1.13.1",
    "approximate": false
  },
  "mesh": {
    "name": "trimesh",
    "version": "4.4.0"
  }
}
```

## Backend tests

Required tests:

```text
backend registry lookup
unknown backend name error message (names the kind and available names)
mesh sampling determinism
nearest-neighbor correctness on small point sets
trajectory backend output normalization
camera convention normalization
T&T wrapper dry-run or fixture output parsing
DTU backend fixture with ObsMask / Plane behavior
```

All backends are always installed, so backend tests do not use `pytest.importorskip`. Only the DTU MATLAB external-tool path may skip when MATLAB is absent, and it must skip with an explicit reason.

## Backend non-goals

Do not add backends that perform these **as reconstruction**:

```text
TSDF integration
RGB-D fusion
volumetric fusion
online mapping
SLAM tracking
learned reconstruction inference
```

Depth IO is allowed. Depth integration into scene reconstructions is not part of eval3r.

**Exception (visibility culling only):** the `visibility` backend kind may render a
prediction's depth from the GT trajectory (pyrender, EGL offscreen) and TSDF-integrate
(open3d) those rendered depths *only to trim the prediction to the observed region* before
scoring — the community ScanNet single-/double-layer convention. This produces no new scene
geometry. It is enabled only when a protocol's masking requests it, and the renderer + TSDF
backend, versions, voxel size, trajectory fingerprint, and per-scene culled fraction are
recorded in result metadata. See CLAUDE.md "Evaluation-time visibility culling exception".
