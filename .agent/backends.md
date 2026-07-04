# eval3r Backends

Backends provide small adapter interfaces around external libraries and official evaluation tools. They keep eval3r focused on protocols, schemas, dataset adapters, metric definitions, and reproducibility records.

The base package should stay light. Heavy or specialized libraries must be optional extras.

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

## Base dependencies

Base install:

```toml
dependencies = [
  "numpy",
  "scipy",
  "pandas",
  "pyyaml",
  "typer",
  "rich",
  "pydantic",
]
```

The base package must not require:

```text
Open3D
PyTorch
PyCOLMAP
FAISS
evo
Waymo tooling
MATLAB
```

## Optional extras

Recommended extras:

```toml
[project.optional-dependencies]
mesh = ["trimesh", "plyfile"]
open3d = ["open3d"]
pose = ["evo"]
colmap = ["pycolmap"]
depth = ["imageio", "opencv-python"]
torch = ["torch"]
faiss = ["faiss-cpu"]
waymo = ["waymo-open-dataset"]
all = [
  "trimesh",
  "plyfile",
  "open3d",
  "evo",
  "pycolmap",
  "imageio",
  "opencv-python",
]
```

The `all` extra intentionally excludes `torch`, `faiss-cpu`, and `waymo-open-dataset`; those remain individually opt-in because they are heavy or platform-sensitive.

The base install contains no point-cloud or mesh file loader. Loading a `.ply` for the single-file quick-check path requires the lightweight `mesh` extra (`trimesh`, `plyfile`); default backend preferences should point at `trimesh`/`plyfile`, not Open3D.

Missing optional dependencies should fail with clear messages:

```text
This protocol requires the optional 'pose' extra because it uses evo.
Install with: pip install 'eval3r[pose]'
```

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
    def require(self, kind: str, name: str, extra: str | None = None) -> Backend: ...
```

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
PyTorch / PyTorch3D
FAISS
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
Waymo parser when implemented
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
Waymo uses per-sensor calibrations and vehicle-to-global frame poses.
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
optional extra required
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
missing optional dependency error message
mesh sampling determinism
nearest-neighbor correctness on small point sets
trajectory backend output normalization
camera convention normalization
T&T wrapper dry-run or fixture output parsing
DTU backend fixture with ObsMask / Plane behavior
```

Optional backend tests should use:

```python
pytest.importorskip("open3d")
pytest.importorskip("trimesh")
pytest.importorskip("evo")
pytest.importorskip("pycolmap")
```

## Backend non-goals

Do not add backends for:

```text
TSDF integration
RGB-D fusion
volumetric fusion
online mapping
SLAM tracking
learned reconstruction inference
```

Depth IO is allowed. Depth integration into scene reconstructions is not part of eval3r.
