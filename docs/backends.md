# Backends

Backends are small adapter interfaces around external libraries and official evaluation
tools. eval3r keeps protocol meaning, schema validation, result structure, failure
accounting, hashing, and metric definitions to itself, and delegates commodity work:

| Kind | Backends | Delegates to |
|---|---|---|
| `mesh` | `trimesh` (default), `open3d` | mesh loading, deterministic surface sampling |
| `pointcloud` | `plyfile` (default), `open3d` | point-cloud IO |
| `nearest_neighbor` | `scipy` (default), `open3d` | NN distance queries (exact; cKDTree) |
| `registration` | `open3d` | ICP (never runs silently; fully recorded) |
| `trajectory` | `evo` | association, ATE/RPE, SE3/Sim3 alignment |
| `camera` | `pycolmap` | COLMAP models, non-pinhole cameras |
| `depth_io` | `imageio` (default), `opencv` | depth/mask loading (OpenCV additionally reads PFM) |
| `official_eval` | `dtu_eval_python`, `tnt_official`, `eth3d_official` | official benchmark behavior |
| `visibility` | pyrender + open3d | protocol-requested visibility culling only |

Protocols select backends via `backend_preferences` (kind → name). Defaults stay on
trimesh/plyfile/scipy; Open3D is used only where a protocol asks for it.

## Version recording

Backend names, library versions, and result-affecting parameters are written into every
run (`backend_versions.json` and `results.json`), including official tool
versions/commits and the exact command used to invoke them:

```json
{
  "nearest_neighbor": {"name": "scipy", "library": "scipy", "version": "1.15.3", "approximate": false},
  "mesh": {"name": "trimesh", "version": "4.4.0"}
}
```

## Official evaluation backends

Official wrappers exist where faithful reimplementation is risky. They drive the **real**
official code end-to-end — eval3r never substitutes a lookalike evaluator, and a missing
tool fails explicitly rather than producing unofficial numbers.

- `dtu_eval_python` — validated Python port of the official DTU MATLAB evaluation
  (ObsMask/Plane included), regression-tested against the reference; the MATLAB script is
  an optional external path, and which evaluator ran is recorded.
- `tnt_official` — subprocess wrapper around the official Tanks and Temples toolbox,
  located via `EVAL3R_TNT_TOOLBOX`, run unmodified under the interpreter given by
  `EVAL3R_TNT_PYTHON` (the toolbox pins `open3d==0.9`; porting it would change scoring).
  Per-scene thresholds come from the official output.
- `eth3d_official` — subprocess wrapper around the official ETH3D
  `multi-view-evaluation` binary, located via `EVAL3R_ETH3D_TOOL`. Voxel/beam parameters
  stay at the official defaults and are recorded per scene.

See [Installation](install.md#external-official-evaluation-tools) for setup.

## The visibility-culling exception

eval3r contains no reconstruction backends (no TSDF integration, RGB-D fusion, or SLAM as
methods). The one narrow exception: the `visibility` backend may render a *prediction's*
depth from the GT camera trajectory (pyrender, EGL offscreen) and TSDF-integrate those
renders (open3d) **solely to trim the prediction to the observed region** before scoring
— the community ScanNet convention. No new scene geometry is produced; it runs only when
a protocol's masking requests it; and the renderer, TSDF backend, versions, voxel size,
trajectory fingerprint, and per-scene culled fraction are recorded in result metadata.

## Non-dependencies

The project deliberately does not depend on PyTorch, FAISS, or Waymo tooling. MATLAB is
the only optional external *tool* (DTU official-script path only); everything
pip-installable is a required base dependency.
