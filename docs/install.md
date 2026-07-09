# Installation

## Package

```bash
pip install eval3r
```

Until the first PyPI release, install from source:

```bash
git clone https://github.com/xingruiy/eval3r
pip install ./eval3r
```

Requires Python >= 3.10. This installs the `e3r` command.

## Dependency policy

All Python dependencies are **required base dependencies** and are always installed:
numpy, scipy, pandas, pyyaml, typer, rich, pydantic, trimesh, plyfile, open3d, evo,
pycolmap, imageio, opencv-python, pyrender, matplotlib.

There is no optional-extra system: a plain `pip install eval3r` pulls in every backend,
so results never depend on which extras happened to be installed. The project
intentionally does not depend on PyTorch, FAISS, or Waymo tooling.

Two dependencies have narrow roles worth knowing:

- `pyrender` powers only offscreen depth rendering for protocol-requested visibility
  culling (ScanNet). It needs a headless GL context; the backend sets
  `PYOPENGL_PLATFORM=egl` itself when unset.
- `matplotlib` is used only for debug outputs (distance histogram PNGs via the headless
  Agg canvas, and the error colormap). It never participates in metric computation.

## External official evaluation tools

eval3r always evaluates against the **real** official code — it never substitutes a
lookalike evaluator. Three official tools are external and user-supplied (they are not
pip packages). Runs that need an absent tool fail with an explicit message; they never
silently emit unofficial numbers.

### Tanks and Temples official toolbox

The `tnt_official` backend wraps the official
[`isl-org/TanksAndTemples`](https://github.com/isl-org/TanksAndTemples)
`python_toolbox/evaluation` as a subprocess. The toolbox pins `open3d==0.9`, whose
registration API and RANSAC semantics differ from modern open3d, so eval3r runs it
**byte-for-byte unmodified under its own pinned interpreter** instead of porting it
(porting would change the alignment used for scoring).

```bash
git clone https://github.com/isl-org/TanksAndTemples
conda create -n tnt_toolbox python=3.7
conda run -n tnt_toolbox pip install open3d==0.9.0.0 matplotlib

export EVAL3R_TNT_TOOLBOX=/path/to/TanksAndTemples/python_toolbox/evaluation
export EVAL3R_TNT_PYTHON=/path/to/envs/tnt_toolbox/bin/python
```

The resolved toolbox directory, its git commit, and the interpreter used are recorded in
result metadata.

### ETH3D multi-view evaluation binary

The `eth3d_official` backend wraps the official
[`ETH3D/multi-view-evaluation`](https://github.com/ETH3D/multi-view-evaluation) C++ tool
as a subprocess:

```bash
git clone https://github.com/ETH3D/multi-view-evaluation
cd multi-view-evaluation && mkdir build && cd build && cmake .. && make

export EVAL3R_ETH3D_TOOL=/path/to/multi-view-evaluation/build/ETH3DMultiViewEvaluation
```

(`ETH3D_MULTI_VIEW_EVALUATION` is accepted as an alternative variable.) Note for builds
against PCL >= 1.12: upstream's `CMakeLists.txt` carries a stale `-std=c++11` flag that
contradicts its own `CMAKE_CXX_STANDARD 17`; changing that one flag to `-std=c++17` is a
mechanical build fix matching upstream's stated intent (no scoring source is touched).

### DTU MATLAB path (optional)

DTU evaluation uses a **validated Python port** of the official MATLAB evaluation by
default (fidelity `native`, regression-tested against the reference code). MATLAB
is the only optional external tool: if you want the literal MATLAB official script path,
eval3r can drive it, and whether MATLAB was used is recorded in result metadata. A
missing MATLAB fails with an explicit message pointing at the validated port.

## Development install

```bash
git clone https://github.com/xingruiy/eval3r
cd eval3r
pip install -e . --group dev   # PEP 735 dependency group (pip >= 25.1)
# or: pip install -e . && pip install ruff mypy pytest mkdocs build twine
ruff check . && mypy eval3r && pytest && mkdocs build
```

Tests that exercise the external official tools skip cleanly, with an explicit reason,
when the tool is not configured (`EVAL3R_TNT_*`, `EVAL3R_ETH3D_TOOL`, MATLAB).
