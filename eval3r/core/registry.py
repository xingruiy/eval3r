"""Backend registry: selection by kind + name, and version recording.

All backends are always installed (CLAUDE.md / ``.agent/backends.md``), so there is
no missing-dependency install path and no install hints. ``require`` fails with an
explicit message that names the kind and the available names when a name is unknown.

Registry kinds match the ``backend_preferences`` keys used in protocol YAMLs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from eval3r.core.errors import Eval3rError

# Registry kinds (match .agent/backends.md and EvalProtocol.backend_preferences keys).
BACKEND_KINDS: tuple[str, ...] = (
    "mesh",
    "pointcloud",
    "nearest_neighbor",
    "registration",
    "trajectory",
    "camera",
    "depth_io",
    "official_eval",
)


class BackendError(Eval3rError):
    """A backend could not be selected or is misconfigured."""


class UnknownBackendKindError(BackendError):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        super().__init__(
            f"unknown backend kind '{kind}'. Known kinds: {', '.join(BACKEND_KINDS)}."
        )


class UnknownBackendError(BackendError):
    """A backend name is not registered for a given kind."""

    def __init__(self, kind: str, name: str, available: list[str]) -> None:
        self.kind = kind
        self.name = name
        self.available = available
        avail_txt = ", ".join(sorted(available)) if available else "(none registered)"
        super().__init__(
            f"no backend named '{name}' is registered for kind '{kind}'. "
            f"Available '{kind}' backends: {avail_txt}."
        )


@dataclass(frozen=True)
class BackendInfo:
    """Version metadata recorded for every backend actually used in a run."""

    kind: str
    name: str
    library: str
    version: str
    approximate: bool = False

    def as_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "library": self.library,
            "version": self.version,
            "approximate": self.approximate,
        }


# --- backend Protocol interfaces (structural typing) ---------------------------


@runtime_checkable
class Backend(Protocol):
    """Common structural interface: a name and version metadata."""

    name: str

    def backend_info(self) -> BackendInfo: ...


@runtime_checkable
class MeshBackend(Protocol):
    name: str

    def backend_info(self) -> BackendInfo: ...

    def load_mesh(self, path: Path) -> Any: ...

    def sample_surface(
        self,
        mesh: Any,
        n_points: int,
        seed: int,
        return_normals: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]: ...

    def export_mesh(self, mesh: Any, path: Path) -> None: ...


@runtime_checkable
class PointCloudBackend(Protocol):
    name: str

    def backend_info(self) -> BackendInfo: ...

    def load_pointcloud(self, path: Path) -> np.ndarray: ...

    def save_pointcloud(
        self,
        points: np.ndarray,
        path: Path,
        colors: np.ndarray | None = None,
    ) -> None: ...


@runtime_checkable
class NNBackend(Protocol):
    name: str

    def backend_info(self) -> BackendInfo: ...

    def nearest_distances(
        self,
        query_points: np.ndarray,
        reference_points: np.ndarray,
    ) -> np.ndarray: ...


# --- registry ------------------------------------------------------------------


class BackendRegistry:
    """Maps ``(kind, name)`` to a backend instance."""

    def __init__(self) -> None:
        self._backends: dict[str, dict[str, Any]] = {k: {} for k in BACKEND_KINDS}

    def register(self, kind: str, backend: Any) -> None:
        if kind not in self._backends:
            raise UnknownBackendKindError(kind)
        name = getattr(backend, "name", None)
        if not isinstance(name, str) or not name:
            raise BackendError(
                f"backend for kind '{kind}' must expose a non-empty string 'name' attribute; "
                f"got {backend!r}"
            )
        self._backends[kind][name] = backend

    def available(self, kind: str) -> list[str]:
        if kind not in self._backends:
            raise UnknownBackendKindError(kind)
        return sorted(self._backends[kind])

    def get(self, kind: str, name: str) -> Any:
        """Alias of :meth:`require`; both raise on an unknown name."""
        return self.require(kind, name)

    def require(self, kind: str, name: str) -> Any:
        if kind not in self._backends:
            raise UnknownBackendKindError(kind)
        backends = self._backends[kind]
        if name not in backends:
            raise UnknownBackendError(kind, name, list(backends))
        return backends[name]

    def info(self, kind: str, name: str) -> BackendInfo:
        return self.require(kind, name).backend_info()

    def backend_versions(self, preferences: dict[str, str]) -> dict[str, dict[str, Any]]:
        """Collect version metadata for a protocol's resolved backend preferences.

        Returns the ``kind -> {name, library, version, approximate}`` structure used
        in ``backend_versions.json`` / result metadata.
        """
        out: dict[str, dict[str, Any]] = {}
        for kind, name in preferences.items():
            out[kind] = self.info(kind, name).as_metadata()
        return out


_DEFAULT_REGISTRY: BackendRegistry | None = None


def default_registry() -> BackendRegistry:
    """Return the process-wide registry, populating built-in backends on first use.

    Backends are registered lazily here (rather than at import time) so importing
    ``eval3r.core.registry`` has no heavy side effects and there is no import cycle.
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        registry = BackendRegistry()
        from eval3r.backends.mesh_trimesh import TrimeshMeshBackend
        from eval3r.backends.nn_scipy import ScipyNNBackend
        from eval3r.backends.pointcloud_plyfile import PlyfilePointCloudBackend

        registry.register("mesh", TrimeshMeshBackend())
        registry.register("pointcloud", PlyfilePointCloudBackend())
        registry.register("nearest_neighbor", ScipyNNBackend())
        _DEFAULT_REGISTRY = registry
    return _DEFAULT_REGISTRY
