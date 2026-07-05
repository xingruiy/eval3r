"""Dataset-adapter registry: name -> adapter factory.

Adapters are registered as factories ``(root: Path | None) -> DatasetAdapter`` because
a real adapter is bound to a dataset root on disk. ``create`` fails with an explicit,
name-listing message when a dataset is unknown (CLAUDE.md error rules). Built-in
adapters are registered lazily on first use to keep imports cheap.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from eval3r.core.errors import DatasetError, UnknownDatasetError
from eval3r.datasets.base import DatasetAdapter

AdapterFactory = Callable[[Path | None], DatasetAdapter]


class DatasetRegistry:
    """Maps a dataset name to a factory that builds its adapter for a given root."""

    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, name: str, factory: AdapterFactory) -> None:
        if not name:
            raise DatasetError("dataset adapter name must be a non-empty string.")
        self._factories[name] = factory

    def available(self) -> list[str]:
        return sorted(self._factories)

    def create(self, name: str, root: Path | None = None) -> DatasetAdapter:
        if name not in self._factories:
            raise UnknownDatasetError(name, list(self._factories))
        return self._factories[name](root)


_DEFAULT_REGISTRY: DatasetRegistry | None = None


def default_registry() -> DatasetRegistry:
    """Process-wide dataset registry with built-in adapters registered on first use."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        registry = DatasetRegistry()
        from eval3r.datasets.custom import CustomAdapter
        from eval3r.datasets.dtu import DTUAdapter
        from eval3r.datasets.eth3d import Eth3dAdapter
        from eval3r.datasets.scannet import ScanNetAdapter
        from eval3r.datasets.tanks_temples import TanksAndTemplesAdapter

        registry.register("custom", CustomAdapter.factory)
        registry.register("dtu", DTUAdapter.factory)
        registry.register("eth3d", Eth3dAdapter.factory)
        registry.register("scannet", ScanNetAdapter.factory)
        registry.register("tanks_temples", TanksAndTemplesAdapter.factory)
        _DEFAULT_REGISTRY = registry
    return _DEFAULT_REGISTRY
