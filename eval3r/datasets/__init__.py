"""eval3r dataset adapters.

Adapters own per-dataset filesystem layout and asset loaders. The benchmark
orchestrator calls them; nothing here knows about metrics.
"""

from __future__ import annotations

from typing import Type

from eval3r.datasets.base import (
    AdapterValidationReport,
    Asset,
    DatasetAdapter,
)
from eval3r.datasets.layout import LayoutEntry, format_path, raise_missing
from eval3r.datasets.scannet import ScanNetAdapter
from eval3r.datasets.tum_rgbd import TumRGBDAdapter
from eval3r.datasets.replica import ReplicaAdapter
from eval3r.datasets.dtu import DTUAdapter
from eval3r.datasets.eth3d import ETH3DAdapter
from eval3r.datasets.tanks_temples import TanksTemplesAdapter
from eval3r.datasets.generic import GenericAdapter

_REGISTRY: dict[str, Type[DatasetAdapter]] = {}
_ALIASES: dict[str, str] = {}


def register_dataset(cls: Type[DatasetAdapter]) -> Type[DatasetAdapter]:
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} must set a non-empty class attribute 'name'")
    _REGISTRY[cls.name] = cls
    return cls


def get_dataset(name: str) -> Type[DatasetAdapter]:
    canonical = _ALIASES.get(name, name)
    if canonical not in _REGISTRY:
        available = sorted(set(_REGISTRY) | set(_ALIASES))
        raise KeyError(
            f"unknown dataset: {name!r}. Available: {available}"
        )
    return _REGISTRY[canonical]


def list_datasets() -> list[str]:
    """List canonical dataset names (aliases are accepted by get_dataset)."""
    return sorted(_REGISTRY)


# Built-in registrations.
register_dataset(ScanNetAdapter)
register_dataset(TumRGBDAdapter)
register_dataset(ReplicaAdapter)
register_dataset(DTUAdapter)
register_dataset(ETH3DAdapter)
register_dataset(TanksTemplesAdapter)
# Back-compat alias for short Tanks & Temples id.
_ALIASES["tnt"] = TanksTemplesAdapter.name
register_dataset(GenericAdapter)


__all__ = [
    "Asset",
    "DatasetAdapter",
    "AdapterValidationReport",
    "ScanNetAdapter",
    "TumRGBDAdapter",
    "ReplicaAdapter",
    "DTUAdapter",
    "ETH3DAdapter",
    "TanksTemplesAdapter",
    "GenericAdapter",
    "LayoutEntry",
    "format_path",
    "raise_missing",
    "register_dataset",
    "get_dataset",
    "list_datasets",
]
