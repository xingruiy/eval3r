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

_REGISTRY: dict[str, Type[DatasetAdapter]] = {}


def register_dataset(cls: Type[DatasetAdapter]) -> Type[DatasetAdapter]:
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} must set a non-empty class attribute 'name'")
    _REGISTRY[cls.name] = cls
    return cls


def get_dataset(name: str) -> Type[DatasetAdapter]:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown dataset: {name!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_datasets() -> list[str]:
    return sorted(_REGISTRY)


# Built-in registrations.
register_dataset(ScanNetAdapter)


__all__ = [
    "Asset",
    "DatasetAdapter",
    "AdapterValidationReport",
    "ScanNetAdapter",
    "LayoutEntry",
    "format_path",
    "raise_missing",
    "register_dataset",
    "get_dataset",
    "list_datasets",
]
