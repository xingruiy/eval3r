"""Dataset adapters: normalize native files to eval3r conventions.

Use :func:`default_registry` to obtain the process-wide dataset registry with the
built-in adapters registered. The :class:`DatasetAdapter` protocol defines the
interface every adapter satisfies.
"""

from __future__ import annotations

from eval3r.datasets.base import DatasetAdapter
from eval3r.datasets.custom import CustomAdapter
from eval3r.datasets.dtu import DTUAdapter
from eval3r.datasets.registry import DatasetRegistry, default_registry
from eval3r.datasets.scannet import ScanNetAdapter

__all__ = [
    "DatasetAdapter",
    "CustomAdapter",
    "DTUAdapter",
    "ScanNetAdapter",
    "DatasetRegistry",
    "default_registry",
]
