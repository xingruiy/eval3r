"""Optional-dependency import helper."""

from __future__ import annotations

import importlib
from types import ModuleType

from eval3r.utils.errors import MissingOptionalDependencyError


def optional_import(module: str, *, extra: str) -> ModuleType:
    """Import ``module`` or raise a clear install hint pointing at extra ``[extra]``."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise MissingOptionalDependencyError(
            f"{module!r} is required for this feature. "
            f"Install with: pip install 'eval3r[{extra}]'"
        ) from e
