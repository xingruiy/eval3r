"""Bundled artefacts shipped with the eval3r package.

Anything that needs to travel inside the wheel — dataset splits, sample
configs, demo geometry — lives here. Layout under ``eval3r/assets/`` is free
to grow; new categories should add their own subdirectory (e.g.
``eval3r/assets/datasets/<name>/``, ``eval3r/assets/presets/``).

Use :func:`asset_text` / :func:`asset_path` to access bundled files instead of
constructing paths from ``__file__``; the helpers go through
``importlib.resources`` so they work the same in editable installs and built
wheels.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

__all__ = ["asset_path", "asset_text", "ASSETS_PACKAGE"]

ASSETS_PACKAGE = "eval3r.assets"


def asset_path(*parts: str) -> Path:
    """Return a filesystem path to a bundled asset.

    Example::

        asset_path("datasets", "scannet", "test.txt")
        # -> Path('/.../eval3r/assets/datasets/scannet/test.txt')
    """
    target = resources.files(ASSETS_PACKAGE).joinpath(*parts)
    return Path(str(target))


def asset_text(*parts: str, encoding: str = "utf-8") -> str:
    """Read the contents of a bundled text asset."""
    target = resources.files(ASSETS_PACKAGE).joinpath(*parts)
    return target.read_text(encoding=encoding)
