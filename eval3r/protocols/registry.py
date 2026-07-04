"""Built-in protocol registry.

Discovers the protocol YAML files shipped under ``protocols/builtin/`` and resolves
a protocol name (or an explicit file path) to a validated :class:`EvalProtocol`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from eval3r.core.errors import ProtocolNotFoundError
from eval3r.core.protocol import EvalProtocol
from eval3r.protocols.loader import load_protocol_file

BUILTIN_DIR = Path(__file__).resolve().parent / "builtin"


@lru_cache(maxsize=1)
def _builtin_index() -> dict[str, Path]:
    """Map built-in protocol name (YAML stem) -> file path."""
    if not BUILTIN_DIR.is_dir():
        return {}
    return {p.stem: p for p in sorted(BUILTIN_DIR.glob("*.yaml"))}


def list_protocols() -> list[str]:
    """Names of all built-in protocols, sorted."""
    return sorted(_builtin_index())


def builtin_path(name: str) -> Path:
    """Path of a built-in protocol by name, or raise :class:`ProtocolNotFoundError`."""
    index = _builtin_index()
    if name not in index:
        raise ProtocolNotFoundError(name, list(index))
    return index[name]


def load_protocol(name_or_path: str | Path) -> EvalProtocol:
    """Resolve a built-in protocol name or a YAML file path to an ``EvalProtocol``.

    A value that exists on disk as a file is loaded directly; otherwise it is looked
    up as a built-in name.
    """
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"} or candidate.is_file():
        return load_protocol_file(candidate)
    return load_protocol_file(builtin_path(str(name_or_path)))
