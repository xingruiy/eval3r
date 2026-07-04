"""Protocol YAML loader.

Parses a protocol YAML document into a validated :class:`EvalProtocol`, raising
:class:`ProtocolValidationError` with the source path and the failing fields when
validation fails (never a bare exception or boolean, per CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from eval3r.core.errors import ProtocolValidationError
from eval3r.core.protocol import EvalProtocol


def _format_validation_error(exc: ValidationError) -> str:
    """Render a Pydantic error as one line per failing field: ``loc: message``."""
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']} (input={err.get('input')!r})")
    return "\n".join(lines)


def load_protocol_data(data: dict[str, Any], *, source: str) -> EvalProtocol:
    """Validate an already-parsed protocol mapping into an :class:`EvalProtocol`."""
    if not isinstance(data, dict):
        raise ProtocolValidationError(
            source, f"top-level YAML must be a mapping, got {type(data).__name__}"
        )
    name = data.get("name") if isinstance(data.get("name"), str) else None
    try:
        return EvalProtocol.model_validate(data)
    except ValidationError as exc:
        raise ProtocolValidationError(
            source, _format_validation_error(exc), name=name
        ) from exc


def load_protocol_text(text: str, *, source: str) -> EvalProtocol:
    """Parse and validate a protocol from a YAML string."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ProtocolValidationError(source, f"YAML parse error: {exc}") from exc
    return load_protocol_data(data, source=source)


def load_protocol_file(path: str | Path) -> EvalProtocol:
    """Load and validate a protocol from a YAML file path."""
    path = Path(path)
    if not path.is_file():
        raise ProtocolValidationError(str(path), "protocol file does not exist")
    return load_protocol_text(path.read_text(encoding="utf-8"), source=str(path))
