"""Explicit eval3r exception types with rich, actionable messages.

Per CLAUDE.md, handled errors must state what failed, why, and the concrete inputs
involved (file path, protocol name, the field/check that failed). These exception
types carry that context rather than reducing failures to booleans or bare exits.
"""

from __future__ import annotations


class Eval3rError(Exception):
    """Base class for all eval3r errors."""


class InvalidGeometryError(Eval3rError):
    """A point/mesh array had the wrong shape, was empty, or held non-finite values."""


class ProtocolError(Eval3rError):
    """A protocol could not be loaded, found, or validated."""


class ProtocolNotFoundError(ProtocolError):
    """A protocol name did not resolve to a known built-in or file."""

    def __init__(self, name: str, known: list[str]) -> None:
        self.name = name
        self.known = known
        known_txt = ", ".join(sorted(known)) if known else "(none registered)"
        super().__init__(
            f"protocol '{name}' is not a known built-in protocol. "
            f"Known protocols: {known_txt}. "
            f"Pass an exact built-in name, or a path to a protocol YAML file."
        )


class ProtocolValidationError(ProtocolError):
    """A protocol YAML failed schema validation.

    Wraps the underlying Pydantic error with the source path and protocol name so
    the message points at the file and field that failed.
    """

    def __init__(self, source: str, detail: str, *, name: str | None = None) -> None:
        self.source = source
        self.detail = detail
        self.name = name
        named = f" (protocol '{name}')" if name else ""
        super().__init__(f"protocol file failed validation: {source}{named}\n{detail}")
