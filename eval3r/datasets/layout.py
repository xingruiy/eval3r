"""Path templating + missing-path error formatting for dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Formatter

from eval3r.utils.errors import MissingArtifactError


def format_path(template: str, /, **subs: object) -> str:
    """``str.format`` with a clearer error when a placeholder is missing."""
    fmt = Formatter()
    needed = {n for _, n, _, _ in fmt.parse(template) if n}
    missing = needed - set(subs)
    if missing:
        raise KeyError(
            f"Path template {template!r} is missing substitutions: {sorted(missing)}"
        )
    return template.format(**subs)


@dataclass(frozen=True)
class LayoutEntry:
    """One row of an adapter's expected-layout table."""

    path: str
    overrides: tuple[str, ...]

    def render(self) -> str:
        return f"  {self.path:<46s} <{', '.join(self.overrides)}>"


def render_layout(entries: list[LayoutEntry]) -> str:
    return "\n".join(e.render() for e in entries)


def raise_missing(
    *,
    dataset: str,
    asset: str,
    tried: Path,
    overrides: tuple[str, ...],
    layout: list[LayoutEntry] | None = None,
) -> None:
    """Raise ``MissingArtifactError`` with the standard hint format."""
    msg = [
        f"{dataset}: {asset} not found at {tried}",
        f"Hint: pass {' or '.join(f'`{o}=...`' for o in overrides)} to override.",
    ]
    if layout:
        msg.append("Expected layout (override flags in <angle brackets>):")
        msg.append(render_layout(layout))
    raise MissingArtifactError("\n".join(msg))
