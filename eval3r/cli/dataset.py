"""`e3r dataset` command group: inspect dataset adapters and capabilities.

`list` shows registered adapters; `inspect` shows an adapter's declared
capabilities and — when a ``--root`` is given — the scenes it discovers for a
split. Capability honesty matters: the output surfaces whether GT is independent,
whether local official evaluation exists, and whether a split is server-only, so a
user cannot mistake an eval3r-native comparison for an official benchmark.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from eval3r.core.errors import Eval3rError
from eval3r.datasets import default_registry as default_dataset_registry

dataset_app = typer.Typer(
    help="Inspect dataset adapters and their capabilities.",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


@dataset_app.command("list")
def dataset_list() -> None:
    """List registered dataset adapter names."""
    names = default_dataset_registry().available()
    if not names:
        err_console.print("[red]No dataset adapters are registered.[/red]")
        raise typer.Exit(code=1)
    console.print("Registered dataset adapters: " + ", ".join(names))


@dataset_app.command("inspect")
def dataset_inspect(
    dataset: str = typer.Argument(..., help="Registered dataset adapter name."),
    root: Path | None = typer.Option(None, "--root", help="Dataset root (enables discovery)."),
    split: str | None = typer.Option(None, "--split", help="Split to list scenes (needs --root)."),
) -> None:
    """Show a dataset adapter's capabilities and (optionally) its discovered scenes."""
    try:
        adapter = default_dataset_registry().create(dataset, root)
    except Eval3rError as exc:
        err_console.print(f"[red]Failed to build adapter:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    caps = adapter.capabilities
    table = Table(title=f"dataset '{adapter.name}' capabilities", show_header=False, box=None)
    table.add_column(style="bold cyan")
    table.add_column()
    for name_ in caps.__class__.model_fields:
        if name_ == "notes":
            continue
        table.add_row(name_, str(getattr(caps, name_)))
    console.print(table)
    for note in caps.notes:
        console.print(f"[dim]note:[/] {note}")

    if split is not None:
        try:
            scenes = list(adapter.iter_scenes(split))
        except Eval3rError as exc:
            err_console.print(f"[red]Failed to list scenes:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        console.print(f"\nsplit '{split}': {len(scenes)} scenes")
        console.print(", ".join(scenes))
