"""`e3r protocol` command group: inspect built-in evaluation protocols.

Verbose by design (CLAUDE.md CLI rules): ``show`` echoes the resolved protocol
identity, its canonical hash, and the alignment / masking / sampling / confidence /
failure / metric policies actually in effect.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from eval3r.core.errors import ProtocolError
from eval3r.core.hashing import protocol_hash
from eval3r.protocols import list_protocols, load_protocol

protocol_app = typer.Typer(
    help="Inspect built-in evaluation protocols.",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


@protocol_app.command("list")
def protocol_list() -> None:
    """List the names of all built-in protocols."""
    names = list_protocols()
    if not names:
        err_console.print("[red]No built-in protocols are registered.[/red]")
        raise typer.Exit(code=1)
    table = Table(title=f"Built-in protocols ({len(names)})")
    table.add_column("name", style="bold cyan")
    table.add_column("fidelity")
    table.add_column("hash", overflow="fold")
    for name in names:
        proto = load_protocol(name)
        table.add_row(name, proto.fidelity, protocol_hash(proto))
    console.print(table)


@protocol_app.command("show")
def protocol_show(
    name: str = typer.Argument(..., help="Built-in protocol name or path to a protocol YAML."),
) -> None:
    """Show a protocol's resolved fields and canonical hash."""
    try:
        proto = load_protocol(name)
    except ProtocolError as exc:
        # Print the full, explicit reason — never a bare non-zero exit.
        err_console.print(f"[red]Failed to load protocol:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    phash = protocol_hash(proto)

    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold")
    header.add_column()
    header.add_row("name", proto.name)
    header.add_row("protocol_version", proto.protocol_version)
    header.add_row("schema_version", str(proto.schema_version))
    header.add_row("fidelity", proto.fidelity)
    header.add_row("protocol_hash", phash)
    header.add_row(
        "dataset",
        f"{proto.dataset.dataset} / {proto.dataset.variant} / {proto.dataset.split}",
    )
    header.add_row("prediction_modality", proto.prediction_modality)
    header.add_row("local_evaluation", proto.local_evaluation.status)
    console.print(header)

    policies = Table(title="Resolved policies", show_header=True)
    policies.add_column("policy", style="bold")
    policies.add_column("value")
    policies.add_row(
        "alignment",
        f"{proto.alignment.mode} (granularity={proto.alignment.granularity})",
    )
    policies.add_row("confidence", proto.confidence.policy)
    policies.add_row(
        "masking",
        f"pred={proto.masking.pred_culling.method}, gt={proto.masking.gt_culling.method}, "
        f"valid_region={proto.masking.valid_region.method}",
    )
    policies.add_row(
        "sampling",
        f"pred={proto.sampling.pred.method}, gt={proto.sampling.gt.method}",
    )
    policies.add_row("failure_policy", proto.failure_policy.policy)
    policies.add_row(
        "metrics",
        ", ".join(m.name for m in proto.metrics) or "(none — server-only / no local metrics)",
    )
    policies.add_row(
        "backend_preferences",
        ", ".join(f"{k}={v}" for k, v in proto.backend_preferences.items()) or "(defaults)",
    )
    console.print(policies)
