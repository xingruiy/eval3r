"""eval3r preset show <name> | preset list"""

from __future__ import annotations

import json

import typer

from eval3r.presets import PRESETS

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command("list")
def list_cmd() -> None:
    """List available presets."""
    for name in sorted(PRESETS):
        typer.echo(name)


@app.command("show")
def show_cmd(name: str = typer.Argument(..., help="Preset name (e.g., scannet).")) -> None:
    """Print a preset as YAML-ish JSON for inspection."""
    if name not in PRESETS:
        raise typer.BadParameter(f"unknown preset: {name}; available: {sorted(PRESETS)}")
    typer.echo(json.dumps(PRESETS[name], indent=2))
