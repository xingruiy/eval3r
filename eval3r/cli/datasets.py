"""e3r datasets — list, show, validate dataset adapters."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from eval3r.datasets import get_dataset, list_datasets

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command("list")
def list_cmd() -> None:
    """List registered dataset adapters."""
    for name in list_datasets():
        typer.echo(name)


@app.command("show")
def show_cmd(
    name: str = typer.Argument(..., help="Dataset adapter name (e.g., scannet)."),
) -> None:
    """Print expected layout + supported assets for a dataset adapter."""
    cls = get_dataset(name)
    console = Console()
    console.print(f"[bold]{cls.name}[/]")
    console.print("[bold]expected layout:[/]")
    console.print(cls.expected_layout)


@app.command("validate")
def validate_cmd(
    name: str = typer.Argument(..., help="Dataset adapter name."),
    root: str = typer.Option(..., "--root", help="Filesystem root for the dataset."),
    split: str | None = typer.Option(
        None,
        "--split",
        help="Path to a split file (one scene id per line). Omit to auto-discover scenes.",
    ),
    scenes: int = typer.Option(1, help="Number of scenes to sample-check."),
    # Generic adapter overrides.
    adapter_opt: list[str] = typer.Option(
        [], "-o", "--adapter-opt",
        help="Adapter-specific override in key=value form (repeatable).",
    ),
    # Legacy ScanNet-style overrides (mapped to adapter_opt keys).
    color_subdir: str | None = typer.Option(None, "--color-subdir"),
    depth_subdir: str | None = typer.Option(None, "--depth-subdir"),
    pose_subdir: str | None = typer.Option(None, "--pose-subdir"),
    intrinsics_subdir: str | None = typer.Option(None, "--intrinsics-subdir"),
    mesh_filename: str | None = typer.Option(None, "--mesh-filename"),
    color_format: str | None = typer.Option(None, "--color-format"),
    depth_format: str | None = typer.Option(None, "--depth-format"),
    pose_format: str | None = typer.Option(None, "--pose-format"),
) -> None:
    """Sample-check a dataset root matches the adapter's expected layout."""
    cls = get_dataset(name)
    # Build overrides from both generic -o and legacy named options.
    overrides: dict[str, str] = {}
    for item in adapter_opt:
        if "=" not in item:
            raise typer.BadParameter(
                f"Adapter opt must be key=value, got: {item!r}"
            )
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()
    # Legacy named options (mapped to standard adapter kwarg names).
    overrides.update({
        k: v
        for k, v in {
            "color_subdir": color_subdir,
            "depth_subdir": depth_subdir,
            "pose_subdir": pose_subdir,
            "intrinsics_subdir": intrinsics_subdir,
            "mesh_filename": mesh_filename,
            "color_format": color_format,
            "depth_format": depth_format,
            "pose_format": pose_format,
        }.items()
        if v is not None
    })
    try:
        ds = cls(root, split=split, validate_on_init=False, **overrides)  # type: ignore[arg-type]
    except Exception as e:
        typer.secho(f"construction failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    report = ds.validate(scenes=scenes)
    table = Table(title=f"validate {name}")
    table.add_column("check", style="cyan")
    table.add_column("ok", style="white")
    table.add_column("detail", style="white")
    for check, ok, detail in report.checks:
        table.add_row(check, "[green]✓[/]" if ok else "[red]✗[/]", detail)
    Console().print(table)
    if not report.ok:
        Console().print(f"[red]{len(report.errors)} error(s)[/]")
        raise typer.Exit(code=1)
