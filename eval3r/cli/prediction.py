"""`e3r prediction` command group: inspect eval3r-native prediction directories.

`validate` resolves every declared per-scene file (and, by default, re-hashes it
against the recorded fingerprints) and reports a per-scene outcome table with
every failure verbatim; `show` summarizes the manifest and the scene/file
layout. Both are verbose by design (``rich``, CLAUDE.md CLI verbosity rules):
partial validity is listed scene by scene, never reduced to a bare exit code.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from eval3r.core.errors import Eval3rError
from eval3r.predictions.layout import ENTRY_PATH_FIELDS
from eval3r.predictions.reader import PredictionCheck, check_prediction_dir

prediction_app = typer.Typer(
    help="Validate or inspect an eval3r-native prediction directory.",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


def _manifest_panel(check: PredictionCheck) -> Panel:
    manifest = check.manifest
    dataset = manifest.dataset
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("prediction root", str(check.root))
    method = manifest.method + (f" ({manifest.version})" if manifest.version else "")
    table.add_row("method", method)
    table.add_row(
        "dataset",
        f"{dataset.dataset}"
        + (f" / {dataset.variant}" if dataset.variant else "")
        + (f" / split {dataset.split}" if dataset.split else ""),
    )
    table.add_row("modality", manifest.prediction_modality)
    table.add_row("scale / unit", f"{manifest.scale} / {manifest.unit}")
    table.add_row("coordinate frame", manifest.coordinate_frame)
    table.add_row("source pose format", manifest.source_pose_format)
    table.add_row(
        "uses GT",
        f"pose={manifest.uses_gt.pose} intrinsics={manifest.uses_gt.intrinsics} "
        f"scale={manifest.uses_gt.scale}",
    )
    table.add_row(
        "confidence",
        f"present={manifest.confidence.present} "
        f"self_filtered={manifest.confidence.self_filtered}",
    )
    table.add_row("layout", str(manifest.metadata.get("layout", "(not recorded)")))
    table.add_row(
        "written by eval3r", str(manifest.metadata.get("eval3r_version", "(not recorded)"))
    )
    table.add_row("scenes", str(len(manifest.scenes)))
    return Panel(table, title="prediction manifest", expand=False)


@prediction_app.command("validate")
def prediction_validate(
    root: Path = typer.Argument(..., help="Prediction directory containing manifest.yaml."),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Re-hash every declared file against the fingerprints recorded in the "
        "manifest (--no-verify checks schema and file existence only).",
    ),
) -> None:
    """Validate manifest schema, per-scene file existence, and (default) fingerprints."""
    try:
        check = check_prediction_dir(root, verify=verify)
    except Eval3rError as exc:
        err_console.print(
            Panel(str(exc), title="prediction validation failed", style="red", expand=False)
        )
        raise typer.Exit(code=1) from exc

    console.print(_manifest_panel(check))
    outcomes = Table(title="per-scene validation", header_style="bold")
    outcomes.add_column("scene")
    outcomes.add_column("files")
    outcomes.add_column("outcome")
    for scene_id in check.manifest.scenes:
        fields = ", ".join(sorted(check.scenes.get(scene_id, {}))) or "-"
        if scene_id in check.issues:
            outcomes.add_row(scene_id, fields, "[red]fail[/]")
        else:
            outcomes.add_row(scene_id, fields, "[green]ok[/]")
    console.print(outcomes)

    n_ok = len(check.manifest.scenes) - len(check.issues)
    console.print(
        f"{n_ok}/{len(check.manifest.scenes)} scenes valid "
        f"({'fingerprints verified' if verify else 'existence only, --no-verify'})"
    )
    for scene_id, problems in check.issues.items():
        for problem in problems:
            err_console.print(f"  [red]fail[/] {scene_id}: {problem}")
    if check.issues:
        raise typer.Exit(code=1)


@prediction_app.command("show")
def prediction_show(
    root: Path = typer.Argument(..., help="Prediction directory containing manifest.yaml."),
) -> None:
    """Show the manifest summary and the per-scene file table without hashing."""
    try:
        check = check_prediction_dir(root, verify=False)
    except Eval3rError as exc:
        err_console.print(
            Panel(str(exc), title="prediction show failed", style="red", expand=False)
        )
        raise typer.Exit(code=1) from exc

    console.print(_manifest_panel(check))
    files = Table(title="scene files", header_style="bold")
    files.add_column("scene")
    files.add_column("field")
    files.add_column("path")
    files.add_column("fingerprint")
    for scene_id, entry in check.manifest.scenes.items():
        recorded = entry.metadata.get("fingerprints", {})
        recorded = recorded if isinstance(recorded, dict) else {}
        for field_name in ENTRY_PATH_FIELDS:
            declared = getattr(entry, field_name)
            if declared is None:
                continue
            missing = field_name not in check.scenes.get(scene_id, {})
            files.add_row(
                scene_id,
                field_name,
                f"[red]{declared} (missing)[/]" if missing else str(declared),
                "recorded" if field_name in recorded else "-",
            )
    console.print(files)
    if check.issues:
        console.print(
            f"[yellow]{len(check.issues)}/{len(check.manifest.scenes)} scenes have "
            f"missing files[/] — run `e3r prediction validate` for details."
        )
