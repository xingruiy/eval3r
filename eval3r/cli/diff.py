"""`e3r diff` command: compare two run result directories.

Strict by default: runs with different protocol hashes are refused with the full
reason (their metric numbers are not comparable). ``--loose`` allows the
comparison but the output is prominently labeled NON-STRICT. Every comparability
warning from ``.agent/reproducibility.md`` (GT provenance, coverage, failure
policy, alignment, confidence, sampling, backend officialness) is printed, and
each run's partial coverage is shown with the same banner behavior as the report
formats — a partial aggregate is never displayed as a full one.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from eval3r.core.errors import Eval3rError
from eval3r.reports.diff import RunDiff, RunIdentity, diff_runs
from eval3r.reports.table import CoverageInfo, coverage_banner

console = Console()
err_console = Console(stderr=True)


def _value(v: float | None) -> str:
    return "-" if v is None else f"{v:.6g}"


def _echo_run(label: str, ident: RunIdentity) -> None:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("run dir", ident.run_dir)
    table.add_row("method", ident.method or "(unnamed)")
    table.add_row(
        "dataset", f"{ident.dataset} / {ident.variant or '-'} (split {ident.split or '-'})"
    )
    table.add_row("protocol", f"{ident.protocol} v{ident.protocol_version}")
    table.add_row("protocol hash", ident.protocol_hash)
    table.add_row("fidelity", ident.fidelity)
    table.add_row(
        "coverage",
        f"{ident.n_scenes_evaluated}/{ident.n_scenes_expected} scenes evaluated, "
        f"{ident.n_scenes_failed} failed",
    )
    console.print(Panel(table, title=label, expand=False))
    banner = coverage_banner(
        CoverageInfo(
            expected=ident.n_scenes_expected,
            evaluated=ident.n_scenes_evaluated,
            failed=ident.n_scenes_failed,
            failure_policy=ident.failure_policy,
        )
    )
    if banner is not None:
        console.print(f"[bold yellow]⚠ {label}: {banner}[/]")


def _echo_diff(diff: RunDiff) -> None:
    if not diff.strict:
        console.print(
            Panel(
                "NON-STRICT COMPARISON: the runs were compared with --loose. "
                "Differences below may come from the protocols, not the methods.",
                style="bold yellow",
                expand=False,
            )
        )

    _echo_run("run A", diff.run_a)
    _echo_run("run B", diff.run_b)

    if diff.warnings:
        warn = Table(title="comparability warnings", header_style="bold yellow")
        warn.add_column("field", no_wrap=True)
        warn.add_column("run A")
        warn.add_column("run B")
        warn.add_column("why it matters")
        for w in diff.warnings:
            warn.add_row(w.field, w.value_a, w.value_b, w.reason)
        console.print(warn)
    else:
        console.print("[green]no comparability warnings: the runs are comparable.[/]")

    metrics = Table(title="aggregate metrics (delta = B - A)", header_style="bold")
    metrics.add_column("metric")
    metrics.add_column("run A", justify="right")
    metrics.add_column("run B", justify="right")
    metrics.add_column("delta", justify="right")
    for m in diff.metrics:
        metrics.add_row(m.name, _value(m.value_a), _value(m.value_b), _value(m.delta))
    console.print(metrics)

    if diff.per_scene:
        per_scene = Table(title="per-scene metrics (delta = B - A)", header_style="bold")
        per_scene.add_column("scene")
        per_scene.add_column("metric")
        per_scene.add_column("run A", justify="right")
        per_scene.add_column("run B", justify="right")
        per_scene.add_column("delta", justify="right")
        for scene, deltas in sorted(diff.per_scene.items()):
            for m in deltas:
                per_scene.add_row(
                    scene, m.name, _value(m.value_a), _value(m.value_b), _value(m.delta)
                )
        console.print(per_scene)

    if diff.scenes_only_in_a:
        console.print(
            f"[yellow]scenes only in run A ({len(diff.scenes_only_in_a)}):[/] "
            + ", ".join(diff.scenes_only_in_a)
        )
    if diff.scenes_only_in_b:
        console.print(
            f"[yellow]scenes only in run B ({len(diff.scenes_only_in_b)}):[/] "
            + ", ".join(diff.scenes_only_in_b)
        )


def diff_command(
    run_a: Path = typer.Argument(..., help="First run directory (or its results.json)."),
    run_b: Path = typer.Argument(..., help="Second run directory (or its results.json)."),
    loose: bool = typer.Option(
        False,
        "--loose",
        help=(
            "Allow comparing runs with different protocol hashes. The output is "
            "labeled NON-STRICT; differences may come from the protocols, not the "
            "methods."
        ),
    ),
) -> None:
    """Compare two run result directories (refuses mismatched protocol hashes)."""
    console.print(f"[bold]comparing runs:[/] A={run_a}  B={run_b}  (loose={loose})")
    try:
        diff = diff_runs(run_a, run_b, loose=loose)
    except Eval3rError as exc:
        err_console.print(Panel(str(exc), title="diff refused", style="red", expand=False))
        raise typer.Exit(code=1) from exc
    _echo_diff(diff)
