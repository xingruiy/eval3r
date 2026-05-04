"""eval3r validate <path>"""

from __future__ import annotations

import json
import sys

from rich.console import Console
from rich.table import Table

from eval3r.prediction.validate import validate_prediction


def run(path: str, *, json_out: bool = False) -> None:
    report = validate_prediction(path)
    if json_out:
        out = {
            "ok": report.ok,
            "errors": report.errors,
            "warnings": report.warnings,
            "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in report.checks],
        }
        print(json.dumps(out, indent=2))
    else:
        table = Table(title=f"validate {path}")
        table.add_column("check", style="cyan")
        table.add_column("ok", style="white")
        table.add_column("detail", style="white")
        for name, ok, detail in report.checks:
            table.add_row(name, "[green]✓[/]" if ok else "[red]✗[/]", detail)
        Console().print(table)
        if report.errors:
            Console().print(f"[red]{len(report.errors)} error(s)[/]")
        if report.warnings:
            Console().print(f"[yellow]{len(report.warnings)} warning(s)[/]")
    if not report.ok:
        sys.exit(1)
