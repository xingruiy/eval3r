"""Markdown report writer (``results.md``).

Renders the :class:`~eval3r.reports.table.ReportData` view of a ``RunResult`` as
GitHub-flavored Markdown. The partial-coverage banner, coverage counts, failure
policy, and per-scene failure reasons are always rendered — a partial aggregate is
never presented as a full one.
"""

from __future__ import annotations

from pathlib import Path

from eval3r.core.result import RunResult
from eval3r.reports.table import ReportData, build_report_data


def _escape(cell: str) -> str:
    return cell.replace("|", "\\|").replace("\n", " ")


def _table(columns: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(_escape(c) for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape(cell) for cell in row) + " |")
    return lines


def render_markdown(result: RunResult) -> str:
    """Render a full Markdown report for one run."""
    data: ReportData = build_report_data(result)
    lines: list[str] = [f"# {data.title}", ""]

    if data.banner is not None:
        lines += [f"> **⚠ {data.banner}**", ""]

    lines += ["## Run configuration", ""]
    lines += _table(["field", "value"], [[k, v] for k, v in data.header])
    lines += [""]

    cov = data.coverage
    lines += [
        "## Scene coverage",
        "",
        f"- expected scenes: {cov.expected}",
        f"- evaluated scenes: {cov.evaluated}",
        f"- failed scenes: {cov.failed}",
        f"- failure policy: {cov.failure_policy}",
        f"- aggregates are {'PARTIAL' if cov.partial else 'complete'}",
        "",
    ]

    lines += ["## Aggregate metrics", ""]
    lines += _table(data.aggregate_columns, data.aggregate_rows)
    lines += [""]

    lines += ["## Per-scene metrics", ""]
    lines += _table(data.per_scene_columns, data.per_scene_rows)
    lines += [""]

    if data.failures:
        lines += ["## Failed scenes", ""]
        lines += _table(
            ["scene", "stage", "reason"],
            [[scene, stage, reason] for scene, stage, reason in data.failures],
        )
        lines += [""]

    return "\n".join(lines)


def write_markdown_report(result: RunResult, path: Path) -> None:
    """Write ``results.md`` for one run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(result), encoding="utf-8")
