"""HTML report writer (``report.html``).

Renders the :class:`~eval3r.reports.table.ReportData` view of a ``RunResult`` as a
single self-contained HTML page (inline CSS, no external assets). The
partial-coverage banner is a prominent colored block at the top of the page so
partial aggregates are impossible to miss.
"""

from __future__ import annotations

import html as _html
from pathlib import Path

from eval3r.core.result import RunResult
from eval3r.reports.table import ReportData, build_report_data

_CSS = """
body { font-family: sans-serif; margin: 2em; color: #222; }
table { border-collapse: collapse; margin: 0.5em 0 1.5em; }
th, td { border: 1px solid #999; padding: 0.3em 0.6em; text-align: left; }
th { background: #eee; }
.banner { background: #b30000; color: #fff; padding: 0.8em 1em; font-weight: bold; }
.coverage { margin: 1em 0; }
"""


def _esc(text: str) -> str:
    return _html.escape(text)


def _table(columns: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["<table>", "<tr>" + "".join(f"<th>{_esc(c)}</th>" for c in columns) + "</tr>"]
    for row in rows:
        lines.append("<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>")
    lines.append("</table>")
    return lines


def render_html(result: RunResult) -> str:
    """Render a self-contained HTML report page for one run."""
    data: ReportData = build_report_data(result)
    lines: list[str] = [
        "<!DOCTYPE html>",
        "<html><head>",
        '<meta charset="utf-8">',
        f"<title>{_esc(data.title)}</title>",
        f"<style>{_CSS}</style>",
        "</head><body>",
        f"<h1>{_esc(data.title)}</h1>",
    ]

    if data.banner is not None:
        lines.append(f'<div class="banner">⚠ {_esc(data.banner)}</div>')

    cov = data.coverage
    lines += [
        '<div class="coverage">',
        f"<p>Scene coverage: {cov.evaluated} of {cov.expected} evaluated, "
        f"{cov.failed} failed (failure policy: {_esc(cov.failure_policy)}); "
        f"aggregates are {'<strong>PARTIAL</strong>' if cov.partial else 'complete'}.</p>",
        "</div>",
    ]

    lines.append("<h2>Run configuration</h2>")
    lines += _table(["field", "value"], [[k, v] for k, v in data.header])

    lines.append("<h2>Aggregate metrics</h2>")
    lines += _table(data.aggregate_columns, data.aggregate_rows)

    lines.append("<h2>Per-scene metrics</h2>")
    lines += _table(data.per_scene_columns, data.per_scene_rows)

    if data.failures:
        lines.append("<h2>Failed scenes</h2>")
        lines += _table(
            ["scene", "stage", "reason"],
            [[scene, stage, reason] for scene, stage, reason in data.failures],
        )

    lines += ["</body></html>", ""]
    return "\n".join(lines)


def write_html_report(result: RunResult, path: Path) -> None:
    """Write ``report.html`` for one run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(result), encoding="utf-8")
