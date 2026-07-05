"""Result reporting and diffing in multiple formats.

Task 006 provides JSON/CSV run-directory output; task 016 adds Markdown/LaTeX/HTML
reports (with partial-coverage banners), run diffing, and debug outputs
(error-colored point clouds, distance histograms).
"""

from __future__ import annotations

from eval3r.reports.csv import (
    PER_SCENE_COLUMNS,
    per_scene_rows,
    write_per_scene_csv,
    write_results_csv,
)
from eval3r.reports.diff import RunDiff, diff_runs, load_run_result
from eval3r.reports.html import render_html, write_html_report
from eval3r.reports.json import (
    dump_json,
    read_run_result_json,
    write_failures_json,
    write_run_result_json,
)
from eval3r.reports.latex import render_latex, write_latex_report
from eval3r.reports.markdown import render_markdown, write_markdown_report
from eval3r.reports.plots import (
    write_distance_histogram,
    write_error_colored_pointcloud,
    write_geometry_debug_outputs,
)
from eval3r.reports.run_directory import default_run_dir_name, write_run_directory
from eval3r.reports.table import ReportData, build_report_data

__all__ = [
    "write_run_directory",
    "default_run_dir_name",
    "write_run_result_json",
    "read_run_result_json",
    "write_failures_json",
    "dump_json",
    "write_results_csv",
    "write_per_scene_csv",
    "per_scene_rows",
    "PER_SCENE_COLUMNS",
    "ReportData",
    "build_report_data",
    "render_markdown",
    "write_markdown_report",
    "render_latex",
    "write_latex_report",
    "render_html",
    "write_html_report",
    "write_distance_histogram",
    "write_error_colored_pointcloud",
    "write_geometry_debug_outputs",
    "RunDiff",
    "diff_runs",
    "load_run_result",
]
