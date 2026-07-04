"""Result reporting and diffing in multiple formats.

Task 006 provides JSON/CSV run-directory output; Markdown/LaTeX/HTML reports and
diffing arrive in task 016.
"""

from __future__ import annotations

from eval3r.reports.csv import (
    PER_SCENE_COLUMNS,
    per_scene_rows,
    write_per_scene_csv,
    write_results_csv,
)
from eval3r.reports.json import (
    dump_json,
    read_run_result_json,
    write_failures_json,
    write_run_result_json,
)
from eval3r.reports.run_directory import default_run_dir_name, write_run_directory

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
]
