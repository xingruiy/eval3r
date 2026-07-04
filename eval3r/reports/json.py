"""JSON result writer/reader.

``results.json`` is the full :class:`RunResult` (self-describing per
``.agent/reproducibility.md``). ``failures.json`` is a coverage-focused view of the
run's scene failures and the failure policy that governed them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval3r.core.result import RunResult


def dump_json(obj: Any, path: Path) -> None:
    """Write ``obj`` as pretty JSON (UTF-8), creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_run_result_json(result: RunResult, path: Path) -> None:
    """Write the full ``RunResult`` to ``results.json``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_run_result_json(path: Path) -> RunResult:
    """Read a ``results.json`` back into a validated ``RunResult``."""
    return RunResult.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_failures_json(result: RunResult, path: Path) -> None:
    """Write a coverage-focused ``failures.json`` for the run's failed scenes."""
    payload = {
        "failure_policy": result.failure_policy.policy,
        "n_scenes_expected": result.n_scenes_expected,
        "n_scenes_evaluated": result.n_scenes_evaluated,
        "n_scenes_failed": len(result.failed_scenes),
        "failed_scenes": [
            {
                "scene_id": f.scene_id,
                "stage": f.stage,
                "reason": f.reason,
                "recoverable": f.recoverable,
                "traceback": f.traceback,
                "policy_action": result.failure_policy.policy,
            }
            for f in result.failed_scenes
        ],
    }
    dump_json(payload, path)
