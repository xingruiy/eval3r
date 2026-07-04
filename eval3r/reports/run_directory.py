"""Run-directory writer: assembles a complete, schema-valid run directory.

Produces the files described in ``.agent/reproducibility.md`` from an
already-computed :class:`RunResult`. Only files with content are written, except
``results.json``, ``environment.json``, and ``logs.txt``, which are always present.

This slice does not execute any pipeline stages or compute metrics — it serializes
results that the runner (task 007) will hand it.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from eval3r.core.protocol import EvalProtocol
from eval3r.core.result import RunResult
from eval3r.reports.csv import write_per_scene_csv, write_results_csv
from eval3r.reports.json import dump_json, write_failures_json, write_run_result_json


def _slug(value: str | None) -> str:
    if not value:
        return "unknown"
    return re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-") or "unknown"


def default_run_dir_name(result: RunResult, *, now: datetime | None = None) -> str:
    """``<timestamp>_<dataset>_<method>`` directory name."""
    now = now or datetime.now(timezone.utc).astimezone()
    stamp = now.strftime("%Y-%m-%d_%H%M%S")
    return f"{stamp}_{_slug(result.dataset.dataset)}_{_slug(result.method)}"


def write_run_directory(
    result: RunResult,
    out_dir: Path,
    *,
    protocol: EvalProtocol | None = None,
    protocol_yaml: str | None = None,
    manifest: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
    backend_versions: dict[str, Any] | None = None,
    alignment_transforms: list[dict[str, Any]] | None = None,
    logs: str | None = None,
) -> Path:
    """Write a complete run directory and return its path.

    ``environment`` / ``backend_versions`` default to the values already on
    ``result`` so a directory is self-describing even without explicit inputs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always-present core files.
    write_run_result_json(result, out_dir / "results.json")
    write_results_csv(result, out_dir / "results.csv")
    write_per_scene_csv(result, out_dir / "per_scene.csv")
    write_failures_json(result, out_dir / "failures.json")

    env = environment if environment is not None else result.environment
    dump_json(env or {}, out_dir / "environment.json")

    backends = backend_versions if backend_versions is not None else result.backend_versions
    dump_json(backends or {}, out_dir / "backend_versions.json")

    (out_dir / "logs.txt").write_text(logs or "", encoding="utf-8")

    # Conditional files.
    if protocol_yaml is not None:
        (out_dir / "protocol.yaml").write_text(protocol_yaml, encoding="utf-8")
    elif protocol is not None:
        (out_dir / "protocol.yaml").write_text(
            yaml.safe_dump(protocol.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )

    if manifest is not None:
        (out_dir / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )

    if config is not None:
        (out_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

    if alignment_transforms is not None:
        dump_json(
            {"alignment_transforms": alignment_transforms},
            out_dir / "alignment_transforms.json",
        )

    return out_dir
